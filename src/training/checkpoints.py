"""Conservative checks for local single-GPU Accelerate resume state."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.dataset.scanner import scan_images
from src.training.log_parser import TrainingLogParser


def read_step_layout(run_dir: Path) -> dict:
    """Use durable run metadata, or read older runs' logs without loading tensors."""
    record_path = run_dir / "run.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Run history must contain a JSON object.")
        layout = record.get("step_layout")
        if layout is not None:
            if (not isinstance(layout, dict) or layout.get("version") != 1
                    or type(layout.get("steps_per_epoch")) is not int or layout["steps_per_epoch"] <= 0):
                raise ValueError("Run history has invalid steps-per-epoch metadata.")
            return layout
    log_path = run_dir / "training.log"
    if log_path.is_file():
        parser = TrainingLogParser()
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                parser.feed(line)
                if parser.step_layout:
                    return parser.step_layout
    raise ValueError(
        "Cannot determine this resumed checkpoint's overall position. Keep its run.json "
        "or original training.log (including batches per epoch and gradient accumulation) "
        "beside settings.json. Only single-GPU, accumulation-1 runs are supported."
    )


def dataset_signature(folder: str) -> str:
    """Detect changed pairing/content cheaply; image bytes are not rehashed."""
    digest = hashlib.sha256()
    for image in scan_images(folder):
        stat = image.stat()
        caption = image.with_suffix(".txt")
        digest.update(f"{image.name}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("utf-8"))
        digest.update(caption.read_bytes() if caption.exists() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    if not path.is_dir():
        raise ValueError("Choose a training-state folder, not a LoRA .safetensors file.")
    required = ["optimizer.bin", "scheduler.bin", "random_states_0.pkl", "train_state.json"]
    if not any((path / name).is_file() and (path / name).stat().st_size
               for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("State folder is missing its saved network weights.")
    for name in required:
        if not (path / name).is_file() or not (path / name).stat().st_size:
            raise ValueError(f"State folder is incomplete: missing or empty {name}.")
    metadata = json.loads((path / "train_state.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(
        type(metadata.get(key)) is not int or metadata[key] < 0
        for key in ("current_step", "current_epoch")
    ):
        raise ValueError("State folder has invalid training-position metadata.")
    settings_path = path.parent / "settings.json"
    if settings_path.is_file():
        from src.training.config import load_config
        source = load_config(settings_path)
        if source.resume_state:
            # sd-scripts can save a step relative to the last skipped epoch, but
            # current_epoch remains absolute. Whole-epoch offsets do not change
            # the within-epoch position. At an epoch end, modulo must yield B,
            # not zero. This works for both reset and absolute saved counters.
            steps_per_epoch = read_step_layout(path.parent)["steps_per_epoch"]
            raw_step, epoch = metadata["current_step"], metadata["current_epoch"]
            if raw_step < 1 or epoch < 1:
                raise ValueError("Resumed checkpoint has no completed training step.")
            overall = (epoch - 1) * steps_per_epoch + (raw_step - 1) % steps_per_epoch + 1
            if raw_step > overall:
                raise ValueError("Saved step and epoch disagree with this run's steps per epoch.")
            metadata = {**metadata, "raw_current_step": raw_step, "current_step": overall,
                        "steps_per_epoch": steps_per_epoch}
    return metadata


def list_checkpoints(run_dir: Path) -> list[Path]:
    result = []
    for path in run_dir.glob("*-state"):
        try:
            info = inspect_checkpoint(path)
            result.append((info["current_step"], path))
        except (OSError, ValueError):
            continue
    return [path for _, path in sorted(result, key=lambda item: (item[0], item[1].name), reverse=True)]


def validate_resume(config) -> list[str]:
    from src.training.config import load_config
    try:
        state = Path(config.resume_state).resolve()
        metadata = inspect_checkpoint(state)
        if config.mixed_precision == "fp16" and (
            not (state / "scaler.pt").is_file() or not (state / "scaler.pt").stat().st_size
        ):
            return ["FP16 resume state is missing or has an empty scaler.pt."]
        source = load_config(state.parent / "settings.json")
        if source.resume_state and metadata["current_step"] >= source.epochs * metadata["steps_per_epoch"]:
            return ["This checkpoint has reached the original training target. Select an earlier checkpoint; extending completed runs is not supported yet."]
        allowed = {
            "output_dir", "output_name", "resume_state", "offline_mode", "save_training_state",
            "save_every_n_steps", "measure_performance", "sd_scripts_dir", "python_executable",
            "trigger_word", "class_token", "keep_tokens_auto",
        }
        differences = [key for key, value in asdict(source).items()
                       if key not in allowed and getattr(config, key) != value]
        if differences:
            return ["Resume must use the original training settings. Different: " + ", ".join(differences)
                    + ". Load the run's settings from History first."]
        record_path = state.parent / "run.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            start = record.get("resume_start_step")
            if type(start) is int and metadata["current_step"] < start:
                return ["Checkpoint position is earlier than this run's starting position. Check its saved epoch/step metadata."]
            total = record.get("metrics", {}).get("total", 0)
            if type(total) is int and total > 0 and metadata["current_step"] >= total:
                return ["This checkpoint has reached the original training target. Select an earlier checkpoint; extending completed runs is not supported yet."]
            signature = record.get("dataset_signature")
            if signature and signature != dataset_signature(config.dataset_path):
                return ["Dataset images or captions changed since this checkpoint. Restore the original dataset before resuming."]
            model_stat = Path(config.model_path).stat()
            if record.get("model_signature") and record["model_signature"] != [model_stat.st_size, model_stat.st_mtime_ns]:
                return ["The base model file changed since this checkpoint. Restore the original base model."]
        if state.name == f"{source.output_name}-state":
            return ["This is an end-of-training state. Select an earlier interrupted-run checkpoint; extending completed runs is not supported yet."]
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return [f"Cannot resume: {error}. Use a complete state folder inside a run with settings.json."]
    return []
