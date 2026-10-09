"""Read run records without changing historical outputs."""

from dataclasses import asdict
import json
from pathlib import Path

from src.training.config import load_config


def read_run(directory: Path) -> dict:
    record_path = directory / "run.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Invalid run record")
        if not isinstance(record.get("metrics", {}), dict):
            raise ValueError("Invalid run metrics")
    else:
        # Older runs have no structured record. Do not claim that absence means failure.
        log_path = directory / "training.log"
        tail = ""
        if log_path.exists():
            with log_path.open("rb") as file:
                file.seek(max(0, log_path.stat().st_size - 8192))
                tail = file.read().decode("utf-8", errors="replace")
        status = "unknown (legacy run)"
        for marker, label in (("Training complete:", "success"), ("Training stopped.", "stopped"),
                              ("Training failed", "failed"), ("training has not started", "exported")):
            if marker in tail:
                status = label
        record = {"status": status, "message": "Legacy run; open its log for details."}
    return {**record, "directory": str(directory)}


def list_runs(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    result = []
    for directory in root.iterdir():
        if directory.is_dir() and (directory / "settings.json").is_file():
            try:
                result.append(read_run(directory))
            except (OSError, ValueError):
                result.append({"directory": str(directory), "status": "unreadable record", "message": "Open the run folder to inspect it."})
    return sorted(result, key=lambda item: Path(item["directory"]).stat().st_mtime, reverse=True)


def compare_runs(baseline: Path, candidate: Path) -> str:
    first, second = read_run(baseline), read_run(candidate)
    config_a, config_b = load_config(baseline / "settings.json"), load_config(candidate / "settings.json")
    ignored = {"output_dir", "output_name", "measure_performance", "offline_mode", "resume_state"}
    changes = [f"{key}: {value} → {getattr(config_b, key)}" for key, value in asdict(config_a).items()
               if key not in ignored and value != getattr(config_b, key)]
    lines = [f"Baseline: {baseline.name}", f"Candidate: {candidate.name}", "", "Settings changed:", *(changes or ["None"])]
    for label, record in (("Baseline", first), ("Candidate", second)):
        metrics = record.get("metrics", {})
        rate, memory = metrics.get("steps_per_second"), metrics.get("peak_gpu_memory_mib")
        lines.append(f"{label}: {rate:.3f} steps/s" if isinstance(rate, (float, int)) else f"{label}: insufficient step timing")
        lines.append(f"  Sampled GPU 0 board-wide peak: {memory:.0f} MiB" if isinstance(memory, (float, int)) else "  GPU memory: not measured")
    lines.extend(["", "Interpretation:"])
    same_data = first.get("dataset_signature") and first.get("dataset_signature") == second.get("dataset_signature")
    comparable = (same_data and first.get("status") == second.get("status") == "success"
                  and config_a.model_path == config_b.model_path
                  and first.get("model_signature") == second.get("model_signature")
                  and config_a.batch_size == config_b.batch_size and config_a.resolution == config_b.resolution
                  and not config_a.resume_state and not config_b.resume_state
                  and config_a.seed >= 0 and config_a.seed == config_b.seed)
    a, b = first.get("metrics", {}).get("steps_per_second"), second.get("metrics", {}).get("steps_per_second")
    if comparable and isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0:
        lines.append(f"Observed step-rate ratio: {b / a:.2f}x. This is not a quality score or guaranteed speedup.")
    else:
        lines.append("No speedup conclusion: missing timing, incomplete/resumed runs, or different data/batch/resolution/seed.")
    lines.append("Timing excludes the first five reported steps but can include checkpoint/validation overhead. GPU memory includes other apps. Change one setting at a time, repeat measurements, and compare output quality separately in ComfyUI. No settings are changed by this comparison.")
    return "\n".join(lines)
