"""Translate app settings into sd-scripts TOML and a shell-free argument list."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from src.training.config import SDXLTrainConfig, save_config, validate_config, validate_runtime
from src.training.checkpoints import inspect_checkpoint


def _toml(value) -> str:
    # JSON basic strings, numbers and booleans are also valid TOML scalar values.
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def training_environment(config: SDXLTrainConfig) -> dict[str, str]:
    """Explicit child-only overrides, also recorded for reproducing an exported run."""
    value = "1" if config.offline_mode else "0"
    return {"HF_HUB_OFFLINE": value, "TRANSFORMERS_OFFLINE": value}


def dataset_toml(config: SDXLTrainConfig) -> str:
    return "\n".join([
        "[general]", 'caption_extension = ".txt"',
        f"shuffle_caption = {_toml(config.shuffle_caption)}",
        f"keep_tokens = {config.keep_tokens}", "",
        "[[datasets]]", f"resolution = {config.resolution}",
        f"batch_size = {config.batch_size}", "enable_bucket = true",
        "bucket_reso_steps = 64", "min_bucket_reso = 64",
        f"max_bucket_reso = {max(1024, config.resolution * 2)}", "",
        "[[datasets.subsets]]",
        f"image_dir = {_toml(str(Path(config.dataset_path).resolve()))}",
        f"num_repeats = {config.num_repeats}", "",
    ])


def training_toml(config: SDXLTrainConfig, run_dir: Path) -> str:
    values = {
        "pretrained_model_name_or_path": str(Path(config.model_path).resolve()),
        "dataset_config": str(run_dir / "dataset.toml"),
        "output_dir": str(run_dir), "output_name": config.output_name,
        "save_model_as": "safetensors", "network_module": "networks.lora",
        "network_dim": config.network_dim, "network_alpha": config.network_alpha,
        "learning_rate": config.learning_rate, "optimizer_type": config.optimizer,
        "lr_scheduler": config.lr_scheduler, "max_train_epochs": config.epochs,
        "mixed_precision": config.mixed_precision,
        "cache_latents": config.cache_latents,
        "cache_latents_to_disk": config.cache_latents_to_disk,
        "cache_text_encoder_outputs": config.cache_text_encoder_outputs,
        "gradient_checkpointing": config.gradient_checkpointing,
        "network_train_unet_only": config.network_train_unet_only,
        config.attention_backend: True,
        "save_every_n_epochs": 1,
        "max_data_loader_n_workers": 0,  # Conservative Windows process/memory usage.
        "gradient_accumulation_steps": 1,
        "no_half_vae": True,
        "save_state": config.save_training_state,
        "save_state_on_train_end": config.save_training_state,
    }
    if config.seed >= 0:
        values["seed"] = config.seed
    if config.save_every_n_steps:
        values["save_every_n_steps"] = config.save_every_n_steps
    if config.resume_state:
        values["resume"] = str(Path(config.resume_state).resolve())
        values["skip_until_initial_step"] = True
        # Restore optimizer/model state normally, but override the trainer's
        # possibly epoch-relative saved counter with the verified overall step.
        values["initial_step"] = inspect_checkpoint(Path(config.resume_state))["current_step"]
    return "\n".join(f"{name} = {_toml(value)}" for name, value in values.items()) + "\n"


@dataclass(frozen=True)
class TrainingRun:
    directory: Path
    command: tuple[str, ...]
    working_directory: Path
    expected_model: Path


def build_command(config: SDXLTrainConfig, run_dir: Path) -> tuple[str, ...]:
    # Direct Python launch is single-process; sd-scripts uses Accelerator internally.
    return (
        str(Path(config.python_executable).resolve()), "-u",
        str(Path(config.sd_scripts_dir).resolve() / "sdxl_train_network.py"),
        "--config_file", str(run_dir.resolve() / "training.toml"),
    )


def create_training_run(config: SDXLTrainConfig) -> TrainingRun:
    errors = validate_config(config) + validate_runtime(config)
    if errors:
        raise ValueError("\n".join(errors))
    # Every run owns a new folder: configs, logs and checkpoints never overwrite old runs.
    run_dir = Path(tempfile.mkdtemp(prefix=f"{config.output_name}-", dir=Path(config.output_dir).resolve()))
    (run_dir / "dataset.toml").write_text(dataset_toml(config), encoding="utf-8")
    (run_dir / "training.toml").write_text(training_toml(config, run_dir), encoding="utf-8")
    save_config(config, run_dir / "settings.json")
    command = build_command(config, run_dir)
    (run_dir / "command.json").write_text(
        json.dumps({
            "argv": command, "cwd": str(Path(config.sd_scripts_dir).resolve()),
            "environment": training_environment(config),
        }, indent=2), encoding="utf-8",
    )
    return TrainingRun(run_dir, command, Path(config.sd_scripts_dir).resolve(), run_dir / f"{config.output_name}.safetensors")
