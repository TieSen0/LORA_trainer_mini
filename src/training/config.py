"""Application-owned SDXL settings and versioned persistence."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import re
import tempfile

from src.dataset.scanner import scan_images


OPTIMIZERS = ("AdamW", "AdamW8bit")
SCHEDULERS = ("constant", "cosine", "linear")
PRECISIONS = ("fp16", "bf16", "no")
ATTENTION_BACKENDS = ("sdpa", "xformers")
SCHEMA_VERSION = 6


@dataclass(frozen=True)
class SDXLTrainConfig:
    model_path: str = ""
    dataset_path: str = ""
    output_dir: str = ""
    output_name: str = "my_lora"
    resolution: int = 1024
    batch_size: int = 1
    epochs: int = 10
    learning_rate: float = 1e-4
    optimizer: str = "AdamW"
    lr_scheduler: str = "constant"
    network_dim: int = 32
    network_alpha: float = 16.0
    mixed_precision: str = "fp16"
    attention_backend: str = "sdpa"
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = False
    gradient_checkpointing: bool = True
    network_train_unet_only: bool = True
    sd_scripts_dir: str = ""
    python_executable: str = ""
    trigger_word: str = ""
    class_token: str = ""
    num_repeats: int = 1
    shuffle_caption: bool = True
    keep_tokens: int = 0
    keep_tokens_auto: bool = True
    offline_mode: bool = False
    save_training_state: bool = True
    save_every_n_steps: int = 0
    resume_state: str = ""
    seed: int = 42
    measure_performance: bool = False


def validate_settings(config: SDXLTrainConfig) -> list[str]:
    """Validate values separately from machine-specific paths (also used on import)."""
    errors = []
    defaults = SDXLTrainConfig()
    for item in fields(config):
        value, default = getattr(config, item.name), getattr(defaults, item.name)
        valid_type = (
            type(value) in (int, float) if type(default) is float
            else type(value) is type(default)
        )
        if not valid_type:
            errors.append(f"Invalid value type for {item.name}.")
    if errors:
        return errors

    if not 0 <= config.save_every_n_steps <= 2147483647:
        errors.append("Checkpoint interval must be 0 (epoch only) or a positive step count.")
    if not -1 <= config.seed <= 2147483647:
        errors.append("Seed must be -1 (random) or between 0 and 2147483647.")
    if config.save_training_state and config.seed < 0:
        errors.append("Choose a fixed seed before saving resumable state.")
    if not 1 <= config.num_repeats <= 2147483647:
        errors.append("Repeats Per Image must be greater than 0 (maximum 2147483647).")
    if not 0 <= config.keep_tokens <= 2147483647:
        errors.append("Keep Tokens must be 0 or greater (maximum 2147483647).")
    for name, maximum in (("batch_size", 1024), ("epochs", 100000), ("network_dim", 1024)):
        if not 1 <= getattr(config, name) <= maximum:
            errors.append(f"{name} must be between 1 and {maximum}.")
    if not 64 <= config.resolution <= 4096 or config.resolution % 64:
        errors.append("Resolution must be a multiple of 64 between 64 and 4096.")
    if not math.isfinite(config.learning_rate) or not 1e-8 <= config.learning_rate <= 1:
        errors.append("Learning rate must be between 0.00000001 and 1.")
    if not math.isfinite(config.network_alpha) or not 0.01 <= config.network_alpha <= 1024:
        errors.append("Network alpha must be between 0.01 and 1024.")
    for label, value, allowed in (
        ("Optimizer", config.optimizer, OPTIMIZERS),
        ("Learning-rate scheduler", config.lr_scheduler, SCHEDULERS),
        ("Mixed precision", config.mixed_precision, PRECISIONS),
        ("Cross-attention", config.attention_backend, ATTENTION_BACKENDS),
    ):
        if value not in allowed:
            errors.append(f"{label} must be one of: {', '.join(allowed)}.")
    if config.cache_text_encoder_outputs and not config.network_train_unet_only:
        errors.append("Caching text-encoder outputs requires U-Net-only training.")
    if config.cache_text_encoder_outputs and config.shuffle_caption:
        errors.append("Shuffle Captions cannot be enabled while caching text-encoder outputs.")

    name = config.output_name
    if (
        not name or name != name.strip() or name.endswith(".")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name in {".", ".."}
        or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name.split(".")[0], re.I)
        or len(name) > 128
    ):
        errors.append("Output name must be a valid Windows filename, up to 128 characters.")
    if name.lower().endswith(".safetensors"):
        errors.append("Enter the output name without .safetensors; the extension is added automatically.")
    return errors


def validate_config(config: SDXLTrainConfig) -> list[str]:
    """Check settings and local paths, without loading a checkpoint or starting training."""
    errors = validate_settings(config)
    if errors:
        return errors
    try:
        model = Path(config.model_path)
        if not config.model_path or not model.is_file():
            errors.append("Choose an existing SDXL base model file.")
        elif model.suffix.lower() not in {".safetensors", ".ckpt"}:
            errors.append("Base model must be a .safetensors or .ckpt file.")
        else:
            with model.open("rb") as file:
                if not file.read(1):
                    errors.append("The base model file is empty.")
    except (OSError, ValueError) as error:
        errors.append(f"Cannot read the base model: {error}")
    try:
        if not config.dataset_path:
            errors.append("Choose a dataset folder.")
        elif not scan_images(config.dataset_path):
            errors.append("The dataset folder contains no supported images directly inside it.")
    except (OSError, ValueError) as error:
        errors.append(f"Cannot scan the dataset folder: {error}")
    try:
        output = Path(config.output_dir)
        if not config.output_dir or not output.is_dir():
            errors.append("Choose an existing output folder (you can create one in the folder picker).")
        elif not os.access(output, os.W_OK):
            errors.append("The output folder is not writable.")
    except (OSError, ValueError) as error:
        errors.append(f"Cannot access the output folder: {error}")
    return errors


def validate_runtime(config: SDXLTrainConfig) -> list[str]:
    errors = []
    if not config.sd_scripts_dir or not (Path(config.sd_scripts_dir) / "sdxl_train_network.py").is_file():
        errors.append("Choose the sd-scripts folder containing sdxl_train_network.py.")
    if not config.python_executable or not Path(config.python_executable).is_file():
        errors.append("Choose the Python executable from your prepared sd-scripts environment.")
    if config.resume_state:
        from src.training.checkpoints import validate_resume
        errors.extend(validate_resume(config))
    return errors


def save_config(config: SDXLTrainConfig, destination: Path) -> None:
    """Save a validated app configuration atomically, never a training command."""
    errors = validate_config(config)
    if errors:
        raise ValueError("\n".join(errors))
    if destination.suffix.lower() != ".json":
        raise ValueError("Training settings must be saved as a .json file.")
    config = replace(config, **{
        name: str(Path(getattr(config, name)).resolve())
        for name in ("model_path", "dataset_path", "output_dir", "sd_scripts_dir", "python_executable")
        if getattr(config, name)
    })
    payload = {"schema_version": SCHEMA_VERSION, "training_type": "sdxl_lora", "settings": asdict(config)}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=".training-config-", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def config_payload(config: SDXLTrainConfig) -> dict:
    return {"schema_version": SCHEMA_VERSION, "training_type": "sdxl_lora", "settings": asdict(config)}


def load_config(source: Path) -> SDXLTrainConfig:
    """Reject malformed/unknown settings; allow unavailable paths to be corrected in the UI."""
    return parse_config(json.loads(source.read_text(encoding="utf-8")))


def parse_config(payload) -> SDXLTrainConfig:
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") not in (1, 2, 3, 4, 5, SCHEMA_VERSION):
        raise ValueError("Unsupported training configuration version.")
    if payload.get("training_type") != "sdxl_lora":
        raise ValueError("Only SDXL LoRA configurations are supported.")
    settings = payload.get("settings")
    if isinstance(settings, dict) and payload["schema_version"] == 1:
        # Settings saved before the selector was introduced use the new default.
        settings = {"attention_backend": "sdpa", **settings}
    if isinstance(settings, dict) and payload["schema_version"] < 3:
        settings = {"sd_scripts_dir": "", "python_executable": "", **settings}
    if isinstance(settings, dict) and payload["schema_version"] < 4:
        # The new shuffle default is incompatible with the old text-cache default.
        # Migrate only the legacy default, never silently repair a new invalid config.
        if "shuffle_caption" not in settings and settings.get("cache_text_encoder_outputs") is True:
            settings = {**settings, "cache_text_encoder_outputs": False}
        settings = {
            "trigger_word": "", "class_token": "", "num_repeats": 1,
            "shuffle_caption": True, "keep_tokens": 0, "keep_tokens_auto": True,
            **settings,
        }
    if isinstance(settings, dict) and payload["schema_version"] < 5:
        settings = {"offline_mode": False, **settings}
    if isinstance(settings, dict) and payload["schema_version"] < 6:
        settings = {
            "save_training_state": False, "save_every_n_steps": 0, "resume_state": "",
            "seed": -1, "measure_performance": False, **settings,
        }
    if not isinstance(settings, dict) or set(settings) != {item.name for item in fields(SDXLTrainConfig)}:
        raise ValueError("The configuration has missing or unknown settings.")
    config = SDXLTrainConfig(**settings)
    errors = validate_settings(config)
    if errors:
        raise ValueError("\n".join(errors))
    return config
