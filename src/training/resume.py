"""Prepare an explicitly selected checkpoint without starting a trainer."""

from dataclasses import replace
import json
from pathlib import Path

from src.training.checkpoints import inspect_checkpoint, list_checkpoints, read_step_layout, validate_resume
from src.training.config import load_config


def resume_progress(state):
    state = Path(state)
    info = inspect_checkpoint(state)
    config = load_config(state.parent / "settings.json")
    total = None
    record_path = state.parent / "run.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not isinstance(record.get("metrics", {}), dict):
            raise ValueError("Invalid run metadata; cannot determine resume progress.")
        candidate = record.get("metrics", {}).get("total")
        if type(candidate) is int and candidate > 0:
            total = candidate
    if total is None:
        try:
            total = read_step_layout(state.parent)["steps_per_epoch"] * config.epochs
        except ValueError:
            pass  # Old runs may have no total; never invent a remaining count.
    step = info["current_step"]
    if total is not None:
        return f"Saved step {step} / {total} · Remaining steps: {max(0, total - step)}"
    return f"Saved step {step} · Target: {config.epochs} epochs (step total not recorded)"


def load_resume_config(state):
    """All entry points restore the selected state's own settings, not GUI drafts."""
    state = Path(state).resolve()
    inspect_checkpoint(state)
    config = replace(load_config(state.parent / "settings.json"), resume_state=str(state))
    errors = validate_resume(config)
    if errors:
        raise ValueError("\n".join(errors))
    return config


def load_resume_from_folder(folder):
    """Choose the highest-step complete state within one explicitly chosen run.

    Never choose another run or silently rewind a completed/incompatible run.
    An explicit state folder remains supported for existing manual workflows.
    """
    folder = Path(folder).resolve()
    if not (folder / "settings.json").is_file():
        if folder.name.endswith("-state") or (folder / "train_state.json").is_file():
            return load_resume_config(folder)
        raise ValueError("Choose one LoRA run folder containing settings.json and its saved state folders, not the output root or a model file.")
    # Validate settings before discovery: malformed metadata must not look like
    # a run which simply has no checkpoints.
    load_config(folder / "settings.json")
    record_path = folder / "run.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Invalid run metadata.")
        if record.get("status") == "success":
            raise ValueError("This run is already complete. Quick resume will not restart it from an earlier checkpoint.")
    states = list_checkpoints(folder)
    if not states:
        raise ValueError("This run has no complete saved state. Resume needs a state folder, not just LoRA weights.")
    # list_checkpoints sorts by recovered overall step, not filename or mtime.
    # Validation failure of the latest complete state must not silently fall
    # back to older training progress.
    return load_resume_config(states[0])
