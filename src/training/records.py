"""Small atomic JSON records; never load pickle files in the GUI."""

import json
import os
from pathlib import Path
import tempfile


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".record-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_session_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "MiniLoRATrainer" / "session.json"


def load_session(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported remembered-settings file; it has been left unchanged.")
    if not isinstance(value.get("dataset", ""), str):
        raise ValueError("Invalid remembered dataset path.")
    prefixes = value.get("prefixes", {})
    if not isinstance(prefixes, dict) or any(
        not isinstance(k, str) or not isinstance(v, list) or len(v) != 2
        or any(not isinstance(tag, str) for tag in v) for k, v in prefixes.items()
    ):
        raise ValueError("Invalid remembered caption prefixes.")
    return value
