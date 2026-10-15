"""Read-only desktop/runtime checks, usable even when Qt cannot import."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.training.environment import training_probe


def probe_process(python, code, cwd):
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run([str(python), "-u", "-c", code], cwd=str(cwd), env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=45, **options)
        detail = (result.stdout + result.stderr).strip()
        return result.returncode == 0, detail or f"Exit code: {result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def collect_report(config=None, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    checks = []

    def add(status, title, detail):
        checks.append({"status": status, "title": title, "detail": detail})

    add("OK" if sys.version_info >= (3, 12) else "FAIL", "Desktop Python", f"{sys.executable}\n{sys.version}")
    imports = (
        "from PySide6 import QtWidgets; import PIL, numpy, onnxruntime, huggingface_hub, psutil; "
        "from importlib.metadata import version; "
        "print('\\n'.join(p + ': ' + version(p) for p in "
        "('PySide6', 'Pillow', 'numpy', 'onnxruntime', 'huggingface-hub', 'psutil')))"
    )
    ok, detail = probe_process(sys.executable, imports, root)
    add("OK" if ok else "FAIL", "Desktop dependency imports", detail + ("" if ok else "\nClose the app and run Setup.cmd; use Repair.cmd if imports still fail."))
    add("OK" if (root / "uv.lock").is_file() else "FAIL", "Locked dependencies", "uv.lock found." if (root / "uv.lock").is_file() else "Restore uv.lock from the same source release; do not guess dependency versions.")
    model_dir = root / "models" / "wd14"
    files = [model_dir / name for name in ("model.onnx", "selected_tags.csv")]
    try:
        found = [p.is_file() and p.stat().st_size > 0 for p in files]
    except OSError:
        found = [False, False]
    add("OK" if all(found) else "WARN", "Optional WD14 files",
        "Both files found (model/tag-list compatibility is checked during tagging)." if all(found)
        else "Matching, readable model.onnx and selected_tags.csv are needed in models/wd14 to use local tagging. Caption editing does not require them.")
    if config is None:
        add("WARN", "Training runtime", "Not configured. Select sd-scripts and its Python in Training Setup, then use Check Environment.")
    else:
        trainer = Path(config.sd_scripts_dir)
        python = Path(config.python_executable)
        if not config.sd_scripts_dir or not (trainer / "sdxl_train_network.py").is_file() or not config.python_executable or not python.is_file():
            add("FAIL", "Training runtime paths", "Select a valid sd-scripts folder and its separate Python executable in Environment. Desktop Setup does not install PyTorch or sd-scripts.")
        else:
            ok, detail = probe_process(python, training_probe(config), trainer)
            add("OK" if ok else "FAIL", "Training imports / CUDA", detail + ("" if ok else "\nRepair the selected sd-scripts environment using its own installation instructions. Do not install training dependencies into the desktop .venv."))
        for title, value, kind in (("Base model", config.model_path, "file"), ("Dataset", config.dataset_path, "dir"), ("Output", config.output_dir, "dir")):
            path = Path(value)
            exists = bool(value) and (path.is_file() if kind == "file" else path.is_dir())
            add("OK" if exists else "WARN", title, value if exists else "Not selected or unavailable. Locate it in Training Setup.")
            if exists and kind == "dir":
                try:
                    free = shutil.disk_usage(path).free / 1024**3
                    add("WARN" if free < 10 else "OK", title + " free disk space", f"{free:.1f} GiB free. Space required depends on caches, models, and checkpoint count; this is not a capacity guarantee.")
                except OSError as error:
                    add("WARN", title + " free disk space", str(error))
        if config.offline_mode:
            add("WARN", "Offline tokenizer cache", "Not verified by this check. Training needs locally cached tokenizers; no network requests or model loads were made.")
    return checks


def format_report(checks):
    return "Mini LoRA Trainer — Environment check\nNo training, downloads, installs, or repairs performed.\nThis report contains local paths; review before sharing.\n\n" + "\n\n".join(
        f"[{item['status']}] {item['title']}\n{item['detail']}" for item in checks
    ) + "\n\nPassing checks does not guarantee sufficient memory, valid model architecture, or a successful training run.\n"
