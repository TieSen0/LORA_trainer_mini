"""Test-only process: exercises the runner contract without GPU work or weights."""

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
args = parser.parse_args()
config = tomllib.loads(Path(args.config_file).read_text(encoding="utf-8"))
assert config.get("sdpa", False) != config.get("xformers", False)
dataset = tomllib.loads(Path(config["dataset_config"]).read_text(encoding="utf-8"))
assert Path(dataset["datasets"][0]["subsets"][0]["image_dir"]).is_dir()
print("Synthetic training fixture: 中文", flush=True)
print("Synthetic stderr message", file=sys.stderr, flush=True)
if config["output_name"] == "fake_fail":
    raise SystemExit(7)
resume_step = 0
if config.get("resume"):
    assert (Path(config["resume"]) / "optimizer.bin").is_file()
    assert config["skip_until_initial_step"] is True
    print("resume training from local state", flush=True)
    resume_step = config["initial_step"]
print("num batches per epoch: 1", flush=True)
print("gradient accumulation steps = 1", flush=True)
print("total optimization steps / 学習ステップ数: 2", flush=True)
if config.get("save_state"):
    state = Path(config["output_dir"]) / f'{config["output_name"]}-step00000001-state'
    state.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
        (state / name).write_bytes(b"TEST STATE ONLY")
    (state / "train_state.json").write_text(json.dumps({"current_step": 1, "current_epoch": 1}), encoding="utf-8")
    print(f"Random states saved in {state / 'random_states_0.pkl'}", flush=True)
for step in range(3 - resume_step):
    print(f"steps: {step * 50}%|fixture| {step}/2 [00:00<00:00]", end="\r", flush=True)
    time.sleep(0.02)
if config["output_name"] != "no_output":
    (Path(config["output_dir"]) / f'{config["output_name"]}.safetensors').write_bytes(
        b"TEST FIXTURE ONLY - not a real LoRA model"
    )
