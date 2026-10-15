"""CPU-only simulation of sd-scripts' reset step counter and state restoration.

These JSON files are deliberately NOT real model/optimizer tensors.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
args = parser.parse_args()
config = tomllib.loads(Path(args.config_file).read_text(encoding="utf-8"))
batches = 126
total = config["max_train_epochs"] * batches
initial = config.get("initial_step", 0)
stop_at = int(os.environ.get("TEST_STOP_AT", total))
assert config["gradient_accumulation_steps"] == 1
optimizer_step = 0
if config.get("resume"):
    assert config["skip_until_initial_step"]
    state = Path(config["resume"])
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
        restored = json.loads((state / name).read_text())
        assert restored["completed"] == initial, (name, restored, initial)
    optimizer_step = initial
    print(f"TEST restored optimizer, scheduler, model, RNG and scaler at {initial}", flush=True)

print(f"num batches per epoch: {batches}", flush=True)
print("gradient accumulation steps = 1", flush=True)
print(f"total optimization steps: {total}", flush=True)
within_epoch = initial % batches
raw_step = within_epoch
raw_total = total - within_epoch
print(f"steps: 0%|fixture| 0/{raw_total}", end="\r", flush=True)
for overall in range(initial + 1, total + 1):
    # The installed trainer resets global_step after skipping full epochs.
    raw_step += 1
    optimizer_step += 1
    assert optimizer_step == overall
    session_step = overall - initial
    print(f"steps: {session_step * 100 // raw_total}%|fixture| {session_step}/{raw_total}", end="\r", flush=True)
    if overall == stop_at:
        state = Path(config["output_dir"]) / f'{config["output_name"]}-step{raw_step:08}-state'
        state.mkdir()
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
            (state / name).write_text(json.dumps({"completed": optimizer_step}), encoding="utf-8")
        (state / "train_state.json").write_text(json.dumps({
            "current_epoch": math.ceil(overall / batches), "current_step": raw_step,
        }), encoding="utf-8")
        print(f"\nRandom states saved in {state / 'random_states_0.pkl'}", flush=True)
        print("TEST CHECKPOINT COMPLETE", flush=True)
        if stop_at < total:
            time.sleep(60)  # The test's owning runner stops this process tree.
            raise RuntimeError("Test runner did not stop at the checkpoint")

(Path(config["output_dir"]) / f'{config["output_name"]}.safetensors').write_text(
    json.dumps({"completed": optimizer_step}), encoding="utf-8",
)
