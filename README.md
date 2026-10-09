# Mini LoRA Trainer

**Prepare your dataset. Write captions. Train an SDXL LoRA. Resume when you need to.**

Mini LoRA Trainer is a Windows desktop app that brings image browsing, WD14
captioning, dataset checks, and SDXL LoRA training into one guided workflow.
It uses a separate [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
installation for training rather than implementing another training engine.

The goal is a focused, approachable interface for local, single-GPU training—not
a replacement for every feature in Kohya. You can also use the dataset and caption
tools without installing the GPU training environment.

**Status:** early-stage, source-run desktop application, version 0.1.0. Successful
GPU training, ComfyUI use, and an interrupted-run resume have been reported by the
project user. Repeated resume has automated simulation coverage; it has not yet
been verified with a new end-to-end GPU experiment. There is no bundled installer.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Guided walkthrough](#guided-walkthrough)
- [Checkpoint saving and repeated resume](#checkpoint-saving-and-repeated-resume)
- [Offline mode](#offline-mode)
- [Settings, history, and performance](#settings-history-and-performance)
- [Files and storage](#files-and-storage)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Development](#development)
- [Licensing and acknowledgments](#licensing-and-acknowledgments)

## Features

| Area | What the app provides |
| --- | --- |
| Image browsing | Choose a folder; preview PNG, JPG, JPEG, and WebP images; navigate with Previous/Next or a position slider. |
| Caption editing | Load, edit, and save matching `.txt` captions; automatically save changed captions when navigating away or closing. |
| WD14 tagging | Generate captions for the current image or the whole dataset, with a confidence threshold and excluded tags. |
| Caption prefixes | Apply a Trigger Word and optional Class Token to existing captions without duplicating those prefix tags. |
| Batch renaming | Rename images and captions together with a filename prefix, starting number, zero-padding, conflict checks, and completion feedback. |
| Dataset review | Find corrupt images, missing or empty captions, unreadable files, duplicate image stems, and orphan captions. |
| Training setup | Configure repeats, epochs, resolution, batch size, learning rate, rank/alpha, optimizer, scheduler, precision, attention, and memory options. |
| Local training | Export training files or launch the selected sd-scripts environment in the background; view logs and stop the owned process. |
| Recovery | Save training states and resume interrupted runs, including resuming again from a resumed run's checkpoint. |
| Monitoring | Display observed startup phases, overall steps, elapsed time, speed, estimated remaining time, and inactivity warnings. |
| Settings and history | Remember drafts, save/load configuration files, browse previous runs, prepare resumes, and compare recorded performance. |

Supported training choices include AdamW/AdamW8bit, constant/cosine/linear
scheduling, fp16/bf16/full precision, and SDPA/xFormers. Availability depends on
the selected training environment and GPU.

## Requirements

There are **two separate Python environments**. Keeping them separate avoids
mixing desktop dependencies with the trainer's PyTorch/CUDA dependencies.

| Component | Required for | Requirements |
| --- | --- | --- |
| Desktop app | Browsing, caption editing, WD14, configuration, history | Windows, [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12 as selected by `.python-version`. |
| Training environment | Running SDXL LoRA training | A separately prepared sd-scripts installation, its compatible Python/PyTorch dependencies, and one supported NVIDIA CUDA GPU. |
| Model and data | Tagging or training | Matching WD14 model/tag-list files for tagging; a compatible SDXL base model and image-caption dataset for training. |

The default app dependencies use CPU ONNX Runtime for WD14. Caption preparation
does not require a CUDA GPU. Training VRAM requirements depend on your model,
resolution, batch size, and memory options; the app does not promise a minimum
that works for every setup. Leave enough disk space for model weights, caches,
and multiple checkpoint states.

## Installation

### 1. Get the source and install uv

Clone this repository using the URL from GitHub's **Code** button, or choose
**Download ZIP** and extract it. Open PowerShell in the folder containing `app.py`
and `pyproject.toml`.

If uv is not installed, one Windows option is:

```powershell
winget install --id=astral-sh.uv -e
```

Reopen PowerShell if needed, then check `uv --version`. See the
[official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for other installation methods.

### 2. Install and launch the desktop app

Run these commands from the repository root:

```powershell
uv python install 3.12
uv sync --locked
uv run app.py
```

`uv sync` installs the GUI and captioning dependencies into this project's
`.venv`. It does **not** install sd-scripts, PyTorch/CUDA, base-model weights,
or optional training packages. Initial dependency/model downloads require network
access; later training can use [Offline Mode](#offline-mode).

Always run the app from this source checkout. A standalone wheel or executable
distribution is not currently supported. To launch it again, use `uv run app.py`.

### 3. Prepare local folders and model files

Optional convenience folders can be created from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path models\wd14, models\sdxl, datasets, output
```

- **WD14:** place `model.onnx` and its matching `selected_tags.csv` in
  `models/wd14/`. Use the app-compatible 448-pixel WD14 ONNX model format.
  The default model is
  [SmilingWolf/wd-v1-4-convnext-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2).
  Both files must come from the same model release. If neither is present, the
  app downloads the default pair on first use. If only one is present, it stops
  with an error rather than mixing models.
- **SDXL:** put your base `.safetensors` or `.ckpt` in `models/sdxl/`, or keep it
  elsewhere and browse to it in Training Setup. A base model is not included.
- **Dataset/output:** these can be anywhere accessible. Choose an existing output
  directory in Training Setup. The folders above are conveniences, not mandatory
  locations.

Review the licenses and usage conditions of models and datasets you obtain.

### 4. Prepare the separate training environment

Follow the installation instructions for your chosen
[sd-scripts revision](https://github.com/kohya-ss/sd-scripts), including its supported
Python version, compatible CUDA-enabled PyTorch build, and dependencies. The
trainer's Python version does not have to match the app's Python 3.12.

In **Training Setup**, provide:

| Field | Example location |
| --- | --- |
| sd-scripts folder | `D:\Tools\sd-scripts` — must contain `sdxl_train_network.py`. |
| Training Python executable | `D:\Tools\sd-scripts\venv\Scripts\python.exe` — select the interpreter actually prepared for training. |

These paths are examples, not required folder names. Some installations use
`.venv` instead of `venv`.

Training performs an environment check for PyTorch, Accelerate, safetensors,
CUDA, and selected optional dependencies. AdamW8bit needs bitsandbytes; xFormers
needs a compatible xformers installation; bf16 requires GPU support. The app
does not install these packages automatically.

## Guided walkthrough

This example prepares a character dataset and trains a LoRA named `my_character`.
Example values illustrate the controls; they are not universal quality settings.

### Step 1 — Open your dataset

Click **Choose Dataset Folder** and select the folder directly containing your
images. The app does not scan subfolders.

```text
datasets/my_character/
├── image_001.png
├── image_001.txt
├── image_002.jpg
└── image_002.txt
```

Captions pair with images by filename stem. Missing `.txt` files can be created
by saving a caption or running WD14. Use **Previous**, **Next**, or the slider to
browse; the position label shows, for example, `13 / 120`.

### Step 2 — Write or generate captions

You can type directly into the caption editor and click **Save**, or generate
tags using WD14. Set a distinctive **Trigger Word**, for example `example_character`,
and optionally a **Class Token**, such as `1girl`.

| Control | What it does |
| --- | --- |
| Apply to Current | Adds the prefix to the current editor text, including unsaved edits. It does not run WD14. |
| Apply to All | Adds the prefix to dataset captions, preserving unrelated existing tags. It does not run WD14. |
| WD14 Tag Current | Generates a replacement caption for one image, then adds the prefix. |
| WD14 Tag All | Generates replacement captions for the entire dataset, then adds the prefix. |

The preview `example_character, 1girl` shows the prefix that will be applied.
Changing those fields alone does not change caption files. Applying a prefix
again does not duplicate its tags. If you change the trigger, remove any unwanted
old trigger yourself; the app does not guess which existing tags to delete.

**WD14 threshold** filters tags by confidence. **Exclude tags** accepts
comma-separated predictions to omit, such as `watermark, signature`. Explicitly
entered prefix tags are added afterward. Review generated captions yourself:
automatic tags can be inaccurate.

WD14 replacement asks for confirmation when appropriate; batch tagging always
asks. Editing and navigation are temporarily locked while tagging. Back up
important captions before batch replacement; there is no caption-version history.

### Step 3 — Optionally rename and review

Use **Batch Rename Dataset** to create names such as `character_001.png` with
matching `character_001.txt`. The filename prefix is separate from caption
Trigger Word/Class Token. Padding `3` means a three-digit number such as `001`.
Conflicting targets are rejected before renaming, and completion reports counts.

Click **Review Dataset**. Resolve corrupt images, missing/empty captions,
unreadable files, and images sharing a caption stem. Orphan captions are warnings.
Review is read-only and checks file integrity, not caption quality.

### Step 4 — Configure training

Open **Training Setup** and select your SDXL base model, dataset, output folder,
sd-scripts folder, and training Python. Enter `my_character` as the output name,
without an extension.

The editable defaults include:

| Setting | Default |
| --- | --- |
| Resolution / batch size | 1024 / 1 |
| Epochs / repeats per image | 10 / 1 |
| Learning rate | 0.0001 |
| LoRA rank / alpha | 32 / 16 |
| Optimizer / scheduler | AdamW / constant |
| Precision / attention | fp16 / SDPA |
| Caption shuffling | Enabled |
| Latent caching / gradient checkpointing | Enabled |
| Text-encoder output caching | Disabled |
| Training scope | U-Net only |
| Seed / resumable-state saving | 42 / enabled for new settings |

Important relationships:

- **Repeats Per Image** is set here. Dataset names such as `2_role 1girl` do not
  control repeats, trigger words, or class tokens. The app generates a dataset
  TOML that explicitly specifies these training options.
- **Keep Tokens** preserves leading comma-separated caption tags during
  shuffling. The example prefix suggests `2`; this counts tags, not CLIP subword
  tokens. Manual edits override the automatic suggestion until you re-enable it.
- Prefix fields in Training Setup are shared settings, not a second caption
  injection. Use Apply or WD14 in the caption window to write the prefix to files.
- Text-encoder output caching requires U-Net-only training and cannot be combined
  with caption shuffling. The interface enforces these constraints.
- Use a fixed seed for resumable training. Older saved configurations may retain
  random seed behavior and disabled state saving; check them explicitly.

### Step 5 — Validate, export, or train

1. Click **Check Settings** and resolve reported problems.
2. Use **Save Settings...** to keep a reusable JSON configuration.
3. Optionally click **Export Training Files** to inspect the generated TOMLs and
   command without starting training.
4. Click **Start Training**. The app reviews the dataset again and checks the
   selected training environment before launching.
5. Follow the **Training Log** tab for loading, caching, step progress, and errors.

Model loading and caching can take time. An indeterminate progress bar during
startup does not mean the app has frozen. Once steps arrive, the app reports
overall progress and eventually a speed/ETA estimate. A no-output warning after
60 seconds is informational; it does not automatically stop training.

Every run receives a new output subfolder. Success requires both a zero process
exit code and a nonempty final `.safetensors` file. This confirms an output was
produced, not that the model has good quality.

### Step 6 — Find and evaluate the model

Click **Open Run Folder** and find `my_character.safetensors`. Load it in your
SDXL workflow in ComfyUI or another compatible inference tool. Evaluate the
results with your trigger word and suitable test prompts.

The app does not generate evaluation images or automatically choose the best
checkpoint. Training speed and generated-image quality are separate measurements.

## Checkpoint saving and repeated resume

**A LoRA model file and a resumable training state are different things.**
Inference uses LoRA weights; continuation also needs optimizer, scheduler,
random-state, position, and applicable precision-scaler files.

- Enable **Save resumable state at checkpoints** before training.
- **Checkpoint interval (steps) = 0** means epoch checkpoints only, not disabled
  saving. A positive value also enables step checkpoints.
- **Stop** forcibly ends the owned training process tree. It does **not** request
  a fresh checkpoint. Work since the last completed checkpoint may be lost.
- Keep the original dataset, base model, training settings, and fixed seed when
  resuming. **Epochs remains the original total target**, not additional epochs.

To continue:

1. Open **History / Compare** and select the latest interrupted run.
2. Select a complete state folder and click **Prepare Resume**.
3. Check the settings and click **Start Training**. Only approve states you trust.
4. If you stop again, repeat these steps using a checkpoint from the new run.

Alternatively, load the relevant run's settings and browse to its state folder
under **Resume state folder**. **New Training** clears this field; loading
settings alone starts nothing and does not restore optimizer state.

For example, a checkpoint at step `378` of `504` resumes with the next step
displayed as `379 / 504`. You can later resume from `400`, then `450`, while
keeping `504` as the target. Each continuation gets a new output folder; its source
checkpoint is not overwritten.

Some sd-scripts versions save epoch-relative step counts after resuming. The app
recovers the overall position from saved epoch/step metadata and recorded steps
per epoch, then passes an explicit `initial_step` alongside `resume`. A trainer
warning about overriding the saved step count concerns position only: model and
optimizer state are still restored. Raw trainer log counters may start at zero;
the app's progress display includes previously completed steps.

Keep each checkpoint inside its run folder with `settings.json` and `run.json`.
Older resumed runs can use their original `training.log` if it contains batches
per epoch and gradient accumulation. Repeated resume supports this app's
single-GPU, accumulation-1 setup. Missing or inconsistent position metadata is
reported rather than guessed. Ancestor checkpoint folders are not needed to
calculate a newer checkpoint's position.

Only load states you created or trust: optimizer/random-state files may contain
executable serialized Python data. The GUI reads JSON and checks file presence;
the trainer loads and validates the actual saved state. Structural checks cannot
guarantee binary integrity or bit-for-bit identical resumed results.

Extending a completed run beyond its original target is not implemented.

## Offline mode

Enable **Offline Mode (use cached Hugging Face files)** to disable Hugging Face
network checks in the training subprocess. This can avoid startup delays when
the required assets are already present locally.

The base SDXL checkpoint alone is not sufficient: required tokenizer files must
also exist in the selected training environment's Hugging Face cache. Missing
cache entries cause an error; there is no silent online retry. Prepare the cache
while connected, or disable Offline Mode when downloads are needed.

The setting applies `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` to the child
processes, not your whole computer. It does not control WD14 downloads. For local
WD14 use, supply both matching model files described above. Exported
`command.json` records the environment overrides for manual launches.

## Settings, history, and performance

- The app remembers the dataset path, dataset-specific prefixes, WD14 options,
  and training draft between launches in
  `%LOCALAPPDATA%/MiniLoRATrainer/session.json`. Resume selection is cleared on
  startup so continuation is always intentional.
- **Save Settings... / Load Settings...** use application JSON configurations,
  not arbitrary sd-scripts TOMLs. Each run also stores its exact settings. Older
  supported configuration versions are migrated on load.
- **History / Compare** lists outcomes, model files, and available state folders;
  opens logs/folders; loads settings; and prepares a resume. **Choose Output
  Folder** changes which runs History lists—it does not select a resume state.
- **Measure performance** enables optional GPU memory sampling. History can
  compare observed speed and sampled GPU memory between suitable baseline and
  candidate runs. Timing excludes the first five new steps; memory measurements
  are GPU-wide, include other applications, and can miss brief peaks.

For a manual comparison, keep model, data, seed, batch size, and resolution the
same; select one run as baseline and change one setting for the candidate.
Comparison does not claim a speedup when required evidence is absent or runs are
incomplete/resumed. It does not score image quality or automatically change settings.

**Automatic tuning** would mean automatically launching trials with different
settings and selecting results using a defined measurement. It is not implemented
and is not a required post-training step.

## Files and storage

A typical run folder contains:

```text
output/
└── my_character-<unique-id>/
    ├── settings.json                  # App configuration
    ├── dataset.toml                   # Dataset, repeats, buckets, caption options
    ├── training.toml                  # Trainer configuration
    ├── command.json                   # Command, working directory, environment
    ├── training.log                   # Raw output and errors
    ├── run.json                       # Outcome, timing, and recovery metadata
    ├── my_character-000001.safetensors # Saved epoch weights, when available
    ├── my_character-000001-state/      # Optional resumable epoch checkpoint
    └── my_character.safetensors        # Final model after successful training
```

Step checkpoints use step-based names; an end-of-training state can also be saved.
Not every file exists in an exported, interrupted, or failed run.

**Training environment** means the software that performs training. **Outputs**
mean models, states, settings, and logs produced by it. Removing unwanted output
folders does not require reinstalling the environment. The app does not
automatically delete old outputs. Back up anything you need before removing it;
deleting a state removes that recovery point. State files can be much larger than
the final LoRA. Keep the full run folder for reliable future resume/history.

Models, datasets, generated outputs, virtual environments, and common state-file
formats are excluded from Git. Never commit private datasets, caption content,
tokens, or training artifacts accidentally. `.gitignore` protects the named local
folders, not every arbitrary output directory you might choose.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv` is not recognized | Install uv, then reopen PowerShell and check `uv --version`. |
| No images appear | Select the folder directly containing supported images, not its parent. Subfolders are not scanned. |
| WD14 reports missing/mismatched files | Provide both matching `model.onnx` and `selected_tags.csv` in `models/wd14/`. |
| WD14 replaced my captions | Tag Current/All generate replacements. Apply Current/All only modifies prefix tags. Back up captions before batch tagging. |
| CUDA or a dependency check fails | Verify the **Training Python executable**, not the app's `.venv`. Repair that environment using its sd-scripts installation instructions. |
| Startup appears stuck | Inspect the latest log phase. Loading/caching can be slow; network checks may also delay startup. Use Offline Mode only after preparing required caches. |
| Offline training reports missing files | Supply the missing cached assets or temporarily disable Offline Mode to obtain them. |
| CUDA out of memory | Review batch size, resolution, cache/checkpointing settings, and other GPU applications. Changes require a new run rather than an incompatible resume. |
| No state exists after Stop | State saving must have been enabled and a checkpoint completed before stopping. Stop does not force a save. |
| Resume reports mismatched settings/data | Load the selected run's settings and restore its original dataset/base model. Do not change the total epoch target. |
| Repeated resume cannot determine position | Keep that checkpoint's `settings.json` and `run.json`, or its original metadata-bearing `training.log`, in the run folder. |
| Log starts from 0 after resume | Raw trainer counters can restart. Use the app's overall step display; inspect state-loading messages if restoration itself failed. |
| Checkpoint has reached the target | The run is complete. Continuing beyond its original target is not supported yet. |
| A history entry says running/interrupted | The app may have closed without recording a final outcome. Inspect its log and saved files. |

## Limitations

- SDXL LoRA, Windows, and one NVIDIA CUDA GPU are the current training target.
  Other model families, operating systems, distributed training, and arbitrary
  external trainer configurations are not promised to work.
- No automatic trainer installation, packaged desktop installer, training queue,
  automatic tuning, in-app generated previews, or model-quality scoring.
- No extension of already-completed runs. Repeated resume continues an unfinished
  run toward its original target; it does not add extra epochs.
- No recursive dataset scan, automatic caption-quality validation, caption undo
  history, or automatic deletion of old models/states.
- Settings checks verify paths, ranges, and option compatibility. They do not
  prove that a selected checkpoint is actually an SDXL architecture.
- Trainer versions can differ. Counter handling was inspected against local
  sd-scripts revision `690ea7f96c23182352ec63def76d431c6120bd2f`; keep a known-working
  trainer environment when resuming. This is not a compatibility guarantee for
  every revision or a claim that all training combinations were tested.

## Development

### Project layout

```text
app.py                      # Desktop entry point
src/
├── ui/                     # Caption window, training setup, review, history
├── dataset/                # Scanning, captions, renaming, validation
├── tagger/                 # WD14 backend and background worker
├── training/               # Configs, commands, runner, monitoring, resume, records
└── mini_lora_trainer/       # Source-checkout command entry point
tests/
├── fixtures/               # Clearly marked synthetic training processes
└── test_*.py               # Dataset, UI, configuration, runner, and resume tests
pyproject.toml              # App dependencies and metadata
uv.lock                     # Locked dependency versions
```

### Run the tests

From the repository root:

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
```

Tests use temporary datasets, offscreen Qt windows, and synthetic CPU training
processes; they do not require a GPU or train real models. Repeated-resume coverage
includes checkpoints at overall steps 378, 400, 450, and completion at 504, checking
position, simulated state restoration, source preservation, and displayed progress.
Passing these tests does not establish real model quality or GPU reproducibility.

When reporting a problem, include the app/trainer revision, Windows/GPU details,
reproduction steps, and relevant log excerpts. Remove private paths, captions,
credentials, and other sensitive content before sharing. Keep contributions
focused and add regression tests for changes to file handling or resume behavior.

## Licensing and acknowledgments

No project license has been selected or included yet. Do not assume a permissive
open-source license merely because the source is available. Third-party software,
models, and datasets retain their own licenses and terms.

Built using PySide6, Pillow, NumPy, ONNX Runtime, Hugging Face Hub, and uv.
Training is powered by [sd-scripts](https://github.com/kohya-ss/sd-scripts);
default automatic tagging uses
[SmilingWolf's WD14 model](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2).
Base models and datasets are supplied by the user, not redistributed here.
