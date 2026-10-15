"""Supervise a separate training process and stream its output off the UI thread."""

import codecs
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

from PySide6.QtCore import QThread, Signal

from src.dataset.validator import validate_dataset
from src.training.command_builder import TrainingRun, create_training_run, training_environment
from src.training.config import SDXLTrainConfig, validate_config, validate_runtime
from src.training.monitor import TrainingMonitor, sample_gpu
from src.training.environment import training_probe
from src.training.resources import ResourceSampler
from src.training.records import write_json
from src.training.checkpoints import dataset_signature, inspect_checkpoint, list_checkpoints
from src.training.checkpoint_monitor import CheckpointMonitor


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Only the process started for this run and its descendants are targeted.
        result = subprocess.run(
            [str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "taskkill.exe"),
             "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode and process.poll() is None:
            raise RuntimeError("Could not stop the training process tree: " + result.stderr.decode(errors="replace"))
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)


def run_process(command, cwd, on_output, stop_requested: threading.Event, *, env_overrides=None, on_tick=None, on_started=None) -> int:
    """No shell; stop remains responsive even when the child produces no output."""
    if stop_requested.is_set():
        return -1
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    # Do not leak the GUI's virtual-environment/Python path into the chosen runtime.
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(key, None)
    if env_overrides:
        environment.update(env_overrides)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(
        list(command), cwd=str(cwd), env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options,
    )
    chunks = queue.Queue()

    def read_pipe():
        try:
            while data := process.stdout.read1(4096):
                chunks.put(data)
        finally:
            chunks.put(None)

    reader = threading.Thread(target=read_pipe, daemon=True)
    reader.start()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        if on_started is not None:
            on_started(process.pid)
        eof = False
        while not eof or process.poll() is None:
            if on_tick is not None:
                on_tick()
            if stop_requested.is_set() and process.poll() is None:
                stop_process_tree(process)
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                eof = True
            else:
                on_output(decoder.decode(chunk))
        on_output(decoder.decode(b"", final=True))
        return process.wait()
    finally:
        if process.poll() is None:
            stop_process_tree(process)
        reader.join(timeout=2)
        if not reader.is_alive():
            process.stdout.close()


@dataclass(frozen=True)
class TrainingResult:
    status: str
    message: str
    run_directory: Path | None = None
    model_path: Path | None = None


class TrainingWorker(QThread):
    output = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    prepared = Signal(object)
    completed = Signal(object)
    telemetry = Signal(object)

    def __init__(self, config: SDXLTrainConfig, export_only=False, parent=None):
        super().__init__(parent)
        self.config = config
        self.export_only = export_only
        self.stop_requested = threading.Event()

    def request_stop(self):
        self.stop_requested.set()

    def run(self):
        run: TrainingRun | None = None
        log = None
        monitor = TrainingMonitor()
        record = {}
        last_tick = last_gpu = last_record = 0.0
        gpu_peak = None
        gpu_name = None
        checkpoint_monitor = None
        checkpoint_status = None
        sampler = ResourceSampler() if self.config.measure_performance and not self.export_only else None
        resources = {} if sampler is not None else None

        def tick(force=False):
            nonlocal last_tick, last_gpu, last_record, gpu_peak, gpu_name, checkpoint_status, resources
            now = time.monotonic()
            if not force and now - last_tick < 1:
                return
            last_tick = now
            if self.config.measure_performance and not self.export_only and now - last_gpu >= 5:
                last_gpu = now
                resources = sampler.sample(now)
                gpu = sample_gpu()
                resources["gpu"] = gpu
                if gpu is not None:
                    gpu_name = gpu["name"]
                    gpu_peak = max(gpu_peak or 0, gpu["memory_mib"])
            if checkpoint_monitor is not None:
                checkpoint_status = checkpoint_monitor.scan(
                    monitor.current, monitor.total, successful_exit=record.get("status") == "success",
                )
                record["checkpoint_status"] = checkpoint_status
            metrics = {**monitor.snapshot(), "peak_gpu_memory_mib": gpu_peak, "gpu_name": gpu_name,
                       "resources": resources,
                       "checkpoint_status": checkpoint_status}
            self.telemetry.emit(metrics)
            if run is not None and record:
                record["metrics"] = metrics
                if force or now - last_record >= 15:
                    last_record = now
                    try:
                        write_json(run.directory / "run.json", record)
                    except (OSError, ValueError) as error:
                        self.output.emit(f"Could not update run history: {error}\n")

        def output(text):
            if log is not None:
                log.write(text)
                log.flush()
            self.output.emit(text)
            checkpoint_finished = checkpoint_monitor.feed(text) if checkpoint_monitor is not None else False
            for step, total in monitor.feed(text):
                self.status_changed.emit(f"Training: step {step} / {total}")
                self.progress_changed.emit(step, total)
            layout = monitor.parser.step_layout
            if run is not None and record and layout and record.get("step_layout") != layout:
                # Persist before checkpoints arrive, including if the app or
                # trainer is later interrupted. Old logs remain a fallback.
                record["step_layout"] = layout
                tick(force=True)
            elif checkpoint_finished:
                tick(force=True)
            self.telemetry.emit({**monitor.snapshot(), "peak_gpu_memory_mib": gpu_peak, "gpu_name": gpu_name,
                                 "resources": resources,
                                 "checkpoint_status": checkpoint_status})

        def check_cancelled():
            if self.stop_requested.is_set():
                raise InterruptedError("Stopped by user.")

        try:
            errors = validate_config(self.config) + validate_runtime(self.config)
            if errors:
                raise ValueError("\n".join(errors))
            if self.config.resume_state:
                state = Path(self.config.resume_state)
                checkpoint = inspect_checkpoint(state)
                resume_step = checkpoint["current_step"]
                expected_total = (checkpoint["steps_per_epoch"] * self.config.epochs
                                  if "steps_per_epoch" in checkpoint else None)
                history_path = state.parent / "run.json"
                if history_path.is_file():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    target = history.get("metrics", {}).get("total")
                    if type(target) is int and target > resume_step:
                        expected_total = target
                monitor = TrainingMonitor(
                    now=monitor.started, resume_step=resume_step, expected_total=expected_total,
                )
            self.status_changed.emit("Reviewing dataset before training...")
            monitor.phase = "Reviewing dataset"

            def review_progress(current, total):
                check_cancelled()
                self.status_changed.emit(f"Reviewing dataset: {current} / {total}")
                tick()

            report = validate_dataset(self.config.dataset_path, review_progress)
            blockers = []
            for issue in report.issues:
                output(f"{issue.severity}: {issue.path.name}: {issue.message}\n")
                if issue.severity == "Error" or issue.message.startswith(("Missing caption", "Caption is empty")):
                    blockers.append(f"{issue.path.name}: {issue.message}")
            if blockers:
                raise ValueError("Fix dataset problems before training:\n" + "\n".join(blockers))
            check_cancelled()
            run = create_training_run(self.config)
            model_stat = Path(self.config.model_path).stat()
            record = {
                "version": 1, "status": "exporting" if self.export_only else "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "resume_from": self.config.resume_state,
                "resume_start_step": monitor.resume_step,
                "dataset_signature": dataset_signature(self.config.dataset_path),
                "model_signature": [model_stat.st_size, model_stat.st_mtime_ns],
                "message": "Run has started; an unfinished record may also indicate an interrupted app.",
            }
            write_json(run.directory / "run.json", record)
            if not self.export_only:
                checkpoint_monitor = CheckpointMonitor(
                    run.directory, resume_state=self.config.resume_state,
                    saving_enabled=self.config.save_training_state,
                )
            self.prepared.emit(run)
            log = (run.directory / "training.log").open("w", encoding="utf-8")
            output(f"Run folder: {run.directory}\n")
            if self.config.resume_state:
                output(
                    f"Overall progress will include {monitor.resume_step} saved checkpoint steps. "
                    "Raw trainer log counters count new steps from zero. "
                    "The trainer receives this overall position as initial_step; its message "
                    "about overriding the state's step count does not mean weights or optimizer state are ignored.\n"
                )
            environment = training_environment(self.config)
            output(
                "Offline Mode: ON. Hugging Face network checks disabled; required files must be cached.\n"
                if self.config.offline_mode else "Offline Mode: OFF. Hugging Face network access allowed.\n"
            )
            if self.export_only:
                result = TrainingResult("exported", "Training files exported; training has not started.", run.directory)
            else:
                monitor.phase = "Checking training environment"
                self.status_changed.emit("Checking the training Python environment...")
                # Check only in the user's selected runtime; no torch install in the GUI environment.
                probe = training_probe(self.config)
                code = run_process(
                    [run.command[0], "-u", "-c", probe], run.working_directory, output, self.stop_requested,
                    env_overrides=environment,
                    on_tick=tick,
                )
                check_cancelled()
                if code:
                    raise RuntimeError("Training environment check failed. See the log for the missing package or GPU error.")
                self.status_changed.emit("Initializing trainer: loading tokenizers, dataset and model; no steps reported yet...")
                monitor.phase = "Initializing trainer / tokenizers"
                code = run_process(
                    run.command, run.working_directory, output, self.stop_requested,
                    env_overrides=environment,
                    on_tick=tick,
                    on_started=sampler.attach if sampler is not None else None,
                )
                check_cancelled()
                output("\n")  # Flush the last progress record, even without a trailing newline.
                if code:
                    offline_hint = (
                        " Offline Mode is enabled. If the log reports missing cached files, "
                        "make the required files available locally or disable Offline Mode to download them."
                        if self.config.offline_mode else ""
                    )
                    raise RuntimeError(f"Training failed with exit code {code}. See training.log for details.{offline_hint}")
                if not run.expected_model.is_file() or not run.expected_model.stat().st_size:
                    raise RuntimeError("Training exited without producing the expected .safetensors file.")
                result = TrainingResult("success", f"Training complete: {run.expected_model}", run.directory, run.expected_model)
        except InterruptedError:
            result = TrainingResult("stopped", "Training stopped. Existing checkpoints were kept.", run.directory if run else None)
        except Exception as error:
            result = TrainingResult("failed", str(error), run.directory if run else None)
        finally:
            if log is not None:
                log.close()
        self.output.emit(f"\n{result.message}\n")
        if run is not None:
            try:
                checkpoints = [str(path) for path in list_checkpoints(run.directory)]
            except OSError as error:
                checkpoints = []
                self.output.emit(f"Could not list saved states: {error}\n")
            record.update(
                status=result.status, message=result.message,
                finished_at=datetime.now(timezone.utc).isoformat(),
                model_path=str(result.model_path) if result.model_path else None,
                checkpoints=checkpoints,
            )
            tick(force=True)
            try:
                with (run.directory / "training.log").open("a", encoding="utf-8") as final_log:
                    final_log.write(f"\n{result.message}\n")
            except OSError as error:
                self.output.emit(f"Could not append final log message: {error}\n")
        self.completed.emit(result)
