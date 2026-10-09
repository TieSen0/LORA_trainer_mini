"""Observed phases and timing, never guessed percentages during initialization."""

import csv
import os
import subprocess
import time

from src.training.log_parser import TrainingLogParser


class TrainingMonitor:
    def __init__(self, now=None, *, resume_step=0, expected_total=None):
        self.started = time.monotonic() if now is None else now
        self.last_output = self.started
        self.phase = "Preparing"
        self.current = self.total = 0
        self.resume_step = resume_step
        self.expected_total = expected_total
        self.session_current = 0
        self.parser = TrainingLogParser()
        self.pending = ""
        self.sample_start = None
        self.sample_step = None
        self.sample_end = None
        self.sample_end_step = None

    def feed(self, text, now=None):
        now = time.monotonic() if now is None else now
        if text:
            self.last_output = now
        self.pending = (self.pending + text)[-16384:]
        lines = self.pending.replace("\r", "\n").split("\n")
        self.pending = lines.pop()
        for line in lines:
            lower = line.lower()
            for words, phase in (
                (("loading settings", "tokenizer"), "Loading settings / tokenizers"),
                (("loading dataset", "prepare images", "read image", "bucket"), "Preparing dataset"),
                (("loading model", "load sdxl", "loading checkpoint", "building unet", "loading unet", "load text encoder"), "Loading model"),
                (("caching latents", "cache latents"), "Caching latents"),
                (("caching text encoder", "cache text encoder"), "Caching text-encoder outputs"),
                (("resume training", "load train state", "skipping epoch", "skipping "), "Restoring training state"),
                (("saving checkpoint", "save train state", "saving model", "save state"), "Saving checkpoint"),
            ):
                if any(word in lower for word in words):
                    self.phase = phase
                    break
        progress = []
        for current, reported_total in self.parser.feed(text):
            if current < self.session_current:
                self.sample_start = self.sample_step = self.sample_end = self.sample_end_step = None
            # Warmup and throughput use newly executed steps, not restored ones.
            if current > self.session_current and current >= 5:
                if self.sample_start is None:
                    self.sample_start, self.sample_step = now, current
                self.sample_end, self.sample_end_step = now, current
            self.session_current = current
            overall = self.resume_step + current
            total = self.parser.total_steps or self.expected_total
            if not self.resume_step:
                total = total or reported_total
            # A resumed tqdm denominator is not necessarily the original target.
            # Do not invent a percentage if neither the log nor history has it.
            if total is None:
                self.phase = "Training (waiting for overall step total)"
                continue
            if overall > total:
                self.phase = "Training (step counter exceeds expected total)"
                continue
            self.current, self.total = overall, total
            self.phase = "Training" if current else "Training ready"
            progress.append((overall, total))
        return progress

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        rate = None
        if self.sample_start is not None and self.sample_end > self.sample_start:
            rate = (self.sample_end_step - self.sample_step) / (self.sample_end - self.sample_start)
        return {
            "phase": self.phase, "elapsed_seconds": max(0, now - self.started),
            "silent_seconds": max(0, now - self.last_output),
            "current": self.current, "total": self.total,
            "resume_step": self.resume_step, "session_current": self.session_current,
            "steps_per_second": rate,
            "eta_seconds": (self.total - self.current) / rate if rate and self.total else None,
            "measured_steps": (self.sample_end_step - self.sample_step) if self.sample_start is not None else 0,
        }


def sample_gpu():
    """GPU 0 board-wide memory, not process allocation or a guaranteed exact peak."""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, **options,
        )
        if result.returncode == 0:
            row = next(csv.reader(result.stdout.splitlines()))
            return {"name": row[0].strip(), "memory_mib": float(row[1]), "utilization": float(row[2])}
    except (OSError, ValueError, StopIteration, subprocess.TimeoutExpired):
        pass
    return None
