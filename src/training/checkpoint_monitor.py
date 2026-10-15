"""Read-only recovery-point tracking; never deserialize optimizer or RNG files."""

from datetime import datetime, timezone
from pathlib import Path
import re

from src.training.checkpoints import inspect_checkpoint


def _compact(value):
    return re.sub(r"\s+", "", str(value)).replace("\\", "/").casefold()


class CheckpointMonitor:
    def __init__(self, run_directory, *, resume_state="", saving_enabled=True):
        self.directory = Path(run_directory).resolve()
        self.resume_state = Path(resume_state).resolve() if resume_state else None
        self.saving_enabled = saving_enabled
        self.pending = ""
        self.completed_files = set()

    def feed(self, text):
        """Recognize Accelerate's final required-file save, including Rich wraps."""
        self.pending = (self.pending + text)[-32768:]
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", self.pending)
        clean = _compact(clean)
        clean = re.sub(r"checkpointing\.py:\d+", "", clean)
        before = len(self.completed_files)
        for match in re.finditer(r"randomstatessavedin(.+?random_states_0\.pkl)", clean):
            self.completed_files.add(match.group(1))
        return len(self.completed_files) != before

    def scan(self, current, total, *, successful_exit=False):
        candidates = []
        pending = 0
        error = None
        try:
            paths = list(self.directory.glob("*-state"))
        except OSError as problem:
            paths = []
            error = str(problem)
        if self.resume_state is not None:
            paths.append(self.resume_state)
        for path in paths:
            source = path == self.resume_state
            try:
                info = inspect_checkpoint(path)
                step = info["current_step"]
                saved = datetime.fromtimestamp(
                    (path / "random_states_0.pkl").stat().st_mtime, timezone.utc,
                ).isoformat()
                # Presence/nonzero size alone is insufficient while writes may
                # still be in flight. Require a completion message, a later
                # training update, or a successful trainer exit.
                if source:
                    evidence = "selected resume state (file checks only)"
                elif _compact(path / "random_states_0.pkl") in self.completed_files:
                    evidence = "trainer reported required state files saved"
                elif current > step:
                    evidence = "training advanced beyond the saved step"
                elif successful_exit:
                    evidence = "trainer exited successfully"
                else:
                    pending += 1
                    continue
                candidates.append({
                    "path": str(path), "step": step, "epoch": info["current_epoch"],
                    "saved_at": saved, "source": source, "evidence": evidence,
                    "can_resume": bool(total and step < total),
                })
            except (OSError, ValueError, TypeError, AttributeError) as problem:
                if source:
                    error = f"Selected resume state is unavailable: {problem}"
                else:
                    pending += 1
        latest = max(candidates, key=lambda item: (item["step"], not item["source"], item["saved_at"]), default=None)
        return {
            "latest": latest, "pending_count": pending, "error": error,
            "saving_enabled": self.saving_enabled,
            "steps_at_risk": max(0, current - (latest["step"] if latest else 0)),
        }
