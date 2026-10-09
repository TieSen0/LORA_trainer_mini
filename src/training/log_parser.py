"""Extract training-step progress without mistaking model downloads for training."""

import re


class TrainingLogParser:
    def __init__(self):
        self.pending = ""
        self.total_steps = None
        self.batches_per_epoch = None
        self.gradient_accumulation = None

    @property
    def step_layout(self):
        # The app launches one process with one optimizer update per batch.
        if self.batches_per_epoch and self.gradient_accumulation == 1:
            return {"version": 1, "steps_per_epoch": self.batches_per_epoch}
        return None

    def feed(self, text: str) -> list[tuple[int, int]]:
        self.pending += text
        parts = re.split(r"[\r\n]", self.pending)
        self.pending = parts.pop()[-16384:]
        result = []
        for part in parts:
            # Strip after assembling a line: escape sequences can span chunks.
            part = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", part)
            # Unlike tqdm's denominator, this remains the full training target
            # when sd-scripts skips batches for a mid-epoch resume.
            target = re.search(r"\btotal optimization steps\b[^:\r\n]*:\s*(\d+)\b", part, re.IGNORECASE)
            if target and int(target.group(1)) > 0:
                self.total_steps = int(target.group(1))
            batches = re.search(r"\bnum batches per epoch\b[^:\r\n]*:\s*(\d+)\b", part, re.IGNORECASE)
            if batches and int(batches.group(1)) > 0:
                self.batches_per_epoch = int(batches.group(1))
            accumulation = re.search(r"\bgradient accumulation steps\b[^=:\r\n]*[=:]\s*(\d+)\b", part, re.IGNORECASE)
            if accumulation:
                self.gradient_accumulation = int(accumulation.group(1))
            match = re.search(r"\bsteps:\s*\d+%.*?\b(\d+)\s*/\s*(\d+)", part)
            if match:
                current, total = map(int, match.groups())
                if total > 0 and 0 <= current <= total:
                    result.append((current, total))
        return result
