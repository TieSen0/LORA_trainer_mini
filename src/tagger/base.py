"""Interfaces shared by automatic image taggers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoredTag:
    name: str
    confidence: float
    category: str


@dataclass(frozen=True)
class TagPrediction:
    tags: tuple[ScoredTag, ...]

    @property
    def caption(self) -> str:
        return ", ".join(tag.name for tag in self.tags)


class Tagger(ABC):
    @abstractmethod
    def predict(self, image_path: Path) -> TagPrediction:
        """Generate structured tags for one image."""
