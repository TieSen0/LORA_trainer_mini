"""Safe, paired image-and-caption batch renaming utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from src.dataset.captions import get_caption_path


@dataclass(frozen=True)
class RenameOperation:
    source_image: Path
    target_image: Path
    source_caption: Path | None
    target_caption: Path | None

    @property
    def file_moves(self) -> tuple[tuple[Path, Path], ...]:
        moves = [(self.source_image, self.target_image)]
        if self.source_caption and self.target_caption:
            moves.append((self.source_caption, self.target_caption))
        return tuple(moves)


@dataclass(frozen=True)
class RenamePlan:
    operations: tuple[RenameOperation, ...]

    @property
    def file_count(self) -> int:
        return sum(len(operation.file_moves) for operation in self.operations)


def create_rename_plan(
    image_paths: Iterable[Path], prefix: str, start_index: int = 1, padding: int = 3
) -> RenamePlan:
    """Build and validate a full rename plan without changing files."""
    normalized_prefix = prefix.strip()
    if not normalized_prefix or any(char in normalized_prefix for char in '\\/:*?"<>|'):
        raise ValueError("The filename prefix must be a valid, non-empty filename.")
    if start_index < 0:
        raise ValueError("The start index cannot be negative.")
    if padding < 1:
        raise ValueError("Padding must be at least 1.")

    images = [Path(image_path).resolve() for image_path in image_paths]
    if len({_path_key(image_path) for image_path in images}) != len(images):
        raise ValueError("The rename plan contains the same image more than once.")

    operations = []
    for offset, image_path in enumerate(images):
        new_stem = f"{normalized_prefix}_{start_index + offset:0{padding}d}"
        source_caption = get_caption_path(image_path)
        has_caption = source_caption.exists()

        operations.append(
            RenameOperation(
                source_image=image_path,
                target_image=image_path.with_name(f"{new_stem}{image_path.suffix}"),
                source_caption=source_caption if has_caption else None,
                target_caption=(
                    source_caption.with_name(f"{new_stem}.txt")
                    if has_caption
                    else None
                ),
            )
        )

    plan = RenamePlan(tuple(operations))
    _validate_plan(plan)
    return plan


def execute_rename_plan(plan: RenamePlan) -> None:
    """Apply a validated plan using temporary names to prevent overwrites."""
    _validate_plan(plan)
    moves = [
        (source, target)
        for operation in plan.operations
        for source, target in operation.file_moves
        if _path_key(source) != _path_key(target)
    ]

    staged_moves: list[tuple[Path, Path, Path]] = []
    try:
        for source, target in moves:
            temporary_path = _temporary_path_for(source)
            source.rename(temporary_path)
            staged_moves.append((source, temporary_path, target))

        for _, temporary_path, target in staged_moves:
            temporary_path.rename(target)
    except OSError:
        _restore_staged_moves(staged_moves)
        raise


def _validate_plan(plan: RenamePlan) -> None:
    moves = [
        (source, target)
        for operation in plan.operations
        for source, target in operation.file_moves
    ]
    source_keys = {_path_key(source) for source, _ in moves}
    target_keys = [_path_key(target) for _, target in moves]

    if len(target_keys) != len(set(target_keys)):
        raise ValueError("Two files in this plan would receive the same filename.")

    for source, target in moves:
        if not source.is_file():
            raise ValueError(f"Source file no longer exists: {source.name}")
        if target.exists() and _path_key(target) not in source_keys:
            raise ValueError(
                f"Target filename already exists and will not be overwritten: {target.name}"
            )


def _temporary_path_for(source: Path) -> Path:
    while True:
        candidate = source.with_name(f".{source.stem}.rename-{uuid4().hex}{source.suffix}")
        if not candidate.exists():
            return candidate


def _restore_staged_moves(staged_moves: list[tuple[Path, Path, Path]]) -> None:
    for source, temporary_path, target in reversed(staged_moves):
        if source.exists():
            continue
        if target.exists():
            target.rename(source)
        elif temporary_path.exists():
            temporary_path.rename(source)


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()
