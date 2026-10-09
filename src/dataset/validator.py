"""Read-only checks for the flat image/caption dataset used by the editor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from src.dataset.captions import get_caption_path, load_caption
from src.dataset.scanner import scan_images


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    path: Path
    message: str


@dataclass
class ValidationReport:
    folder: Path
    image_count: int = 0
    caption_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "Error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "Warning" for issue in self.issues)


def validate_dataset(
    folder_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> ValidationReport:
    """Inspect direct children only; never create or modify dataset files."""
    folder = Path(folder_path)
    report = ValidationReport(folder)
    try:
        images = scan_images(str(folder))
        captions = [
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ]
    except OSError as error:
        report.issues.append(ValidationIssue("Error", folder, f"Cannot scan folder: {error}"))
        return report

    report.image_count = len(images)
    report.caption_count = len(captions)
    if not images:
        report.issues.append(ValidationIssue(
            "Error", folder,
            "No supported images found directly in this folder. Subfolders are not scanned.",
        ))

    stems: dict[str, list[Path]] = {}
    for image in images:
        stems.setdefault(image.stem.casefold(), []).append(image)
    for shared_images in stems.values():
        if len(shared_images) > 1:
            report.issues.append(ValidationIssue(
                "Error", shared_images[0],
                "Multiple images share one caption filename: "
                + ", ".join(image.name for image in shared_images)
                + ". Give each image a unique filename stem.",
            ))

    for index, image in enumerate(images, 1):
        try:
            with Image.open(image) as source:
                source.verify()
            # Decode pixels too: some formats defer corruption checks until loading.
            with Image.open(image) as source:
                source.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            report.issues.append(ValidationIssue("Error", image, f"Cannot read image: {error}"))

        caption_path = get_caption_path(image)
        try:
            if not caption_path.exists():
                report.issues.append(ValidationIssue(
                    "Warning", caption_path, "Missing caption. Write a caption or generate WD14 tags.",
                ))
            elif not load_caption(image).strip():
                report.issues.append(ValidationIssue("Warning", caption_path, "Caption is empty."))
        except (OSError, UnicodeError) as error:
            report.issues.append(ValidationIssue(
                "Error", caption_path, f"Cannot read caption as UTF-8 text: {error}",
            ))
        if progress:
            progress(index, len(images))

    for caption in sorted(captions):
        if caption.stem.casefold() not in stems:
            report.issues.append(ValidationIssue(
                "Warning", caption, "Caption has no matching supported image in this folder.",
            ))
    return report
