from pathlib import Path
from typing import Iterable


def get_caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def load_caption(image_path: Path) -> str:
    caption_path = get_caption_path(image_path)

    if not caption_path.exists():
        return ""

    return caption_path.read_text(encoding="utf-8")


def save_caption(image_path: Path, caption: str) -> None:
    caption_path = get_caption_path(image_path)
    caption_path.write_text(caption, encoding="utf-8")


def build_caption_prefix(trigger_word: str, class_token: str = "") -> str:
    """Normalize explicit comma-separated prefix tags, preserving their order."""
    tags = {}
    for tag in f"{trigger_word},{class_token}".split(","):
        tag = tag.strip()
        if tag:
            tags.setdefault(tag.casefold(), tag)
    return ", ".join(tags.values())


def suggest_keep_tokens(trigger_word: str, class_token: str = "") -> int:
    prefix = build_caption_prefix(trigger_word, class_token)
    return len(prefix.split(",")) if prefix else 0


def add_caption_prefix(caption: str, prefix: str) -> str:
    """Move each prefix tag to the front; retain all unrelated caption tags."""
    normalized_prefix = build_caption_prefix(prefix)
    if not normalized_prefix:
        raise ValueError("Enter a Trigger Word or Class Token first.")
    prefix_tags = normalized_prefix.split(", ")
    prefix_keys = {tag.casefold() for tag in prefix_tags}

    tags = [tag.strip() for tag in caption.split(",") if tag.strip()]
    remaining_tags = [
        tag for tag in tags if tag.casefold() not in prefix_keys
    ]

    return ", ".join([normalized_prefix, *remaining_tags])


def apply_caption_prefix(
    image_paths: Iterable[Path], prefix: str
) -> list[Path]:
    """Apply a trigger word to captions and return paths whose captions changed."""
    changed_images = []

    for image_path in image_paths:
        caption = load_caption(image_path)
        updated_caption = add_caption_prefix(caption, prefix)

        if updated_caption != caption:
            save_caption(image_path, updated_caption)
            changed_images.append(image_path)

    return changed_images
