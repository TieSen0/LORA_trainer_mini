"""WD14 ONNX tagger backend.

Model files are downloaded on first use and kept outside source control in
``models/wd14``. The UI only constructs this backend; all inference stays here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.tagger.base import ScoredTag, TagPrediction, Tagger


IMAGE_SIZE = 448
DEFAULT_MODEL_REPOSITORY = "SmilingWolf/wd-v1-4-convnext-tagger-v2"


@dataclass(frozen=True)
class WD14Config:
    threshold: float = 0.35
    excluded_tags: frozenset[str] = field(default_factory=frozenset)
    model_repository: str = DEFAULT_MODEL_REPOSITORY
    model_directory: Path = Path(__file__).resolve().parents[2] / "models" / "wd14"
    replace_underscores: bool = True


@dataclass(frozen=True)
class _Label:
    name: str
    category: str


class WD14Tagger(Tagger):
    def __init__(self, config: WD14Config):
        if not 0.0 <= config.threshold <= 1.0:
            raise ValueError("WD14 confidence threshold must be between 0 and 1.")

        self.config = config
        self._session = None
        self._input_name = ""
        self._labels: tuple[_Label, ...] = ()

    def predict(self, image_path: Path) -> TagPrediction:
        self._load_model()
        image = self._preprocess_image(image_path)
        scores = self._session.run(None, {self._input_name: image[None, ...]})[0][0]
        return TagPrediction(self._tags_from_scores(scores))

    def _load_model(self) -> None:
        if self._session is not None:
            return

        try:
            import numpy  # noqa: F401
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "WD14 dependencies are missing. Run 'uv sync' and try again."
            ) from error

        model_directory = self.config.model_directory
        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / "model.onnx"
        tags_path = model_directory / "selected_tags.csv"
        if model_path.is_file() != tags_path.is_file():
            raise RuntimeError(
                f"Place both model.onnx and its matching selected_tags.csv in {model_directory}. "
                "Only one file was found; no download was attempted to avoid mixing models."
            )
        if not (model_path.is_file() and tags_path.is_file()):
            model_path = hf_hub_download(
                repo_id=self.config.model_repository,
                filename="model.onnx",
                local_dir=model_directory,
            )
            tags_path = hf_hub_download(
                repo_id=self.config.model_repository,
                filename="selected_tags.csv",
                local_dir=model_directory,
            )

        self._labels = self._load_labels(Path(tags_path))
        providers = ort.get_available_providers()
        preferred_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(model_path), providers=preferred_providers)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _load_labels(tags_path: Path) -> tuple[_Label, ...]:
        with tags_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            labels = tuple(
                _Label(row["name"], row["category"])
                for row in reader
                if row.get("name") and row.get("category")
            )

        if not labels:
            raise RuntimeError("The WD14 tag list is empty or invalid.")
        return labels

    def _tags_from_scores(self, scores) -> tuple[ScoredTag, ...]:
        if len(scores) != len(self._labels):
            raise RuntimeError("The WD14 model and selected_tags.csv do not match.")

        excluded_tags = {tag.casefold() for tag in self.config.excluded_tags}
        tags = []
        for label, score in zip(self._labels, scores):
            if label.category not in {"0", "4"} or float(score) < self.config.threshold:
                continue

            name = label.name.replace("_", " ") if self.config.replace_underscores else label.name
            if label.name.casefold() in excluded_tags or name.casefold() in excluded_tags:
                continue
            tags.append(ScoredTag(name, float(score), label.category))

        return tuple(sorted(tags, key=lambda tag: tag.confidence, reverse=True))

    @staticmethod
    def _preprocess_image(image_path: Path):
        import numpy as np

        with Image.open(image_path) as source:
            if source.mode in ("RGBA", "LA") or "transparency" in source.info:
                rgba = source.convert("RGBA")
                image = Image.new("RGB", rgba.size, "white")
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = source.convert("RGB")

        square_size = max(image.size)
        square = Image.new("RGB", (square_size, square_size), "white")
        square.paste(image, ((square_size - image.width) // 2, (square_size - image.height) // 2))
        resized = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32)[:, :, ::-1]
