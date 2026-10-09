"""Background execution for automatic caption generation."""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.tagger.base import Tagger


class TaggingWorker(QThread):
    progress_changed = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, tagger: Tagger, image_paths: list[Path]):
        super().__init__()
        self.tagger = tagger
        self.image_paths = image_paths

    def run(self):
        predictions = {}
        try:
            for index, image_path in enumerate(self.image_paths, start=1):
                predictions[image_path] = self.tagger.predict(image_path)
                self.progress_changed.emit(index, len(self.image_paths))
        except Exception as error:
            self.failed.emit(str(error))
            return

        self.completed.emit(predictions)
