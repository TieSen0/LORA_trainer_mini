"""Background dataset review with a readable, selectable report."""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextEdit, QVBoxLayout

from src.dataset.validator import ValidationReport, validate_dataset


class DatasetReviewWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, folder_path: str, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path

    def run(self):
        try:
            self.completed.emit(validate_dataset(self.folder_path, self.progress.emit))
        except Exception as error:
            self.failed.emit(str(error))


class DatasetReviewDialog(QDialog):
    def __init__(self, folder_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Review")
        self.resize(760, 500)
        self.summary = QLabel("Checking dataset...")
        self.summary.setWordWrap(True)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.reject)
        self.buttons.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.details)
        layout.addWidget(self.buttons)
        self.worker = DatasetReviewWorker(folder_path, self)
        self.worker.progress.connect(self.show_progress)
        self.worker.completed.connect(self.show_report)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.review_finished)
        self.running = True
        self.worker.start()

    def show_progress(self, completed: int, total: int):
        self.summary.setText(f"Checking images and captions: {completed} / {total}")

    def show_report(self, report: ValidationReport):
        self.summary.setText(
            f"Review complete: {report.image_count} images, {report.caption_count} caption files. "
            f"{report.error_count} errors, {report.warning_count} warnings."
        )
        lines = [f"Folder: {report.folder}", "Direct folder only; subfolders excluded.", ""]
        lines.extend(
            f"{issue.severity}: {issue.path.name}\n{issue.message}\n"
            for issue in report.issues
        )
        if not report.issues:
            lines.append("No dataset file problems found.")
        lines.append("\nUse Training Setup to check base model paths, output folders, and training parameters.")
        self.details.setPlainText("\n".join(lines))

    def show_error(self, message: str):
        self.summary.setText("Dataset review failed.")
        self.details.setPlainText(message)

    def review_finished(self):
        self.running = False
        self.buttons.setEnabled(True)

    def reject(self):
        if not self.running:
            super().reject()

    def closeEvent(self, event):
        if self.running:
            event.ignore()
        else:
            super().closeEvent(event)
