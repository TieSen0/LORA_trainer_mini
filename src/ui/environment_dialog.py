"""Read-only environment diagnostics; imports run away from the UI thread."""

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout

from src.diagnostics import collect_report, format_report
from src.ui.design import apply_theme


class EnvironmentWorker(QThread):
    completed = Signal(object)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            checks = collect_report(self.config)
        except Exception as error:
            checks = [{"status": "FAIL", "title": "Environment check interrupted", "detail": str(error)}]
        self.completed.emit(checks)


class EnvironmentDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        apply_theme()
        self.setWindowTitle("Check Environment")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.summary = QLabel("Checking desktop and selected training runtime. Each import check has a 45-second timeout.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText("No training, downloads or repairs will be started. Please wait for the checks to finish.")
        layout.addWidget(self.details)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.save_button = QPushButton("Save Report...")
        self.buttons.addButton(self.save_button, QDialogButtonBox.ActionRole)
        self.save_button.clicked.connect(self.save_report)
        self.buttons.rejected.connect(self.reject)
        self.buttons.setEnabled(False)
        layout.addWidget(self.buttons)
        self.running = True
        self.worker = EnvironmentWorker(config, self)
        self.worker.completed.connect(self.show_report)
        self.worker.finished.connect(self.check_finished)
        self.worker.start()

    def show_report(self, checks):
        counts = {status: sum(item["status"] == status for item in checks) for status in ("OK", "WARN", "FAIL")}
        self.summary.setText(f"Check complete: {counts['OK']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed. No changes made.")
        self.details.setPlainText(format_report(checks))

    def check_finished(self):
        self.running = False
        self.buttons.setEnabled(True)

    def save_report(self):
        destination, _ = QFileDialog.getSaveFileName(self, "Save Environment Report", "environment-report.txt", "Text files (*.txt)")
        if destination:
            try:
                # Refuse overwrites even if another process creates the file after selection.
                with Path(destination).open("x", encoding="utf-8") as file:
                    file.write(self.details.toPlainText())
            except OSError as error:
                QMessageBox.warning(self, "Report Not Saved", f"Choose a new filename.\n{error}")

    def accept(self):
        if not self.running:
            super().accept()

    def reject(self):
        if not self.running:
            super().reject()

    def closeEvent(self, event):
        if self.running:
            event.ignore()
        else:
            super().closeEvent(event)
