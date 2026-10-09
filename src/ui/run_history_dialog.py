"""Browse previous runs, prepare a resume, and compare measurements without launching."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from src.training.checkpoints import inspect_checkpoint, list_checkpoints
from src.training.config import load_config
from src.training.history import compare_runs, list_runs


class RunHistoryDialog(QDialog):
    settings_selected = Signal(object)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training History and Performance Comparison")
        self.resize(950, 700)
        self.root = Path(root) if root else None
        self.runs = []
        self.baseline = None
        layout = QVBoxLayout(self)
        self.folder_label = QLabel()
        layout.addWidget(self.folder_label)
        buttons = QHBoxLayout()
        for title, callback in (("Choose Output Folder", self.choose_folder), ("Refresh", self.refresh)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Run", "Status", "Started (UTC)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_selection)
        layout.addWidget(self.table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.checkpoints = QComboBox()
        layout.addWidget(QLabel("Complete-looking state folders (integrity is verified by the trainer on load):"))
        layout.addWidget(self.checkpoints)
        actions = QHBoxLayout()
        for index, (title, callback) in enumerate((
            ("Open Run Folder", self.open_folder), ("Open Log", self.open_log),
            ("Load Settings", self.load_selected), ("Prepare Resume", self.prepare_resume),
            ("Use as Baseline", self.set_baseline), ("Compare to Baseline", self.compare),
        )):
            if index == 3:
                layout.addLayout(actions)
                actions = QHBoxLayout()
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        note = QLabel("Nothing starts automatically. Resume requires a trusted local training state, not just LoRA weights. Memory measurements are GPU-wide and include other apps; compare quality separately in ComfyUI.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.refresh()

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose Output Folder", str(self.root or ""))
        if selected:
            self.root = Path(selected)
            self.refresh()

    def refresh(self):
        self.folder_label.setText(str(self.root or "Choose an output folder to browse previous runs."))
        try:
            self.runs = list_runs(self.root) if self.root else []
        except OSError as error:
            QMessageBox.warning(self, "Cannot Read History", str(error))
            self.runs = []
        self.table.setRowCount(0)
        for record in self.runs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            status = record.get("status", "unknown")
            if status in {"running", "exporting"}:
                status = "Running or interrupted — check log"
            for column, value in enumerate((Path(record["directory"]).name, status, record.get("started_at", "Legacy run"))):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self.checkpoints.clear()
        self.details.setPlainText("Select a run. Older runs may not have timing or resume state.")

    def selected_directory(self):
        row = self.table.currentRow()
        return Path(self.runs[row]["directory"]) if 0 <= row < len(self.runs) else None

    def show_selection(self):
        directory = self.selected_directory()
        self.checkpoints.clear()
        if directory is None:
            return
        record = self.runs[self.table.currentRow()]
        try:
            states = list_checkpoints(directory)
            for state in states:
                info = inspect_checkpoint(state)
                self.checkpoints.addItem(f"{state.name} — saved step {info['current_step']}", str(state))
            models = [p.name for p in directory.glob("*.safetensors") if p.is_file()]
            metrics = record.get("metrics", {})
            self.details.setPlainText(
                f"{directory}\n{record.get('message', '')}\n\nModels: {', '.join(models) or 'None'}"
                f"\nResume states: {len(states)}\nElapsed seconds: {metrics.get('elapsed_seconds', 'Not recorded')}"
                f"\nObserved steps/s: {metrics.get('steps_per_second') or 'Not enough timing'}"
                f"\nSampled GPU-wide peak MiB: {metrics.get('peak_gpu_memory_mib') or 'Not measured'}"
                f"\nResumed from: {record.get('resume_from') or 'Fresh run'}"
            )
        except (OSError, ValueError) as error:
            self.details.setPlainText(str(error))

    def open_folder(self):
        directory = self.selected_directory()
        if directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def open_log(self):
        directory = self.selected_directory()
        if directory and (directory / "training.log").is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory / "training.log")))

    def load_selected(self, resume=False):
        directory = self.selected_directory()
        if directory is None:
            return
        try:
            config = load_config(directory / "settings.json")
            state = self.checkpoints.currentData() if resume else ""
            if resume and not state:
                raise ValueError("No usable state folder. Old LoRA files alone cannot restore optimizer/scheduler state.")
            config = replace(config, resume_state=state or "")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Load Run", str(error))
            return
        self.settings_selected.emit(config)
        self.accept()

    def prepare_resume(self):
        self.load_selected(resume=True)

    def set_baseline(self):
        self.baseline = self.selected_directory()
        if self.baseline:
            self.details.appendPlainText(f"\nBaseline selected: {self.baseline.name}. Select another run and click Compare to Baseline.")

    def compare(self):
        candidate = self.selected_directory()
        if not self.baseline or not candidate:
            QMessageBox.information(self, "Choose Runs", "Select a baseline first, then select a candidate run.")
            return
        try:
            self.details.setPlainText(compare_runs(self.baseline, candidate))
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Cannot Compare Runs", str(error))
