"""SDXL setup and monitoring UI; execution lives in the training modules."""

from pathlib import Path
import re
import time

from PySide6.QtCore import QSignalBlocker, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QDesktopServices, QTextCursor

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget, QPlainTextEdit, QProgressBar, QTabWidget,
)

from src.training.config import (
    OPTIMIZERS, PRECISIONS, SCHEDULERS, SDXLTrainConfig,
    load_config, save_config, validate_config, validate_runtime,
)
from src.training.runner import TrainingWorker
from src.dataset.captions import build_caption_prefix, suggest_keep_tokens
from src.ui.dataset_review_dialog import DatasetReviewDialog
from src.ui.run_history_dialog import RunHistoryDialog


class TrainingDialog(QDialog):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SDXL LoRA Training Setup")
        self.resize(740, 800)
        self.controls = {}
        self.worker = None
        self.run_directory = None
        layout = QVBoxLayout(self)
        note = QLabel(
            "Use a prepared sd-scripts Python environment. Each run saves its model and logs "
            "in a new subfolder of your output folder."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        self.tabs = QTabWidget()
        self.tabs.addTab(scroll, "Settings")
        layout.addWidget(self.tabs)
        self.settings_panel = content

        paths = self.add_group(content_layout, "Files and folders")
        for key, label in (
            ("model_path", "SDXL base model:"),
            ("dataset_path", "Image dataset folder:"),
            ("output_dir", "Output folder:"),
            ("sd_scripts_dir", "sd-scripts folder:"),
            ("python_executable", "Training Python executable:"),
        ):
            edit = QLineEdit()
            self.controls[key] = edit
            button = QPushButton("Browse...")
            button.clicked.connect(lambda checked=False, key=key: self.browse_path(key))
            row = QHBoxLayout()
            row.addWidget(edit)
            row.addWidget(button)
            paths.addRow(label, row)
        self.controls["sd_scripts_dir"].setPlaceholderText("Folder containing sdxl_train_network.py")
        self.controls["python_executable"].setPlaceholderText("Prepared environment, e.g. .venv/Scripts/python.exe")
        self.controls["output_name"] = QLineEdit()
        self.controls["output_name"].setPlaceholderText("Name without .safetensors")
        paths.addRow("Output model name:", self.controls["output_name"])
        review_button = QPushButton("Review This Dataset")
        review_button.clicked.connect(self.review_dataset)
        paths.addRow(review_button)

        self.controls["offline_mode"] = QCheckBox("Offline Mode (use cached Hugging Face files)")
        paths.addRow(self.controls["offline_mode"])
        offline_note = QLabel(
            "Skips Hugging Face network checks during training. Required tokenizers must already "
            "be cached; missing files cause an error, not a download. This does not affect WD14."
        )
        offline_note.setWordWrap(True)
        paths.addRow(offline_note)

        recovery = self.add_group(content_layout, "Checkpoints, resume and measurement")
        for key, title in (("save_training_state", "Save resumable state at checkpoints (extra disk space)"),
                           ("measure_performance", "Measure performance (optional GPU memory sampling)")):
            self.controls[key] = QCheckBox(title)
            recovery.addRow(self.controls[key])
        for key, title, default in (("save_every_n_steps", "Checkpoint interval (steps):", 0),
                                    ("seed", "Seed:", 42)):
            self.controls[key] = QSpinBox()
            self.controls[key].setRange(0, 2147483647)
            if key == "seed":
                self.controls[key].setMinimum(-1)
                self.controls[key].setSpecialValueText("Random (not resumable)")
            self.controls[key].setValue(default)
            recovery.addRow(title, self.controls[key])
        self.controls["save_every_n_steps"].setToolTip("0 saves at epoch checkpoints only; a positive value also saves every N steps.")
        self.controls["seed"].setToolTip("Use the same fixed seed for resume and performance comparisons.")
        self.controls["resume_state"] = QLineEdit()
        self.controls["resume_state"].setPlaceholderText("Empty = new training; choose a trusted state folder")
        resume_row = QHBoxLayout()
        resume_row.addWidget(self.controls["resume_state"])
        resume_button = QPushButton("Browse...")
        resume_button.clicked.connect(lambda: self.browse_path("resume_state"))
        resume_row.addWidget(resume_button)
        clear_resume = QPushButton("New Training")
        clear_resume.clicked.connect(self.controls["resume_state"].clear)
        resume_row.addWidget(clear_resume)
        recovery.addRow("Resume state folder:", resume_row)
        recovery_note = QLabel(
            "Checkpoint interval 0 saves at epoch boundaries only; a positive value adds step checkpoints. "
            "Resume restores a saved state into a NEW run folder using the original training settings and dataset. "
            "You can resume again from that run's checkpoints; History shows their overall saved steps. "
            "Epochs is the original total target, not additional epochs. Stop does not save immediately: work after the last checkpoint is lost. "
            "Only load states you created or trust. For tuning, measure a baseline, change one setting, run again, then compare in History. No automatic tuning or quality scoring."
        )
        recovery_note.setWordWrap(True)
        recovery.addRow(recovery_note)

        training = self.add_group(content_layout, "Training parameters")
        for key, label, minimum, maximum, step in (
            ("resolution", "Resolution (square target):", 64, 4096, 64),
            ("batch_size", "Batch size:", 1, 1024, 1),
            ("epochs", "Epochs:", 1, 100000, 1),
            ("network_dim", "LoRA rank:", 1, 1024, 1),
        ):
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            self.controls[key] = spin
            training.addRow(label, spin)
        for key, label, minimum, maximum, decimals, step in (
            ("learning_rate", "Learning rate:", 1e-8, 1, 8, 0.00001),
            ("network_alpha", "LoRA alpha:", 0.01, 1024, 2, 1),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            self.controls[key] = spin
            training.addRow(label, spin)
        for key, label, options in (
            ("optimizer", "Optimizer:", OPTIMIZERS),
            ("lr_scheduler", "Learning-rate scheduler:", SCHEDULERS),
            ("mixed_precision", "Mixed precision:", PRECISIONS),
        ):
            combo = QComboBox()
            combo.addItems(options)
            self.controls[key] = combo
            training.addRow(label, combo)

        attention = QComboBox()
        attention.addItem("SDPA", "sdpa")
        attention.addItem("xFormers", "xformers")
        attention.setToolTip(
            "Select one attention backend. xFormers requires a compatible xformers "
            "installation in the training environment."
        )
        self.controls["attention_backend"] = attention
        training.addRow("Cross-attention:", attention)

        dataset = self.add_group(content_layout, "Dataset and captions")
        for key, label in (("trigger_word", "Trigger Word:"), ("class_token", "Class Token (optional):")):
            self.controls[key] = QLineEdit()
            dataset.addRow(label, self.controls[key])
            self.controls[key].textChanged.connect(self.update_prefix_suggestion)
        self.prefix_preview = QLabel()
        self.prefix_preview.setTextFormat(Qt.PlainText)
        self.prefix_preview.setWordWrap(True)
        dataset.addRow("Caption Prefix Preview:", self.prefix_preview)
        for key, label, minimum in (("num_repeats", "Repeats Per Image:", 1), ("keep_tokens", "Keep Tokens:", 0)):
            spin = QSpinBox()
            spin.setRange(minimum, 2147483647)
            self.controls[key] = spin
            dataset.addRow(label, spin)
        self.controls["shuffle_caption"] = QCheckBox("Shuffle Captions")
        self.controls["keep_tokens_auto"] = QCheckBox("Automatically suggest Keep Tokens from prefix")
        dataset.addRow(self.controls["shuffle_caption"])
        dataset.addRow(self.controls["keep_tokens_auto"])
        self.keep_tokens_hint = QLabel()
        self.keep_tokens_hint.setWordWrap(True)
        dataset.addRow(self.keep_tokens_hint)
        prefix_note = QLabel(
            "Prefix fields are saved with settings. Use Apply to Current/All or WD14 in the caption "
            "window to write them to captions. Keep Tokens counts comma-separated tags, not words. "
            "Repeats and trigger words are never taken from folder names."
        )
        prefix_note.setWordWrap(True)
        dataset.addRow(prefix_note)
        self.controls["keep_tokens"].valueChanged.connect(self.keep_tokens_edited)
        self.controls["keep_tokens_auto"].toggled.connect(self.update_prefix_suggestion)
        self.controls["shuffle_caption"].toggled.connect(self.update_shuffle_constraint)

        memory = self.add_group(content_layout, "Memory and training scope")
        for key, label in (
            ("cache_latents", "Cache latents"),
            ("cache_text_encoder_outputs", "Cache text-encoder outputs"),
            ("gradient_checkpointing", "Gradient checkpointing"),
            ("network_train_unet_only", "Train U-Net only (keep text encoders frozen)"),
        ):
            check = QCheckBox(label)
            self.controls[key] = check
            memory.addRow(check)
        explanation = QLabel(
            "Text-encoder caching requires U-Net-only training and disables caption shuffling. "
            "Enabling shuffling turns this cache off. Turn caching off to train text encoders. "
            "Training checks for a CUDA GPU in the selected environment. AdamW8bit requires "
            "bitsandbytes; xFormers requires a compatible xformers installation."
        )
        explanation.setWordWrap(True)
        memory.addRow(explanation)
        self.controls["cache_text_encoder_outputs"].toggled.connect(self.update_cache_constraint)
        self.apply_config(SDXLTrainConfig())

        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(3000)
        monitor_layout.addWidget(self.progress)
        self.activity_label = QLabel("No active run.")
        self.activity_label.setWordWrap(True)
        monitor_layout.addWidget(self.activity_label)
        self.latest_telemetry = None
        self.telemetry_received_at = None
        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(1000)
        self.activity_timer.timeout.connect(self.refresh_activity)
        monitor_layout.addWidget(self.logs)
        self.open_output_button = QPushButton("Open Run Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_run_folder)
        monitor_layout.addWidget(self.open_output_button)
        self.tabs.addTab(monitor, "Training Log")

        self.status = QLabel("Settings have not been saved.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.settings_buttons = []
        for label, handler in (
            ("Check Settings", self.check_settings),
            ("Save Settings...", self.save_settings),
            ("Load Settings...", self.load_settings),
            ("History / Compare", self.open_history),
            ("Close", self.accept),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
            self.settings_buttons.append(button)
        layout.addLayout(buttons)
        run_buttons = QHBoxLayout()
        self.export_button = QPushButton("Export Training Files")
        self.export_button.clicked.connect(lambda: self.begin_training(export_only=True))
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(lambda: self.begin_training(export_only=False))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Ends the training process tree. Progress since the last saved checkpoint is lost.")
        self.stop_button.clicked.connect(self.stop_training)
        for button in (self.export_button, self.start_button, self.stop_button):
            run_buttons.addWidget(button)
        layout.addLayout(run_buttons)
        for widget in self.controls.values():
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.mark_changed)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.mark_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.mark_changed)
            else:
                widget.valueChanged.connect(self.mark_changed)

    @staticmethod
    def add_group(layout, title):
        group = QGroupBox(title)
        layout.addWidget(group)
        return QFormLayout(group)

    def mark_changed(self, *_):
        self.status.setText("Settings changed. The app remembers your draft; Save Settings keeps a separate copy.")
        self.settings_changed.emit()

    def browse_path(self, key):
        current = self.controls[key].text()
        if key in {"model_path", "python_executable"}:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose SDXL Base Model" if key == "model_path" else "Choose Training Python", current,
                "Model files (*.safetensors *.ckpt)" if key == "model_path" else "All files (*)",
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "Choose Folder", current)
        if selected:
            self.controls[key].setText(selected)
            if key == "sd_scripts_dir" and not self.controls["python_executable"].text():
                for relative in (".venv/Scripts/python.exe", "venv/Scripts/python.exe", ".venv/bin/python"):
                    candidate = Path(selected) / relative
                    if candidate.is_file():
                        self.controls["python_executable"].setText(str(candidate))
                        break

    def update_cache_constraint(self, enabled):
        unet_only = self.controls["network_train_unet_only"]
        if enabled:
            self.controls["shuffle_caption"].setChecked(False)
            unet_only.setChecked(True)
        unet_only.setEnabled(not enabled)

    def update_shuffle_constraint(self, enabled):
        if enabled:
            self.controls["cache_text_encoder_outputs"].setChecked(False)

    def keep_tokens_edited(self, *_):
        # Only direct edits reach here; automatic suggestions block this signal.
        with QSignalBlocker(self.controls["keep_tokens_auto"]):
            self.controls["keep_tokens_auto"].setChecked(False)
        self.update_prefix_suggestion()

    def update_prefix_suggestion(self, *_, apply_suggestion=True):
        trigger = self.controls["trigger_word"].text()
        class_token = self.controls["class_token"].text()
        prefix = build_caption_prefix(trigger, class_token)
        self.prefix_preview.setText(prefix or "(none)")
        suggested = suggest_keep_tokens(trigger, class_token)
        automatic = self.controls["keep_tokens_auto"].isChecked()
        if automatic and apply_suggestion:
            with QSignalBlocker(self.controls["keep_tokens"]):
                self.controls["keep_tokens"].setValue(suggested)
        self.keep_tokens_hint.setText(
            f"Suggested: {suggested} — {'automatic' if automatic else 'manual override'}. "
            "Editing Keep Tokens switches to manual mode."
        )

    def set_caption_prefix(self, trigger_word, class_token):
        changed = (
            self.controls["trigger_word"].text() != trigger_word
            or self.controls["class_token"].text() != class_token
        )
        for key, value in (("trigger_word", trigger_word), ("class_token", class_token)):
            with QSignalBlocker(self.controls[key]):
                self.controls[key].setText(value)
        self.update_prefix_suggestion()
        if changed:
            self.mark_changed()

    def apply_config(self, config):
        blockers = [QSignalBlocker(widget) for widget in self.controls.values()]
        for key, widget in self.controls.items():
            value = getattr(config, key)
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QComboBox):
                if key == "attention_backend":
                    widget.setCurrentIndex(widget.findData(value))
                else:
                    widget.setCurrentText(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            else:
                widget.setValue(value)
        del blockers
        self.update_cache_constraint(config.cache_text_encoder_outputs)
        # Loading preserves the saved numeric value and automatic/manual mode.
        self.update_prefix_suggestion(apply_suggestion=False)
        if hasattr(self, "status"):
            self.settings_changed.emit()

    def read_config(self):
        values = {}
        for key, widget in self.controls.items():
            if isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentData() if key == "attention_backend" else widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            else:
                values[key] = widget.value()
        return SDXLTrainConfig(**values)

    def validated_config(self):
        config = self.read_config()
        errors = validate_config(config)
        if errors:
            self.status.setText("Settings need attention.")
            QMessageBox.warning(self, "Check Training Settings", "\n".join(errors))
            return None
        return config

    def check_settings(self):
        if self.validated_config() is not None:
            self.status.setText("Settings and paths checked. Review the dataset before training.")
            QMessageBox.information(
                self, "Settings Checked",
                "Settings and file paths look valid. This does not verify the model's SDXL architecture "
                "or GPU compatibility. Use Review This Dataset to inspect images and captions.",
            )

    def save_settings(self):
        config = self.validated_config()
        if config is None:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save Training Settings",
            str(Path(config.output_dir) / f"{config.output_name}.training.json"),
            "Training settings (*.json)",
        )
        if not destination:
            return
        path = Path(destination)
        if not path.suffix:
            path = path.with_suffix(".json")
            if path.exists() and QMessageBox.question(
                self, "Replace Settings File", f"Replace the existing file {path}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                return
        try:
            save_config(config, path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Save Settings", str(error))
            return
        self.status.setText(f"Settings saved: {path}")
        QMessageBox.information(self, "Settings Saved", f"Saved to {path}\nTraining has not started.")

    def load_settings(self):
        source, _ = QFileDialog.getOpenFileName(self, "Load Training Settings", "", "Training settings (*.json)")
        if not source:
            return
        try:
            config = load_config(Path(source))
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Load Settings", str(error))
            return
        self.apply_config(config)
        self.status.setText(f"Loaded: {source}. Check settings to verify paths on this computer.")

    def review_dataset(self):
        folder = self.controls["dataset_path"].text().strip()
        if not folder:
            QMessageBox.warning(self, "Choose Dataset", "Choose the image dataset folder first.")
            return
        dialog = DatasetReviewDialog(folder, self)
        dialog.exec()
        dialog.deleteLater()

    def begin_training(self, export_only=False):
        if self.worker is not None:
            return
        config = self.validated_config()
        if config is None:
            return
        errors = validate_runtime(config)
        if errors:
            QMessageBox.warning(self, "Training Environment Required", "\n".join(errors))
            return
        if config.resume_state and not export_only:
            answer = QMessageBox.question(
                self, "Resume Trusted Training State",
                "Resume loads optimizer and random-state files, which may contain executable Python data. "
                "Only continue with a state you created or trust. The original run will not be overwritten. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.settings_changed.emit()
        self.logs.clear()
        self.run_directory = None
        self.open_output_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setText("Preparing training files..." if export_only else "Preparing training...")
        self.tabs.setCurrentIndex(1)
        self.settings_panel.setEnabled(False)
        for button in (*self.settings_buttons, self.start_button, self.export_button):
            button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.latest_telemetry = None
        self.telemetry_received_at = time.monotonic()
        self.activity_label.setText("Preparing; waiting for activity...")
        self.activity_timer.start()
        self.worker = TrainingWorker(config, export_only=export_only, parent=self)
        self.worker.output.connect(self.append_log)
        self.worker.status_changed.connect(self.status.setText)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.prepared.connect(self.run_prepared)
        self.worker.completed.connect(self.run_completed)
        self.worker.telemetry.connect(self.receive_telemetry)
        self.worker.finished.connect(self.release_worker)
        self.worker.start()

    def append_log(self, text):
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
        self.logs.moveCursor(QTextCursor.End)
        self.logs.insertPlainText(text)
        self.logs.ensureCursorVisible()

    def update_progress(self, current, total):
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.progress.setFormat(f"Overall step {current} / {total} (%p%)")

    def run_prepared(self, run):
        self.run_directory = run.directory
        self.open_output_button.setEnabled(True)

    def run_completed(self, result):
        self.activity_timer.stop()
        self.status.setText(result.message)
        if result.status == "success":
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("Training complete")
        elif self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        if result.status != "success":
            self.progress.setFormat(result.status.capitalize())

    def release_worker(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.settings_panel.setEnabled(True)
        self.update_cache_constraint(self.controls["cache_text_encoder_outputs"].isChecked())
        for button in (*self.settings_buttons, self.start_button, self.export_button):
            button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.activity_timer.stop()

    def receive_telemetry(self, metrics):
        self.latest_telemetry = metrics
        self.telemetry_received_at = time.monotonic()
        self.refresh_activity()

    def refresh_activity(self):
        if self.latest_telemetry is None:
            return
        metrics = self.latest_telemetry
        extra = max(0, time.monotonic() - self.telemetry_received_at) if self.worker else 0
        elapsed = int(metrics["elapsed_seconds"] + extra)
        silence = int(metrics["silent_seconds"] + extra)
        text = f"{metrics['phase']} | Elapsed {elapsed // 60}:{elapsed % 60:02d} | Last output {silence}s ago"
        rate = metrics.get("steps_per_second")
        if rate:
            if metrics.get("eta_seconds") is not None:
                text += f" | {rate:.3f} steps/s | ETA ~{int(metrics['eta_seconds'] // 60)} min (remaining steps)"
        memory = metrics.get("peak_gpu_memory_mib")
        if memory is not None:
            text += f" | Sampled GPU-wide peak {memory:.0f} MiB"
        if silence >= 60 and self.worker is not None:
            text += "\nNo log output for over a minute. Initialization may be slow; check the log/network/GPU. This alone does not mean the process has failed."
        self.activity_label.setText(text)

    def open_history(self):
        if self.worker is not None:
            return
        dialog = RunHistoryDialog(self.controls["output_dir"].text(), self)
        dialog.settings_selected.connect(self.apply_config)
        dialog.exec()
        dialog.deleteLater()

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.status.setText("Stopping... latest saved checkpoints will be kept.")
            self.stop_button.setEnabled(False)

    def open_run_folder(self):
        if self.run_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_directory)))

    def accept(self):
        if self.worker is None:
            super().accept()

    def reject(self):
        if self.worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.status.setText("Use Stop and wait for the process to finish before closing.")
            event.ignore()
        else:
            super().closeEvent(event)
