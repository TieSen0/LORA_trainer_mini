"""SDXL setup and monitoring UI; execution lives in the training modules."""

from pathlib import Path
from dataclasses import replace
import re
import time

from PySide6.QtCore import QSignalBlocker, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QAction, QDesktopServices, QTextCursor

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QMenu,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget, QPlainTextEdit, QProgressBar, QTabWidget,
)

from src.training.config import (
    OPTIMIZERS, PRECISIONS, SCHEDULERS, SDXLTrainConfig,
    load_config, save_config, validate_config, validate_runtime,
)
from src.training.runner import TrainingWorker
from src.training.resources import resource_summary
from src.training.resume import load_resume_config, load_resume_from_folder, resume_progress
from src.dataset.captions import build_caption_prefix, suggest_keep_tokens
from src.ui.dataset_review_dialog import DatasetReviewDialog
from src.ui.environment_dialog import EnvironmentDialog
from src.ui.run_history_dialog import RunHistoryDialog
from src.ui.design import apply_theme, heading


class TrainingDialog(QDialog):
    settings_changed = Signal()
    release_preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        apply_theme()
        self.setWindowTitle("SDXL LoRA Training Setup")
        self.resize(900, 800)
        self.controls = {}
        self.worker = None
        self.run_directory = None
        self.checkpoint_status = None
        self.new_training_draft = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addLayout(heading("Train your LoRA", "SDXL  /  Configure, train, and recover"), 1)
        layout.addLayout(header)
        self.mode_label = QLabel("New training — no checkpoint selected.")
        self.mode_label.setTextFormat(Qt.PlainText)
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)
        self.settings_sections = QTabWidget()
        section_layouts = {}
        for title in ("Training", "Captions", "Recovery", "Environment"):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setSpacing(12)
            scroll.setWidget(page)
            self.settings_sections.addTab(scroll, title)
            section_layouts[title] = page_layout
        content_layout = section_layouts["Training"]
        self.tabs = QTabWidget()
        self.tabs.addTab(self.settings_sections, "Settings")
        layout.addWidget(self.tabs)
        self.settings_panel = self.settings_sections

        paths = self.add_group(content_layout, "Model && dataset")
        environment = self.add_group(section_layouts["Environment"], "Training runtime")
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
            (environment if key in {"sd_scripts_dir", "python_executable"} else paths).addRow(label, row)
        self.controls["sd_scripts_dir"].setPlaceholderText("Folder containing sdxl_train_network.py")
        self.controls["python_executable"].setPlaceholderText("Prepared environment, e.g. .venv/Scripts/python.exe")
        self.controls["output_name"] = QLineEdit()
        self.controls["output_name"].setPlaceholderText("Name without .safetensors")
        paths.addRow("Output model name:", self.controls["output_name"])
        review_button = QPushButton("Review This Dataset")
        review_button.clicked.connect(self.review_dataset)
        paths.addRow("Dataset check:", review_button)

        self.controls["offline_mode"] = QCheckBox("Offline Mode (use cached Hugging Face files)")
        environment.addRow(self.controls["offline_mode"])
        offline_note = QLabel(
            "Offline mode requires cached tokenizers. It does not affect WD14."
        )
        offline_note.setWordWrap(True)
        offline_note.setProperty("role", "muted")
        environment.addRow(offline_note)
        self.environment_button = QPushButton("Check Environment")
        self.environment_button.setToolTip("Read-only desktop, training imports, CUDA and path checks. No training, downloads or repairs.")
        self.environment_button.clicked.connect(self.check_environment)
        environment.addRow(self.environment_button)

        recovery = self.add_group(section_layouts["Recovery"], "Checkpoints && resume")
        for key, title in (("save_training_state", "Save resumable state at checkpoints (extra disk space)"),
                           ("measure_performance", "Measure performance (RAM, CPU, disk and GPU)")):
            self.controls[key] = QCheckBox(title)
            (environment if key == "measure_performance" else recovery).addRow(self.controls[key])
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
        self.controls["resume_state"].setReadOnly(True)
        self.controls["resume_state"].setPlaceholderText("Choose a run folder to select its latest checkpoint")
        resume_row = QHBoxLayout()
        resume_row.addWidget(self.controls["resume_state"])
        resume_button = QPushButton("Choose Run...")
        resume_button.setToolTip("Select one LoRA run folder. Its latest complete state and original settings are selected automatically. Use History to choose an older state.")
        resume_button.clicked.connect(lambda: self.browse_path("resume_state"))
        resume_row.addWidget(resume_button)
        clear_resume = QPushButton("New Training")
        clear_resume.setToolTip("Leave resume mode and restore the new-training draft you had before selecting a checkpoint.")
        clear_resume.clicked.connect(self.start_new_training)
        resume_row.addWidget(clear_resume)
        recovery.addRow("Selected checkpoint:", resume_row)
        recovery_note = QLabel(
            "Set an interval above 0 (e.g. 100) to also save every N steps; keep Save resumable state enabled for recovery. "
            "0 = epoch checkpoints only. Resume keeps the original epoch target and uses a new run folder. "
            "Stop does not save a new checkpoint. Only load states you trust."
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

        dataset = self.add_group(section_layouts["Captions"], "Caption settings")
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
            "Shared with the caption workspace. Use Apply or WD14 there to write prefix tags to files. "
            "Repeats are configured here, not read from folder names."
        )
        prefix_note.setWordWrap(True)
        dataset.addRow(prefix_note)
        self.controls["keep_tokens"].valueChanged.connect(self.keep_tokens_edited)
        self.controls["keep_tokens_auto"].toggled.connect(self.update_prefix_suggestion)
        self.controls["shuffle_caption"].toggled.connect(self.update_shuffle_constraint)

        memory = self.add_group(section_layouts["Environment"], "Memory && training scope")
        # Desktop preference, intentionally not a trainer/resume configuration field.
        self.release_preview_checkbox = QCheckBox("Release preview memory when training starts")
        self.release_preview_checkbox.setChecked(True)
        self.release_preview_checkbox.setToolTip("Pauses the dataset preview until you click Load Preview. Does not affect export-only runs or training images.")
        self.release_preview_checkbox.toggled.connect(self.mark_changed)
        memory.addRow(self.release_preview_checkbox)
        for key, label in (
            ("cache_latents", "Cache latents"),
            ("cache_latents_to_disk", "Cache latents to disk (reduce RAM use)"),
            ("cache_text_encoder_outputs", "Cache text-encoder outputs"),
            ("gradient_checkpointing", "Gradient checkpointing"),
            ("network_train_unet_only", "Train U-Net only (keep text encoders frozen)"),
        ):
            check = QCheckBox(label)
            self.controls[key] = check
            memory.addRow(check)
        explanation = QLabel(
            "Disk latents: automatically creates/reuses .npz files in the selected dataset. Switching folders uses that folder's caches. "
            "Replacing images in place or changing the model/VAE requires rebuilding old caches. "
            "Text-encoder caching requires U-Net-only training and disables caption shuffling."
        )
        explanation.setWordWrap(True)
        memory.addRow(explanation)
        self.controls["cache_latents"].setToolTip("Reuse VAE-encoded images during training. Stored in RAM unless Cache latents to disk is enabled.")
        self.controls["cache_latents_to_disk"].setToolTip("Requires Cache latents and a writable dataset folder. Uses disk space and disk reads; does not guarantee faster training. Existing caches are not automatically invalidated when the image contents or model/VAE change.")
        self.controls["optimizer"].setToolTip("AdamW8bit requires bitsandbytes in the training environment.")
        self.controls["measure_performance"].setToolTip("Samples every 5 seconds. CPU/disk/RAM pressure are system-wide; GPU readings are GPU 0 board-wide. Trainer RAM sums its process tree and may count shared pages more than once. Hints are not diagnoses.")
        self.controls["offline_mode"].setToolTip("Disables Hugging Face network checks for training. Missing cached files cause an error; there is no online fallback.")
        for page_layout in section_layouts.values():
            page_layout.addStretch()
        self.controls["cache_text_encoder_outputs"].toggled.connect(self.update_cache_constraint)
        self.controls["cache_latents"].toggled.connect(self.update_latent_cache_constraint)
        self.apply_config(SDXLTrainConfig())

        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(16, 16, 16, 16)
        monitor_layout.setSpacing(12)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(3000)
        monitor_layout.addWidget(self.progress)
        self.activity_label = QLabel("No active run.")
        self.activity_label.setWordWrap(True)
        self.activity_label.setProperty("role", "muted")
        monitor_layout.addWidget(self.activity_label)
        self.resource_label = QLabel("Resource monitoring is optional; enable it in Environment.")
        self.resource_label.setTextFormat(Qt.PlainText)
        self.resource_label.setWordWrap(True)
        self.resource_label.setToolTip(self.controls["measure_performance"].toolTip())
        monitor_layout.addWidget(self.resource_label)
        self.checkpoint_label = QLabel("No checkpoint information yet.")
        self.checkpoint_label.setTextFormat(Qt.PlainText)
        self.checkpoint_label.setWordWrap(True)
        self.checkpoint_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        checkpoint_card = QGroupBox("Recovery point")
        checkpoint_layout = QVBoxLayout(checkpoint_card)
        checkpoint_layout.addWidget(self.checkpoint_label)
        monitor_layout.addWidget(checkpoint_card)
        checkpoint_buttons = QHBoxLayout()
        self.open_checkpoint_button = QPushButton("Open Checkpoint Folder")
        self.open_checkpoint_button.setEnabled(False)
        self.open_checkpoint_button.clicked.connect(self.open_checkpoint_folder)
        checkpoint_buttons.addWidget(self.open_checkpoint_button)
        self.prepare_latest_button = QPushButton("Prepare Latest Resume")
        self.prepare_latest_button.setEnabled(False)
        self.prepare_latest_button.clicked.connect(self.prepare_latest_resume)
        checkpoint_buttons.addWidget(self.prepare_latest_button)
        checkpoint_layout.addLayout(checkpoint_buttons)
        self.latest_telemetry = None
        self.telemetry_received_at = None
        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(1000)
        self.activity_timer.timeout.connect(self.refresh_activity)
        monitor_layout.addWidget(self.logs, 1)
        self.open_output_button = QPushButton("Open Run Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_run_folder)
        log_actions = QHBoxLayout()
        log_actions.addStretch()
        log_actions.addWidget(self.open_output_button)
        monitor_layout.addLayout(log_actions)
        self.tabs.addTab(monitor, "Training Log")

        self.status = QLabel("Draft settings are remembered automatically.")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "muted")
        layout.addWidget(self.status)
        self.settings_buttons = []
        settings_button = QPushButton("Settings && export")
        self.settings_menu = QMenu(settings_button)
        self.settings_menu.addAction("Save Settings...", self.save_settings)
        self.settings_menu.addAction("Load Settings...", self.load_settings)
        self.settings_menu.addSeparator()
        self.export_button = QAction("Export Training Files", self)
        self.export_button.triggered.connect(lambda: self.begin_training(export_only=True))
        self.settings_menu.addAction(self.export_button)
        settings_button.setMenu(self.settings_menu)
        history_button = QPushButton("History / Compare")
        history_button.clicked.connect(self.open_history)
        header.addWidget(history_button)
        header.addWidget(settings_button)
        self.settings_buttons.extend((history_button, settings_button))
        run_buttons = QHBoxLayout()
        for title, handler in (("Check Settings", self.check_settings), ("Close", self.accept)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            run_buttons.addWidget(button)
            self.settings_buttons.append(button)
        run_buttons.addStretch()
        self.choose_resume_button = QPushButton("Resume from checkpoint...")
        self.choose_resume_button.clicked.connect(self.open_history)
        self.choose_resume_button.setToolTip("Choose any run and saved checkpoint; its original settings and paths load automatically.")
        run_buttons.addWidget(self.choose_resume_button)
        self.settings_buttons.append(self.choose_resume_button)
        self.start_button = QPushButton("Start Training")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(lambda: self.begin_training(export_only=False))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Ends the training process tree. Progress since the last saved checkpoint is lost.")
        self.stop_button.clicked.connect(self.stop_training)
        for button in (self.stop_button, self.start_button):
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
        self.controls["resume_state"].textChanged.connect(self.update_training_mode)
        self.update_training_mode()

    @staticmethod
    def add_group(layout, title):
        group = QGroupBox(title)
        layout.addWidget(group)
        form = QFormLayout(group)
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        return form

    def mark_changed(self, *_):
        self.status.setText("Settings changed. The app remembers your draft; Save Settings keeps a separate copy.")
        self.settings_changed.emit()

    def browse_path(self, key):
        current = self.controls[key].text()
        if key == "resume_state":
            initial = str(Path(current).parent) if current else self.controls["output_dir"].text()
            selected = QFileDialog.getExistingDirectory(self, "Choose a LoRA Run Folder (contains settings.json and state folders)", initial)
        elif key in {"model_path", "python_executable"}:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose SDXL Base Model" if key == "model_path" else "Choose Training Python", current,
                "Model files (*.safetensors *.ckpt)" if key == "model_path" else "All files (*)",
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "Choose Folder", current)
        if selected:
            if key == "resume_state":
                self.prepare_resume_from_folder(selected)
                return
            self.controls[key].setText(selected)
            if key == "sd_scripts_dir" and not self.controls["python_executable"].text():
                for relative in (".venv/Scripts/python.exe", "venv/Scripts/python.exe", ".venv/bin/python"):
                    candidate = Path(selected) / relative
                    if candidate.is_file():
                        self.controls["python_executable"].setText(str(candidate))
                        break

    def update_latent_cache_constraint(self, enabled):
        disk_cache = self.controls["cache_latents_to_disk"]
        if not enabled:
            disk_cache.setChecked(False)
        disk_cache.setEnabled(enabled)

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
        if config.resume_state and not self.controls["resume_state"].text():
            self.new_training_draft = self.read_config()
        elif not config.resume_state:
            self.new_training_draft = None
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
        self.update_latent_cache_constraint(config.cache_latents)
        # Loading preserves the saved numeric value and automatic/manual mode.
        self.update_prefix_suggestion(apply_suggestion=False)
        self.update_training_mode()
        if hasattr(self, "status"):
            self.settings_changed.emit()

    def update_training_mode(self, *_):
        state = self.controls["resume_state"].text().strip()
        self.controls["resume_state"].setToolTip(state or "No checkpoint selected; new training.")
        if hasattr(self, "start_button"):
            self.start_button.setText("Resume Training" if state else "Start Training")
        if not state:
            self.mode_label.setText("New training — no checkpoint selected.")
            return
        try:
            progress = resume_progress(state)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            progress = f"Checkpoint needs attention: {error}"
        self.mode_label.setText(f"Resume selected: {Path(state).parent.name} / {Path(state).name}\n{progress}")

    def start_new_training(self):
        if self.worker is not None:
            return
        config = self.new_training_draft or replace(self.read_config(), resume_state="")
        self.apply_config(replace(config, resume_state=""))
        self.tabs.setCurrentIndex(0)
        self.settings_sections.setCurrentIndex(0)
        self.status.setText("New training selected. No checkpoint will be loaded.")

    def prepare_resume_from_path(self, state):
        if self.worker is not None:
            return
        try:
            config = load_resume_config(state)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            QMessageBox.warning(self, "Cannot Prepare Resume", str(error))
            return
        self.apply_config(config)
        self.tabs.setCurrentIndex(0)
        self.settings_sections.setCurrentIndex(2)
        self.status.setText("Original settings and paths loaded. Review the selected checkpoint, then click Resume Training.")

    def prepare_resume_from_folder(self, folder):
        if self.worker is not None:
            return
        try:
            config = load_resume_from_folder(folder)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            QMessageBox.warning(self, "Cannot Prepare Resume", str(error))
            return
        self.apply_config(config)
        self.tabs.setCurrentIndex(0)
        self.settings_sections.setCurrentIndex(2)
        self.status.setText("Checkpoint selected and original settings loaded. Check the saved step above, then click Resume Training.")

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

    def check_environment(self):
        if self.worker is not None:
            return
        dialog = EnvironmentDialog(self.read_config(), self)
        dialog.exec()
        dialog.deleteLater()

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
        if config.resume_state:
            self.prepare_resume_from_path(config.resume_state)
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
                self.mode_label.text() + "\n\nResume loads optimizer and random-state files, which may contain executable Python data. "
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
        self.checkpoint_status = None
        self.checkpoint_label.setText("Checking recovery points..." if not export_only else "Export only; no checkpoint will be saved.")
        self.open_checkpoint_button.setEnabled(False)
        self.prepare_latest_button.setEnabled(False)
        self.activity_label.setText("Preparing; waiting for activity...")
        self.resource_label.setText(resource_summary({} if config.measure_performance and not export_only else None, "Preparing"))
        self.activity_timer.start()
        self.worker = TrainingWorker(config, export_only=export_only, parent=self)
        self.worker.output.connect(self.append_log)
        self.worker.status_changed.connect(self.status.setText)
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.prepared.connect(self.run_prepared)
        self.worker.completed.connect(self.run_completed)
        self.worker.telemetry.connect(self.receive_telemetry)
        self.worker.finished.connect(self.release_worker)
        if not export_only and self.release_preview_checkbox.isChecked():
            self.release_preview_requested.emit()
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
        self.refresh_checkpoint_status()
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
        self.update_latent_cache_constraint(self.controls["cache_latents"].isChecked())
        for button in (*self.settings_buttons, self.start_button, self.export_button):
            button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.activity_timer.stop()
        self.refresh_checkpoint_status()

    def receive_telemetry(self, metrics):
        self.latest_telemetry = metrics
        self.checkpoint_status = metrics.get("checkpoint_status")
        self.telemetry_received_at = time.monotonic()
        self.refresh_activity()
        self.refresh_checkpoint_status()

    def refresh_checkpoint_status(self):
        status = self.checkpoint_status
        if status is None:
            return
        latest = status.get("latest")
        metrics = self.latest_telemetry or {}
        current, total = metrics.get("current", 0), metrics.get("total", 0)
        if latest:
            title = "Selected resume checkpoint (files checked)" if latest["source"] else "Last complete checkpoint"
            text = f"{title}: step {latest['step']}" + (f" / {total}" if total else "")
            text += f"\nFolder: {latest['path']}\nSaved (UTC): {latest['saved_at']}"
        else:
            text = "No complete checkpoint available yet."
        risk = max(0, current - (latest["step"] if latest else 0))
        text += f"\nEstimated steps at risk: {risk} (reported progress since the recovery point)."
        if not latest:
            text += " Without a recovery point, this run cannot be resumed."
        if status.get("pending_count"):
            text += "\nA checkpoint is incomplete or its save is not yet confirmed; it is not counted as a recovery point."
        if not status.get("saving_enabled", True):
            text += "\nState saving is disabled; no new resumable checkpoints are expected."
        if status.get("error"):
            text += f"\nCheckpoint check: {status['error']}"
        text += "\nStop does not save a new checkpoint. The trainer verifies state contents when loading."
        self.checkpoint_label.setText(text)
        self.open_checkpoint_button.setEnabled(bool(latest))
        self.prepare_latest_button.setEnabled(bool(latest and latest["can_resume"] and self.worker is None))

    def open_checkpoint_folder(self):
        latest = (self.checkpoint_status or {}).get("latest")
        if latest:
            path = Path(latest["path"])
            if not path.is_dir():
                QMessageBox.warning(self, "Checkpoint Unavailable", "The checkpoint folder no longer exists.")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def prepare_latest_resume(self):
        latest = (self.checkpoint_status or {}).get("latest")
        if self.worker is not None or not latest or not latest["can_resume"]:
            return
        self.prepare_resume_from_path(latest["path"])

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
        self.resource_label.setText(resource_summary(metrics.get("resources"), metrics["phase"]))

    def open_history(self):
        if self.worker is not None:
            return
        dialog = RunHistoryDialog(self.controls["output_dir"].text(), self)
        dialog.settings_selected.connect(self.apply_history_selection)
        dialog.exec()
        dialog.deleteLater()

    def apply_history_selection(self, config):
        if config.resume_state:
            self.prepare_resume_from_path(config.resume_state)
        else:
            self.apply_config(config)
            self.tabs.setCurrentIndex(0)
            self.settings_sections.setCurrentIndex(0)
            self.status.setText("Settings copied for NEW training. No checkpoint will be loaded.")

    def stop_training(self):
        if self.worker is not None:
            answer = QMessageBox.question(
                self, "Stop Without Saving a New Checkpoint?",
                self.checkpoint_label.text() + "\n\nTraining may advance while this question is open. "
                "Unsaved steps will be lost. Stop now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes or self.worker is None:
                return
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
