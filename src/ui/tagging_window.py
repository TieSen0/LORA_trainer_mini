from pathlib import Path
from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.dataset.captions import (
    add_caption_prefix,
    apply_caption_prefix,
    build_caption_prefix,
    load_caption,
    save_caption,
)
from src.dataset.renamer import create_rename_plan, execute_rename_plan
from src.dataset.scanner import scan_images
from src.tagger.wd14 import WD14Config, WD14Tagger
from src.tagger.worker import TaggingWorker
from src.ui.dataset_review_dialog import DatasetReviewDialog
from src.ui.training_dialog import TrainingDialog
from src.training.config import config_payload, parse_config
from src.training.records import load_session, write_json
from src.ui.design import apply_theme, heading, label, ImagePreview


class TaggingWindow(QWidget):
    def __init__(self, folder_path: str | None = None, *, session_path: Path | None = None):
        super().__init__()
        apply_theme()

        self.folder_path = ""
        self.images = []
        self.index = 0
        self.dataset_prefixes = {}
        self.tagging_prefix = ""
        self.session_path = session_path
        self.remembered_training = None
        self.restoring_session = False
        self.preview_paused = False
        self.release_preview_on_training = True

        self.setWindowTitle("Mini LoRA Trainer")
        self.resize(1120, 780)

        self.image_label = ImagePreview()
        self.filename_label = label("Image preview", "section")
        self.preview_memory_button = QPushButton("Release Preview Memory")
        self.preview_memory_button.setToolTip("Unload the preview without changing images or captions. Navigation remains available.")
        self.preview_memory_button.clicked.connect(self.toggle_preview)

        self.caption_edit = QTextEdit()
        self.caption_edit.setAcceptRichText(False)
        self.caption_edit.setPlaceholderText("Write a caption, or generate tags with WD14 below.")
        self.caption_edit.setMinimumHeight(130)
        self.caption_edit.setAccessibleName("Image caption")

        self.choose_folder_button = QPushButton("Choose Dataset Folder")
        self.choose_folder_button.clicked.connect(self.choose_dataset_folder)
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setPlaceholderText("No dataset selected")

        self.previous_button = QPushButton("Previous")
        self.save_button = QPushButton("Save")
        self.next_button = QPushButton("Next")
        self.position_label = QLabel()
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setAccessibleName("Image position")
        self.position_slider.setRange(0, max(0, len(self.images) - 1))
        self.position_slider.setEnabled(bool(self.images))

        self.trigger_word_edit = QLineEdit()
        self.trigger_word_edit.setPlaceholderText("e.g. tsnr01")
        self.class_token_edit = QLineEdit()
        self.class_token_edit.setPlaceholderText("Optional, e.g. 1girl")
        self.prefix_preview = QLabel("(none)")
        self.prefix_preview.setTextFormat(Qt.PlainText)
        self.prefix_preview.setWordWrap(True)
        self.trigger_word_edit.textChanged.connect(self.update_caption_prefix)
        self.class_token_edit.textChanged.connect(self.update_caption_prefix)
        self.apply_prefix_current_button = QPushButton("Apply to Current")
        self.apply_prefix_all_button = QPushButton("Apply to All")
        self.rename_button = QPushButton("Batch Rename Dataset")
        self.review_button = QPushButton("Review Dataset")
        self.training_button = QPushButton("Training Setup")
        self.training_button.setProperty("role", "primary")
        self.training_dialog = None
        self.training_dataset_path = None
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)

        self.wd14_threshold = QDoubleSpinBox()
        self.wd14_threshold.setRange(0.0, 1.0)
        self.wd14_threshold.setSingleStep(0.05)
        self.wd14_threshold.setValue(0.35)
        self.wd14_excluded_tags = QLineEdit()
        self.wd14_excluded_tags.setPlaceholderText("Excluded tags, separated by commas")
        self.tag_current_button = QPushButton("WD14 Tag Current")
        self.tag_all_button = QPushButton("WD14 Tag All")
        model_directory = WD14Config().model_directory
        local_model_available = all(
            (model_directory / name).is_file()
            for name in ("model.onnx", "selected_tags.csv")
        )
        self.tagging_status = QLabel(
            "Local WD14 model ready."
            if local_model_available
            else "Place model.onnx and selected_tags.csv in models/wd14, or download on first use."
        )
        self.tagging_status.setWordWrap(True)
        self.tagging_worker = None

        self.previous_button.clicked.connect(self.previous_image)
        self.save_button.clicked.connect(self.save_current_caption)
        self.next_button.clicked.connect(self.next_image)
        self.position_slider.valueChanged.connect(self.go_to_image)
        self.apply_prefix_current_button.clicked.connect(self.apply_prefix_to_current)
        self.apply_prefix_all_button.clicked.connect(self.apply_prefix_to_all)
        self.rename_button.clicked.connect(self.batch_rename_dataset)
        self.review_button.clicked.connect(self.review_dataset)
        self.training_button.clicked.connect(self.open_training_setup)
        self.tag_current_button.clicked.connect(self.tag_current_image)
        self.tag_all_button.clicked.connect(self.tag_all_images)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.position_slider, 1)
        button_layout.addWidget(self.position_label)
        button_layout.addWidget(self.next_button)

        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(self.apply_prefix_current_button)
        prefix_layout.addWidget(self.apply_prefix_all_button)
        prefix_form = QFormLayout()
        prefix_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        prefix_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        prefix_form.addRow("Trigger Word:", self.trigger_word_edit)
        prefix_form.addRow("Class Token:", self.class_token_edit)
        prefix_form.addRow("Prefix preview:", self.prefix_preview)
        prefix_form.addRow(prefix_layout)

        tagger_layout = QFormLayout()
        tagger_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        tagger_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        tagger_layout.addRow("WD14 threshold:", self.wd14_threshold)
        tagger_layout.addRow("Exclude tags:", self.wd14_excluded_tags)
        tagger_buttons = QHBoxLayout()
        tagger_buttons.addWidget(self.tag_current_button)
        tagger_buttons.addWidget(self.tag_all_button)

        self.caption_tools = QTabWidget()
        prefix_page, tagger_page = QWidget(), QWidget()
        prefix_page_layout = QVBoxLayout(prefix_page)
        prefix_page_layout.addLayout(prefix_form)
        prefix_page_layout.addWidget(label("Adds prefix tags without replacing your caption."))
        prefix_page_layout.addStretch()
        tagger_page_layout = QVBoxLayout(tagger_page)
        tagger_page_layout.addLayout(tagger_layout)
        tagger_page_layout.addLayout(tagger_buttons)
        tagger_page_layout.addWidget(self.tagging_status)
        tagger_page_layout.addWidget(label("WD14 replaces captions. Review the generated tags."))
        tagger_page_layout.addStretch()
        self.caption_tools.addTab(prefix_page, "Caption prefix")
        self.caption_tools.addTab(tagger_page, "Auto-tag · WD14")

        inspector = QWidget()
        right_layout = QVBoxLayout(inspector)
        right_layout.setContentsMargins(12, 0, 0, 0)
        caption_header = QHBoxLayout()
        caption_header.addWidget(label("Caption", "section"))
        caption_header.addStretch()
        caption_header.addWidget(self.save_button)
        right_layout.addLayout(caption_header)
        right_layout.addWidget(self.caption_edit, 1)
        right_layout.addWidget(label("Changes are saved when you move to another image."))
        right_layout.addWidget(self.caption_tools)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setWidget(inspector)
        inspector_scroll.setMinimumWidth(360)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 12, 0)
        preview_header = QHBoxLayout()
        preview_header.addWidget(self.filename_label, 1)
        preview_header.addWidget(self.preview_memory_button)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.image_label, 1)
        preview_layout.addLayout(button_layout)
        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(preview)
        self.workspace_splitter.addWidget(inspector_scroll)
        self.workspace_splitter.setSizes([660, 440])

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.choose_folder_button)
        folder_layout.addWidget(self.dataset_path_edit, 1)
        folder_layout.addWidget(self.review_button)
        folder_layout.addWidget(self.rename_button)
        window_layout = QVBoxLayout()
        window_layout.setContentsMargins(24, 20, 24, 16)
        window_layout.setSpacing(16)
        header = QHBoxLayout()
        header.addLayout(heading("Mini LoRA Trainer", "Dataset workspace  /  Prepare images and captions"), 1)
        header.addWidget(self.training_button)
        window_layout.addLayout(header)
        window_layout.addLayout(folder_layout)
        window_layout.addWidget(self.workspace_splitter, 1)
        window_layout.addWidget(self.dataset_status)
        self.setLayout(window_layout)

        self.position_label.setText("0 / 0")
        self.image_label.setText("Choose a dataset folder to view images.")
        self.set_tagging_busy(False)
        self.session_timer = QTimer(self)
        self.session_timer.setSingleShot(True)
        self.session_timer.setInterval(500)
        self.session_timer.timeout.connect(self.save_session)
        self.wd14_threshold.valueChanged.connect(self.schedule_session_save)
        self.wd14_excluded_tags.textChanged.connect(self.schedule_session_save)
        if session_path is not None:
            self.restore_session(restore_dataset=not folder_path)
        if folder_path:
            self.open_dataset(folder_path)

    def schedule_session_save(self, *_):
        if self.session_path is not None and not self.restoring_session and hasattr(self, "session_timer"):
            self.session_timer.start()

    def save_session(self):
        if self.session_path is None or self.restoring_session:
            return
        config = self.training_dialog.read_config() if self.training_dialog else self.remembered_training
        try:
            write_json(self.session_path, {
                "version": 1, "dataset": self.folder_path,
                "prefixes": self.dataset_prefixes,
                "training": config_payload(config) if config else None,
                "wd14_threshold": self.wd14_threshold.value(),
                "wd14_excluded_tags": self.wd14_excluded_tags.text(),
                "release_preview_on_training": (
                    self.training_dialog.release_preview_checkbox.isChecked()
                    if self.training_dialog else self.release_preview_on_training
                ),
            })
        except (OSError, ValueError) as error:
            self.dataset_status.setText(f"Could not remember settings: {error}. Save a training settings copy manually.")

    def restore_session(self, *, restore_dataset=True):
        self.restoring_session = True
        try:
            data = load_session(self.session_path)
            release_preview = data.get("release_preview_on_training", True)
            if type(release_preview) is not bool:
                raise ValueError("Invalid remembered preview-memory preference.")
            self.release_preview_on_training = release_preview
            if data.get("training") is not None:
                # Resume must always be an explicit action after reopening the app.
                self.remembered_training = replace(parse_config(data["training"]), resume_state="")
            self.dataset_prefixes = data.get("prefixes", {})
            threshold = data.get("wd14_threshold", 0.35)
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                raise ValueError("Invalid remembered WD14 threshold.")
            excluded = data.get("wd14_excluded_tags", "")
            if not isinstance(excluded, str):
                raise ValueError("Invalid remembered excluded tags.")
            self.wd14_threshold.setValue(threshold)
            self.wd14_excluded_tags.setText(excluded)
            folder = data.get("dataset", "")
            if restore_dataset and folder:
                if Path(folder).is_dir():
                    self.open_dataset(folder)
                else:
                    self.dataset_status.setText("Previous dataset is unavailable. Choose its new location; saved training paths are still retained.")
            if self.remembered_training:
                self.training_dataset_path = self.folder_path or None
        except (OSError, ValueError, TypeError) as error:
            self.dataset_status.setText(f"Could not restore remembered settings: {error}. Original file left unchanged; automatic saving disabled for this session.")
            self.session_path = None
        finally:
            self.restoring_session = False

    @staticmethod
    def dataset_key(folder_path):
        return str(Path(folder_path).resolve())

    def effective_caption_prefix(self):
        return build_caption_prefix(self.trigger_word_edit.text(), self.class_token_edit.text())

    def update_caption_prefix(self, *_):
        self.prefix_preview.setText(self.effective_caption_prefix() or "(none)")
        if self.folder_path:
            self.dataset_prefixes[self.dataset_key(self.folder_path)] = (
                self.trigger_word_edit.text(), self.class_token_edit.text(),
            )
        self.schedule_session_save()

    def set_caption_prefix(self, trigger_word, class_token):
        with QSignalBlocker(self.trigger_word_edit), QSignalBlocker(self.class_token_edit):
            self.trigger_word_edit.setText(trigger_word)
            self.class_token_edit.setText(class_token)
        self.update_caption_prefix()

    def choose_dataset_folder(self):
        if self.tagging_worker is not None:
            return
        folder_path = QFileDialog.getExistingDirectory(
            self, "Choose the folder containing your images", self.folder_path
        )
        if folder_path:
            self.open_dataset(folder_path)

    def open_dataset(self, folder_path: str):
        if self.tagging_worker is not None:
            return
        try:
            images = scan_images(folder_path)
            self.save_caption_if_changed()
            caption = load_caption(images[0]) if images else ""
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "Cannot Open Dataset", str(error))
            return

        self.folder_path = folder_path
        self.set_caption_prefix(*self.dataset_prefixes.get(self.dataset_key(folder_path), ("", "")))
        self.images = images
        self.index = 0
        self.dataset_path_edit.setText(folder_path)
        self.dataset_status.clear()
        with QSignalBlocker(self.position_slider):
            self.position_slider.setRange(0, max(0, len(images) - 1))
            self.position_slider.setValue(0)
        self.set_tagging_busy(False)
        self.schedule_session_save()
        if images:
            self.load_current_image(caption)
        else:
            self.image_label.clear()
            self.image_label.setText(
                "No supported images in this folder.\n"
                "Choose a folder containing PNG, JPG, JPEG, or WebP files.\n"
                "Subfolders are not scanned."
            )
            self.caption_edit.clear()
            self.position_label.setText("0 / 0")
            self.filename_label.setText("Image preview")
            self.setWindowTitle("Mini LoRA Trainer")

    def load_current_image(self, caption: str | None = None):
        image_path = self.images[self.index]

        self.refresh_preview()
        self.filename_label.setText(image_path.name)

        if caption is None:
            caption = load_caption(image_path)
        self.caption_edit.setPlainText(caption)

        # Keep button and slider navigation synchronized without re-entering
        # go_to_image when this value is changed by code.
        with QSignalBlocker(self.position_slider):
            self.position_slider.setValue(self.index)

        self.position_label.setText(f"{self.index + 1} / {len(self.images)}")

        self.setWindowTitle(
            f"Mini LoRA Trainer - {self.index + 1}/{len(self.images)}"
        )

    def refresh_preview(self):
        """Update only image pixels, never reload the caption editor."""
        if self.preview_paused:
            self.image_label.clear()
            self.image_label.setText("Preview paused to save memory.\nClick Load Preview to view the selected image.")
        elif self.images:
            # QPixmap(path) implicitly caches file-loaded pixmaps on the GUI
            # thread. Decode via QImageReader so clearing our two pixmaps really
            # drops all preview-owned pixel references, without flushing UI icons.
            reader = QImageReader(str(self.images[self.index]))
            self.image_label.set_source(QPixmap.fromImage(reader.read()))

    def release_preview_memory(self):
        if not self.images:
            return
        self.preview_paused = True
        self.preview_memory_button.setText("Load Preview")
        self.refresh_preview()

    def toggle_preview(self):
        if not self.images or self.tagging_worker is not None:
            return
        if self.preview_paused:
            self.preview_paused = False
            self.preview_memory_button.setText("Release Preview Memory")
            self.refresh_preview()
        else:
            self.release_preview_memory()

    def save_current_caption(self):
        if not self.images:
            return

        image_path = self.images[self.index]
        caption = self.caption_edit.toPlainText()

        save_caption(image_path, caption)

        print("Saved:", image_path)

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.save_caption_if_changed()
            self.index += 1
            self.load_current_image()

    def previous_image(self):
        if self.index > 0:
            self.save_caption_if_changed()
            self.index -= 1
            self.load_current_image()

    def go_to_image(self, index: int):
        if not self.images or index == self.index:
            return

        self.save_caption_if_changed()
        self.index = index
        self.load_current_image()

    def save_caption_if_changed(self):
        if not self.images:
            return

        image_path = self.images[self.index]
        caption = self.caption_edit.toPlainText()
        if caption != load_caption(image_path):
            save_caption(image_path, caption)

    def apply_prefix_to_current(self):
        if not self.images:
            return

        try:
            caption = add_caption_prefix(
                self.caption_edit.toPlainText(), self.effective_caption_prefix()
            )
        except ValueError as error:
            QMessageBox.warning(self, "Caption Prefix", str(error))
            return

        self.caption_edit.setPlainText(caption)
        self.save_current_caption()

    def apply_prefix_to_all(self):
        if not self.images:
            return

        prefix = self.effective_caption_prefix()
        try:
            add_caption_prefix("", prefix)
        except ValueError as error:
            QMessageBox.warning(self, "Caption Prefix", str(error))
            return

        answer = QMessageBox.question(
            self,
            "Apply Caption Prefix",
            f'Add "{prefix.strip().strip(",")}" to every caption in this dataset?',
        )
        if answer != QMessageBox.Yes:
            return

        self.save_caption_if_changed()
        changed_images = apply_caption_prefix(self.images, prefix)
        self.load_current_image()
        QMessageBox.information(
            self,
            "Caption Prefix",
            f"Updated {len(changed_images)} caption(s).",
        )

    def batch_rename_dataset(self):
        if not self.images:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Rename Dataset")

        prefix_edit = QLineEdit()
        prefix_edit.setPlaceholderText("e.g. role")
        start_spin = QSpinBox()
        start_spin.setRange(0, 1_000_000)
        start_spin.setValue(1)
        padding_spin = QSpinBox()
        padding_spin.setRange(1, 10)
        padding_spin.setValue(3)

        form_layout = QFormLayout(dialog)
        form_layout.addRow("Name prefix:", prefix_edit)
        form_layout.addRow("Start index:", start_spin)
        form_layout.addRow("Padding:", padding_spin)

        dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        dialog_buttons.accepted.connect(dialog.accept)
        dialog_buttons.rejected.connect(dialog.reject)
        form_layout.addRow(dialog_buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        try:
            # Include a newly created caption file in the paired rename plan.
            self.save_caption_if_changed()
            plan = create_rename_plan(
                self.images,
                prefix_edit.text(),
                start_spin.value(),
                padding_spin.value(),
            )
        except (ValueError, OSError, UnicodeError) as error:
            QMessageBox.warning(self, "Cannot Rename Dataset", str(error))
            return

        answer = QMessageBox.question(
            self,
            "Confirm Batch Rename",
            f"Rename {plan.file_count} file(s), including {len(plan.operations)} image(s)?",
        )
        if answer != QMessageBox.Yes:
            return

        try:
            execute_rename_plan(plan)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Rename Failed", str(error))
            return

        renamed_images = sum(
            operation.source_image != operation.target_image for operation in plan.operations
        )
        renamed_captions = sum(
            operation.source_caption is not None
            and operation.source_caption != operation.target_caption
            for operation in plan.operations
        )
        summary = (
            f"Rename complete: {renamed_images} image(s) and {renamed_captions} caption file(s) renamed."
            if renamed_images or renamed_captions
            else "Rename complete: filenames already match; no files needed renaming."
        )
        self.dataset_status.setText(summary)
        current_image = plan.operations[self.index].target_image
        try:
            self.images = scan_images(self.folder_path)
            self.index = next(
                index for index, image in enumerate(self.images)
                if image.resolve() == current_image
            )
            with QSignalBlocker(self.position_slider):
                self.position_slider.setRange(0, len(self.images) - 1)
                self.position_slider.setValue(self.index)
            self.load_current_image()
        except (OSError, UnicodeError, StopIteration) as error:
            QMessageBox.warning(
                self, "Rename Completed; Refresh Failed",
                f"{summary}\nPlease reopen the dataset to refresh the preview.\n{error}",
            )
            return
        QMessageBox.information(self, "Rename Complete", summary)

    def review_dataset(self):
        if self.tagging_worker is not None:
            return
        folder_path = self.folder_path or QFileDialog.getExistingDirectory(
            self, "Choose a dataset folder to review"
        )
        if not folder_path:
            return
        try:
            self.save_caption_if_changed()
        except (OSError, UnicodeError) as error:
            QMessageBox.warning(
                self, "Caption Not Saved",
                f"{error}\nReview will inspect files on disk. Your editor text is unchanged.",
            )
        dialog = DatasetReviewDialog(folder_path, self)
        dialog.exec()
        dialog.deleteLater()

    def tag_current_image(self):
        if self.images:
            if self.caption_edit.toPlainText().strip():
                answer = QMessageBox.question(
                    self,
                    "WD14 Tag Current Image",
                    "Replace this image's caption with generated WD14 tags?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return
            self.start_tagging([self.images[self.index]])

    def open_training_setup(self):
        if self.tagging_worker is not None:
            return
        try:
            self.save_caption_if_changed()
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "Cannot Save Caption", str(error))
            return
        if self.training_dialog is None:
            self.training_dialog = TrainingDialog(self)
            self.training_dialog.release_preview_checkbox.setChecked(self.release_preview_on_training)
            self.training_dialog.release_preview_requested.connect(self.release_preview_memory)
            if self.remembered_training is not None:
                self.training_dialog.apply_config(self.remembered_training)
            self.training_dialog.settings_changed.connect(self.schedule_session_save)
        if self.folder_path and self.folder_path != self.training_dataset_path:
            if self.training_dialog.read_config().resume_state:
                self.training_dialog.start_new_training()
            self.training_dialog.controls["dataset_path"].setText(self.folder_path)
            self.training_dataset_path = self.folder_path
        if (self.folder_path and not self.training_dialog.read_config().resume_state
                and self.dataset_key(self.training_dialog.read_config().dataset_path) == self.dataset_key(self.folder_path)):
            self.training_dialog.set_caption_prefix(self.trigger_word_edit.text(), self.class_token_edit.text())
        self.training_dialog.exec()
        config = self.training_dialog.read_config()
        if config.dataset_path:
            key = self.dataset_key(config.dataset_path)
            self.dataset_prefixes[key] = (config.trigger_word, config.class_token)
            if self.folder_path and key == self.dataset_key(self.folder_path):
                self.set_caption_prefix(config.trigger_word, config.class_token)
        self.save_session()

    def tag_all_images(self):
        if not self.images:
            return

        answer = QMessageBox.question(
            self,
            "WD14 Tag Dataset",
            "Replace captions for every image with generated WD14 tags?",
        )
        if answer == QMessageBox.Yes:
            self.start_tagging(self.images)

    def start_tagging(self, image_paths):
        if self.tagging_worker is not None:
            return

        excluded_tags = frozenset(
            tag.strip()
            for tag in self.wd14_excluded_tags.text().split(",")
            if tag.strip()
        )
        try:
            tagger = WD14Tagger(
                WD14Config(
                    threshold=self.wd14_threshold.value(),
                    excluded_tags=excluded_tags,
                )
            )
        except ValueError as error:
            QMessageBox.warning(self, "WD14 Tagger", str(error))
            return

        try:
            self.save_caption_if_changed()
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "Cannot Save Caption", str(error))
            return
        self.tagging_prefix = self.effective_caption_prefix()
        self.tagging_worker = TaggingWorker(tagger, list(image_paths))
        self.tagging_worker.progress_changed.connect(self.update_tagging_progress)
        self.tagging_worker.completed.connect(self.save_generated_captions)
        self.tagging_worker.failed.connect(self.show_tagging_error)
        self.tagging_worker.finished.connect(self.finish_tagging)
        self.set_tagging_busy(True)
        self.tagging_status.setText("Loading WD14 model and generating captions...")
        self.tagging_worker.start()

    def update_tagging_progress(self, completed: int, total: int):
        self.tagging_status.setText(f"WD14 tagged {completed} / {total} image(s).")

    def save_generated_captions(self, predictions):
        saved_count = 0
        try:
            for image_path, prediction in predictions.items():
                caption = (
                    add_caption_prefix(prediction.caption, self.tagging_prefix)
                    if self.tagging_prefix else prediction.caption
                )
                save_caption(image_path, caption)
                saved_count += 1
            self.load_current_image()
        except (OSError, UnicodeError) as error:
            self.show_tagging_error(
                f"Saved {saved_count} of {len(predictions)} captions before an error: {error}"
            )
            return
        self.tagging_status.setText(f"WD14 updated {len(predictions)} caption(s).")

    def show_tagging_error(self, message: str):
        self.tagging_status.setText("WD14 tagging failed.")
        QMessageBox.critical(self, "WD14 Tagger", message)

    def finish_tagging(self):
        worker = self.tagging_worker
        self.tagging_worker = None
        self.set_tagging_busy(False)
        if worker is not None:
            worker.deleteLater()

    def set_tagging_busy(self, busy: bool):
        self.choose_folder_button.setEnabled(not busy)
        self.review_button.setEnabled(not busy)
        self.training_button.setEnabled(not busy)
        for widget in (
            self.caption_edit, self.previous_button, self.next_button,
            self.save_button, self.position_slider, self.trigger_word_edit, self.class_token_edit,
            self.preview_memory_button,
            self.apply_prefix_current_button, self.apply_prefix_all_button,
            self.rename_button, self.wd14_threshold, self.wd14_excluded_tags,
            self.tag_current_button, self.tag_all_button,
        ):
            widget.setEnabled(not busy and bool(self.images))

    def closeEvent(self, event):
        if self.training_dialog is not None and self.training_dialog.worker is not None:
            QMessageBox.information(
                self, "Training in Progress", "Stop training and wait for it to finish before closing."
            )
            event.ignore()
            return
        if self.tagging_worker is not None:
            QMessageBox.information(
                self, "Tagging in Progress", "Please wait for tagging to finish before closing."
            )
            event.ignore()
            return
        try:
            self.save_caption_if_changed()
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "Cannot Save Caption", str(error))
            event.ignore()
            return
        self.save_session()
        event.accept()
