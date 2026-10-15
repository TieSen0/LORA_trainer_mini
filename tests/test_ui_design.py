"""Presentation regressions; no model loading or GPU training."""

import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtGui import QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

from src.training.config import SDXLTrainConfig
from src.ui.run_history_dialog import RunHistoryDialog
from src.ui.tagging_window import TaggingWindow
from src.ui.training_dialog import TrainingDialog


class UIDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Windows' offscreen plugin has no system fonts. Supply installed fonts
        # for meaningful geometry checks without changing the desktop app.
        if not QFontDatabase.families():
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for name in ("segoeui.ttf", "seguisb.ttf", "segoeuib.ttf"):
                if (fonts / name).is_file():
                    QFontDatabase.addApplicationFont(str(fonts / name))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, color in (("one", "red"), ("two", "blue")):
            Image.new("RGB", (600, 800), color).save(self.root / f"{name}.png")
            (self.root / f"{name}.txt").write_text(name, encoding="utf-8")

    def settle(self):
        # Nested layouts post geometry updates on subsequent event passes.
        for _ in range(8):
            self.app.processEvents()

    def show(self, window):
        self.addCleanup(window.close)
        window.show()
        self.settle()
        return window

    def test_dark_palette_covers_native_controls_and_inactive_windows(self):
        self.show(TrainingDialog())
        palette = self.app.palette()
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            for role in (QPalette.Window, QPalette.Base, QPalette.Button, QPalette.ToolTipBase):
                self.assertLess(palette.color(group, role).lightnessF(), .3)
            self.assertGreater(palette.color(group, QPalette.Text).lightnessF(), .45)
        self.assertNotEqual(palette.color(QPalette.Active, QPalette.Text),
                            palette.color(QPalette.Disabled, QPalette.Text))

    def test_navigation_and_autosave_survive_new_layout(self):
        window = self.show(TaggingWindow(str(self.root)))
        self.assertEqual(window.position_slider.maximum(), 1)
        window.caption_edit.setPlainText("edited caption")
        with patch.object(window, "load_current_image", wraps=window.load_current_image) as load:
            window.next_button.click()
            self.assertEqual(load.call_count, 1)  # Programmatic slider sync must not reload.
            self.assertEqual(window.position_slider.value(), 1)
            self.assertEqual(window.position_label.text(), "2 / 2")
            self.assertEqual(window.caption_edit.toPlainText(), "two")
            window.position_slider.setValue(0)
            self.assertEqual(load.call_count, 2)
        self.assertEqual(window.caption_edit.toPlainText(), "edited caption")
        self.assertEqual((self.root / "one.txt").read_text(encoding="utf-8"), "edited caption")
        window.next_button.click()
        window.previous_button.click()
        self.assertEqual(window.position_slider.value(), 0)
        window.caption_edit.setPlainText("saved explicitly")
        window.save_button.click()
        self.assertEqual((self.root / "one.txt").read_text(encoding="utf-8"), "saved explicitly")

    def test_resize_preserves_unsaved_caption_and_scales_preview(self):
        window = self.show(TaggingWindow(str(self.root)))
        window.caption_edit.setPlainText("not saved yet")
        with patch.object(window, "load_current_image") as load:
            window.resize(900, 680)
            window.workspace_splitter.setSizes([470, 360])
            self.settle()
            load.assert_not_called()
        self.assertEqual(window.index, 0)
        self.assertEqual(window.caption_edit.toPlainText(), "not saved yet")
        self.assertEqual((self.root / "one.txt").read_text(encoding="utf-8"), "one")
        preview = window.image_label
        self.assertLessEqual(preview.pixmap().width(), preview.contentsRect().width())
        self.assertLessEqual(preview.pixmap().height(), preview.contentsRect().height())
        self.assertAlmostEqual(preview.pixmap().width() / preview.pixmap().height(), .75, delta=.01)
        inspector = window.workspace_splitter.widget(1)
        for index in (0, 1):
            window.caption_tools.setCurrentIndex(index)
            self.settle()
            self.assertEqual(inspector.horizontalScrollBar().maximum(), 0)

    def test_unreadable_image_does_not_keep_previous_preview(self):
        window = self.show(TaggingWindow(str(self.root)))
        (self.root / "two.png").write_bytes(b"invalid image")
        window.next_image()
        self.assertTrue(window.image_label.source.isNull())
        self.assertTrue(window.image_label.pixmap().isNull())
        self.assertIn("Unable to preview", window.image_label.text())
        self.assertEqual(window.caption_edit.toPlainText(), "two")
        window.previous_image()
        self.assertFalse(window.image_label.pixmap().isNull())

    def test_release_and_reload_do_not_touch_unsaved_caption(self):
        window = self.show(TaggingWindow(str(self.root)))
        window.caption_edit.setPlainText("unsaved edits")
        window.preview_memory_button.click()
        self.assertTrue(window.preview_paused)
        self.assertTrue(window.image_label.source.isNull())
        self.assertTrue(window.image_label.pixmap().isNull())
        self.assertEqual(window.preview_memory_button.text(), "Load Preview")
        self.assertEqual(window.caption_edit.toPlainText(), "unsaved edits")
        self.assertEqual((self.root / "one.txt").read_text(encoding="utf-8"), "one")
        window.preview_memory_button.click()
        self.assertFalse(window.preview_paused)
        self.assertFalse(window.image_label.source.isNull())
        self.assertEqual(window.caption_edit.toPlainText(), "unsaved edits")
        self.assertEqual((self.root / "one.txt").read_text(encoding="utf-8"), "one")

    def test_paused_navigation_and_resize_never_decode_images(self):
        window = self.show(TaggingWindow(str(self.root)))
        window.release_preview_memory()
        with patch("src.ui.tagging_window.QImageReader") as reader:
            window.caption_edit.setPlainText("updated first caption")
            window.next_button.click()
            self.assertEqual(window.caption_edit.toPlainText(), "two")
            self.assertEqual(window.position_label.text(), "2 / 2")
            window.previous_button.click()
            self.assertEqual(window.caption_edit.toPlainText(), "updated first caption")
            window.position_slider.setValue(1)
            window.resize(900, 680)
            self.settle()
            reader.assert_not_called()
        self.assertTrue(window.image_label.source.isNull())
        self.assertTrue(window.image_label.pixmap().isNull())
        window.preview_memory_button.click()
        self.assertEqual(window.index, 1)
        self.assertEqual(window.image_label.source.toImage().pixelColor(0, 0).name(), "#0000ff")

    def test_empty_dataset_disables_preview_button(self):
        window = self.show(TaggingWindow())
        self.assertFalse(window.preview_memory_button.isEnabled())
        window.release_preview_memory()
        self.assertFalse(window.preview_paused)

    def test_training_release_is_opt_in_and_skips_export(self):
        for enabled, export_only in ((True, False), (False, False), (True, True)):
            with self.subTest(enabled=enabled, export_only=export_only):
                window = self.show(TaggingWindow(str(self.root)))
                with patch.object(TrainingDialog, "exec", return_value=0):
                    window.open_training_setup()
                dialog = window.training_dialog
                self.assertFalse(window.preview_paused)  # Opening setup is not training.
                dialog.release_preview_checkbox.setChecked(enabled)
                with (patch.object(dialog, "validated_config", return_value=SDXLTrainConfig()),
                      patch("src.ui.training_dialog.validate_runtime", return_value=[]),
                      patch("src.ui.training_dialog.TrainingWorker") as worker):
                    worker.return_value.start.side_effect = lambda: self.assertEqual(
                        window.preview_paused, enabled and not export_only)
                    dialog.begin_training(export_only=export_only)
                    worker.return_value.start.assert_called_once()
                    dialog.release_worker()
                self.assertEqual(window.preview_paused, enabled and not export_only)
                window.close()

    def test_failed_validation_or_cancelled_resume_keeps_preview(self):
        dialog = self.show(TrainingDialog())
        with (patch("src.ui.training_dialog.TrainingWorker") as worker,
              patch.object(dialog, "validated_config", return_value=None),
              patch.object(dialog, "release_preview_requested") as release):
            dialog.begin_training()
            worker.assert_not_called()
            release.emit.assert_not_called()
        with (patch.object(dialog, "validated_config", return_value=replace(SDXLTrainConfig(), resume_state="state")),
              patch("src.ui.training_dialog.validate_runtime", return_value=[]),
              patch.object(QMessageBox, "question", return_value=QMessageBox.No),
              patch.object(dialog, "release_preview_requested") as release):
            dialog.begin_training()
            release.emit.assert_not_called()
        self.assertIsNone(dialog.worker)

    def test_preview_training_preference_remembered_separately_from_run_config(self):
        session = self.root / "session.json"
        window = self.show(TaggingWindow(str(self.root), session_path=session))
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
        self.assertTrue(window.training_dialog.release_preview_checkbox.isChecked())
        window.training_dialog.release_preview_checkbox.setChecked(False)
        window.release_preview_memory()
        window.save_session()
        window.close()
        restored = self.show(TaggingWindow(session_path=session))
        self.assertFalse(restored.preview_paused)  # Fresh captioning session shows images.
        with patch.object(TrainingDialog, "exec", return_value=0):
            restored.open_training_setup()
        self.assertFalse(restored.training_dialog.release_preview_checkbox.isChecked())
        self.assertFalse(hasattr(restored.training_dialog.read_config(), "release_preview_on_training"))

    def test_settings_sections_keep_configuration_and_export_action(self):
        dialog = self.show(TrainingDialog())
        expected = {
            "Training": ("model_path", "dataset_path", "attention_backend", "epochs"),
            "Captions": ("trigger_word", "class_token", "num_repeats", "keep_tokens"),
            "Recovery": ("resume_state", "save_training_state", "save_every_n_steps", "seed"),
            "Environment": ("sd_scripts_dir", "python_executable", "offline_mode", "measure_performance"),
        }
        self.assertEqual(dialog.settings_sections.count(), len(expected))
        for index, (name, keys) in enumerate(expected.items()):
            self.assertEqual(dialog.settings_sections.tabText(index), name)
            page = dialog.settings_sections.widget(index)
            for key in keys:
                self.assertTrue(page.isAncestorOf(dialog.controls[key]))
        config = SDXLTrainConfig(output_name="ui_roundtrip", attention_backend="xformers", epochs=14)
        dialog.apply_config(config)
        self.assertEqual(dialog.read_config(), config)
        actions = [a.text() for a in dialog.settings_menu.actions() if not a.isSeparator()]
        self.assertEqual(actions, ["Save Settings...", "Load Settings...", "Export Training Files"])
        with patch.object(dialog, "begin_training") as begin:
            dialog.export_button.trigger()
            begin.assert_called_once_with(export_only=True)

    def test_training_sections_fit_compact_window(self):
        dialog = self.show(TrainingDialog())
        dialog.resize(800, 680)
        for index in range(dialog.settings_sections.count()):
            dialog.settings_sections.setCurrentIndex(index)
            self.settle()
            page = dialog.settings_sections.widget(index)
            self.assertEqual(page.horizontalScrollBar().maximum(), 0)
            self.assertTrue(dialog.start_button.isVisible())
            self.assertTrue(dialog.stop_button.isVisible())

    def test_multiline_checkpoint_information_fits(self):
        dialog = self.show(TrainingDialog())
        dialog.tabs.setCurrentIndex(1)
        dialog.checkpoint_label.setText(
            "Last complete checkpoint: step 120 / 504\n"
            "D:/output/example-run/example-step00000120-state\n"
            "Steps at risk: 12\nStop does not save a new checkpoint."
        )
        self.settle()
        label = dialog.checkpoint_label
        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))

    def test_history_disables_actions_until_selection(self):
        dialog = self.show(RunHistoryDialog(self.root))
        self.assertTrue(all(not action.isEnabled() for action in dialog.run_actions))

    def test_history_disables_resume_if_checkpoint_listing_fails(self):
        with patch("src.ui.run_history_dialog.list_runs", return_value=[{
            "directory": str(self.root), "status": "stopped",
        }]):
            dialog = self.show(RunHistoryDialog(self.root))
        with patch("src.ui.run_history_dialog.list_checkpoints", side_effect=OSError("unavailable")):
            dialog.table.selectRow(0)
        self.assertFalse(dialog.resume_button.isEnabled())
        self.assertEqual(dialog.checkpoints.count(), 0)
        self.assertIn("unavailable", dialog.details.toPlainText())


if __name__ == "__main__":
    unittest.main()
