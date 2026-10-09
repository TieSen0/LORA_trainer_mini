import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLineEdit, QMessageBox

from src.dataset.validator import validate_dataset
from src.ui.dataset_review_dialog import DatasetReviewDialog
from src.ui.tagging_window import TaggingWindow


class DatasetReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def image(self, name):
        path = self.folder / name
        Image.new("RGB", (16, 16), "red").save(path)
        return path

    def test_clean_dataset_ignores_subfolders_and_does_not_modify_files(self):
        self.image("one.png")
        (self.folder / "one.txt").write_text("trigger, red", encoding="utf-8")
        nested = self.folder / "nested"
        nested.mkdir()
        (nested / "broken.png").write_bytes(b"broken")
        before = {p: p.read_bytes() for p in self.folder.rglob("*") if p.is_file()}
        report = validate_dataset(self.folder)
        self.assertEqual((report.image_count, report.caption_count), (1, 1))
        self.assertEqual(report.issues, [])
        self.assertEqual(before, {p: p.read_bytes() for p in self.folder.rglob("*") if p.is_file()})

    def test_reports_corruption_missing_empty_unreadable_shared_and_orphan_captions(self):
        for name in ("missing.png", "empty.png", "badtext.png", "shared.jpg", "shared.png"):
            self.image(name)
        (self.folder / "broken.png").write_bytes(b"not an image")
        (self.folder / "empty.txt").write_text(" \n", encoding="utf-8")
        (self.folder / "badtext.txt").write_bytes(b"\xff\xfe")
        (self.folder / "shared.txt").write_text("shared", encoding="utf-8")
        (self.folder / "orphan.txt").write_text("orphan", encoding="utf-8")
        report = validate_dataset(self.folder)
        self.assertEqual(report.image_count, 6)
        self.assertEqual(report.error_count, 3)
        self.assertEqual(report.warning_count, 4)
        for text in ("Cannot read image", "Missing caption", "Caption is empty", "UTF-8", "Multiple images", "no matching"):
            self.assertTrue(any(text in issue.message for issue in report.issues), text)

    def test_missing_empty_and_non_directory(self):
        self.assertEqual(validate_dataset(self.folder).error_count, 1)
        self.assertEqual(validate_dataset(self.folder / "missing").error_count, 1)
        image = self.image("one.png")
        self.assertEqual(validate_dataset(image).error_count, 1)

    def test_background_review_delivers_report_and_releases_dialog(self):
        self.image("one.png")
        dialog = DatasetReviewDialog(str(self.folder))
        deadline = time.monotonic() + 10
        while dialog.running and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        dialog.worker.wait()
        self.app.processEvents()
        self.assertFalse(dialog.running)
        self.assertIn("Review complete: 1 images", dialog.summary.text())
        self.assertIn("Missing caption", dialog.details.toPlainText())
        self.assertTrue(dialog.buttons.isEnabled())
        dialog.close()

    def test_rename_completion_counts_and_noop(self):
        self.image("photo.png")
        (self.folder / "photo.txt").write_text("trigger, red", encoding="utf-8")
        window = TaggingWindow(str(self.folder))

        def accept(dialog):
            dialog.findChild(QLineEdit).setText("role")
            return QDialog.Accepted

        with patch.object(QDialog, "exec", accept), patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), patch.object(QMessageBox, "information") as message:
            window.batch_rename_dataset()
            self.assertIn("1 image(s) and 1 caption file(s)", message.call_args.args[2])
            self.assertEqual(window.dataset_status.text(), message.call_args.args[2])
            self.assertEqual((self.folder / "role_001.txt").read_text(encoding="utf-8"), "trigger, red")
            window.batch_rename_dataset()
            self.assertIn("no files needed renaming", message.call_args.args[2])
        self.assertTrue(window.review_button.isEnabled())
        window.set_tagging_busy(True)
        self.assertFalse(window.review_button.isEnabled())
        window.set_tagging_busy(False)
        window.close()

    def test_rename_failure_never_reports_success(self):
        self.image("photo.png")
        window = TaggingWindow(str(self.folder))

        def accept(dialog):
            dialog.findChild(QLineEdit).setText("role")
            return QDialog.Accepted

        with patch.object(QDialog, "exec", accept), patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), patch.object(QMessageBox, "information") as success, patch.object(QMessageBox, "critical") as failure, patch("src.ui.tagging_window.execute_rename_plan", side_effect=OSError("locked file")):
            window.batch_rename_dataset()
            success.assert_not_called()
            failure.assert_called_once()
        window.close()

    def test_review_without_open_dataset_and_after_save_failure(self):
        window = TaggingWindow()
        self.assertTrue(window.review_button.isEnabled())
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.folder)), patch("src.ui.tagging_window.DatasetReviewDialog") as dialog:
            window.review_dataset()
            dialog.assert_called_once_with(str(self.folder), window)
            dialog.return_value.exec.assert_called_once()
        self.image("photo.png")
        window.open_dataset(str(self.folder))
        window.caption_edit.setPlainText("unsaved edit")
        with patch("src.ui.tagging_window.save_caption", side_effect=OSError("locked")), patch.object(QMessageBox, "warning"), patch("src.ui.tagging_window.DatasetReviewDialog") as dialog:
            window.review_dataset()
            dialog.return_value.exec.assert_called_once()
            self.assertEqual(window.caption_edit.toPlainText(), "unsaved edit")
        window.close()


if __name__ == "__main__":
    unittest.main()
