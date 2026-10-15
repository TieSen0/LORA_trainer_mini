from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox

from src.training.checkpoint_monitor import CheckpointMonitor
from src.training.checkpoints import list_checkpoints
from src.training.config import SDXLTrainConfig, save_config
from src.training.records import write_json
from src.ui.training_dialog import TrainingDialog


class CheckpointMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="checkpoint 空格 ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        dataset = self.root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (16, 16)).save(dataset / "one.png")
        (dataset / "one.txt").write_text("solo", encoding="utf-8")
        model = self.root / "base.safetensors"
        model.write_bytes(b"TEST ONLY")
        self.config = SDXLTrainConfig(
            model_path=str(model), dataset_path=str(dataset), output_dir=str(self.root),
            python_executable=sys.executable,
            sd_scripts_dir=str(Path(__file__).parent.resolve() / "fixtures" / "fake_sd_scripts"),
        )
        save_config(self.config, self.run / "settings.json")
        write_json(self.run / "run.json", {"metrics": {"total": 12}})

    def make_state(self, step=4, epoch=1):
        state = self.run / f"my_lora-step{step:08}-state"
        state.mkdir()
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
            (state / name).write_bytes(b"TEST ONLY")
        write_json(state / "train_state.json", {"current_step": step, "current_epoch": epoch})
        return state

    @staticmethod
    def saved(state):
        return f"Random states saved in {state / 'random_states_0.pkl'}\n"

    def test_folder_presence_is_not_save_confirmation(self):
        state = self.make_state()
        monitor = CheckpointMonitor(self.run)
        status = monitor.scan(4, 12)
        self.assertIsNone(status["latest"])
        self.assertEqual(status["pending_count"], 1)
        self.assertEqual(status["steps_at_risk"], 4)
        self.assertTrue(monitor.feed(self.saved(state)))
        status = monitor.scan(4, 12)
        self.assertEqual(status["latest"]["path"], str(state))
        self.assertEqual(status["steps_at_risk"], 0)
        self.assertTrue(status["latest"]["can_resume"])
        self.assertFalse(monitor.feed(self.saved(state)))

    def test_partial_or_fp16_scaler_missing_never_confirmed(self):
        state = self.make_state()
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(state))
        for name in ("optimizer.bin", "scaler.pt"):
            with self.subTest(name=name):
                (state / name).write_bytes(b"")
                self.assertIsNone(monitor.scan(8, 12, successful_exit=True)["latest"])
                self.assertEqual(list_checkpoints(self.run), [])
                (state / name).write_bytes(b"TEST ONLY")

    def test_rich_wrapping_chunk_boundaries_and_other_paths(self):
        state = self.make_state()
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(self.root / "different-state"))
        self.assertIsNone(monitor.scan(4, 12)["latest"])
        target = str(state / "random_states_0.pkl")
        wrapped = "\r\n                         ".join(target[i:i+20] for i in range(0, len(target), 20))
        message = "\x1b[32mINFO [RANK 0] Random states saved checkpointing.py:179\r\nin\r\n" + wrapped + "\x1b[0m\n"
        for char in message:
            monitor.feed(char)
        self.assertEqual(monitor.scan(4, 12)["latest"]["step"], 4)

    def test_progress_and_success_are_completion_evidence_but_stop_is_not(self):
        self.make_state()
        monitor = CheckpointMonitor(self.run)
        self.assertIsNone(monitor.scan(4, 12)["latest"])
        status = monitor.scan(7, 12)
        self.assertEqual(status["latest"]["step"], 4)
        self.assertEqual(status["steps_at_risk"], 3)
        self.assertIsNotNone(monitor.scan(4, 12, successful_exit=True)["latest"])

    def test_source_checkpoint_remains_available_without_new_save(self):
        state = self.make_state()
        new_run = self.root / "new-run"
        new_run.mkdir()
        monitor = CheckpointMonitor(new_run, resume_state=str(state), saving_enabled=False)
        status = monitor.scan(7, 12)
        self.assertTrue(status["latest"]["source"])
        self.assertEqual(status["steps_at_risk"], 3)
        self.assertFalse(status["saving_enabled"])
        (state / "optimizer.bin").unlink()
        status = monitor.scan(7, 12)
        self.assertIsNone(status["latest"])
        self.assertIn("unavailable", status["error"])

    def test_newest_confirmed_state_wins_over_partial_state(self):
        first, second = self.make_state(4), self.make_state(8)
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(first))
        self.assertEqual(monitor.scan(8, 12)["latest"]["step"], 4)
        monitor.feed(self.saved(second))
        self.assertEqual(monitor.scan(8, 12)["latest"]["step"], 8)
        (second / "optimizer.bin").write_bytes(b"")
        self.assertEqual(monitor.scan(8, 12)["latest"]["step"], 4)

    def test_final_checkpoint_is_visible_but_cannot_extend_target(self):
        state = self.make_state(12)
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(state))
        self.assertFalse(monitor.scan(12, 12)["latest"]["can_resume"])

    def test_resumed_checkpoint_uses_overall_step(self):
        state = self.make_state(2, epoch=3)
        save_config(replace(self.config, resume_state="previous state"), self.run / "settings.json")
        write_json(self.run / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 4}})
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(state))
        status = monitor.scan(11, 12)
        self.assertEqual(status["latest"]["step"], 10)
        self.assertEqual(status["steps_at_risk"], 1)

    def dialog_with_status(self):
        state = self.make_state()
        monitor = CheckpointMonitor(self.run)
        monitor.feed(self.saved(state))
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.receive_telemetry({
            "current": 7, "total": 12, "phase": "Training", "elapsed_seconds": 10, "silent_seconds": 0,
            "checkpoint_status": monitor.scan(7, 12),
        })
        return dialog, state

    def test_ui_shows_risk_opens_exact_folder_and_prepares_without_starting(self):
        dialog, state = self.dialog_with_status()
        self.assertIn("Last complete checkpoint: step 4 / 12", dialog.checkpoint_label.text())
        self.assertIn("steps at risk: 3", dialog.checkpoint_label.text())
        self.assertTrue(dialog.open_checkpoint_button.isEnabled())
        self.assertTrue(dialog.prepare_latest_button.isEnabled())
        with patch("src.ui.training_dialog.QDesktopServices.openUrl") as open_url:
            dialog.open_checkpoint_folder()
        self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), state)
        with patch("src.ui.training_dialog.TrainingWorker") as worker:
            dialog.prepare_latest_resume()
            worker.assert_not_called()
        self.assertEqual(dialog.read_config().resume_state, str(state))
        self.assertEqual(dialog.tabs.currentIndex(), 0)

    def test_stop_requires_confirmation_and_resume_enables_after_worker_release(self):
        dialog, state = self.dialog_with_status()
        worker = Mock()
        dialog.worker = worker
        dialog.refresh_checkpoint_status()
        self.assertFalse(dialog.prepare_latest_button.isEnabled())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            dialog.stop_training()
        worker.request_stop.assert_not_called()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as question:
            dialog.stop_training()
        worker.request_stop.assert_called_once()
        self.assertIn("steps at risk: 3", question.call_args.args[2])
        dialog.release_worker()
        self.assertTrue(dialog.prepare_latest_button.isEnabled())

    def test_no_checkpoint_and_disabled_saving_have_explicit_ui_warning(self):
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        status = CheckpointMonitor(self.run, saving_enabled=False).scan(7, 12)
        dialog.receive_telemetry({"phase": "Training", "current": 7, "total": 12,
                                  "elapsed_seconds": 0, "silent_seconds": 0, "checkpoint_status": status})
        self.assertIn("No complete checkpoint", dialog.checkpoint_label.text())
        self.assertIn("State saving is disabled", dialog.checkpoint_label.text())
        self.assertFalse(dialog.open_checkpoint_button.isEnabled())
        self.assertFalse(dialog.prepare_latest_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
