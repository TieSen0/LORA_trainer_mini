from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.training.checkpoints import inspect_checkpoint, list_checkpoints, read_step_layout, validate_resume
from src.training.command_builder import training_toml
from src.training.config import SDXLTrainConfig, save_config
from src.training.records import write_json
from src.training.runner import TrainingWorker, run_process
from src.ui.run_history_dialog import RunHistoryDialog


class ChainedResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="chained resume 空格 ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / "dataset"
        dataset.mkdir()
        Image.new("RGB", (16, 16)).save(dataset / "one.png")
        (dataset / "one.txt").write_text("solo", encoding="utf-8")
        model = self.root / "base.safetensors"
        model.write_bytes(b"TEST ONLY")
        self.config = SDXLTrainConfig(
            model_path=str(model), dataset_path=str(dataset), output_dir=str(self.root),
            epochs=4, python_executable=sys.executable,
            sd_scripts_dir=str(Path(__file__).parent.resolve() / "fixtures" / "chain_sd_scripts"),
        )

    def make_resumed_state(self, raw_step, epoch, *, layout=True):
        directory = self.root / f"resumed-{raw_step}-{epoch}"
        directory.mkdir()
        # No ancestor directory exists: resolving this checkpoint must be local.
        save_config(replace(self.config, resume_state=str(self.root / "unavailable-ancestor")), directory / "settings.json")
        if layout:
            write_json(directory / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 126}})
        state = directory / "my_lora-step-state"
        state.mkdir()
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
            (state / name).write_bytes(b"TEST ONLY")
        write_json(state / "train_state.json", {"current_step": raw_step, "current_epoch": epoch})
        return state

    def test_resolves_mid_epoch_boundary_and_already_absolute_counters(self):
        for raw, epoch, overall in ((22, 4, 400), (126, 3, 378), (150, 4, 402), (400, 4, 400)):
            with self.subTest(raw=raw, epoch=epoch):
                state = self.make_resumed_state(raw, epoch)
                before = (state / "train_state.json").read_bytes()
                self.assertEqual(inspect_checkpoint(state)["current_step"], overall)
                config = replace(self.config, resume_state=str(state))
                self.assertEqual(validate_resume(config), [])
                exported = tomllib.loads(training_toml(config, self.root))
                self.assertEqual(exported["initial_step"], overall)
                self.assertEqual(exported["resume"], str(state))
                self.assertEqual((state / "train_state.json").read_bytes(), before)

    def test_legacy_log_fallback_and_missing_or_unsupported_metadata(self):
        state = self.make_resumed_state(22, 4, layout=False)
        config = replace(self.config, resume_state=str(state))
        self.assertIn("overall position", " ".join(validate_resume(config)))
        log = state.parent / "training.log"
        log.write_text(
            "num batches per epoch / 1epochのバッチ数: 126\n"
            "gradient accumulation steps / 勾配を合計するステップ数 = 2\n", encoding="utf-8",
        )
        self.assertTrue(validate_resume(config))
        log.write_text(log.read_text(encoding="utf-8").replace("= 2", "= 1"), encoding="utf-8")
        self.assertEqual(inspect_checkpoint(state)["current_step"], 400)
        self.assertEqual(validate_resume(config), [])
        write_json(state.parent / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 0}})
        self.assertIn("invalid steps-per-epoch", " ".join(validate_resume(config)))

    def test_invalid_positions_and_partial_states_are_rejected(self):
        for raw, epoch in ((0, 4), (20, 0), (505, 1)):
            state = self.make_resumed_state(raw, epoch)
            self.assertTrue(validate_resume(replace(self.config, resume_state=str(state))))
        state = self.make_resumed_state(22, 4)
        (state / "optimizer.bin").write_bytes(b"")
        self.assertEqual(list_checkpoints(state.parent), [])

    def test_completed_or_backward_checkpoint_cannot_restart_training(self):
        state = self.make_resumed_state(126, 4)
        config = replace(self.config, resume_state=str(state))
        # No metrics.total is needed to recognize the full target from layout.
        self.assertIn("reached the original training target", " ".join(validate_resume(config)))
        state = self.make_resumed_state(22, 4)
        write_json(state.parent / "run.json", {
            "step_layout": {"version": 1, "steps_per_epoch": 126}, "resume_start_step": 450,
        })
        self.assertIn("earlier than", " ".join(validate_resume(replace(self.config, resume_state=str(state)))))

    def test_multiple_real_subprocess_resumes_preserve_position_and_state(self):
        config = self.config
        saved_files = {}
        states = []
        for initial, stop_at in ((0, 378), (378, 400), (400, 450), (450, 504)):
            worker = TrainingWorker(config)
            results, progress, logs = [], [], []
            worker.completed.connect(results.append)
            worker.progress_changed.connect(lambda a, b: progress.append((a, b)))
            worker.output.connect(logs.append)

            def simulated_process(command, cwd, output, stop_requested, **kwargs):
                if command[2] == "-c":
                    return 0
                def receive(text):
                    output(text)
                    if "TEST CHECKPOINT COMPLETE" in "".join(logs) and stop_at < 504:
                        worker.request_stop()
                kwargs["env_overrides"] = {**kwargs.get("env_overrides", {}), "TEST_STOP_AT": str(stop_at)}
                return run_process(command, cwd, receive, stop_requested, **kwargs)

            with patch("src.training.runner.run_process", side_effect=simulated_process):
                worker.run()
            result = results[0]
            self.assertEqual(result.status, "success" if stop_at == 504 else "stopped", result.message)
            self.assertEqual(progress[0], (initial, 504))
            self.assertIn((initial + 1, 504), progress)
            self.assertEqual(progress[-1], (stop_at, 504))
            record = json.loads((result.run_directory / "run.json").read_text())
            self.assertEqual(record["metrics"]["current"], stop_at)
            self.assertEqual(record["resume_start_step"], initial)
            self.assertEqual(read_step_layout(result.run_directory)["steps_per_epoch"], 126)
            state = list_checkpoints(result.run_directory)[0]
            self.assertEqual(inspect_checkpoint(state)["current_step"], stop_at)
            if initial:
                self.assertIn(f"restored optimizer, scheduler, model, RNG and scaler at {initial}", "".join(logs))
            for path, contents in saved_files.items():
                self.assertEqual(path.read_bytes(), contents)
            for path in state.iterdir():
                saved_files[path] = path.read_bytes()
            states.append(state)
            config = replace(config, resume_state=str(state))
            if stop_at < 504:
                self.assertEqual(validate_resume(config), [])
        self.assertIn("reached the original training target", " ".join(validate_resume(config)))
        # History shows the absolute step and can select a second-generation state.
        history = RunHistoryDialog(str(self.root))
        self.addCleanup(history.close)
        row = next(i for i, r in enumerate(history.runs) if Path(r["directory"]) == states[2].parent)
        history.table.selectRow(row)
        history.show_selection()
        self.assertIn("saved step 450", history.checkpoints.currentText())
        selected = []
        history.settings_selected.connect(selected.append)
        history.prepare_resume()
        self.assertEqual(selected[0].resume_state, str(states[2]))


if __name__ == "__main__":
    unittest.main()
