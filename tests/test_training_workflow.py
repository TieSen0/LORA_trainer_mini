from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.training.checkpoints import dataset_signature, inspect_checkpoint, list_checkpoints, validate_resume
from src.training.command_builder import create_training_run, training_toml
from src.training.config import SDXLTrainConfig, config_payload, load_config, parse_config, save_config, validate_settings
from src.training.history import compare_runs, list_runs
from src.training.monitor import TrainingMonitor, sample_gpu
from src.training.records import load_session, write_json
from src.training.runner import TrainingWorker, run_process
from src.training.resume import load_resume_config, load_resume_from_folder
from src.ui.run_history_dialog import RunHistoryDialog
from src.ui.tagging_window import TaggingWindow
from src.ui.training_dialog import TrainingDialog


class TrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        Image.new("RGB", (16, 16)).save(self.dataset / "one.png")
        (self.dataset / "one.txt").write_text("solo", encoding="utf-8")
        self.model = self.root / "base.safetensors"
        self.model.write_bytes(b"test model")
        self.output = self.root / "outputs"
        self.output.mkdir()
        self.config = SDXLTrainConfig(
            model_path=str(self.model), dataset_path=str(self.dataset), output_dir=str(self.output),
            sd_scripts_dir=str(Path(__file__).parent.resolve() / "fixtures" / "fake_sd_scripts"),
            python_executable=sys.executable,
        )

    def make_state(self, directory):
        state = directory / "my_lora-step00000001-state"
        state.mkdir()
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt"):
            (state / name).write_bytes(b"fixture state")
        write_json(state / "train_state.json", {"current_step": 1, "current_epoch": 1})
        return state

    def test_resume_config_mapping_validation_and_no_overwrites(self):
        source = create_training_run(self.config)
        state = self.make_state(source.directory)
        resumed = replace(self.config, resume_state=str(state), save_every_n_steps=10)
        self.assertEqual(validate_resume(resumed), [])
        run = create_training_run(resumed)
        values = tomllib.loads((run.directory / "training.toml").read_text(encoding="utf-8"))
        self.assertEqual(values["resume"], str(state))
        self.assertTrue(values["skip_until_initial_step"])
        self.assertEqual(values["initial_step"], 1)
        self.assertTrue(values["save_state"])
        self.assertTrue(values["save_state_on_train_end"])
        self.assertEqual(values["save_every_n_steps"], 10)
        self.assertEqual(values["seed"], 42)
        self.assertNotEqual(run.directory, source.directory)
        self.assertEqual((state / "optimizer.bin").read_bytes(), b"fixture state")
        self.assertEqual(load_config(run.directory / "settings.json"), resumed)
        self.assertTrue(validate_resume(replace(resumed, optimizer="AdamW8bit")))
        self.assertTrue(validate_resume(replace(resumed, epochs=11)))
        next_state = self.make_state(run.directory)
        write_json(run.directory / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 1}})
        self.assertEqual(validate_resume(replace(resumed, resume_state=str(next_state))), [])

    def test_resume_rejects_model_files_partial_states_and_changed_captions(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        config = replace(self.config, resume_state=str(state))
        write_json(run.directory / "run.json", {"dataset_signature": dataset_signature(self.config.dataset_path)})
        self.assertEqual(validate_resume(config), [])
        (self.dataset / "one.txt").write_text("changed", encoding="utf-8")
        self.assertIn("Dataset", " ".join(validate_resume(config)))
        self.assertTrue(validate_resume(replace(config, resume_state=str(self.model))))
        (state / "optimizer.bin").write_bytes(b"")
        self.assertEqual(list_checkpoints(run.directory), [])
        with self.assertRaises(ValueError):
            inspect_checkpoint(state)
        self.assertTrue(validate_resume(config))

    def test_disk_cache_resume_keeps_original_choice_and_migrates_legacy_runs(self):
        for enabled in (False, True):
            config = replace(self.config, cache_latents_to_disk=enabled)
            run = create_training_run(config)
            state = self.make_state(run.directory)
            resumed = replace(config, resume_state=str(state))
            self.assertEqual(validate_resume(resumed), [])
            self.assertIn("cache_latents_to_disk", " ".join(validate_resume(
                replace(resumed, cache_latents_to_disk=not enabled))))
            if not enabled:
                payload = config_payload(config)
                payload["schema_version"] = 6
                del payload["settings"]["cache_latents_to_disk"]
                write_json(run.directory / "settings.json", payload)
                self.assertEqual(validate_resume(resumed), [])

    def test_manual_resume_restores_all_settings_and_new_training_restores_draft(self):
        source = replace(self.config, output_name="selected_lora", epochs=20, network_dim=16,
                         cache_latents_to_disk=True, offline_mode=True)
        run = create_training_run(source)
        state = self.make_state(run.directory)
        write_json(run.directory / "run.json", {"metrics": {"total": 20}})
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        draft = replace(self.config, output_name="new_lora_draft", epochs=15)
        dialog.apply_config(draft)
        with (patch.object(QFileDialog, "getExistingDirectory", return_value=str(state)),
              patch("src.ui.training_dialog.TrainingWorker") as worker):
            dialog.browse_path("resume_state")
            worker.assert_not_called()
        self.assertEqual(dialog.read_config(), replace(source, resume_state=str(state)))
        self.assertEqual(dialog.start_button.text(), "Resume Training")
        self.assertIn("Saved step 1 / 20", dialog.mode_label.text())
        self.assertIn("Remaining steps: 19", dialog.mode_label.text())
        self.assertTrue(dialog.controls["resume_state"].isReadOnly())
        dialog.start_new_training()
        self.assertEqual(dialog.read_config(), draft)
        self.assertEqual(dialog.start_button.text(), "Start Training")

    def test_run_folder_picker_selects_latest_complete_state_only_in_selected_run(self):
        source = replace(self.config, output_name="chosen_lora", epochs=20)
        run = create_training_run(source)
        older = self.make_state(run.directory)
        newer = run.directory / "chosen_lora-step00000002-state"
        newer.mkdir()
        for file in older.iterdir():
            (newer / file.name).write_bytes(file.read_bytes())
        write_json(newer / "train_state.json", {"current_step": 2, "current_epoch": 2})
        # A copied old state may have a newer timestamp; order by saved progress.
        os.utime(older, (2000000000, 2000000000))
        incomplete = run.directory / "chosen_lora-step00000003-state"
        incomplete.mkdir()
        write_json(incomplete / "train_state.json", {"current_step": 3, "current_epoch": 3})
        other = create_training_run(replace(self.config, output_name="unrelated_lora"))
        other_state = self.make_state(other.directory)
        write_json(other_state / "train_state.json", {"current_step": 9, "current_epoch": 9})
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        with (patch.object(QFileDialog, "getExistingDirectory", return_value=str(run.directory)),
              patch("src.ui.training_dialog.TrainingWorker") as worker):
            dialog.browse_path("resume_state")
            worker.assert_not_called()
        self.assertEqual(dialog.read_config(), replace(source, resume_state=str(newer)))
        self.assertEqual(dialog.start_button.text(), "Resume Training")
        self.assertIsNone(dialog.worker)

    def test_quick_resume_does_not_rewind_completed_run_or_checkpoint(self):
        run = create_training_run(self.config)
        older = self.make_state(run.directory)
        newer = run.directory / "my_lora-step00000002-state"
        newer.mkdir()
        for file in older.iterdir():
            (newer / file.name).write_bytes(file.read_bytes())
        write_json(newer / "train_state.json", {"current_step": 2, "current_epoch": 2})
        write_json(run.directory / "run.json", {"metrics": {"total": 2}, "status": "stopped"})
        with self.assertRaisesRegex(ValueError, "reached"):
            load_resume_from_folder(run.directory)
        write_json(run.directory / "run.json", {"metrics": {"total": 10}, "status": "success"})
        with self.assertRaisesRegex(ValueError, "already complete"):
            load_resume_from_folder(run.directory)

    def test_quick_resume_no_states_or_wrong_folder_preserves_draft(self):
        run = create_training_run(self.config)
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.apply_config(self.config)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.prepare_resume_from_folder(run.directory)
            self.assertIn("no complete saved state", warning.call_args.args[2])
            dialog.prepare_resume_from_folder(self.output)
            self.assertIn("one LoRA run folder", warning.call_args.args[2])
        self.assertEqual(dialog.read_config(), self.config)
        self.assertIsNone(dialog.worker)

    def test_quick_resume_orders_resumed_states_by_overall_not_raw_step(self):
        run = create_training_run(self.config)
        later = self.make_state(run.directory)
        earlier = run.directory / "my_lora-step00000002-state"
        earlier.mkdir()
        for file in later.iterdir():
            (earlier / file.name).write_bytes(file.read_bytes())
        source = replace(self.config, resume_state=str(self.root / "missing-ancestor-state"))
        save_config(source, run.directory / "settings.json")
        write_json(run.directory / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 2}})
        write_json(later / "train_state.json", {"current_step": 1, "current_epoch": 3})
        write_json(earlier / "train_state.json", {"current_step": 2, "current_epoch": 2})
        self.assertEqual(load_resume_from_folder(run.directory).resume_state, str(later))

    def test_history_can_resume_older_selected_state_and_copy_settings_for_new_run(self):
        run = create_training_run(self.config)
        older = self.make_state(run.directory)
        newer = run.directory / "my_lora-step00000002-state"
        newer.mkdir()
        for file in older.iterdir():
            (newer / file.name).write_bytes(file.read_bytes())
        write_json(newer / "train_state.json", {"current_step": 2, "current_epoch": 2})
        write_json(run.directory / "run.json", {"status": "stopped", "metrics": {"total": 10}})
        history = RunHistoryDialog(self.output)
        self.addCleanup(history.close)
        history.table.selectRow(0)
        self.assertEqual(history.checkpoints.currentData(), str(newer))
        history.checkpoints.setCurrentIndex(history.checkpoints.findData(str(older)))
        selected = []
        history.settings_selected.connect(selected.append)
        history.prepare_resume()
        self.assertEqual(selected[-1], replace(self.config, resume_state=str(older)))
        history.load_selected()
        self.assertEqual(selected[-1], self.config)

    def test_invalid_or_completed_resume_does_not_replace_new_training_draft(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        write_json(run.directory / "run.json", {"metrics": {"total": 1}})
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.apply_config(self.config)
        with patch.object(QMessageBox, "warning") as warning:
            dialog.prepare_resume_from_path(state)
            self.assertIn("reached", warning.call_args.args[2])
            dialog.prepare_resume_from_path(self.root / "missing-state")
        self.assertEqual(dialog.read_config(), self.config)
        self.assertIsNone(dialog.worker)
        self.assertEqual(dialog.start_button.text(), "Start Training")

    def test_reopening_setup_keeps_resume_settings_but_new_dataset_clears_selection(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        window = TaggingWindow(str(self.dataset))
        self.addCleanup(window.close)
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
            window.training_dialog.prepare_resume_from_path(state)
            window.set_caption_prefix("different_draft_prefix", "1girl")
            window.open_training_setup()
            self.assertEqual(window.training_dialog.read_config(), replace(self.config, resume_state=str(state)))
            other = self.root / "new-dataset"
            other.mkdir()
            Image.new("RGB", (16, 16)).save(other / "new.png")
            window.open_dataset(str(other))
            window.open_training_setup()
        self.assertEqual(window.training_dialog.read_config().resume_state, "")
        self.assertEqual(window.training_dialog.read_config().dataset_path, str(other))
        self.assertEqual(window.training_dialog.start_button.text(), "Start Training")

    def test_resume_preparation_needs_no_ancestor_state(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        source = replace(self.config, resume_state=str(self.root / "deleted-ancestor-state"))
        save_config(source, run.directory / "settings.json")
        write_json(state / "train_state.json", {"current_step": 1, "current_epoch": 3})
        write_json(run.directory / "run.json", {"step_layout": {"version": 1, "steps_per_epoch": 2}})
        config = load_resume_config(state)
        self.assertEqual(config, replace(source, resume_state=str(state)))
        self.assertEqual(inspect_checkpoint(state)["current_step"], 5)

    def test_checkpoint_options_default_migration_and_ranges(self):
        self.assertTrue(self.config.save_training_state)
        for changes in ({"seed": -2}, {"save_every_n_steps": -1}, {"measure_performance": "yes"}, {"seed": -1}):
            self.assertTrue(validate_settings(replace(self.config, **changes)))
        for version in (1, 2, 3, 4, 5):
            payload = config_payload(self.config)
            payload["schema_version"] = version
            for key in ("save_training_state", "save_every_n_steps", "resume_state", "seed", "measure_performance"):
                del payload["settings"][key]
            loaded = parse_config(payload)
            self.assertFalse(loaded.save_training_state)
            self.assertEqual(loaded.seed, -1)
            values = tomllib.loads(training_toml(loaded, self.output))
            self.assertNotIn("seed", values)
            self.assertNotIn("resume", values)
            self.assertNotIn("save_every_n_steps", values)

    def test_checkpoint_at_total_target_is_not_offered_as_continuation(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        write_json(run.directory / "run.json", {"metrics": {"total": 1}})
        errors = validate_resume(replace(self.config, resume_state=str(state)))
        self.assertIn("reached the original training target", " ".join(errors))

    def test_monitor_split_phases_warmup_duplicate_steps_eta_and_silence(self):
        monitor = TrainingMonitor(now=0)
        monitor.feed("caching lat", now=1)
        monitor.feed("ents\n", now=2)
        self.assertEqual(monitor.phase, "Caching latents")
        monitor.feed("caching text encoder outputs\n", now=3)
        self.assertEqual(monitor.phase, "Caching text-encoder outputs")
        monitor.feed("steps: 50%|x| 5/10\r", now=10)
        self.assertIsNone(monitor.snapshot(now=10)["steps_per_second"])
        monitor.feed("steps: 60%|x| 6/10\r", now=12)
        monitor.feed("steps: 60%|x| 6/10\r", now=14)
        metrics = monitor.snapshot(now=75)
        self.assertEqual(metrics["steps_per_second"], 0.5)
        self.assertEqual(metrics["eta_seconds"], 8)
        self.assertEqual(metrics["silent_seconds"], 61)
        self.assertEqual(metrics["elapsed_seconds"], 75)

    def test_silent_process_still_sends_ticks_and_stops(self):
        stop = threading.Event()
        ticks = []
        def tick():
            ticks.append(time.monotonic())
            if len(ticks) >= 3:
                stop.set()
        code = run_process([sys.executable, "-c", "import time; time.sleep(60)"], self.root,
                           lambda text: None, stop, on_tick=tick)
        self.assertNotEqual(code, 0)
        self.assertGreaterEqual(len(ticks), 3)

    def test_gpu_sampling_is_optional_and_missing_nvidia_smi_is_safe(self):
        with patch("src.training.monitor.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(sample_gpu())
        with patch("src.training.monitor.subprocess.run") as command:
            command.return_value.returncode = 0
            command.return_value.stdout = "Test GPU, 2048, 75\n"
            self.assertEqual(sample_gpu()["memory_mib"], 2048)

    def test_stopped_worker_records_outcome_without_claiming_success(self):
        worker = TrainingWorker(self.config)
        results = []
        worker.completed.connect(results.append)
        def stopping_process(command, *args, **kwargs):
            if command[2] == "-c":
                return 0
            worker.request_stop()
            return -1
        with patch("src.training.runner.run_process", side_effect=stopping_process):
            worker.run()
        self.assertEqual(results[0].status, "stopped")
        record = json.loads((results[0].run_directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "stopped")
        self.assertIsNone(record["model_path"])

    def test_failed_atomic_record_write_preserves_previous_file(self):
        path = self.root / "record.json"
        write_json(path, {"original": True})
        with patch("pathlib.Path.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                write_json(path, {"original": False})
        self.assertEqual(json.loads(path.read_text()), {"original": True})
        self.assertEqual(list(self.root.glob(".record-*")), [])

    def test_stop_during_unconfirmed_save_does_not_claim_recovery_point(self):
        worker = TrainingWorker(self.config)
        results = []
        worker.completed.connect(results.append)

        def interrupted_save(command, cwd, output, stop_requested, **kwargs):
            if command[2] == "-c":
                return 0
            run_dir = Path(command[-1]).parent
            # Files are nonempty but the trainer has not confirmed save completion
            # or advanced beyond this step. Forced stop must not bless this state.
            self.make_state(run_dir)
            output("total optimization steps: 2\nsteps: 50%|x| 1/2\r")
            worker.request_stop()
            return -1

        with patch("src.training.runner.run_process", side_effect=interrupted_save):
            worker.run()
        self.assertEqual(results[0].status, "stopped")
        record = json.loads((results[0].run_directory / "run.json").read_text())
        self.assertIsNone(record["checkpoint_status"]["latest"])
        self.assertEqual(record["checkpoint_status"]["pending_count"], 1)
        self.assertEqual(record["checkpoint_status"]["steps_at_risk"], 1)

    def test_stopped_resume_records_overall_steps_without_changing_source(self):
        source = create_training_run(self.config)
        state = self.make_state(source.directory)
        write_json(state / "train_state.json", {"current_step": 378, "current_epoch": 3})
        write_json(source.directory / "run.json", {"metrics": {"current": 453, "total": 504}})
        original_state = (state / "train_state.json").read_bytes()
        original_record = (source.directory / "run.json").read_bytes()
        worker = TrainingWorker(replace(self.config, resume_state=str(state)))
        results, progress = [], []
        worker.completed.connect(results.append)
        worker.progress_changed.connect(lambda current, total: progress.append((current, total)))

        def stopping_process(command, cwd, output, stop_requested, **kwargs):
            if command[2] == "-c":
                return 0
            output("total optimization steps: 504\nsteps: 0%|x| 0/504\r")
            output("steps: 0%|x| 1/504\r")
            worker.request_stop()
            return -1

        with patch("src.training.runner.run_process", side_effect=stopping_process):
            worker.run()
        self.assertEqual(results[0].status, "stopped")
        self.assertEqual(progress, [(378, 504), (379, 504)])
        record = json.loads((results[0].run_directory / "run.json").read_text())
        self.assertEqual(record["metrics"]["current"], 379)
        self.assertEqual(record["metrics"]["total"], 504)
        self.assertEqual((state / "train_state.json").read_bytes(), original_state)
        self.assertEqual((source.directory / "run.json").read_bytes(), original_record)

    @staticmethod
    def fake_probe(command, cwd, output, stop_requested, **kwargs):
        if command[2] == "-c":
            return 0
        return run_process(command, cwd, output, stop_requested, **kwargs)

    def test_real_fixture_run_state_resume_history_and_profile(self):
        results = []
        with patch("src.training.runner.run_process", side_effect=self.fake_probe), patch(
            "src.training.runner.sample_gpu", return_value={"name": "Test GPU", "memory_mib": 1234, "utilization": 10}
        ) as gpu:
            for config in (self.config, replace(self.config, measure_performance=True)):
                worker = TrainingWorker(config)
                worker.completed.connect(results.append)
                worker.run()
                self.assertEqual(results[-1].status, "success")
                if not config.measure_performance:
                    gpu.assert_not_called()
            self.assertTrue(gpu.called)
            state = list_checkpoints(results[0].run_directory)[0]
            worker = TrainingWorker(replace(self.config, resume_state=str(state)))
            progress, statuses = [], []
            worker.progress_changed.connect(lambda step, total: progress.append((step, total)))
            worker.status_changed.connect(statuses.append)
            worker.completed.connect(results.append)
            worker.run()
        self.assertEqual(results[-1].status, "success")
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertIn("Training: step 2 / 2", statuses)
        resumed_record = json.loads((results[-1].run_directory / "run.json").read_text())
        self.assertEqual(resumed_record["metrics"]["current"], 2)
        self.assertEqual(resumed_record["metrics"]["total"], 2)
        self.assertEqual(resumed_record["metrics"]["session_current"], 1)
        self.assertEqual(resumed_record["metrics"]["resume_step"], 1)
        self.assertIn("1/2", (results[-1].run_directory / "training.log").read_text(encoding="utf-8"))
        records = list_runs(self.output)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["status"] == "success" for record in records))
        profile = next(record for record in records if record["metrics"]["peak_gpu_memory_mib"] is not None)
        self.assertEqual(profile["metrics"]["peak_gpu_memory_mib"], 1234)
        self.assertEqual(json.loads((results[-1].run_directory / "run.json").read_text())["resume_from"], str(state))

    def test_history_legacy_and_comparison_are_read_only(self):
        first, second = create_training_run(self.config), create_training_run(self.config)
        signature = dataset_signature(self.config.dataset_path)
        for run, rate in ((first, 1.0), (second, 1.2)):
            write_json(run.directory / "run.json", {"status": "success", "dataset_signature": signature,
                       "metrics": {"steps_per_second": rate, "peak_gpu_memory_mib": 1234}})
        original = (first.directory / "settings.json").read_bytes()
        report = compare_runs(first.directory, second.directory)
        self.assertIn("1.20x", report)
        self.assertIn("not a quality score", report)
        self.assertEqual((first.directory / "settings.json").read_bytes(), original)
        third = create_training_run(self.config)
        self.assertIn("unknown", next(r for r in list_runs(self.output) if r["directory"] == str(third.directory))["status"])
        self.assertIn("No speedup conclusion", compare_runs(first.directory, third.directory))

    def test_remembered_settings_restore_paths_offline_and_caption_state(self):
        session = self.root / "session.json"
        window = TaggingWindow(str(self.dataset), session_path=session)
        window.set_caption_prefix("trigger", "1girl")
        window.wd14_excluded_tags.setText("watermark")
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
        config = replace(self.config, offline_mode=True, trigger_word="trigger", class_token="1girl", keep_tokens=2,
                         resume_state="saved path must not auto-resume")
        window.training_dialog.apply_config(config)
        window.save_session()
        window.close()
        restored = TaggingWindow(session_path=session)
        self.addCleanup(restored.close)
        self.assertEqual(restored.folder_path, str(self.dataset))
        self.assertEqual(restored.effective_caption_prefix(), "trigger, 1girl")
        self.assertEqual(restored.wd14_excluded_tags.text(), "watermark")
        with patch.object(TrainingDialog, "exec", return_value=0):
            restored.open_training_setup()
        self.assertEqual(restored.training_dialog.read_config(), replace(config, resume_state=""))
        self.assertIsNone(restored.training_dialog.worker)

    def test_corrupt_session_preserved_and_invalid_paths_can_be_remembered(self):
        session = self.root / "session.json"
        session.write_text("broken", encoding="utf-8")
        window = TaggingWindow(session_path=session)
        self.assertIsNone(window.session_path)
        self.assertIn("left unchanged", window.dataset_status.text())
        window.close()
        self.assertEqual(session.read_text(), "broken")
        write_json(session, {"version": 1, "dataset": "missing folder", "prefixes": {},
                   "training": config_payload(replace(self.config, dataset_path="missing folder"))})
        window = TaggingWindow(session_path=session)
        self.assertEqual(window.remembered_training.dataset_path, "missing folder")
        self.assertIn("unavailable", window.dataset_status.text())
        window.close()

    def test_resume_ui_history_selection_and_silence_warning(self):
        run = create_training_run(self.config)
        state = self.make_state(run.directory)
        history = RunHistoryDialog(str(self.output))
        self.addCleanup(history.close)
        history.table.selectRow(0)
        history.show_selection()
        selected = []
        history.settings_selected.connect(selected.append)
        history.prepare_resume()
        self.assertEqual(selected[0].resume_state, str(state))
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.update_progress(379, 504)
        self.assertEqual(dialog.progress.value(), 379)
        self.assertEqual(dialog.progress.maximum(), 504)
        self.assertIn("Overall step 379 / 504", dialog.progress.format())
        dialog.apply_config(selected[0])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            dialog.begin_training()
        self.assertIsNone(dialog.worker)
        dialog.worker = object()
        dialog.receive_telemetry({"phase": "Loading model", "elapsed_seconds": 70, "silent_seconds": 65})
        self.assertIn("does not mean", dialog.activity_label.text())
        dialog.worker = None


if __name__ == "__main__":
    unittest.main()
