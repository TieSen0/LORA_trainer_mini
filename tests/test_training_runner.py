from dataclasses import replace
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
from PySide6.QtWidgets import QApplication

from src.training.command_builder import create_training_run, training_environment
from src.training.config import SDXLTrainConfig, load_config
from src.training.log_parser import TrainingLogParser
from src.training.runner import TrainingWorker, run_process
from src.ui.training_dialog import TrainingDialog


class TrainingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lora test 空格 ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        image = self.root / "one.png"
        Image.new("RGB", (16, 16)).save(image)
        image.with_suffix(".txt").write_text("red", encoding="utf-8")
        model = self.root / "base.safetensors"
        model.write_bytes(b"fixture")
        self.config = SDXLTrainConfig(
            model_path=str(model), dataset_path=str(self.root), output_dir=str(self.root),
            sd_scripts_dir=str(Path(__file__).parent.resolve() / "fixtures" / "fake_sd_scripts"),
            python_executable=sys.executable,
        )

    @staticmethod
    def skip_gpu_probe(command, cwd, on_output, stop_requested, **kwargs):
        if command[2] == "-c":
            on_output("Mocked environment probe; no GPU training performed.\n")
            return 0
        return run_process(command, cwd, on_output, stop_requested, **kwargs)

    def test_export_maps_fields_and_attention_without_overwrites(self):
        for backend in ("sdpa", "xformers"):
            config = replace(self.config, attention_backend=backend)
            run = create_training_run(config)
            dataset = tomllib.loads((run.directory / "dataset.toml").read_text(encoding="utf-8"))
            training = tomllib.loads((run.directory / "training.toml").read_text(encoding="utf-8"))
            self.assertEqual(dataset["datasets"][0]["subsets"][0]["image_dir"], str(self.root))
            self.assertEqual(dataset["datasets"][0]["batch_size"], config.batch_size)
            self.assertEqual(training["max_train_epochs"], config.epochs)
            self.assertEqual(training["optimizer_type"], config.optimizer)
            self.assertEqual(training["network_module"], "networks.lora")
            self.assertTrue(training[backend])
            self.assertNotIn("xformers" if backend == "sdpa" else "sdpa", training)
            self.assertEqual(run.command[0], str(Path(sys.executable).resolve()))
            self.assertEqual(run.command[-1], str(run.directory / "training.toml"))
            self.assertEqual(load_config(run.directory / "settings.json"), config)
            another = create_training_run(config)
            self.assertNotEqual(run.directory, another.directory)

    def test_streams_real_fixture_process_and_finds_output(self):
        for name, status in (("my_lora", "success"), ("fake_fail", "failed"), ("no_output", "failed")):
            with self.subTest(name=name):
                worker = TrainingWorker(replace(self.config, output_name=name))
                results, progress, logs = [], [], []
                worker.completed.connect(results.append)
                worker.progress_changed.connect(lambda a, b: progress.append((a, b)))
                worker.output.connect(logs.append)
                with patch("src.training.runner.run_process", side_effect=self.skip_gpu_probe):
                    worker.run()
                result = results[0]
                self.assertEqual(result.status, status)
                record = json.loads((result.run_directory / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(record["status"], status)
                self.assertIn("Synthetic stderr", "".join(logs))
                self.assertIn("中文", (result.run_directory / "training.log").read_text(encoding="utf-8"))
                if status == "success":
                    self.assertTrue(result.model_path.exists())
                    self.assertIn((2, 2), progress)

    def test_no_process_for_export_and_bad_dataset(self):
        for valid in (True, False):
            if not valid:
                (self.root / "one.txt").unlink()
            worker = TrainingWorker(self.config, export_only=True)
            results = []
            worker.completed.connect(results.append)
            with patch("src.training.runner.run_process") as process:
                worker.run()
                process.assert_not_called()
            self.assertEqual(results[0].status, "exported" if valid else "failed")

    def test_parser_ignores_download_progress_and_handles_split_steps(self):
        parser = TrainingLogParser()
        self.assertEqual(parser.feed("download: 50%|x| 5/10\rsteps: 20%|x| 2/"), [])
        self.assertEqual(parser.feed("10 [00:01]\r"), [(2, 10)])
        self.assertEqual(parser.feed("steps: 0%|x| 0/0\n"), [])

    def test_stop_silent_process_and_failed_launch(self):
        stop = threading.Event()
        output = []

        def received(text):
            output.append(text)
            if "ready" in "".join(output):
                stop.set()

        started = time.monotonic()
        code = run_process(
            [sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(60)"],
            self.root, received, stop,
        )
        self.assertNotEqual(code, 0)
        self.assertLess(time.monotonic() - started, 20)
        with self.assertRaises(OSError):
            run_process([str(self.root / "missing.exe")], self.root, output.append, threading.Event())

    def test_monitor_releases_controls_after_run(self):
        dialog = TrainingDialog()
        dialog.apply_config(self.config)
        with patch("src.training.runner.run_process", side_effect=self.skip_gpu_probe):
            dialog.begin_training()
            self.assertFalse(dialog.start_button.isEnabled())
            dialog.reject()
            self.assertIsNotNone(dialog.worker)
            deadline = time.monotonic() + 15
            while dialog.worker is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIsNone(dialog.worker)
        self.assertTrue(dialog.start_button.isEnabled())
        self.assertFalse(dialog.stop_button.isEnabled())
        self.assertEqual(dialog.progress.value(), 100)
        self.assertIn("Training complete:", dialog.status.text())
        self.assertTrue(dialog.open_output_button.isEnabled())
        dialog.close()

    def test_failed_environment_probe_does_not_launch_training(self):
        worker = TrainingWorker(self.config)
        results = []
        worker.completed.connect(results.append)
        with patch("src.training.runner.run_process", return_value=1) as process:
            worker.run()
            process.assert_called_once()
            self.assertEqual(process.call_args.args[0][2], "-c")
        self.assertEqual(results[0].status, "failed")
        self.assertIn("environment check failed", results[0].message)
        self.assertTrue((results[0].run_directory / "training.log").exists())

    def test_offline_environment_is_child_only_and_online_can_be_restored(self):
        script = "import os,json; print(json.dumps({k:os.environ.get(k) for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}))"
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "parent-value", "TRANSFORMERS_OFFLINE": "parent-value"}):
            for offline in (True, False):
                with self.subTest(offline=offline):
                    config = replace(self.config, offline_mode=offline)
                    output = []
                    code = run_process(
                        [sys.executable, "-c", script], self.root, output.append, threading.Event(),
                        env_overrides=training_environment(config),
                    )
                    self.assertEqual(code, 0)
                    self.assertEqual(json.loads("".join(output)), {
                        "HF_HUB_OFFLINE": "1" if offline else "0",
                        "TRANSFORMERS_OFFLINE": "1" if offline else "0",
                    })
                    self.assertEqual(os.environ["HF_HUB_OFFLINE"], "parent-value")
                    self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "parent-value")

    def test_offline_worker_probe_launch_and_export_environment(self):
        config = replace(self.config, offline_mode=True)
        worker = TrainingWorker(config)
        results, statuses, logs = [], [], []
        worker.completed.connect(results.append)
        worker.status_changed.connect(statuses.append)
        worker.output.connect(logs.append)
        with patch("src.training.runner.run_process", side_effect=self.skip_gpu_probe) as process:
            worker.run()
        self.assertEqual(results[0].status, "success")
        self.assertEqual(process.call_count, 2)
        for call in process.call_args_list:
            self.assertEqual(call.kwargs["env_overrides"], training_environment(config))
        self.assertIn("Offline Mode: ON", "".join(logs))
        self.assertTrue(any(s.startswith("Initializing trainer:") for s in statuses))
        self.assertIn("Training: step 2 / 2", statuses)
        directory = results[0].run_directory
        command = json.loads((directory / "command.json").read_text(encoding="utf-8"))
        self.assertEqual(command["environment"], training_environment(config))
        self.assertTrue(load_config(directory / "settings.json").offline_mode)
        self.assertNotIn("offline_mode", tomllib.loads((directory / "training.toml").read_text(encoding="utf-8")))

    def test_offline_failure_explains_missing_cache_without_online_retry(self):
        worker = TrainingWorker(replace(self.config, offline_mode=True))
        results = []
        worker.completed.connect(results.append)
        with patch("src.training.runner.run_process", side_effect=[0, 1]) as process:
            worker.run()
        self.assertEqual(process.call_count, 2)
        self.assertEqual(results[0].status, "failed")
        self.assertIn("missing cached files", results[0].message)

    @unittest.skipUnless(os.name == "nt", "Windows process-tree stop")
    def test_stop_ends_child_process_too(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handles = []
        text = []
        stop = threading.Event()

        def output(chunk):
            text.append(chunk)
            if "\n" in "".join(text) and not handles:
                pid = int("".join(text).strip())
                handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
                self.assertTrue(handle)
                handles.append(handle)
                stop.set()

        script = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid,flush=True); time.sleep(60)"
        )
        try:
            run_process([sys.executable, "-u", "-c", script], self.root, output, stop)
            self.assertTrue(handles)
            self.assertEqual(kernel.WaitForSingleObject(handles[0], 5000), 0)
        finally:
            for handle in handles:
                kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
