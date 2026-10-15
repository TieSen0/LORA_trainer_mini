"""Maintenance and telemetry checks without model loading or GPU training."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import diagnose
from src.diagnostics import collect_report, format_report, probe_process
from src.training.config import SDXLTrainConfig
from src.training.monitor import sample_gpu
from src.training.resources import ResourceSampler, resource_summary
from src.training.runner import run_process
from src.ui.environment_dialog import EnvironmentDialog
from src.ui.training_dialog import TrainingDialog


class DiagnosticsTests(unittest.TestCase):
    def test_probes_are_offline_isolated_and_bounded(self):
        with patch.dict(os.environ, {"VIRTUAL_ENV": "wrong", "PYTHONPATH": "wrong", "PYTHONHOME": "wrong"}), patch(
            "src.diagnostics.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="ok", stderr="")
        ) as run:
            self.assertEqual(probe_process(sys.executable, "print('ok')", Path.cwd()), (True, "ok"))
        options = run.call_args.kwargs
        self.assertEqual(options["timeout"], 45)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(options["env"]["TRANSFORMERS_OFFLINE"], "1")
        for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
            self.assertNotIn(key, options["env"])

    def test_import_failure_and_timeout_are_reported(self):
        for error in (FileNotFoundError("missing interpreter"), subprocess.TimeoutExpired("probe", 45)):
            with patch("src.diagnostics.subprocess.run", side_effect=error):
                ok, detail = probe_process("missing", "pass", Path.cwd())
                self.assertFalse(ok)
                self.assertTrue(detail)

    def test_unconfigured_report_is_actionable_without_changing_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.diagnostics.probe_process", return_value=(False, "missing package")):
            root = Path(directory)
            report = format_report(collect_report(root=root))
            self.assertIn("Setup.cmd", report)
            self.assertIn("Not configured", report)
            self.assertIn("[FAIL] Locked dependencies", report)
            self.assertEqual(list(root.iterdir()), [])

    def test_configured_runtime_uses_selected_python(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.diagnostics.probe_process", return_value=(True, "imports OK")) as probe:
            root = Path(directory)
            (root / "sdxl_train_network.py").touch()
            config = replace(SDXLTrainConfig(), sd_scripts_dir=str(root), python_executable=sys.executable, offline_mode=True)
            report = format_report(collect_report(config, root=root))
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(probe.call_args.args[0], Path(sys.executable))
            self.assertIn("Offline tokenizer cache", report)
            self.assertIn("Not verified", report)

    def test_cli_refuses_to_overwrite_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text("keep me", encoding="utf-8")
            with patch.object(sys, "argv", ["diagnose.py", "--report", str(path)]), patch(
                "diagnose.load_session", return_value={}
            ), patch("diagnose.collect_report", return_value=[]), patch("builtins.print"):
                self.assertEqual(diagnose.main(), 1)
            self.assertEqual(path.read_text(), "keep me")


class ResourceTests(unittest.TestCase):
    def sampler(self):
        sampler = ResourceSampler()
        ps = Mock()
        ps.Error = RuntimeError
        ps.virtual_memory.return_value = SimpleNamespace(percent=93, available=0.5 * 1024**3)
        ps.Process.return_value.memory_info.return_value = SimpleNamespace(rss=40 * 1024**2)
        ps.cpu_percent.return_value = 80
        sampler.psutil = ps
        return sampler, ps

    def test_first_sample_is_baseline_then_disk_rate_is_delta(self):
        sampler, ps = self.sampler()
        ps.disk_io_counters.side_effect = [SimpleNamespace(read_bytes=0, write_bytes=0),
                                         SimpleNamespace(read_bytes=10 * 1024**2, write_bytes=5 * 1024**2)]
        first = sampler.sample(now=10)
        self.assertIsNone(first["cpu_percent"])
        self.assertNotIn("disk_read_mib_s", first)
        second = sampler.sample(now=15)
        self.assertEqual(second["cpu_percent"], 80)
        self.assertEqual(second["disk_read_mib_s"], 2)
        self.assertEqual(second["disk_write_mib_s"], 1)
        ps.cpu_percent.assert_called_with(interval=None)

    def test_trainer_tree_handles_disappearing_children_and_pid_reuse(self):
        sampler, ps = self.sampler()
        ps.disk_io_counters.return_value = None
        child = Mock()
        child.memory_info.side_effect = RuntimeError("exited")
        root = ps.Process.return_value
        root.children.return_value = [child]
        sampler.attach(123)
        first = sampler.sample()
        self.assertEqual(first["trainer_ram_mib"], 40)
        self.assertTrue(first["trainer_sample_partial"])
        root.is_running.return_value = False
        second = sampler.sample()
        self.assertNotIn("trainer_ram_mib", second)

    def test_missing_monitor_dependency_does_not_break_training(self):
        sampler = ResourceSampler()
        sampler.psutil = None
        sampler.attach(123)
        self.assertIn("Setup.cmd", sampler.sample()["unavailable"])

    def test_hints_are_not_bottleneck_diagnoses_or_initialization_alarms(self):
        data = {"ram_percent": 94, "available_ram_gib": 0.3,
                "gpu": {"utilization": 3, "memory_mib": 4567}}
        text = resource_summary(data, "Training")
        self.assertIn("paging may", text)
        self.assertIn("does not identify a bottleneck", text)
        self.assertNotIn("Low sampled GPU", resource_summary(data, "Loading model"))
        self.assertIn("off", resource_summary(None, "Training"))
        self.assertIn("Waiting", resource_summary({}, "Training"))

    def test_malformed_gpu_output_is_unavailable(self):
        with patch("src.training.monitor.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="broken\n")):
            self.assertIsNone(sample_gpu())

    def test_started_callback_gets_only_the_launched_child(self):
        pids, output = [], []
        code = run_process([sys.executable, "-c", "import os; print(os.getpid(), os.getppid())"], Path.cwd(), output.append,
                           threading.Event(), on_started=pids.append)
        self.assertEqual(code, 0)
        # Windows venv executables may redirect to a child interpreter.
        self.assertEqual(len(pids), 1)
        self.assertIn(pids[0], [int(pid) for pid in "".join(output).split()])


class EnvironmentUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_async_report_and_close_guard(self):
        gate = threading.Event()

        def report(config):
            gate.wait(timeout=2)
            return [{"status": "WARN", "title": "Test", "detail": "No model loaded"}]

        with patch("src.ui.environment_dialog.collect_report", side_effect=report):
            dialog = EnvironmentDialog(SDXLTrainConfig())
            try:
                self.assertTrue(dialog.running)
                self.assertFalse(dialog.buttons.isEnabled())
                dialog.accept()
                self.assertEqual(dialog.result(), 0)
                gate.set()
                self.assertTrue(dialog.worker.wait(3000))
                for _ in range(5):
                    self.app.processEvents()
                self.assertFalse(dialog.running)
                self.assertIn("1 warnings", dialog.summary.text())
                self.assertIn("No model loaded", dialog.details.toPlainText())
                self.assertTrue(dialog.buttons.isEnabled())
            finally:
                gate.set()
                dialog.worker.wait(3000)
                self.app.processEvents()
                dialog.close()

    def test_training_dialog_displays_resource_snapshot(self):
        dialog = TrainingDialog()
        try:
            dialog.receive_telemetry({"phase": "Training", "elapsed_seconds": 10, "silent_seconds": 0,
                                      "resources": {"ram_percent": 91, "available_ram_gib": .7}})
            self.assertIn("System RAM 91%", dialog.resource_label.text())
            self.assertIn("GPU reading unavailable", dialog.resource_label.text())
            self.assertEqual(dialog.environment_button.text(), "Check Environment")
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
