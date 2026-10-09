import unittest

from src.training.monitor import TrainingMonitor


def step(current, total=504):
    return f"steps: {int(current / total * 100)}%|fixture| {current}/{total}\r"


class ResumeProgressTests(unittest.TestCase):
    def test_epoch_boundary_overall_progress_and_completion(self):
        monitor = TrainingMonitor(now=0, resume_step=378)
        monitor.feed("total optimization steps / 学習ステップ数: 504\n", now=1)
        self.assertEqual(monitor.feed(step(0), now=2), [(378, 504)])
        self.assertEqual(monitor.feed(step(1), now=3), [(379, 504)])
        self.assertEqual(monitor.feed(step(40), now=4), [(418, 504)])
        self.assertEqual(monitor.feed(step(126), now=5), [(504, 504)])
        self.assertEqual(monitor.snapshot(now=5)["eta_seconds"], 0)

    def test_mid_epoch_uses_full_target_not_shortened_tqdm_total(self):
        monitor = TrainingMonitor(now=0, resume_step=400)
        monitor.feed("total optimization steps: 504\n", now=1)
        # sd-scripts subtracts only the within-epoch 22 batches from tqdm's
        # denominator, not all 400 saved steps. There are 104 updates left.
        self.assertEqual(monitor.feed(step(0, 482), now=2), [(400, 504)])
        self.assertEqual(monitor.feed(step(1, 482), now=3), [(401, 504)])
        self.assertEqual(monitor.feed(step(104, 482), now=4), [(504, 504)])

    def test_warmup_duplicate_steps_and_eta_use_new_steps_only(self):
        monitor = TrainingMonitor(now=0, resume_step=378, expected_total=504)
        for current in range(5):
            monitor.feed(step(current), now=current * 2)
        self.assertIsNone(monitor.sample_start)
        monitor.feed(step(5), now=10)
        self.assertIsNone(monitor.snapshot(now=10)["steps_per_second"])
        monitor.feed(step(6), now=12)
        monitor.feed(step(6), now=14)
        metrics = monitor.snapshot(now=75)
        self.assertEqual(metrics["current"], 384)
        self.assertEqual(metrics["session_current"], 6)
        self.assertEqual(metrics["steps_per_second"], 0.5)
        self.assertEqual(metrics["eta_seconds"], 240)
        self.assertEqual(metrics["measured_steps"], 1)
        self.assertEqual(metrics["silent_seconds"], 61)

    def test_chunked_unicode_ansi_and_unrelated_progress(self):
        monitor = TrainingMonitor(now=0, resume_step=378)
        content = (
            "caching latents: 100%|x| 126/126\r\n"
            "\x1b[32mtotal optimization steps / 学習ステップ数: 504\x1b[0m\r\n"
            + step(0) + step(1)
        )
        progress = []
        for char in content:
            progress.extend(monitor.feed(char, now=1))
        self.assertEqual(progress, [(378, 504), (379, 504)])

    def test_unknown_target_waits_and_log_takes_precedence_over_history(self):
        monitor = TrainingMonitor(now=0, resume_step=400)
        self.assertEqual(monitor.feed(step(1, 482), now=1), [])
        self.assertEqual(monitor.snapshot(now=1)["total"], 0)
        self.assertIsNone(monitor.snapshot(now=1)["eta_seconds"])
        monitor.feed("total optimization steps: 504\n", now=2)
        self.assertEqual(monitor.feed(step(2, 482), now=3), [(402, 504)])
        monitor = TrainingMonitor(now=0, resume_step=378, expected_total=600)
        monitor.feed("total optimization steps: 504\n", now=1)
        self.assertEqual(monitor.feed(step(1), now=2), [(379, 504)])

    def test_unexpected_counter_does_not_emit_more_than_target(self):
        monitor = TrainingMonitor(now=0, resume_step=378, expected_total=504)
        monitor.feed(step(125), now=1)
        self.assertEqual(monitor.feed(step(127), now=2), [])
        self.assertEqual(monitor.snapshot(now=2)["current"], 503)
        self.assertIn("exceeds expected total", monitor.phase)

    def test_fresh_run_without_target_log_keeps_original_counter(self):
        monitor = TrainingMonitor(now=0)
        self.assertEqual(monitor.feed(step(0), now=1), [(0, 504)])
        self.assertEqual(monitor.feed(step(1), now=2), [(1, 504)])

    def test_step_layout_handles_split_unicode_and_ansi_metadata(self):
        monitor = TrainingMonitor(now=0)
        text = (
            "\x1b[32mnum batches per epoch / 1epochのバッチ数: 126\x1b[0m\r\n"
            "gradient accumulation steps / 勾配を合計するステップ数 = 1\r\n"
        )
        for char in text:
            monitor.feed(char, now=1)
        self.assertEqual(monitor.parser.step_layout, {"version": 1, "steps_per_epoch": 126})


if __name__ == "__main__":
    unittest.main()
