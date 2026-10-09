from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox

from src.dataset.captions import (
    add_caption_prefix, apply_caption_prefix, build_caption_prefix,
    load_caption, save_caption, suggest_keep_tokens,
)
from src.training.command_builder import create_training_run, dataset_toml
from src.training.config import SDXLTrainConfig, load_config, save_config, validate_settings
from src.training.runner import TrainingWorker
from src.ui.tagging_window import TaggingWindow
from src.ui.training_dialog import TrainingDialog


class CaptionDatasetSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "10_role 1girl"
        self.dataset.mkdir()
        self.images = [self.dataset / "one.png", self.dataset / "two.png"]
        for image in self.images:
            Image.new("RGB", (16, 16)).save(image)
            save_caption(image, "solo, handmade tag, blue eyes")
        model = self.root / "base.safetensors"
        model.write_bytes(b"fixture, not real model weights")
        self.config = SDXLTrainConfig(
            model_path=str(model), dataset_path=str(self.dataset), output_dir=str(self.root),
            sd_scripts_dir=str(Path(__file__).parent.resolve() / "fixtures" / "fake_sd_scripts"),
            python_executable=sys.executable,
        )

    def test_prefix_combinations_and_suggestions(self):
        for trigger, class_token, prefix, count in (
            ("tsnr01", "", "tsnr01", 1),
            ("tsnr01", "1girl", "tsnr01, 1girl", 2),
            (" tsnr01 ", "  ", "tsnr01", 1),
            ("", "1girl", "1girl", 1),
            ("", "", "", 0),
            ("tsnr01, TSNR01", ", 1girl, 1girl,", "tsnr01, 1girl", 2),
        ):
            with self.subTest(trigger=trigger, class_token=class_token):
                self.assertEqual(build_caption_prefix(trigger, class_token), prefix)
                self.assertEqual(suggest_keep_tokens(trigger, class_token), count)
        with self.assertRaises(ValueError):
            add_caption_prefix("manual", ", ,")

    def test_prefix_insertion_deduplication_and_manual_tags_preserved(self):
        expected = "tsnr01, 1girl, solo, handmade tag, blue eyes"
        for caption in (
            "solo, handmade tag, blue eyes",
            expected,
            "1girl, solo, TSNR01, handmade tag, 1girl, blue eyes, tsnr01",
        ):
            with self.subTest(caption=caption):
                actual = add_caption_prefix(caption, "tsnr01, 1girl")
                self.assertEqual(actual, expected)
                self.assertEqual(add_caption_prefix(actual, "tsnr01, 1girl"), expected)
        self.assertEqual(add_caption_prefix("", "tsnr01"), "tsnr01")
        # Do not guess that an unrelated tag is an old trigger to delete.
        self.assertEqual(add_caption_prefix("old trigger, handmade tag", "new trigger"),
                         "new trigger, old trigger, handmade tag")

    def test_existing_caption_files_updated_without_wd14(self):
        self.assertEqual(apply_caption_prefix(self.images, "tsnr01, 1girl"), self.images)
        self.assertEqual(apply_caption_prefix(self.images, "tsnr01, 1girl"), [])
        for image in self.images:
            self.assertEqual(load_caption(image), "tsnr01, 1girl, solo, handmade tag, blue eyes")

    def test_apply_buttons_use_unsaved_editor_and_preserve_navigation(self):
        window = TaggingWindow(str(self.dataset))
        self.addCleanup(window.close)
        self.assertEqual(window.effective_caption_prefix(), "")  # Never infer the folder name.
        window.set_caption_prefix("tsnr01", "1girl")
        self.assertEqual(window.prefix_preview.text(), "tsnr01, 1girl")
        window.caption_edit.setPlainText("manual edit, 1girl, solo")
        window.apply_prefix_to_current()
        self.assertEqual(load_caption(self.images[0]), "tsnr01, 1girl, manual edit, solo")
        window.caption_edit.setPlainText("manual edit, additional tag")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), patch.object(QMessageBox, "information"):
            window.apply_prefix_to_all()
        self.assertEqual(load_caption(self.images[0]), "tsnr01, 1girl, manual edit, additional tag")
        window.next_image()
        self.assertEqual(window.position_slider.value(), 1)
        self.assertEqual(window.position_label.text(), "2 / 2")
        self.assertEqual(window.caption_edit.toPlainText(), load_caption(self.images[1]))
        window.previous_image()
        self.assertEqual(window.position_slider.value(), 0)
        window.position_slider.setValue(1)
        self.assertEqual(window.index, 1)

    def test_wd14_current_and_all_apply_prefix_with_existing_confirmations(self):
        window = TaggingWindow(str(self.dataset))
        self.addCleanup(window.close)
        window.set_caption_prefix("tsnr01", "1girl")
        window.wd14_excluded_tags.setText("watermark, text")
        for batch in (False, True):
            with self.subTest(batch=batch), patch("src.ui.tagging_window.WD14Tagger") as tagger, patch.object(
                QMessageBox, "question", return_value=QMessageBox.Yes
            ):
                tagger.return_value.predict.return_value = SimpleNamespace(caption="1girl, solo, tsnr01, blue eyes")
                if batch:
                    window.tag_all_images()
                else:
                    window.tag_current_image()
                self.assertFalse(window.trigger_word_edit.isEnabled())
                self.assertFalse(window.class_token_edit.isEnabled())
                worker = window.tagging_worker
                self.assertTrue(worker.wait(5000))
                deadline = time.monotonic() + 5
                while window.tagging_worker is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                self.assertIsNone(window.tagging_worker)
                self.assertEqual(tagger.call_args.args[0].excluded_tags, frozenset({"watermark", "text"}))
                self.assertEqual(tagger.return_value.predict.call_count, 2 if batch else 1)
                for image in self.images if batch else self.images[:1]:
                    self.assertEqual(load_caption(image), "tsnr01, 1girl, solo, blue eyes")
        window.set_caption_prefix("", "")
        window.tagging_prefix = ""
        window.save_generated_captions({self.images[0]: SimpleNamespace(caption="solo, blue eyes")})
        self.assertEqual(load_caption(self.images[0]), "solo, blue eyes")

    def test_defaults_and_validation(self):
        self.assertEqual(self.config.num_repeats, 1)
        self.assertTrue(self.config.shuffle_caption)
        self.assertEqual(self.config.keep_tokens, 0)
        self.assertFalse(validate_settings(self.config))
        for changes, message in (
            ({"num_repeats": 0}, "Repeats Per Image"),
            ({"num_repeats": -1}, "Repeats Per Image"),
            ({"keep_tokens": -1}, "Keep Tokens"),
            ({"num_repeats": True}, "type"),
            ({"num_repeats": 1.5}, "type"),
            ({"keep_tokens": False}, "type"),
            ({"keep_tokens": 1.5}, "type"),
            ({"shuffle_caption": "true"}, "type"),
            ({"cache_text_encoder_outputs": True}, "Shuffle Captions"),
        ):
            with self.subTest(changes=changes):
                config = replace(self.config, **changes)
                self.assertIn(message, "\n".join(validate_settings(config)))
                with self.assertRaises(ValueError):
                    create_training_run(config)
                worker = TrainingWorker(config)
                results = []
                worker.completed.connect(results.append)
                with patch("src.training.runner.run_process") as process:
                    worker.run()
                    process.assert_not_called()
                self.assertEqual(results[0].status, "failed")

    def test_save_load_and_generated_toml(self):
        for repeats, shuffle, keep in ((1, True, 0), (10, True, 2), (20, False, 3)):
            with self.subTest(repeats=repeats):
                config = replace(self.config, num_repeats=repeats, shuffle_caption=shuffle,
                                 keep_tokens=keep, keep_tokens_auto=False, trigger_word="tsnr01", class_token="1girl")
                path = self.root / "settings.json"
                save_config(config, path)
                self.assertEqual(load_config(path), config)
                run = create_training_run(load_config(path))
                parsed = tomllib.loads((run.directory / "dataset.toml").read_text(encoding="utf-8"))
                self.assertEqual(parsed["general"]["shuffle_caption"], shuffle)
                self.assertEqual(parsed["general"]["keep_tokens"], keep)
                subset = parsed["datasets"][0]["subsets"][0]
                self.assertEqual(subset["num_repeats"], repeats)
                self.assertEqual(subset["image_dir"], str(self.dataset))
                self.assertNotIn("caption_prefix", subset)  # Already in .txt; don't apply twice.

    def test_old_versions_default_new_fields_without_folder_inference(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                settings = asdict(self.config)
                for name in ("trigger_word", "class_token", "num_repeats", "shuffle_caption", "keep_tokens", "keep_tokens_auto"):
                    del settings[name]
                if version == 1:
                    del settings["attention_backend"]
                if version < 3:
                    del settings["python_executable"]
                    del settings["sd_scripts_dir"]
                settings["cache_text_encoder_outputs"] = True
                path = self.root / "legacy.json"
                path.write_text(json.dumps({"schema_version": version, "training_type": "sdxl_lora", "settings": settings}), encoding="utf-8")
                config = load_config(path)
                self.assertEqual(config.num_repeats, 1)
                self.assertTrue(config.shuffle_caption)
                self.assertEqual(config.keep_tokens, 0)
                self.assertEqual(config.trigger_word, "")
                self.assertEqual(config.class_token, "")
                self.assertFalse(config.cache_text_encoder_outputs)
                self.assertFalse(validate_settings(config))

    def test_keep_tokens_suggestion_manual_override_and_file_roundtrip(self):
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.apply_config(self.config)
        dialog.set_caption_prefix("tsnr01", "")
        self.assertEqual(dialog.read_config().keep_tokens, 1)
        dialog.controls["class_token"].setText("1girl")
        self.assertEqual(dialog.read_config().keep_tokens, 2)
        dialog.controls["keep_tokens"].setValue(4)
        self.assertFalse(dialog.read_config().keep_tokens_auto)
        dialog.controls["class_token"].setText("")
        self.assertEqual(dialog.read_config().keep_tokens, 4)
        path = self.root / "manual.json"
        save_config(dialog.read_config(), path)
        dialog.apply_config(self.config)
        dialog.apply_config(load_config(path))
        dialog.set_caption_prefix("newtrigger", "1boy")
        self.assertEqual(dialog.read_config().keep_tokens, 4)
        self.assertFalse(dialog.read_config().keep_tokens_auto)
        dialog.controls["keep_tokens_auto"].setChecked(True)
        self.assertEqual(dialog.read_config().keep_tokens, 2)
        dialog.controls["keep_tokens"].setValue(0)
        dialog.set_caption_prefix("another", "")
        self.assertEqual(dialog.read_config().keep_tokens, 0)

    def test_new_configuration_rejects_invalid_or_missing_dataset_settings(self):
        path = self.root / "invalid.json"
        for changes in (
            {"num_repeats": 0}, {"keep_tokens": -1}, {"shuffle_caption": "yes"},
            {"cache_text_encoder_outputs": True}, {"unknown_field": 1},
        ):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({
                    "schema_version": 4, "training_type": "sdxl_lora",
                    "settings": {**asdict(self.config), **changes},
                }), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)
        settings = asdict(self.config)
        del settings["num_repeats"]
        path.write_text(json.dumps({
            "schema_version": 4, "training_type": "sdxl_lora", "settings": settings,
        }), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_shuffle_and_text_cache_ui_are_mutually_exclusive(self):
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.read_config().shuffle_caption)
        dialog.controls["cache_text_encoder_outputs"].setChecked(True)
        self.assertFalse(dialog.read_config().shuffle_caption)
        self.assertTrue(dialog.read_config().network_train_unet_only)
        dialog.controls["shuffle_caption"].setChecked(True)
        self.assertFalse(dialog.read_config().cache_text_encoder_outputs)
        self.assertTrue(dialog.controls["network_train_unet_only"].isEnabled())
        config = replace(self.config, shuffle_caption=False, cache_text_encoder_outputs=True)
        dialog.apply_config(config)
        self.assertEqual(dialog.read_config(), config)

    def test_caption_state_dataset_switch_and_training_config_roundtrip(self):
        window = TaggingWindow(str(self.dataset))
        self.addCleanup(window.close)
        window.set_caption_prefix("tsnr01", "1girl")
        other = self.root / "another dataset"
        other.mkdir()
        Image.new("RGB", (16, 16)).save(other / "one.png")
        window.open_dataset(str(other))
        self.assertEqual(window.effective_caption_prefix(), "")
        window.set_caption_prefix("other", "1boy")
        window.open_dataset(str(self.dataset))
        self.assertEqual(window.effective_caption_prefix(), "tsnr01, 1girl")
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
        dialog = window.training_dialog
        self.assertEqual(dialog.read_config().keep_tokens, 2)
        dialog.apply_config(replace(self.config, trigger_word="tsnr01", class_token="1girl", num_repeats=10, keep_tokens=2))
        path = self.root / "workflow.json"
        save_config(dialog.read_config(), path)
        restored = load_config(path)
        with patch.object(TrainingDialog, "exec", side_effect=lambda: dialog.apply_config(restored)):
            window.set_caption_prefix("temporary", "")
            window.open_training_setup()
        self.assertEqual(window.effective_caption_prefix(), "tsnr01, 1girl")
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
        self.assertEqual(dialog.read_config(), restored)
        parsed = tomllib.loads(dataset_toml(dialog.read_config()))
        self.assertEqual(parsed["general"]["keep_tokens"], 2)
        self.assertTrue(parsed["general"]["shuffle_caption"])
        self.assertEqual(parsed["datasets"][0]["subsets"][0]["num_repeats"], 10)


if __name__ == "__main__":
    unittest.main()
