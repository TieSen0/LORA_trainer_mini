from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.training.config import SCHEMA_VERSION, SDXLTrainConfig, config_payload, load_config, parse_config, save_config, validate_config, validate_settings
from src.ui.tagging_window import TaggingWindow
from src.ui.training_dialog import TrainingDialog


class TrainingConfigTests(unittest.TestCase):
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
        self.model = self.root / "base.safetensors"
        # Path validation intentionally does not inspect model architecture.
        self.model.write_bytes(b"test file, not model weights")
        self.config = SDXLTrainConfig(
            model_path=str(self.model), dataset_path=str(self.dataset), output_dir=str(self.root),
        )

    def test_save_load_roundtrip_all_settings(self):
        config = replace(
            self.config, epochs=20, network_dim=64, network_alpha=32.5,
            learning_rate=0.00005, optimizer="AdamW8bit", lr_scheduler="cosine",
            mixed_precision="bf16", cache_text_encoder_outputs=False,
            network_train_unet_only=False,
            attention_backend="xformers",
        )
        destination = self.root / "training.json"
        save_config(config, destination)
        self.assertEqual(load_config(destination), config)
        self.assertEqual(list(self.root.glob(".training-config-*")), [])

    def test_invalid_types_ranges_names_and_cache_combination(self):
        for changes in (
            {"epochs": True}, {"epochs": 0}, {"resolution": 1000},
            {"learning_rate": float("nan")}, {"learning_rate": -1},
            {"network_alpha": float("inf")}, {"optimizer": "unknown"},
            {"network_train_unet_only": False, "cache_text_encoder_outputs": True, "shuffle_caption": False},
            {"output_name": "../outside"},
            {"output_name": "CON"}, {"output_name": "lora."},
            {"output_name": "lora.safetensors"},
            {"attention_backend": "unknown"},
            {"cache_latents_to_disk": "yes"},
            {"cache_latents": False, "cache_latents_to_disk": True},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(validate_settings(replace(self.config, **changes)))

    def test_disk_latent_cache_roundtrip_and_ui_dependency(self):
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        config = replace(self.config, cache_latents_to_disk=True)
        dialog.apply_config(config)
        self.assertEqual(dialog.read_config(), config)
        destination = self.root / "disk-cache.json"
        save_config(dialog.read_config(), destination)
        self.assertEqual(load_config(destination), config)
        dialog.controls["cache_latents"].setChecked(False)
        self.assertFalse(dialog.controls["cache_latents_to_disk"].isChecked())
        self.assertFalse(dialog.controls["cache_latents_to_disk"].isEnabled())
        dialog.controls["cache_latents"].setChecked(True)
        self.assertTrue(dialog.controls["cache_latents_to_disk"].isEnabled())
        self.assertFalse(dialog.controls["cache_latents_to_disk"].isChecked())

    def test_legacy_disk_cache_default_and_current_schema_strictness(self):
        payload = config_payload(self.config)
        del payload["settings"]["cache_latents_to_disk"]
        for version in range(1, SCHEMA_VERSION):
            payload["schema_version"] = version
            self.assertFalse(parse_config(payload).cache_latents_to_disk)
        payload["schema_version"] = SCHEMA_VERSION
        with self.assertRaises(ValueError):
            parse_config(payload)

    def test_disk_cache_requires_writable_dataset(self):
        with patch("src.training.config.os.access", return_value=False):
            errors = validate_config(replace(self.config, cache_latents_to_disk=True))
        self.assertIn("writable dataset folder", " ".join(errors))

    def test_bad_paths_and_existing_model_output_is_preserved_by_separate_runs(self):
        self.assertFalse(validate_config(self.config))
        self.assertTrue(validate_config(replace(self.config, model_path="")))
        self.assertTrue(validate_config(replace(self.config, dataset_path=str(self.root / "missing"))))
        self.assertTrue(validate_config(replace(self.config, output_dir=str(self.model))))
        self.model.write_bytes(b"")
        self.assertTrue(any("empty" in e for e in validate_config(self.config)))
        self.model.write_bytes(b"nonempty")
        (self.root / "my_lora.safetensors").write_bytes(b"keep existing")
        self.assertEqual(validate_config(self.config), [])

    def test_invalid_save_preserves_existing_settings(self):
        path = self.root / "settings.json"
        path.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            save_config(replace(self.config, epochs=0), path)
        self.assertEqual(path.read_text(encoding="utf-8"), "original")
        with patch("pathlib.Path.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                save_config(self.config, path)
        self.assertEqual(path.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(self.root.glob(".training-config-*")), [])

    def test_import_rejects_malformed_config_but_allows_moved_paths(self):
        path = self.root / "settings.json"
        payload = {"schema_version": 1, "training_type": "sdxl_lora", "settings": asdict(self.config)}
        for bad in ([], {**payload, "schema_version": 99}, {**payload, "training_type": "other"}, {**payload, "settings": {"epochs": 2}}):
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        payload["settings"]["model_path"] = "missing.safetensors"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(load_config(path).model_path, "missing.safetensors")

    def test_dialog_roundtrip_and_cache_constraint(self):
        dialog = TrainingDialog()
        dialog.apply_config(self.config)
        self.assertEqual(dialog.read_config(), self.config)
        self.assertTrue(dialog.controls["network_train_unet_only"].isEnabled())
        dialog.controls["cache_text_encoder_outputs"].setChecked(False)
        dialog.controls["network_train_unet_only"].setChecked(False)
        dialog.controls["cache_text_encoder_outputs"].setChecked(True)
        self.assertTrue(dialog.read_config().network_train_unet_only)
        self.assertFalse(dialog.controls["network_train_unet_only"].isEnabled())
        self.assertFalse(dialog.read_config().shuffle_caption)
        dialog.apply_config(self.config)
        path = self.root / "ui-settings.json"
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")), patch.object(QMessageBox, "information"):
            dialog.save_settings()
        dialog.controls["epochs"].setValue(50)
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
            dialog.load_settings()
        self.assertEqual(dialog.read_config(), self.config)
        self.assertIn("Loaded:", dialog.status.text())
        dialog.close()

    def test_main_window_reuses_form_and_carries_selected_dataset(self):
        window = TaggingWindow(str(self.dataset))
        with patch.object(TrainingDialog, "exec", return_value=0):
            window.open_training_setup()
            self.assertEqual(window.training_dialog.read_config().dataset_path, str(self.dataset))
            window.training_dialog.controls["epochs"].setValue(25)
            window.open_training_setup()
            self.assertEqual(window.training_dialog.read_config().epochs, 25)
        window.set_tagging_busy(True)
        self.assertFalse(window.training_button.isEnabled())
        window.set_tagging_busy(False)
        window.close()

    def test_attention_selection_persists_through_form_and_file(self):
        dialog = TrainingDialog()
        self.assertEqual(dialog.controls["attention_backend"].currentText(), "SDPA")
        for backend, label in (("sdpa", "SDPA"), ("xformers", "xFormers")):
            with self.subTest(backend=backend):
                config = replace(self.config, attention_backend=backend)
                dialog.apply_config(config)
                self.assertEqual(dialog.controls["attention_backend"].currentText(), label)
                destination = self.root / "attention.json"
                save_config(dialog.read_config(), destination)
                self.assertEqual(load_config(destination), config)
                self.assertEqual(json.loads(destination.read_text())["schema_version"], SCHEMA_VERSION)
        dialog.close()

    def test_legacy_attention_default_and_required_new_field(self):
        settings = asdict(self.config)
        del settings["attention_backend"]
        path = self.root / "legacy.json"
        payload = {"schema_version": 1, "training_type": "sdxl_lora", "settings": settings}
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(load_config(path).attention_backend, "sdpa")
        payload["schema_version"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_version_two_settings_load_with_unconfigured_runtime(self):
        settings = asdict(self.config)
        del settings["sd_scripts_dir"]
        del settings["python_executable"]
        path = self.root / "version-two.json"
        path.write_text(json.dumps({
            "schema_version": 2, "training_type": "sdxl_lora", "settings": settings,
        }), encoding="utf-8")
        loaded = load_config(path)
        self.assertEqual(loaded.sd_scripts_dir, "")
        self.assertEqual(loaded.python_executable, "")
        self.assertEqual(loaded.attention_backend, "sdpa")

    def test_offline_mode_ui_and_file_roundtrip(self):
        dialog = TrainingDialog()
        self.addCleanup(dialog.close)
        dialog.apply_config(self.config)
        self.assertFalse(dialog.controls["offline_mode"].isChecked())
        for enabled in (True, False):
            dialog.controls["offline_mode"].setChecked(enabled)
            path = self.root / "offline.json"
            save_config(dialog.read_config(), path)
            restored = load_config(path)
            self.assertEqual(restored.offline_mode, enabled)
            dialog.apply_config(restored)
            self.assertEqual(dialog.read_config(), restored)
        self.assertTrue(validate_settings(replace(self.config, offline_mode="yes")))

    def test_legacy_offline_default_and_new_schema_strictness(self):
        settings = asdict(self.config)
        del settings["offline_mode"]
        path = self.root / "legacy-offline.json"
        for version in (1, 2, 3, 4, 5):
            path.write_text(json.dumps({
                "schema_version": version, "training_type": "sdxl_lora", "settings": settings,
            }), encoding="utf-8")
            if version < 5:
                self.assertFalse(load_config(path).offline_mode)
            else:
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
