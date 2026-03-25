"""
TDD: Section 1 — LoRA / custom .safetensors + Metadata management
Run: python -m unittest genbox.test_utils_s1 -v
"""
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestParseLoraSpec(unittest.TestCase):
    def _import(self):
        from genbox.utils.utils import parse_lora_spec
        return parse_lora_spec

    def test_string_spec_defaults_weight_1(self):
        f = self._import()
        path, weight = f("style.safetensors")
        self.assertEqual(path, "style.safetensors")
        self.assertAlmostEqual(weight, 1.0)

    def test_tuple_spec(self):
        f = self._import()
        path, weight = f(("style.safetensors", 0.7))
        self.assertEqual(path, "style.safetensors")
        self.assertAlmostEqual(weight, 0.7)

    def test_list_spec(self):
        f = self._import()
        path, weight = f(["detail.safetensors", 0.5])
        self.assertEqual(path, "detail.safetensors")
        self.assertAlmostEqual(weight, 0.5)

    def test_weight_coerced_to_float(self):
        f = self._import()
        _, weight = f(("a.safetensors", 1))
        self.assertIsInstance(weight, float)


class TestLoraMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.lora_file = self.base / "mystyle.safetensors"
        self.lora_file.write_bytes(b"\x00" * 8)

    def tearDown(self):
        self.tmp.cleanup()

    def _import(self):
        from genbox.utils.utils import write_lora_metadata, read_lora_metadata
        return write_lora_metadata, read_lora_metadata

    def test_write_and_read_metadata(self):
        write_meta, read_meta = self._import()
        write_meta(
            self.lora_file,
            architecture="flux",
            trigger="masterpiece, best quality",
            description="A stylized watercolor lora",
            preview_url="https://example.com/preview.jpg",
        )
        sidecar = self.lora_file.with_suffix(".json")
        self.assertTrue(sidecar.exists())

        meta = read_meta(self.lora_file)
        self.assertEqual(meta["architecture"], "flux")
        self.assertEqual(meta["trigger"], "masterpiece, best quality")
        self.assertEqual(meta["description"], "A stylized watercolor lora")
        self.assertEqual(meta["preview_url"], "https://example.com/preview.jpg")

    def test_read_missing_returns_empty(self):
        _, read_meta = self._import()
        meta = read_meta(self.base / "nonexistent.safetensors")
        self.assertEqual(meta, {})

    def test_write_minimal_no_optional_fields(self):
        write_meta, read_meta = self._import()
        write_meta(self.lora_file, architecture="sd15")
        meta = read_meta(self.lora_file)
        self.assertEqual(meta["architecture"], "sd15")
        self.assertNotIn("preview_url", meta)


class TestListLoras(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lora_dir = Path(self.tmp.name) / "loras"
        self.lora_dir.mkdir()
        # create 2 loras
        f1 = self.lora_dir / "flux" / "style_a.safetensors"
        f1.parent.mkdir()
        f1.write_bytes(b"\x00" * 100)
        (f1.with_suffix(".json")).write_text(json.dumps({
            "architecture": "flux",
            "trigger": "style_a",
            "description": "Style A",
            "preview_url": "https://example.com/a.jpg",
        }))

        f2 = self.lora_dir / "sd15" / "portrait.safetensors"
        f2.parent.mkdir()
        f2.write_bytes(b"\x00" * 200)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_all_loras(self):
        from genbox.utils.utils import list_loras
        results = list_loras(loras_dir=self.lora_dir)
        self.assertEqual(len(results), 2)

    def test_filter_by_architecture(self):
        from genbox.utils.utils import list_loras
        results = list_loras(loras_dir=self.lora_dir, architecture="flux")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "style_a")

    def test_metadata_included(self):
        from genbox.utils.utils import list_loras
        results = list_loras(loras_dir=self.lora_dir, architecture="flux")
        self.assertEqual(results[0]["description"], "Style A")
        self.assertEqual(results[0]["preview_url"], "https://example.com/a.jpg")

    def test_no_architecture_filter_returns_all(self):
        from genbox.utils.utils import list_loras
        results = list_loras(loras_dir=self.lora_dir)
        names = {r["name"] for r in results}
        self.assertIn("style_a", names)
        self.assertIn("portrait", names)


class TestCustomSafetensorsManagement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name) / "models"
        self.models_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_register_custom_safetensors(self):
        from genbox.utils.utils import register_custom_file
        src = Path(self.tmp.name) / "custom_model.safetensors"
        src.write_bytes(b"\x00" * 16)

        entry = register_custom_file(
            src,
            architecture="sdxl",
            models_dir=self.models_dir,
            description="My custom SDXL fine-tune",
            preview_url="https://example.com/img.png",
        )
        self.assertEqual(entry["architecture"], "sdxl")
        self.assertIn("custom_model", entry["id"])

        # sidecar written
        dest = self.models_dir / "sdxl" / "custom_model.safetensors"
        sidecar = dest.with_suffix(".json")
        self.assertTrue(sidecar.exists())
        meta = json.loads(sidecar.read_text())
        self.assertEqual(meta["description"], "My custom SDXL fine-tune")
        self.assertEqual(meta["preview_url"], "https://example.com/img.png")

    def test_register_custom_gguf(self):
        from genbox.utils.utils import register_custom_file
        src = Path(self.tmp.name) / "mymodel.gguf"
        src.write_bytes(b"\x00" * 16)

        entry = register_custom_file(
            src,
            architecture="flux",
            models_dir=self.models_dir,
        )
        self.assertIn("gguf", entry["quant"])

    def test_register_missing_file_raises(self):
        from genbox.utils.utils import register_custom_file
        with self.assertRaises(FileNotFoundError):
            register_custom_file(
                Path("/nonexistent/model.safetensors"),
                architecture="flux",
                models_dir=self.models_dir,
            )


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name) / "models"

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_read_model_metadata(self):
        from genbox.utils.utils import write_model_metadata, read_model_metadata
        model_path = self.models_dir / "flux" / "my_model.safetensors"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"\x00" * 8)

        write_model_metadata(
            model_path,
            description="Custom FLUX fine-tune for portraits",
            preview_url="https://civitai.com/img/123.jpg",
            tags=["portrait", "photorealistic"],
        )
        meta = read_model_metadata(model_path)
        self.assertEqual(meta["description"], "Custom FLUX fine-tune for portraits")
        self.assertEqual(meta["preview_url"], "https://civitai.com/img/123.jpg")
        self.assertEqual(meta["tags"], ["portrait", "photorealistic"])

    def test_read_missing_model_metadata_returns_empty(self):
        from genbox.utils.utils import read_model_metadata
        meta = read_model_metadata(Path("/nonexistent/model.safetensors"))
        self.assertEqual(meta, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)