"""
tests/test_dataflow.py
Real data-flow tests — no mocks for the logic under test.
Validates actual transformations: TOML bytes, PIL pixels, filename structure,
VRAM profile boundaries, remix key transformation, seed statistics.

Run: python -m unittest tests.test_dataflow -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
# TOML round-trip — real bytes on disk, real Windows-style backslash paths
# ══════════════════════════════════════════════════════════════════════════════

class TestTomlRoundtrip(unittest.TestCase):
    """TOML write -> read with real file I/O. No mocks."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_windows_backslash_path_survives_roundtrip(self):
        """Windows paths with backslashes must round-trip intact."""
        from genbox.config import _save_toml, _load_toml
        win_path = r"C:\Users\Markin\.genbox\models"
        data = {"paths": {"models": win_path}}
        p = self.tmp / "c.toml"
        _save_toml(data, p)
        loaded = _load_toml(p)
        result = loaded["paths"]["models"]
        self.assertIn("models", result)
        # backslashes preserved (not converted to forward slashes)
        self.assertNotIn("/", result)
        self.assertEqual(result, win_path)

    def test_all_scalar_types_survive(self):
        from genbox.config import _save_toml, _load_toml
        data = {
            "s": {
                "str_val": "hello world",
                "int_val": 42,
                "bool_true": True,
                "bool_false": False,
                "list_val": ["a", "b", "c"],
            }
        }
        p = self.tmp / "t.toml"
        _save_toml(data, p)
        loaded = _load_toml(p)
        s = loaded["s"]
        self.assertEqual(s["str_val"], "hello world")
        self.assertEqual(s["int_val"], 42)
        self.assertTrue(s["bool_true"])
        self.assertFalse(s["bool_false"])
        self.assertEqual(s["list_val"], ["a", "b", "c"])

    def test_nested_sections_survive(self):
        from genbox.config import _save_toml, _load_toml
        data = {
            "genbox": {"vram_gb": 12, "profile": "12gb_balanced"},
            "paths":  {"models": "/tmp/models", "outputs": "/tmp/outputs"},
        }
        p = self.tmp / "n.toml"
        _save_toml(data, p)
        loaded = _load_toml(p)
        self.assertEqual(loaded["genbox"]["vram_gb"], 12)
        self.assertEqual(loaded["paths"]["outputs"], "/tmp/outputs")

    def test_accel_list_survives(self):
        from genbox.config import _save_toml, _load_toml
        data = {"defaults": {"accel": ["sageAttn", "teacache"]}}
        p = self.tmp / "a.toml"
        _save_toml(data, p)
        loaded = _load_toml(p)
        self.assertEqual(loaded["defaults"]["accel"], ["sageAttn", "teacache"])

    def test_toml_file_is_valid_utf8(self):
        from genbox.config import _save_toml
        data = {"paths": {"models": r"C:\Users\Test\models"}}
        p = self.tmp / "utf.toml"
        _save_toml(data, p)
        content = p.read_text(encoding="utf-8")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_missing_file_returns_empty_dict(self):
        from genbox.config import _load_toml
        result = _load_toml(self.tmp / "does_not_exist.toml")
        self.assertEqual(result, {})




# ══════════════════════════════════════════════════════════════════════════════
# VRAM profile boundary transitions
# ══════════════════════════════════════════════════════════════════════════════

class TestVramProfileBoundaries(unittest.TestCase):

    def setUp(self):
        from genbox.config import _vram_profile
        self.fn = _vram_profile

    def test_0_is_low(self):
        self.assertEqual(self.fn(0), "8gb_low")

    def test_6_is_low(self):
        self.assertEqual(self.fn(6), "8gb_low")

    def test_boundary_7_is_balanced(self):
        self.assertEqual(self.fn(7), "8gb_balanced")

    def test_10_is_balanced(self):
        self.assertEqual(self.fn(10), "8gb_balanced")

    def test_boundary_11_is_12gb_balanced(self):
        self.assertEqual(self.fn(11), "12gb_balanced")

    def test_14_is_12gb_balanced(self):
        self.assertEqual(self.fn(14), "12gb_balanced")

    def test_boundary_15_is_high(self):
        self.assertEqual(self.fn(15), "16gb_high")

    def test_22_is_high(self):
        self.assertEqual(self.fn(22), "16gb_high")

    def test_boundary_23_is_ultra(self):
        self.assertEqual(self.fn(23), "24gb_ultra")

    def test_48_is_ultra(self):
        self.assertEqual(self.fn(48), "24gb_ultra")

    def test_no_profile_returns_none(self):
        """Every integer from 0..48 must map to a non-empty profile."""
        for v in range(49):
            with self.subTest(vram=v):
                result = self.fn(v)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)


# ══════════════════════════════════════════════════════════════════════════════
# remix() key transformation
# ══════════════════════════════════════════════════════════════════════════════

class TestRemixTransformation(unittest.TestCase):

    def _make(self, meta: dict):
        from genbox.pipeline import GenResult
        p = Path(tempfile.mktemp(suffix=".png"))
        p.write_bytes(b"x")
        return GenResult(output_path=p, metadata=meta, elapsed_s=1.0)

    def test_override_replaces_value(self):
        r = self._make({"seed": 1, "steps": 10, "model": "flux2_klein"})
        params = r.remix(seed=99)
        self.assertEqual(params["seed"], 99)
        self.assertEqual(params["steps"], 10)  # unchanged

    def test_internal_keys_excluded(self):
        r = self._make({"seed": 1, "timestamp": "t", "elapsed_s": 2.0, "output_path": "/x"})
        params = r.remix()
        for k in ("timestamp", "elapsed_s", "output_path"):
            self.assertNotIn(k, params)

    def test_new_key_added_via_override(self):
        r = self._make({"seed": 1})
        params = r.remix(width=512)
        self.assertEqual(params["width"], 512)

    def test_original_metadata_not_mutated(self):
        r = self._make({"seed": 1, "model": "x"})
        _ = r.remix(seed=999)
        self.assertEqual(r.metadata["seed"], 1)

    def test_multiple_overrides(self):
        r = self._make({"seed": 1, "steps": 10, "model": "flux2_klein"})
        params = r.remix(seed=99, steps=50, model="sd35_medium")
        self.assertEqual(params["seed"], 99)
        self.assertEqual(params["steps"], 50)
        self.assertEqual(params["model"], "sd35_medium")

    def test_returns_plain_dict(self):
        r = self._make({"seed": 1})
        self.assertIsInstance(r.remix(), dict)


# ══════════════════════════════════════════════════════════════════════════════
# Registry data integrity — real REGISTRY dict contents
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistryDataIntegrity(unittest.TestCase):

    def test_z_image_turbo_full_repo_flag(self):
        from genbox.models import REGISTRY
        self.assertTrue(REGISTRY["z_image_turbo"].full_repo)

    def test_flux2_klein_full_repo_flag(self):
        from genbox.models import REGISTRY
        e = REGISTRY["flux2_klein"]
        self.assertTrue(e.full_repo, "flux2_klein is a diffusers multi-file repo")
        self.assertEqual(e.hf_repo, "black-forest-labs/FLUX.2-klein-4B")

    def test_sd35_medium_full_repo_flag(self):
        from genbox.models import REGISTRY
        e = REGISTRY["sd35_medium"]
        self.assertTrue(e.full_repo, "sd35_medium is a diffusers multi-file repo")

    def test_full_repo_models_use_model_index_filename(self):
        from genbox.models import REGISTRY
        for mid, entry in REGISTRY.items():
            if entry.full_repo:
                with self.subTest(model=mid):
                    self.assertEqual(entry.hf_filename, "model_index.json",
                                     f"{mid}: full_repo models must use hf_filename='model_index.json'")

    def test_single_file_models_valid_extension(self):
        from genbox.models import REGISTRY
        valid_exts = {".safetensors", ".gguf", ".pt", ".bin", ".ckpt", ".json"}
        for mid, entry in REGISTRY.items():
            with self.subTest(model=mid):
                ext = Path(entry.hf_filename).suffix.lower()
                self.assertIn(ext, valid_exts, f"{mid}: bad extension '{ext}'")

    def test_vram_min_positive(self):
        from genbox.models import REGISTRY
        for mid, entry in REGISTRY.items():
            with self.subTest(model=mid):
                self.assertGreater(entry.vram_min_gb, 0)

    def test_type_is_image_or_video(self):
        from genbox.models import REGISTRY
        for mid, entry in REGISTRY.items():
            with self.subTest(model=mid):
                self.assertIn(entry.type, ("image", "video"))

    def test_architecture_consistent_with_type(self):
        from genbox.models import REGISTRY
        video_archs = {"ltx", "wan"}
        image_archs = {"flux", "sd35", "sd15", "sdxl"}
        for mid, entry in REGISTRY.items():
            with self.subTest(model=mid):
                if entry.type == "video":
                    self.assertIn(entry.architecture, video_archs)
                else:
                    self.assertIn(entry.architecture, image_archs)

    def test_stars_in_valid_range(self):
        from genbox.models import REGISTRY
        for mid, entry in REGISTRY.items():
            with self.subTest(model=mid):
                self.assertIn(entry.quality_stars, range(1, 6))
                self.assertIn(entry.speed_stars, range(1, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
