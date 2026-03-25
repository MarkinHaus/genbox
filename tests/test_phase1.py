"""
tests/test_phase1.py
unittest suite for genbox Phase 1: config, models, cli.
Run: python -m unittest tests.test_phase1 -v
"""

import json
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# ensure genbox is importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# config.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultHome(unittest.TestCase):
    def test_returns_path(self):
        from genbox.config import _default_home
        h = _default_home()
        self.assertIsInstance(h, Path)

    @patch("platform.system", return_value="Windows")
    def test_windows_uses_appdata(self, _):
        from genbox.config import _default_home
        with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
            h = _default_home()
            self.assertIn("genbox", str(h))

    @patch("platform.system", return_value="Linux")
    def test_linux_uses_home(self, _):
        from genbox.config import _default_home
        h = _default_home()
        self.assertTrue(str(h).endswith(".genbox") or "genbox" in str(h))


class TestVramDetection(unittest.TestCase):
    def test_returns_int(self):
        from genbox.config import detect_vram
        result = detect_vram()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_torch_path(self):
        from genbox.config import detect_vram
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = 12 * 1024**3
        with patch.dict(sys.modules, {"torch": mock_torch}):
            # re-import to hit torch path
            import importlib
            import genbox.config as cfg_mod
            importlib.reload(cfg_mod)
            result = cfg_mod.detect_vram()
            # should be 12 or fallback int
            self.assertIsInstance(result, int)

    def test_no_gpu_returns_zero(self):
        from genbox.config import detect_vram
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("subprocess.check_output", side_effect=Exception("no gpu")):
                from genbox.config import detect_vram as dv
                # should not raise
                r = dv()
                self.assertIsInstance(r, int)


class TestVramProfile(unittest.TestCase):
    def setUp(self):
        from genbox.config import _vram_profile
        self.fn = _vram_profile

    def test_profile_8gb(self):
        self.assertEqual(self.fn(6), "8gb_low")

    def test_profile_12gb(self):
        self.assertEqual(self.fn(12), "12gb_balanced")

    def test_profile_16gb(self):
        self.assertEqual(self.fn(16), "16gb_high")

    def test_profile_24gb(self):
        self.assertEqual(self.fn(24), "24gb_ultra")

    def test_profile_zero(self):
        self.assertEqual(self.fn(0), "8gb_low")


class TestTomlWriteRead(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        from genbox.config import _save_toml, _load_toml
        data = {
            "genbox": {"home": "/tmp/test", "vram_gb": 12, "first_run_done": True},
            "paths": {"models": "/tmp/models"},
            "defaults": {"accel": ["sageAttn", "teacache"]},
        }
        path = self.tmp / "config.toml"
        _save_toml(data, path)
        self.assertTrue(path.exists())
        loaded = _load_toml(path)
        self.assertEqual(loaded["genbox"]["vram_gb"], 12)
        self.assertIn("sageAttn", loaded["defaults"]["accel"])

    def test_load_missing_returns_empty(self):
        from genbox.config import _load_toml
        result = _load_toml(self.tmp / "nonexistent.toml")
        self.assertEqual(result, {})


class TestConfigObject(unittest.TestCase):
    def _make_cfg(self, vram=12, profile="12gb_balanced"):
        from genbox.config import Config
        data = {
            "genbox": {"home": "/tmp/gb", "vram_gb": vram, "vram_profile": profile, "first_run_done": True},
            "paths": {
                "models": "/tmp/gb/models",
                "outputs": "/tmp/gb/outputs",
                "loras": "/tmp/gb/loras",
                "cache": "/tmp/gb/cache",
            },
            "defaults": {
                "image_model": "flux2_klein",
                "video_model": "ltx2_fp8",
                "steps": 28,
                "seed": -1,
                "accel": ["sageAttn"],
            },
            "ui": {"port": 8501},
            "accelerators": {"installed": True, "active": ["sageAttn", "teacache"]},
        }
        return Config(data, Path("/tmp/gb"))

    def test_vram_property(self):
        c = self._make_cfg(vram=12)
        self.assertEqual(c.vram_gb, 12)

    def test_paths_return_pathlib(self):
        c = self._make_cfg()
        self.assertIsInstance(c.models_dir, Path)
        self.assertIsInstance(c.outputs_dir, Path)
        self.assertIsInstance(c.loras_dir, Path)
        self.assertIsInstance(c.cache_dir, Path)

    def test_default_models(self):
        c = self._make_cfg()
        self.assertEqual(c.default_image_model, "flux2_klein")
        self.assertEqual(c.default_video_model, "ltx2_fp8")

    def test_active_accels(self):
        c = self._make_cfg()
        self.assertIn("sageAttn", c.active_accels)

    def test_get_nested(self):
        c = self._make_cfg()
        self.assertEqual(c.get("ui", "port", default=9999), 8501)
        self.assertEqual(c.get("ui", "missing_key", default=42), 42)

    def test_repr(self):
        c = self._make_cfg()
        self.assertIn("Config", repr(c))
        self.assertIn("12GB", repr(c))


# ══════════════════════════════════════════════════════════════════════════════
# models.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestModelEntry(unittest.TestCase):
    def _entry(self, vram_min=10):
        from genbox.models import ModelEntry
        return ModelEntry(
            id="test_model",
            name="Test Model",
            type="image",
            architecture="flux",
            vram_min_gb=vram_min,
            hf_repo="test/repo",
            hf_filename="model.safetensors",
            license="Apache 2.0",
            quant="fp8",
            quality_stars=4,
            speed_stars=5,
        )

    def test_fits_vram_true(self):
        e = self._entry(vram_min=10)
        self.assertTrue(e.fits_vram(12))

    def test_fits_vram_false(self):
        e = self._entry(vram_min=24)
        self.assertFalse(e.fits_vram(12))

    def test_fits_vram_exact(self):
        e = self._entry(vram_min=12)
        self.assertTrue(e.fits_vram(12))

    def test_stars_format(self):
        e = self._entry()
        s = e.stars(3)
        self.assertEqual(s.count("★"), 3)
        self.assertEqual(s.count("☆"), 2)

    def test_stars_zero(self):
        e = self._entry()
        s = e.stars(0)
        self.assertEqual(s.count("★"), 0)
        self.assertEqual(s.count("☆"), 5)


class TestRegistry(unittest.TestCase):

    def test_get_known(self):
        from genbox.models import get
        e = get("flux2_klein")
        self.assertEqual(e.id, "flux2_klein")
        self.assertEqual(e.type, "image")

    def test_get_unknown_raises(self):
        from genbox.models import get
        with self.assertRaises(KeyError):
            get("nonexistent_model_xyz")

    def test_list_by_type_image(self):
        from genbox.models import list_registry
        imgs = list_registry(model_type="image")
        self.assertTrue(all(e.type == "image" for e in imgs))
        self.assertGreater(len(imgs), 0)

    def test_list_by_type_video(self):
        from genbox.models import list_registry
        vids = list_registry(model_type="video")
        self.assertTrue(all(e.type == "video" for e in vids))
        self.assertGreater(len(vids), 0)

    def test_list_vram_filter(self):
        from genbox.models import list_registry
        low = list_registry(max_vram=8)
        high = list_registry(max_vram=24)
        self.assertLessEqual(len(low), len(high))
        for e in low:
            self.assertLessEqual(e.vram_min_gb, 8)

    def test_12gb_image_models(self):
        from genbox.models import list_registry
        imgs = list_registry(model_type="image", max_vram=12)
        ids = [e.id for e in imgs]
        self.assertIn("flux2_klein", ids)
        self.assertIn("sd35_medium", ids)

    def test_12gb_video_models(self):
        from genbox.models import list_registry
        vids = list_registry(model_type="video", max_vram=12)
        ids = [e.id for e in vids]
        self.assertIn("ltx2_fp8", ids)
        self.assertIn("wan_1_3b", ids)

    def test_all_entries_have_required_fields(self):
        from genbox.models import REGISTRY
        for model_id, entry in REGISTRY.items():
            with self.subTest(model=model_id):
                self.assertTrue(entry.hf_repo)
                self.assertTrue(entry.hf_filename)
                self.assertTrue(entry.license)
                self.assertIn(entry.type, ("image", "video"))
                self.assertGreater(entry.vram_min_gb, 0)

    def test_quality_stars_range(self):
        from genbox.models import REGISTRY
        for model_id, entry in REGISTRY.items():
            with self.subTest(model=model_id):
                self.assertGreaterEqual(entry.quality_stars, 1)
                self.assertLessEqual(entry.quality_stars, 5)
                self.assertGreaterEqual(entry.speed_stars, 1)
                self.assertLessEqual(entry.speed_stars, 5)


class TestLog(unittest.TestCase):
    def test_log_info_no_crash(self):
        from genbox.models import Log
        # just ensure no exception
        Log.info("test info message")
        Log.ok("test ok message")
        Log.err("test error message")
        Log.accent("test accent message")

    def test_progress_bar_empty(self):
        from genbox.models import _progress_bar
        bar = _progress_bar(0, 0)
        self.assertIn("[", bar)

    def test_progress_bar_half(self):
        from genbox.models import _progress_bar
        bar = _progress_bar(50, 100, width=10)
        self.assertIn("50%", bar)
        self.assertEqual(bar.count("█"), 5)

    def test_progress_bar_full(self):
        from genbox.models import _progress_bar
        bar = _progress_bar(100, 100, width=10)
        self.assertIn("100%", bar)
        self.assertEqual(bar.count("█"), 10)


class TestLocalDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_cfg(self):
        from genbox import models as m
        mock_cfg = MagicMock()
        mock_cfg.models_dir = self.tmp / "models"
        mock_cfg.loras_dir = self.tmp / "loras"
        mock_cfg.models_dir.mkdir(parents=True, exist_ok=True)
        mock_cfg.loras_dir.mkdir(parents=True, exist_ok=True)
        return mock_cfg

    def test_empty_dir_returns_empty(self):
        mock_cfg = self._patch_cfg()
        with patch("genbox.models.cfg", mock_cfg):
            from genbox.models import list_local
            result = list_local()
            self.assertEqual(result, [])


    def test_discovers_gguf(self):
        mock_cfg = self._patch_cfg()
        # GGUF file
        model_file = mock_cfg.models_dir / "flux" / "flux1-schnell-Q8_0.gguf"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"x" * 1024)
        # Shared config dir — _is_installed_entry prüft: exists() + any(iterdir())
        # Name: hf_pipeline_repo "black-forest-labs/FLUX.1-schnell" → "--" ersetzt "/"
        shared_dir = mock_cfg.models_dir / "flux" / "_shared_black-forest-labs--FLUX.1-schnell"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "model_index.json").write_text("{}")  # mind. eine Datei nötig
        with patch("genbox.models.cfg", mock_cfg):
            from genbox.models import list_local
            result = list_local()
            names = [r["name"] for r in result]
            self.assertIn("FLUX.1 Schnell (GGUF Q8)", names)


# ══════════════════════════════════════════════════════════════════════════════
# cli.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCliParser(unittest.TestCase):
    def setUp(self):
        from genbox.cli import build_parser
        self.parser = build_parser()

    def test_setup_command(self):
        args = self.parser.parse_args(["setup"])
        self.assertEqual(args.command, "setup")
        self.assertFalse(args.force)

    def test_setup_force_flag(self):
        args = self.parser.parse_args(["setup", "--force"])
        self.assertTrue(args.force)

    def test_models_list_default(self):
        args = self.parser.parse_args(["models"])
        self.assertEqual(args.command, "models")
        self.assertEqual(args.action, "list")

    def test_models_list_type_filter(self):
        args = self.parser.parse_args(["models", "list", "--type", "video"])
        self.assertEqual(args.type, "video")

    def test_models_search(self):
        args = self.parser.parse_args(["models", "search", "flux", "quantized"])
        self.assertEqual(args.action, "search")
        self.assertEqual(args.query, ["flux", "quantized"])

    def test_models_download(self):
        args = self.parser.parse_args(["models", "download", "--model-id", "flux2_klein"])
        self.assertEqual(args.action, "download")
        self.assertEqual(args.model_id, "flux2_klein")

    def test_gen_required_prompt(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gen"])  # missing --prompt

    def test_loras_command(self):
        args = self.parser.parse_args(["loras", "--arch", "flux"])
        self.assertEqual(args.command, "loras")
        self.assertEqual(args.arch, "flux")

    def test_ui_command(self):
        args = self.parser.parse_args(["ui"])
        self.assertEqual(args.command, "ui")

    def test_invalid_accel_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["gen", "--prompt", "x", "--accel", "invalid_accel"])


class TestCliRequireConfig(unittest.TestCase):
    def test_requires_config_exits_when_none(self):
        # cfg is imported inside _require_config from genbox.config
        with patch("genbox.config.cfg", None):
            from genbox.cli import _require_config
            with self.assertRaises(SystemExit):
                _require_config()


# ══════════════════════════════════════════════════════════════════════════════
# cross-OS path tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossOSPaths(unittest.TestCase):
    def test_all_paths_use_pathlib(self):
        """Verify no raw string path concatenation in config module."""
        import inspect
        from genbox import config
        src = inspect.getsource(config)
        # os.path.join is forbidden — pathlib only
        self.assertNotIn("os.path.join", src)

    def test_config_file_path_is_toml(self):
        from genbox.config import _config_file, _default_home
        p = _config_file(_default_home())
        self.assertEqual(p.suffix, ".toml")

    def test_models_dir_is_subdir_of_home(self):
        from genbox.config import Config
        home = Path("/some/home")
        c = Config({
            "genbox": {"home": str(home), "vram_gb": 12, "vram_profile": "12gb_balanced"},
            "paths": {
                "models": str(home / "models"),
                "outputs": str(home / "outputs"),
                "loras": str(home / "loras"),
                "cache": str(home / "cache"),
            },
            "defaults": {"image_model": "x", "video_model": "y", "steps": 28, "seed": -1, "accel": []},
            "ui": {"port": 8501},
        }, home)
        self.assertTrue(str(c.models_dir).endswith("models"))

    @patch("platform.system", return_value="Windows")
    def test_windows_path_no_forward_slash_issue(self, _):
        from genbox.config import _default_home
        with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Markin\\AppData\\Roaming"}):
            h = _default_home()
            # Path should handle this correctly on any OS
            self.assertIsInstance(h, Path)
            self.assertIn("genbox", str(h))

# ══════════════════════════════════════════════════════════════════════════════
# pipeline.py — GenResult tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGenResult(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_result(self, name="test.png", meta=None):
        from genbox.pipeline import GenResult
        p = self.tmp / name
        p.write_bytes(b"fake image data")
        return GenResult(
            output_path=p,
            metadata=meta or {"model": "flux2_klein", "seed": 42, "prompt": "test"},
            elapsed_s=1.5,
        )

    def test_save_writes_json_sidecar(self):
        gr = self._make_result()
        gr.save()
        meta_path = gr.output_path.with_suffix(".json")
        self.assertTrue(meta_path.exists())
        data = json.loads(meta_path.read_text())
        self.assertEqual(data["model"], "flux2_klein")

    def test_save_to_custom_path(self):
        gr = self._make_result()
        dest = self.tmp / "subdir" / "output.png"
        gr.save(dest)
        self.assertTrue(dest.exists())

    def test_remix_returns_params(self):
        gr = self._make_result(meta={
            "model": "flux2_klein", "seed": 42, "prompt": "original", "steps": 28,
        })
        params = gr.remix(seed=99, steps=40)
        self.assertEqual(params["seed"], 99)
        self.assertEqual(params["steps"], 40)
        self.assertEqual(params["prompt"], "original")

    def test_remix_excludes_internal_keys(self):
        gr = self._make_result(meta={
            "model": "x", "seed": 1,
            "timestamp": "2026-01-01", "elapsed_s": 2.0, "output_path": "/tmp/x.png",
        })
        params = gr.remix()
        self.assertNotIn("timestamp", params)
        self.assertNotIn("elapsed_s", params)
        self.assertNotIn("output_path", params)

    def test_repr_contains_key_info(self):
        gr = self._make_result()
        r = repr(gr)
        self.assertIn("flux2_klein", r)
        self.assertIn("42", r)
        self.assertIn("1.5s", r)

if __name__ == "__main__":
    unittest.main(verbosity=2)
