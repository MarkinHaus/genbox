"""
tests/test_cli.py
CLI argument parsing, command routing, and output-format tests.
All commands are tested without running actual ML — only parser
validation, dispatch logic, and output formatting.

Slow/integration tests are gated behind GENBOX_SLOW_TESTS=1.

Run fast only:
    python -m unittest tests.test_cli -v

Run including slow tests:
    GENBOX_SLOW_TESTS=1 python -m unittest tests.test_cli -v

Or via CLI:
    genbox test tests.test_cli
    genbox test tests.test_cli --slow
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ── Slow-test gate ────────────────────────────────────────────────────────────

def slow(test_fn):
    """
    Decorator: skip unless GENBOX_SLOW_TESTS=1 or --slow is in sys.argv.
    Usage:
        @slow
        def test_real_download(self): ...
    """
    skip_msg = (
        "slow test — run with GENBOX_SLOW_TESTS=1 or genbox test --slow"
    )
    condition = (
        os.environ.get("GENBOX_SLOW_TESTS", "").strip() == "1"
        or "--slow" in sys.argv
    )
    return test_fn if condition else unittest.skip(skip_msg)(test_fn)


# ── Parser helpers ────────────────────────────────────────────────────────────

def _parser():
    from genbox.cli import build_parser
    return build_parser()


def _parse(*argv):
    """Parse argv and return Namespace."""
    return _parser().parse_args(list(argv))


# ══════════════════════════════════════════════════════════════════════════════
# Parser: setup
# ══════════════════════════════════════════════════════════════════════════════

class TestParserSetup(unittest.TestCase):
    def test_setup_no_force(self):
        args = _parse("setup")
        self.assertEqual(args.command, "setup")
        self.assertFalse(args.force)

    def test_setup_force(self):
        args = _parse("setup", "--force")
        self.assertTrue(args.force)


# ══════════════════════════════════════════════════════════════════════════════
# Parser: gen — all modes
# ══════════════════════════════════════════════════════════════════════════════

class TestParserGenModes(unittest.TestCase):
    def _gen(self, *extra):
        return _parse("gen", "-p", "test prompt", *extra)

    def test_default_mode_t2i(self):
        args = self._gen()
        self.assertEqual(args.mode, "t2i")
        self.assertEqual(args.prompt, "test prompt")

    def test_explicit_t2i(self):
        args = self._gen("--mode", "t2i")
        self.assertEqual(args.mode, "t2i")

    def test_i2i_mode(self):
        args = self._gen("--mode", "i2i", "--input", "img.png")
        self.assertEqual(args.mode, "i2i")
        self.assertEqual(args.input, "img.png")

    def test_inpaint_mode(self):
        args = self._gen("--mode", "inpaint", "--input", "img.png", "--mask", "mask.png")
        self.assertEqual(args.mode, "inpaint")
        self.assertEqual(args.mask, "mask.png")

    def test_outpaint_mode(self):
        args = self._gen("--mode", "outpaint", "--input", "img.png", "--expand-left", "128")
        self.assertEqual(args.mode, "outpaint")
        self.assertEqual(args.expand_left, 128)

    def test_outpaint_all_sides(self):
        args = self._gen("--mode", "outpaint", "--input", "img.png",
                         "--expand-left", "64", "--expand-right", "64",
                         "--expand-top", "32", "--expand-bottom", "32")
        self.assertEqual(args.expand_left, 64)
        self.assertEqual(args.expand_right, 64)
        self.assertEqual(args.expand_top, 32)
        self.assertEqual(args.expand_bottom, 32)

    def test_t2v_mode(self):
        args = self._gen("--mode", "t2v", "--frames", "81", "--fps", "16")
        self.assertEqual(args.mode, "t2v")
        self.assertEqual(args.frames, 81)
        self.assertEqual(args.fps, 16)

    def test_i2v_mode(self):
        args = self._gen("--mode", "i2v", "--input", "start.png")
        self.assertEqual(args.mode, "i2v")
        self.assertEqual(args.input, "start.png")

    def test_i2v_end_frame(self):
        args = self._gen("--mode", "i2v", "--input", "s.png", "--end-frame", "e.png")
        self.assertEqual(args.end_frame, "e.png")

    def test_invalid_mode_raises(self):
        with self.assertRaises(SystemExit):
            _parse("gen", "-p", "test", "--mode", "invalid_mode")


class TestParserGenParams(unittest.TestCase):
    def test_steps_default(self):
        args = _parse("gen", "-p", "p")
        self.assertEqual(args.steps, 0)  # 0 = model default

    def test_steps_explicit(self):
        args = _parse("gen", "-p", "p", "--steps", "50")
        self.assertEqual(args.steps, 50)

    def test_guidance(self):
        args = _parse("gen", "-p", "p", "--guidance", "7.5")
        self.assertAlmostEqual(args.guidance, 7.5)

    def test_seed(self):
        args = _parse("gen", "-p", "p", "--seed", "42")
        self.assertEqual(args.seed, 42)

    def test_seed_default_random(self):
        args = _parse("gen", "-p", "p")
        self.assertEqual(args.seed, -1)

    def test_strength(self):
        args = _parse("gen", "-p", "p", "--strength", "0.8")
        self.assertAlmostEqual(args.strength, 0.8)

    def test_blur_radius(self):
        args = _parse("gen", "-p", "p", "--blur-radius", "5.0")
        self.assertAlmostEqual(args.blur_radius, 5.0)

    def test_dilate(self):
        args = _parse("gen", "-p", "p", "--dilate", "8")
        self.assertEqual(args.dilate, 8)

    def test_mask_mode_default(self):
        args = _parse("gen", "-p", "p")
        self.assertEqual(args.mask_mode, "white_inpaint")

    def test_mask_mode_black(self):
        args = _parse("gen", "-p", "p", "--mask-mode", "black_inpaint")
        self.assertEqual(args.mask_mode, "black_inpaint")

    def test_feather_radius(self):
        args = _parse("gen", "-p", "p", "--feather-radius", "20.0")
        self.assertAlmostEqual(args.feather_radius, 20.0)

    def test_negative_prompt(self):
        args = _parse("gen", "-p", "p", "--negative-prompt", "blurry")
        self.assertEqual(args.negative_prompt, "blurry")

    def test_multiple_loras(self):
        args = _parse("gen", "-p", "p", "--lora", "a.safetensors", "--lora", "b.safetensors")
        self.assertEqual(args.lora, ["a.safetensors", "b.safetensors"])

    def test_accel_choices(self):
        args = _parse("gen", "-p", "p", "--accel", "sageAttn", "teacache")
        self.assertIn("sageAttn", args.accel)
        self.assertIn("teacache", args.accel)

    def test_output_shortflag(self):
        args = _parse("gen", "-p", "p", "-o", "out.png")
        self.assertEqual(args.output, "out.png")

    def test_model_shortflag(self):
        args = _parse("gen", "-p", "p", "-m", "flux2_klein")
        self.assertEqual(args.model, "flux2_klein")


# ══════════════════════════════════════════════════════════════════════════════
# Parser: models
# ══════════════════════════════════════════════════════════════════════════════

class TestParserModels(unittest.TestCase):
    def test_default_action_list(self):
        args = _parse("models")
        self.assertEqual(args.action, "list")

    def test_list_action(self):
        args = _parse("models", "list")
        self.assertEqual(args.action, "list")

    def test_local_action(self):
        args = _parse("models", "local")
        self.assertEqual(args.action, "local")

    def test_search_action(self):
        args = _parse("models", "search", "flux", "quantized")
        self.assertEqual(args.action, "search")
        self.assertEqual(args.query, ["flux", "quantized"])

    def test_download_action_with_query(self):
        args = _parse("models", "download", "flux2_klein")
        self.assertEqual(args.action, "download")
        self.assertEqual(args.query, ["flux2_klein"])

    def test_download_with_model_id_flag(self):
        args = _parse("models", "download", "--model-id", "flux2_klein")
        self.assertEqual(args.model_id, "flux2_klein")

    def test_install_defaults_action(self):
        args = _parse("models", "install-defaults")
        self.assertEqual(args.action, "install-defaults")

    def test_install_defaults_profile(self):
        args = _parse("models", "install-defaults", "--profile", "12gb_balanced")
        self.assertEqual(args.profile, "12gb_balanced")

    def test_install_defaults_dry_run(self):
        args = _parse("models", "install-defaults", "--dry-run")
        self.assertTrue(args.dry_run)

    def test_uninstall_action(self):
        args = _parse("models", "uninstall", "flux2_klein")
        self.assertEqual(args.action, "uninstall")
        self.assertEqual(args.query, ["flux2_klein"])

    def test_all_flag(self):
        args = _parse("models", "list", "--all")
        self.assertTrue(args.all)

    def test_type_filter(self):
        args = _parse("models", "list", "--type", "video")
        self.assertEqual(args.type, "video")

    def test_force_flag(self):
        args = _parse("models", "download", "m", "--force")
        self.assertTrue(args.force)


# ══════════════════════════════════════════════════════════════════════════════
# Parser: loras
# ══════════════════════════════════════════════════════════════════════════════

class TestParserLoras(unittest.TestCase):
    def test_default_action_list(self):
        args = _parse("loras")
        self.assertEqual(args.action, "list")

    def test_tag_action(self):
        args = _parse("loras", "tag", "style.safetensors", "--arch", "flux",
                      "--trigger", "cinematic", "--desc", "My style LoRA")
        self.assertEqual(args.action, "tag")
        self.assertEqual(args.query, ["style.safetensors"])
        self.assertEqual(args.arch, "flux")
        self.assertEqual(args.trigger, "cinematic")
        self.assertEqual(args.desc, "My style LoRA")

    def test_arch_filter(self):
        args = _parse("loras", "list", "--arch", "sdxl")
        self.assertEqual(args.arch, "sdxl")

    def test_preview_flag(self):
        args = _parse("loras", "tag", "f.safetensors", "--arch", "flux",
                      "--preview", "https://example.com/img.jpg")
        self.assertEqual(args.preview, "https://example.com/img.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Parser: other commands
# ══════════════════════════════════════════════════════════════════════════════

class TestParserMisc(unittest.TestCase):
    def test_info_command(self):
        args = _parse("info")
        self.assertEqual(args.command, "info")

    def test_ui_default_port(self):
        args = _parse("ui")
        self.assertIsNone(args.port)

    def test_ui_port(self):
        args = _parse("ui", "--port", "9000")
        self.assertEqual(args.port, 9000)

    def test_run_script(self):
        args = _parse("run", "myscript.py")
        self.assertEqual(args.script, "myscript.py")

    def test_test_verbose(self):
        args = _parse("test", "--verbose")
        self.assertTrue(args.verbose)

    def test_test_specific_module(self):
        args = _parse("test", "tests.test_cli", "tests.test_ui_helpers")
        self.assertEqual(args.tests, ["tests.test_cli", "tests.test_ui_helpers"])

    def test_version_exits(self):
        with self.assertRaises(SystemExit) as cm:
            _parse("--version")
        self.assertEqual(cm.exception.code, 0)

    def test_no_command_exits_cleanly(self):
        """No command → shows help, does not crash."""
        with self.assertRaises(SystemExit) as cm:
            _parse("--help")
        self.assertEqual(cm.exception.code, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Command dispatch & logic (no real ML / filesystem writes)
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdSetupDispatch(unittest.TestCase):
    def test_already_configured_no_force(self):
        """Without --force, prints info and returns without calling run_setup."""
        fake_cfg = MagicMock()
        fake_cfg.home = Path("/fake/home")
        fake_cfg.vram_gb = 12
        fake_cfg.vram_profile = "12gb_balanced"

        from genbox.cli import cmd_setup
        args = SimpleNamespace(force=False)
        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.config.cfg", fake_cfg), \
             patch("genbox.config.run_setup") as mock_setup:
            # Capture stdout
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                try:
                    cmd_setup(args)
                except SystemExit:
                    pass
        mock_setup.assert_not_called()

    def test_force_calls_run_setup(self):
        from genbox.cli import cmd_setup
        args = SimpleNamespace(force=True)
        with patch("genbox.config.cfg", None), \
             patch("genbox.config.run_setup") as mock_setup:
            mock_setup.return_value = MagicMock()
            try:
                cmd_setup(args)
            except Exception:
                pass
        mock_setup.assert_called_once()


class TestCmdModelsDispatch(unittest.TestCase):
    def _fake_cfg(self):
        c = MagicMock()
        c.vram_gb = 12
        c.vram_profile = "12gb_balanced"
        c.models_dir = Path(tempfile.mkdtemp())
        return c

    def test_list_action_runs(self):
        from genbox.cli import cmd_models
        args = SimpleNamespace(action="list", type=None, all=False, query=[], model_id=None)
        fake_cfg = self._fake_cfg()
        fake_entry = MagicMock()
        fake_entry.id = "flux2_klein"
        fake_entry.name = "FLUX.2 Klein 4B"
        fake_entry.vram_min_gb = 10
        fake_entry.quant = "fp8"
        fake_entry.quality_stars = 5
        fake_entry.speed_stars = 4
        fake_entry.license = "Apache 2.0"
        fake_entry.fits_vram.return_value = True

        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.models.list_registry", return_value=[fake_entry]), \
             patch("genbox.models._is_installed_entry", return_value=False):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_models(args)
            output = buf.getvalue()
        self.assertIn("flux2_klein", output)

    def test_local_action_empty(self):
        from genbox.cli import cmd_models
        args = SimpleNamespace(action="local", type=None, query=[], model_id=None)
        with patch("genbox.models.list_local", return_value=[]):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_models(args)
            self.assertIn("None", buf.getvalue())

    def test_uninstall_missing_model_exits(self):
        from genbox.cli import cmd_models
        args = SimpleNamespace(action="uninstall", query=[], model_id=None)
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            with self.assertRaises(SystemExit):
                cmd_models(args)

    def test_download_unknown_model_exits(self):
        from genbox.cli import cmd_models
        fake_cfg = self._fake_cfg()
        args = SimpleNamespace(action="download", query=["nonexistent_model"],
                               model_id=None, force=False)
        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.models.get", side_effect=KeyError("nonexistent_model")):
            with self.assertRaises(SystemExit):
                cmd_models(args)

    def test_install_defaults_dry_run_no_download(self):
        from genbox.cli import cmd_models
        fake_cfg = self._fake_cfg()
        args = SimpleNamespace(action="install-defaults", profile="8gb_low",
                               dry_run=True, query=[], model_id=None)

        fake_entry = MagicMock()
        fake_entry.id = "flux1_schnell_q4"
        fake_entry.name = "FLUX.1 Schnell Q4"
        fake_entry.vram_min_gb = 6
        fake_entry.hf_repo = "city96/FLUX.1-schnell-gguf"
        fake_entry.quant = "gguf-q4"

        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.models.get_default_models", return_value=["flux1_schnell_q4"]), \
             patch("genbox.models.REGISTRY", {"flux1_schnell_q4": fake_entry}), \
             patch("genbox.models._is_installed_entry", return_value=False), \
             patch("genbox.models.install_defaults") as mock_install:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_models(args)
        # dry_run → install not called
        mock_install.assert_not_called()


class TestCmdGenDispatch(unittest.TestCase):
    def _fake_cfg(self):
        c = MagicMock()
        c.active_accels = []
        c.home = Path(tempfile.mkdtemp())
        return c

    @slow
    def test_empty_prompt_exits(self):
        from genbox.cli import cmd_gen
        args = SimpleNamespace(
            mode="t2i", prompt="", model=None, input=None, mask=None,
            end_frame=None, negative_prompt="", width=0, height=0,
            steps=0, guidance=0.0, seed=-1, strength=0.75,
            blur_radius=0, dilate=0, mask_mode="white_inpaint",
            expand_left=0, expand_right=0, expand_top=0, expand_bottom=0,
            feather_radius=16.0, frames=0, fps=0,
            lora=[], accel=None, output=None, verbose=False,
        )
        fake_cfg = self._fake_cfg()
        with patch("genbox.cli._require_config", return_value=fake_cfg):
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                with self.assertRaises(SystemExit):
                    cmd_gen(args)

    @slow
    def test_outpaint_no_expansion_exits(self):
        from genbox.cli import cmd_gen
        args = SimpleNamespace(
            mode="outpaint", prompt="test", model=None, input="img.png",
            mask=None, end_frame=None, negative_prompt="",
            width=0, height=0, steps=0, guidance=0.0, seed=-1,
            strength=0.99, blur_radius=0, dilate=0, mask_mode="white_inpaint",
            expand_left=0, expand_right=0, expand_top=0, expand_bottom=0,
            feather_radius=16.0, frames=0, fps=0,
            lora=[], accel=None, output=None, verbose=False,
        )
        fake_cfg = self._fake_cfg()
        mock_pipeline = MagicMock()
        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.cli._require_pipeline", return_value=mock_pipeline):
            with self.assertRaises(SystemExit):
                cmd_gen(args)
        mock_pipeline.outpaint.assert_not_called()

    @slow
    def test_gen_t2i_calls_text_to_image(self):
        from genbox.cli import cmd_gen
        args = SimpleNamespace(
            mode="t2i", prompt="a cat", model="flux2_klein", input=None,
            mask=None, end_frame=None, negative_prompt="",
            width=1024, height=1024, steps=28, guidance=3.5, seed=42,
            strength=0.75, blur_radius=0, dilate=0, mask_mode="white_inpaint",
            expand_left=0, expand_right=0, expand_top=0, expand_bottom=0,
            feather_radius=16.0, frames=0, fps=0,
            lora=[], accel=[], output=None, verbose=False,
        )
        fake_cfg = self._fake_cfg()
        mock_pipeline = MagicMock()
        fake_result = MagicMock()
        fake_result.elapsed_s = 1.5
        fake_result.metadata = {"seed": 42}
        fake_result.output_path = Path("/tmp/x.png")
        mock_pipeline.text_to_image.return_value = fake_result

        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.cli._require_pipeline", return_value=mock_pipeline):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_gen(args)

        mock_pipeline.text_to_image.assert_called_once()
        call_kwargs = mock_pipeline.text_to_image.call_args[1]
        self.assertEqual(call_kwargs["prompt"], "a cat")
        self.assertEqual(call_kwargs["model"], "flux2_klein")
        self.assertEqual(call_kwargs["seed"], 42)


class TestCmdLorasDispatch(unittest.TestCase):
    def test_list_no_loras(self):
        from genbox.cli import cmd_loras
        args = SimpleNamespace(action="list", query=[], arch=None,
                               trigger="", desc="", preview="")
        fake_cfg = MagicMock()
        fake_cfg.loras_dir = Path("/tmp/loras")
        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.models.list_loras", return_value=[]):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_loras(args)
        self.assertIn("No LoRAs", buf.getvalue())

    def test_list_with_loras(self):
        from genbox.cli import cmd_loras
        args = SimpleNamespace(action="list", query=[], arch=None,
                               trigger="", desc="", preview="")
        fake_cfg = MagicMock()
        fake_cfg.loras_dir = Path("/tmp/loras")
        fake_lora = {"name": "cinematic", "architecture": "flux",
                     "size_mb": 150.0, "trigger": "cinematic style",
                     "description": "", "preview_url": ""}
        with patch("genbox.cli._require_config", return_value=fake_cfg), \
             patch("genbox.models.list_loras", return_value=[fake_lora]):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_loras(args)
        self.assertIn("cinematic", buf.getvalue())

    def test_tag_missing_path_exits(self):
        from genbox.cli import cmd_loras
        args = SimpleNamespace(action="tag", query=[], arch="flux",
                               trigger="", desc="", preview="")
        fake_cfg = MagicMock()
        with patch("genbox.cli._require_config", return_value=fake_cfg):
            with self.assertRaises(SystemExit):
                cmd_loras(args)

    def test_tag_missing_arch_exits(self):
        from genbox.cli import cmd_loras
        args = SimpleNamespace(action="tag", query=["f.safetensors"],
                               arch=None, trigger="", desc="", preview="")
        fake_cfg = MagicMock()
        with patch("genbox.cli._require_config", return_value=fake_cfg):
            with self.assertRaises(SystemExit):
                cmd_loras(args)


# ══════════════════════════════════════════════════════════════════════════════
# pipeline.py — is_installed guard & require_installed
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineIsInstalled(unittest.TestCase):
    def test_is_installed_false_for_unknown_model(self):
        from genbox.pipeline import is_installed
        self.assertFalse(is_installed("definitely_not_a_real_model_id_12345"))

    def test_is_installed_returns_bool(self):
        from genbox.pipeline import is_installed
        result = is_installed("flux2_klein")
        self.assertIsInstance(result, bool)

    def test_require_installed_raises_when_missing(self):
        from genbox.pipeline import require_installed
        with self.assertRaises(RuntimeError) as cm:
            require_installed("definitely_not_installed_xyz")
        self.assertIn("not installed", str(cm.exception).lower())

    def test_require_installed_msg_contains_model_id(self):
        from genbox.pipeline import require_installed
        try:
            require_installed("my_missing_model")
        except RuntimeError as e:
            self.assertIn("my_missing_model", str(e))

    def test_require_installed_passes_when_installed(self):
        from genbox.pipeline import require_installed
        with patch("genbox.pipeline.is_installed", return_value=True):
            # Must not raise
            require_installed("any_model_id")

    def test_text_to_image_raises_if_not_installed(self):
        """text_to_image should check installation before loading model."""
        from genbox.pipeline import text_to_image
        with patch("genbox.pipeline._resolve_model") as mock_resolve, \
             patch("genbox.pipeline.is_installed", return_value=False):
            entry = MagicMock()
            entry.architecture = "flux"
            entry.id = "missing_flux"
            mock_resolve.return_value = ("missing_flux", entry)
            with self.assertRaises(RuntimeError):
                text_to_image("test prompt", model="missing_flux")

    def test_text_to_video_raises_if_not_installed(self):
        from genbox.pipeline import text_to_video
        with patch("genbox.pipeline._resolve_model") as mock_resolve, \
             patch("genbox.pipeline.is_installed", return_value=False):
            entry = MagicMock()
            entry.architecture = "wan"
            entry.id = "missing_wan"
            mock_resolve.return_value = ("missing_wan", entry)
            with self.assertRaises(RuntimeError):
                text_to_video("test prompt", model="missing_wan")


# ══════════════════════════════════════════════════════════════════════════════
# SLOW tests — only run with GENBOX_SLOW_TESTS=1
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdSlowIntegration(unittest.TestCase):
    """
    Integration tests that require a real genbox installation.
    Gated behind GENBOX_SLOW_TESTS=1 or --slow flag.
    """

    @slow
    def test_info_runs_configured(self):
        """genbox info completes without error when properly configured."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "genbox.cli", "info"],
            capture_output=True, text=True, timeout=30,
        )
        # Info should succeed (exit 0) when configured
        self.assertEqual(result.returncode, 0)

    @slow
    def test_models_list_runs(self):
        """genbox models list returns at least one entry."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "genbox.cli", "models", "list", "--all"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("flux", result.stdout.lower())

    @slow
    def test_gen_t2i_requires_installed_model(self):
        """
        genbox gen -p 'test' --model flux2_klein fails gracefully when
        the model is not installed (raises RuntimeError, exits 1).
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "genbox.cli", "gen",
             "-p", "test generation",
             "--model", "flux2_klein",
             "--steps", "1"],
            capture_output=True, text=True, timeout=120,
        )
        # Either succeeds (model installed) or exits 1 with useful message
        if result.returncode != 0:
            self.assertIn(
                ("not installed", "not found", "error"),
                result.stderr.lower() + result.stdout.lower(),
            )

    @slow
    def test_loras_list_runs(self):
        """genbox loras list completes without error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "genbox.cli", "loras"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    # Support --slow flag when running directly
    if "--slow" in sys.argv:
        sys.argv.remove("--slow")
        os.environ["GENBOX_SLOW_TESTS"] = "1"
    unittest.main(verbosity=2)