"""
TDD: genbox/ui.py helper functions
Only tests pure data-transformation helpers — no Streamlit imports.
Run: python -m unittest genbox.test_ui_helpers -v
"""
import json
import unittest
import tempfile
from pathlib import Path


# ── Helpers extracted from ui.py (pure functions, no streamlit) ───────────────

class TestGetPipeTypes(unittest.TestCase):
    """Pipeline type list must NOT contain Multi-Image."""

    def test_no_multi_image(self):
        from genbox.genbox_ui.ui_helpers import get_pipe_types
        types = get_pipe_types()
        self.assertNotIn("Multi-Image", types)

    def test_contains_required_modes(self):
        from genbox.genbox_ui.ui_helpers import get_pipe_types
        types = get_pipe_types()
        for m in ("Text → Image", "Image → Image", "Inpaint", "Outpaint",
                  "Text → Video", "Image → Video"):
            self.assertIn(m, types)

    def test_image_before_video(self):
        from genbox.genbox_ui.ui_helpers import get_pipe_types
        types = get_pipe_types()
        t2i_idx = types.index("Text → Image")
        t2v_idx = types.index("Text → Video")
        self.assertLess(t2i_idx, t2v_idx)


class TestMapPipelineToMode(unittest.TestCase):
    """_map_pipeline_to_mode maps metadata pipeline string → UI pipe_type."""

    def _f(self, s):
        from genbox.genbox_ui.ui_helpers import map_pipeline_to_mode
        return map_pipeline_to_mode(s)

    def test_text_to_image(self):
        self.assertEqual(self._f("flux_text_to_image"), "Text → Image")
        self.assertEqual(self._f("sdl_sdxl_text_to_image"), "Text → Image")

    def test_img2img(self):
        self.assertEqual(self._f("img2img_flux"), "Image → Image")
        self.assertEqual(self._f("img2img_sd15"), "Image → Image")

    def test_inpaint(self):
        self.assertEqual(self._f("inpaint_sdxl"), "Inpaint")

    def test_outpaint(self):
        self.assertEqual(self._f("outpaint_sdxl"), "Outpaint")

    def test_text_to_video(self):
        self.assertEqual(self._f("wan_t2v"), "Text → Video")
        self.assertEqual(self._f("ltx_t2v"), "Text → Video")

    def test_image_to_video(self):
        self.assertEqual(self._f("wan_i2v"), "Image → Video")
        self.assertEqual(self._f("img2video_ltx"), "Image → Video")

    def test_unknown_defaults_t2i(self):
        self.assertEqual(self._f("unknown_pipeline"), "Text → Image")


class TestDetectUploadType(unittest.TestCase):
    """Auto-detect file type from extension for drag+drop zone."""

    def _f(self, name):
        from genbox.genbox_ui.ui_helpers import detect_upload_type
        return detect_upload_type(name)

    def test_gguf(self):
        self.assertEqual(self._f("flux1-dev-Q8_0.gguf"), "gguf")
        self.assertEqual(self._f("model.GGUF"), "gguf")

    def test_safetensors_model(self):
        # Large files or names without "lora" → model
        self.assertEqual(self._f("pony_diffusion_v6.safetensors"), "model")

    def test_safetensors_lora(self):
        self.assertEqual(self._f("my_lora_style.safetensors"), "lora")
        self.assertEqual(self._f("cinematic_lora.safetensors"), "lora")

    def test_safetensors_lora_keyword(self):
        from genbox.genbox_ui.ui_helpers import detect_upload_type
        self.assertEqual(detect_upload_type("flux_lora_v2.safetensors"), "lora")

    def test_unknown_extension_returns_unknown(self):
        self.assertEqual(self._f("weights.bin"), "unknown")
        self.assertEqual(self._f("model.pt"), "unknown")


class TestGuessArchFromFilename(unittest.TestCase):
    """Heuristic arch detection from filename for auto-registration."""

    def _f(self, name):
        from genbox.genbox_ui.ui_helpers import guess_arch_from_filename
        return guess_arch_from_filename(name)

    def test_flux_variants(self):
        self.assertEqual(self._f("flux1-schnell-Q8_0.gguf"), "flux")
        self.assertEqual(self._f("flux2_klein_4b.safetensors"), "flux")

    def test_wan(self):
        self.assertEqual(self._f("wan21_1_3b_bf16.safetensors"), "wan")
        self.assertEqual(self._f("Wan2.1-T2V-14B.safetensors"), "wan")

    def test_ltx(self):
        self.assertEqual(self._f("ltx_video_0.9.5.safetensors"), "ltx")
        self.assertEqual(self._f("LTX-Video-distilled.gguf"), "ltx")

    def test_sdxl(self):
        self.assertEqual(self._f("pony_xl_v6.safetensors"), "sdxl")
        self.assertEqual(self._f("animagine_xl_4.safetensors"), "sdxl")

    def test_sd15(self):
        self.assertEqual(self._f("realistic_vision_v6.safetensors"), "sd15")

    def test_fallback_flux(self):
        # Unknown → default "flux"
        self.assertEqual(self._f("unknown_model.gguf"), "flux")


class TestLoadOutputs(unittest.TestCase):
    """_load_outputs scans the outputs dir for .json sidecars."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.outputs_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_sidecar(self, subdir, name, data):
        d = self.outputs_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(json.dumps(data), encoding="utf-8")
        # Also create a fake png
        (d / name.replace(".json", ".png")).write_bytes(b"\x89PNG")
        return p

    def test_returns_list(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        result = load_outputs(self.outputs_dir)
        self.assertIsInstance(result, list)

    def test_reads_sidecar(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        self._write_sidecar("2026-01-01", "img_0001_flux_seed42.json",
                            {"prompt": "test", "model": "flux2_klein",
                             "pipeline": "flux_text_to_image", "seed": 42})
        result = load_outputs(self.outputs_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prompt"], "test")

    def test_tag_from_subdir_name(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        self._write_sidecar("2026-01-01", "img_0001_flux_seed42.json",
                            {"pipeline": "flux_text_to_image"})
        result = load_outputs(self.outputs_dir)
        self.assertEqual(result[0]["_tag"], "2026-01-01")

    def test_file_path_set(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        self._write_sidecar("2026-01-01", "img_0001_flux_seed42.json",
                            {"pipeline": "flux_text_to_image"})
        result = load_outputs(self.outputs_dir)
        self.assertIn("_file_path", result[0])

    def test_empty_dir_returns_empty(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        result = load_outputs(self.outputs_dir)
        self.assertEqual(result, [])

    def test_malformed_json_skipped(self):
        from genbox.genbox_ui.ui_helpers import load_outputs
        bad = self.outputs_dir / "bad.json"
        bad.write_text("{ not valid json", encoding="utf-8")
        result = load_outputs(self.outputs_dir)
        self.assertEqual(result, [])


class TestBuildRemixData(unittest.TestCase):
    """build_remix_data maps a metadata dict to session_state keys."""

    def test_basic_mapping(self):
        from genbox.genbox_ui.ui_helpers import build_remix_data
        meta = {
            "pipeline": "flux_text_to_image",
            "model": "flux2_klein",
            "prompt": "a cat",
            "seed": 42,
            "steps": 28,
        }
        result = build_remix_data(meta)
        self.assertEqual(result["prompt"], "a cat")
        self.assertEqual(result["sel_model"], "flux2_klein")
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["pipe_type"], "Text → Image")

    def test_video_pipeline_mapped(self):
        from genbox.genbox_ui.ui_helpers import build_remix_data
        meta = {"pipeline": "wan_t2v", "model": "wan_1_3b",
                "prompt": "drone", "seed": 7, "steps": 50}
        result = build_remix_data(meta)
        self.assertEqual(result["pipe_type"], "Text → Video")

    def test_missing_keys_have_defaults(self):
        from genbox.genbox_ui.ui_helpers import build_remix_data
        result = build_remix_data({})
        self.assertIn("pipe_type", result)
        self.assertIn("prompt", result)


class TestValidateOutpaintExpansion(unittest.TestCase):
    """Outpaint requires at least one expansion > 0."""

    def test_valid_expansion(self):
        from genbox.genbox_ui.ui_helpers import validate_outpaint_expansion
        ok, msg = validate_outpaint_expansion(left=128, right=0, top=0, bottom=0)
        self.assertTrue(ok)

    def test_all_zero_invalid(self):
        from genbox.genbox_ui.ui_helpers import validate_outpaint_expansion
        ok, msg = validate_outpaint_expansion(left=0, right=0, top=0, bottom=0)
        self.assertFalse(ok)
        self.assertIn("expansion", msg.lower())

    def test_multiple_sides(self):
        from genbox.genbox_ui.ui_helpers import validate_outpaint_expansion
        ok, _ = validate_outpaint_expansion(left=64, right=64, top=32, bottom=32)
        self.assertTrue(ok)


class TestFormatLoraLabel(unittest.TestCase):
    """Format a LoRA dict as a display label."""

    def test_basic_label(self):
        from genbox.genbox_ui.ui_helpers import format_lora_label
        lo = {"name": "cinematic", "architecture": "flux", "size_mb": 150.0}
        label = format_lora_label(lo)
        self.assertIn("cinematic", label)
        self.assertIn("flux", label)

    def test_trigger_included_when_present(self):
        from genbox.genbox_ui.ui_helpers import format_lora_label
        lo = {"name": "style", "architecture": "sdxl", "size_mb": 80.0,
              "trigger": "cinematic style"}
        label = format_lora_label(lo)
        self.assertIn("cinematic style", label)

    def test_no_trigger_no_crash(self):
        from genbox.genbox_ui.ui_helpers import format_lora_label
        lo = {"name": "style", "architecture": "sdxl", "size_mb": 80.0, "trigger": ""}
        label = format_lora_label(lo)
        self.assertIsInstance(label, str)


class TestGetInstallDefaultsForProfile(unittest.TestCase):
    """get_install_defaults_for_profile returns correct model list."""

    def test_8gb_returns_small_models(self):
        from genbox.genbox_ui.ui_helpers import get_install_defaults_for_profile
        ids = get_install_defaults_for_profile("8gb_low")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        # Should include a small, fast model
        self.assertTrue(any("schnell" in i or "1_3b" in i for i in ids))

    def test_24gb_returns_more_models(self):
        from genbox.genbox_ui.ui_helpers import get_install_defaults_for_profile
        ids_8 = get_install_defaults_for_profile("8gb_low")
        ids_24 = get_install_defaults_for_profile("24gb_ultra")
        self.assertGreater(len(ids_24), len(ids_8))

    def test_unknown_profile_falls_back(self):
        from genbox.genbox_ui.ui_helpers import get_install_defaults_for_profile
        ids = get_install_defaults_for_profile("nonexistent_profile")
        self.assertIsInstance(ids, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
