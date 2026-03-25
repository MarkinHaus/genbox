"""
TDD: Section 2 — Image model utilities
        Section 3 — Video model utilities
Run: python -m unittest genbox.test_utils_s2s3 -v
"""
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _make_entry(**kw):
    defaults = dict(
        id="flux1_schnell_q8",
        name="FLUX.1 Schnell Q8",
        type="image",
        architecture="flux",
        quant="gguf-q8",
        hf_filename="flux1-schnell-Q8_0.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-schnell",
        full_repo=False,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIsGguf(unittest.TestCase):
    def test_gguf_q8(self):
        from genbox.utils.utils import is_gguf
        self.assertTrue(is_gguf("gguf-q8"))

    def test_gguf_q4(self):
        from genbox.utils.utils import is_gguf
        self.assertTrue(is_gguf("gguf-q4"))

    def test_fp16_not_gguf(self):
        from genbox.utils.utils import is_gguf
        self.assertFalse(is_gguf("fp16"))

    def test_bf16_not_gguf(self):
        from genbox.utils.utils import is_gguf
        self.assertFalse(is_gguf("bf16"))


class TestIsFlux2(unittest.TestCase):
    def test_flux2_klein(self):
        from genbox.utils.utils import is_flux2
        self.assertTrue(is_flux2("flux2_klein"))

    def test_flux1_schnell(self):
        from genbox.utils.utils import is_flux2
        self.assertFalse(is_flux2("flux1_schnell_q8"))

    def test_case_insensitive(self):
        from genbox.utils.utils import is_flux2
        self.assertTrue(is_flux2("FLUX.2 Klein 4B"))


class TestIsPonyVariant(unittest.TestCase):
    def test_pony_xl_id(self):
        from genbox.utils.utils import is_pony_variant
        self.assertTrue(is_pony_variant("pony_xl"))

    def test_custom_pony_name(self):
        from genbox.utils.utils import is_pony_variant
        self.assertTrue(is_pony_variant("pony_diffusion_v6"))

    def test_animagine_not_pony(self):
        from genbox.utils.utils import is_pony_variant
        self.assertFalse(is_pony_variant("animagine_xl"))


class TestGetRecommendedSampler(unittest.TestCase):
    def test_flux_returns_flowmatch(self):
        from genbox.utils.utils import get_recommended_sampler
        self.assertEqual(get_recommended_sampler("flux"), "FlowMatchEuler")

    def test_wan_returns_unipc(self):
        from genbox.utils.utils import get_recommended_sampler
        self.assertIn("UniPC", get_recommended_sampler("wan"))

    def test_unknown_arch_returns_default(self):
        from genbox.utils.utils import get_recommended_sampler
        self.assertEqual(get_recommended_sampler("unknown_arch"), "default")


class TestDetectFluxVariantFromPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_flux1_from_model_index(self):
        from genbox.utils.utils import detect_flux_variant_from_path
        mi = self.base / "model_index.json"
        mi.write_text(json.dumps({"_class_name": "FluxPipeline"}))
        pipe_cls, transformer_cls = detect_flux_variant_from_path(self.base)
        self.assertEqual(pipe_cls, "FluxPipeline")
        self.assertEqual(transformer_cls, "FluxTransformer2DModel")

    def test_flux2_from_model_index(self):
        from genbox.utils.utils import detect_flux_variant_from_path
        mi = self.base / "model_index.json"
        mi.write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}))
        pipe_cls, transformer_cls = detect_flux_variant_from_path(self.base)
        self.assertIn("Flux2", pipe_cls)
        self.assertIn("Flux2", transformer_cls)

    def test_gguf_from_transformer_config(self):
        from genbox.utils.utils import detect_flux_variant_from_path
        tc = self.base / "transformer" / "config.json"
        tc.parent.mkdir()
        tc.write_text(json.dumps({"_class_name": "FluxTransformer2DModel"}))
        pipe_cls, transformer_cls = detect_flux_variant_from_path(self.base)
        self.assertEqual(pipe_cls, "FluxPipeline")

    def test_fallback_when_no_files(self):
        from genbox.utils.utils import detect_flux_variant_from_path
        pipe_cls, transformer_cls = detect_flux_variant_from_path(self.base)
        self.assertEqual(pipe_cls, "FluxPipeline")
        self.assertEqual(transformer_cls, "FluxTransformer2DModel")


class TestGetImageModelLocalPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_repo_found(self):
        from genbox.utils.utils import get_image_model_local_path
        entry = _make_entry(architecture="flux", full_repo=True, id="flux2_klein", quant="fp8")
        repo_dir = self.models_dir / "flux" / "flux2_klein"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model_index.json").write_text("{}")
        path = get_image_model_local_path(entry, self.models_dir)
        self.assertEqual(path, repo_dir)

    def test_full_repo_missing_raises(self):
        from genbox.utils.utils import get_image_model_local_path
        entry = _make_entry(architecture="flux", full_repo=True, id="flux2_klein", quant="fp8")
        with self.assertRaises(FileNotFoundError):
            get_image_model_local_path(entry, self.models_dir)

    def test_gguf_file_found(self):
        from genbox.utils.utils import get_image_model_local_path
        entry = _make_entry()
        gguf = self.models_dir / "flux" / "flux1-schnell-Q8_0.gguf"
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"\x00" * 8)
        path = get_image_model_local_path(entry, self.models_dir)
        self.assertEqual(path, gguf)

    def test_gguf_missing_raises(self):
        from genbox.utils.utils import get_image_model_local_path
        entry = _make_entry()
        with self.assertRaises(FileNotFoundError):
            get_image_model_local_path(entry, self.models_dir)


class TestGetGgufSharedConfigDir(unittest.TestCase):
    def test_returns_expected_path(self):
        from genbox.utils.utils import get_gguf_shared_config_dir
        entry = _make_entry()
        result = get_gguf_shared_config_dir(entry, Path("/models"))
        self.assertEqual(
            result,
            Path("/models/flux/_shared_black-forest-labs--FLUX.1-schnell")
        )

    def test_no_pipeline_repo_returns_none(self):
        from genbox.utils.utils import get_gguf_shared_config_dir
        entry = _make_entry(hf_pipeline_repo="")
        result = get_gguf_shared_config_dir(entry, Path("/models"))
        self.assertIsNone(result)


class TestListImageModelsLocal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_full_repo(self, arch, name):
        d = self.models_dir / arch / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_finds_full_repo(self):
        from genbox.utils.utils import list_image_models_local
        self._make_full_repo("flux", "flux2_klein")
        results = list_image_models_local(self.models_dir, architecture="flux")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "flux2_klein")

    def test_finds_safetensors(self):
        from genbox.utils.utils import list_image_models_local
        f = self.models_dir / "sdxl" / "my_model.safetensors"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"\x00" * 8)
        results = list_image_models_local(self.models_dir, architecture="sdxl")
        self.assertEqual(len(results), 1)

    def test_finds_gguf(self):
        from genbox.utils.utils import list_image_models_local
        f = self.models_dir / "flux" / "model_q8.gguf"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"\x00" * 8)
        results = list_image_models_local(self.models_dir, architecture="flux")
        self.assertEqual(results[0]["quant"], "gguf-q8")

    def test_variant_filter_flux2(self):
        from genbox.utils.utils import list_image_models_local
        self._make_full_repo("flux", "flux2_klein")
        self._make_full_repo("flux", "flux1_dev")
        results = list_image_models_local(self.models_dir, architecture="flux", variant="flux2")
        self.assertEqual(len(results), 1)
        self.assertIn("flux2", results[0]["name"])

    def test_pony_variant_detection(self):
        from genbox.utils.utils import list_image_models_local
        self._make_full_repo("sdxl", "pony_diffusion_v6")
        results = list_image_models_local(self.models_dir, architecture="sdxl", variant="pony")
        self.assertEqual(len(results), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapFrames(unittest.TestCase):
    def test_ltx_already_valid(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(97, "ltx"), 97)   # 8*12+1

    def test_ltx_snaps_up(self):
        from genbox.utils.utils import snap_frames
        # 95 → 97 (next 8n+1)
        self.assertEqual(snap_frames(95, "ltx"), 97)

    def test_ltx_minimum(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(1, "ltx"), 9)

    def test_wan_already_valid(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(81, "wan"), 81)   # 4*20+1

    def test_wan_snaps_up(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(80, "wan"), 81)

    def test_wan_minimum(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(1, "wan"), 5)

    def test_other_arch_unchanged(self):
        from genbox.utils.utils import snap_frames
        self.assertEqual(snap_frames(30, "other"), 30)


class TestGetWanFlowShift(unittest.TestCase):
    def test_720p_high_flow_shift(self):
        from genbox.utils.utils import get_wan_flow_shift
        self.assertAlmostEqual(get_wan_flow_shift(720), 5.0)

    def test_1080p_high_flow_shift(self):
        from genbox.utils.utils import get_wan_flow_shift
        self.assertAlmostEqual(get_wan_flow_shift(1080), 5.0)

    def test_480p_low_flow_shift(self):
        from genbox.utils.utils import get_wan_flow_shift
        self.assertAlmostEqual(get_wan_flow_shift(480), 3.0)


class TestGetVideoModelLocalPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_video_repo(self, arch, name):
        d = self.models_dir / arch / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_ltx_full_repo_found(self):
        from genbox.utils.utils import get_video_model_local_path
        entry = _make_entry(architecture="ltx", id="ltx2_fp8", quant="bf16", full_repo=True)
        self._make_video_repo("ltx", "ltx2_fp8")
        path = get_video_model_local_path(entry, self.models_dir)
        self.assertTrue(path.exists())

    def test_wan_full_repo_found(self):
        from genbox.utils.utils import get_video_model_local_path
        entry = _make_entry(architecture="wan", id="wan_1_3b", quant="bf16", full_repo=True)
        self._make_video_repo("wan", "wan_1_3b")
        path = get_video_model_local_path(entry, self.models_dir)
        self.assertTrue(path.exists())

    def test_missing_raises(self):
        from genbox.utils.utils import get_video_model_local_path
        entry = _make_entry(architecture="ltx", id="ltx2_fp8", quant="bf16", full_repo=True)
        with self.assertRaises(FileNotFoundError):
            get_video_model_local_path(entry, self.models_dir)

    def test_gguf_raises_value_error(self):
        from genbox.utils.utils import get_video_model_local_path
        entry = _make_entry(architecture="wan", id="wan_gguf", quant="gguf-q4", full_repo=False)
        with self.assertRaises(ValueError):
            get_video_model_local_path(entry, self.models_dir)


class TestListVideoModelsLocal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_video_repo(self, arch, name):
        d = self.models_dir / arch / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_lists_ltx_models(self):
        from genbox.utils.utils import list_video_models_local
        self._make_video_repo("ltx", "ltx2_fp8")
        results = list_video_models_local(self.models_dir, architecture="ltx")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["architecture"], "ltx")

    def test_lists_wan_models(self):
        from genbox.utils.utils import list_video_models_local
        self._make_video_repo("wan", "wan_1_3b")
        self._make_video_repo("wan", "wan21_14b_diffusers")
        results = list_video_models_local(self.models_dir, architecture="wan")
        self.assertEqual(len(results), 2)

    def test_variant_detected_distilled(self):
        from genbox.utils.utils import list_video_models_local
        self._make_video_repo("ltx", "ltx23_distilled")
        results = list_video_models_local(self.models_dir, architecture="ltx")
        self.assertEqual(results[0]["variant"], "ltx_distilled")

    def test_variant_wan_14b(self):
        from genbox.utils.utils import list_video_models_local
        self._make_video_repo("wan", "wan21_14b_diffusers")
        results = list_video_models_local(self.models_dir, architecture="wan")
        self.assertEqual(results[0]["variant"], "wan_14b")

    def test_all_archs_without_filter(self):
        from genbox.utils.utils import list_video_models_local
        self._make_video_repo("ltx", "ltx2_fp8")
        self._make_video_repo("wan", "wan_1_3b")
        results = list_video_models_local(self.models_dir)
        archs = {r["architecture"] for r in results}
        self.assertIn("ltx", archs)
        self.assertIn("wan", archs)


class TestGetVideoGenerationDefaults(unittest.TestCase):
    def test_ltx_defaults(self):
        from genbox.utils.utils import get_video_generation_defaults
        d = get_video_generation_defaults("ltx")
        self.assertEqual(d["frames"], 97)
        self.assertAlmostEqual(d["guidance_scale"], 3.0)

    def test_wan_1_3b_defaults(self):
        from genbox.utils.utils import get_video_generation_defaults
        d = get_video_generation_defaults("wan", "wan_1_3b")
        self.assertEqual(d["frames"], 81)
        self.assertEqual(d["height"], 480)

    def test_wan_14b_defaults_720p(self):
        from genbox.utils.utils import get_video_generation_defaults
        d = get_video_generation_defaults("wan", "wan_14b")
        self.assertEqual(d["height"], 720)

    def test_unknown_returns_empty(self):
        from genbox.utils.utils import get_video_generation_defaults
        d = get_video_generation_defaults("unknown")
        self.assertEqual(d, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)