"""
TDD: pipeline_flux.py, pipeline_sdl.py, pipeline_pony.py
All tests use mocks for diffusers/torch — no real ML imports.
Tests verify: data transformation, model loading routing, config
resolution, metadata correctness, lora wiring, accel application.

Run: python -m unittest genbox.test_pipelines -v
"""
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _entry(**kw):
    defaults = dict(
        id="test_model", name="Test Model", type="image",
        architecture="flux", quant="fp8",
        hf_filename="model_index.json",
        hf_pipeline_repo="", full_repo=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _mock_torch():
    t = MagicMock()
    t.cuda.is_available.return_value = False
    t.backends.mps.is_available.return_value = False
    t.float16 = "float16"
    t.bfloat16 = "bfloat16"
    t.float32 = "float32"
    t.Generator.return_value.manual_seed.return_value = MagicMock()
    return t


def _mock_diffusers():
    d = MagicMock()
    d.FluxPipeline = MagicMock()
    d.Flux2KleinPipeline = MagicMock()
    d.FluxTransformer2DModel = MagicMock()
    d.Flux2Transformer2DModel = MagicMock()
    d.GGUFQuantizationConfig = MagicMock()
    d.StableDiffusionPipeline = MagicMock()
    d.StableDiffusionXLPipeline = MagicMock()
    d.StableDiffusion3Pipeline = MagicMock()
    d.FlowMatchEulerDiscreteScheduler = MagicMock()
    d.DPMSolverMultistepScheduler = MagicMock()
    return d


def _patch_sys_modules(extras=None):
    """Provide a baseline fake module context."""
    mods = {
        "torch": _mock_torch(),
        "diffusers": _mock_diffusers(),
        "transformers": MagicMock(),
        "peft": MagicMock(),
        "gguf": MagicMock(),
        "genbox.config": MagicMock(cfg=None),
    }
    if extras:
        mods.update(extras)
    return mods


# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_flux.py tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFluxPipelineConfig(unittest.TestCase):
    """Test FluxPipelineConfig data class — pure data transforms."""

    def test_default_values(self):
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig
        cfg = FluxPipelineConfig(model_id="flux2_klein")
        self.assertEqual(cfg.steps, 28)
        self.assertAlmostEqual(cfg.guidance_scale, 3.5)
        self.assertEqual(cfg.t5_mode, "fp16")
        self.assertEqual(cfg.width, 1024)
        self.assertEqual(cfg.height, 1024)

    def test_seed_negative_means_random(self):
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig
        cfg = FluxPipelineConfig(model_id="m", seed=-1)
        self.assertEqual(cfg.seed, -1)  # raw value preserved

    def test_loras_defaults_to_empty(self):
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig
        cfg = FluxPipelineConfig(model_id="m")
        self.assertEqual(cfg.loras, [])

    def test_accel_defaults_to_empty(self):
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig
        cfg = FluxPipelineConfig(model_id="m")
        self.assertEqual(cfg.accel, [])

    def test_custom_values_stored(self):
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig
        cfg = FluxPipelineConfig(
            model_id="flux1_dev_q8",
            steps=50,
            guidance_scale=7.0,
            width=768,
            height=512,
            t5_mode="int8",
            loras=[("style.safetensors", 0.8)],
            accel=["xformers"],
        )
        self.assertEqual(cfg.steps, 50)
        self.assertEqual(cfg.width, 768)
        self.assertEqual(cfg.t5_mode, "int8")
        self.assertEqual(len(cfg.loras), 1)


class TestFluxLocalPathResolution(unittest.TestCase):
    """Test that _resolve_flux_local_path returns the right path type."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_full_repo(self, name="flux2_klein"):
        d = self.models_dir / "flux" / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}))
        return d

    def _make_gguf(self, fname="flux1-schnell-Q8_0.gguf"):
        d = self.models_dir / "flux"
        d.mkdir(parents=True, exist_ok=True)
        f = d / fname
        f.write_bytes(b"\x00" * 8)
        return f

    def test_full_repo_path_returned(self):
        from genbox.pipline_image.pipeline_flux import _resolve_flux_local_path
        repo = self._make_full_repo()
        entry = _entry(full_repo=True, id="flux2_klein", quant="fp8")
        path, is_gguf = _resolve_flux_local_path(entry, self.models_dir)
        self.assertEqual(path, repo)
        self.assertFalse(is_gguf)

    def test_gguf_path_returned(self):
        from genbox.pipline_image.pipeline_flux import _resolve_flux_local_path
        gguf = self._make_gguf()
        entry = _entry(full_repo=False, quant="gguf-q8", hf_filename="flux1-schnell-Q8_0.gguf")
        path, is_gguf = _resolve_flux_local_path(entry, self.models_dir)
        self.assertEqual(path, gguf)
        self.assertTrue(is_gguf)

    def test_missing_full_repo_raises(self):
        from genbox.pipline_image.pipeline_flux import _resolve_flux_local_path
        entry = _entry(full_repo=True, id="missing_model", quant="fp8")
        with self.assertRaises(FileNotFoundError):
            _resolve_flux_local_path(entry, self.models_dir)

    def test_missing_gguf_raises(self):
        from genbox.pipline_image.pipeline_flux import _resolve_flux_local_path
        entry = _entry(full_repo=False, quant="gguf-q8", hf_filename="missing.gguf")
        with self.assertRaises(FileNotFoundError):
            _resolve_flux_local_path(entry, self.models_dir)


class TestFluxGgufSharedConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_config_path_computed(self):
        from genbox.pipline_image.pipeline_flux import _flux_gguf_shared_config
        entry = _entry(
            full_repo=False, quant="gguf-q8",
            hf_filename="flux1-schnell-Q8_0.gguf",
            hf_pipeline_repo="black-forest-labs/FLUX.1-schnell",
        )
        path = _flux_gguf_shared_config(entry, self.models_dir)
        self.assertIn("_shared_black-forest-labs--FLUX.1-schnell", str(path))

    def test_no_pipeline_repo_returns_none(self):
        from genbox.pipline_image.pipeline_flux import _flux_gguf_shared_config
        entry = _entry(full_repo=False, quant="gguf-q8", hf_pipeline_repo="")
        result = _flux_gguf_shared_config(entry, self.models_dir)
        self.assertIsNone(result)


class TestFluxT5KwargsBuilder(unittest.TestCase):
    """Test that T5/Qwen encoder kwargs are built correctly per mode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.local_repo = Path(self.tmp.name)
        # create a fake text_encoder_2 dir to simulate T5 presence
        (self.local_repo / "text_encoder_2").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_t5_none_mode_disables_encoder(self):
        from genbox.pipline_image.pipeline_flux import build_t5_kwargs
        kwargs = build_t5_kwargs(
            local_repo=self.local_repo,
            t5_mode="none",
            is_flux2=False,
            dtype=None,
        )
        self.assertIsNone(kwargs.get("text_encoder_2"))
        self.assertIsNone(kwargs.get("tokenizer_2"))

    def test_flux2_skips_t5_handling(self):
        from genbox.pipline_image.pipeline_flux import build_t5_kwargs
        kwargs = build_t5_kwargs(
            local_repo=self.local_repo,
            t5_mode="fp16",
            is_flux2=True,
            dtype=None,
        )
        # FLUX.2 uses Qwen, no T5 kwargs needed
        self.assertEqual(kwargs, {})

    def test_t5_fp16_returns_empty_kwargs(self):
        from genbox.pipline_image.pipeline_flux import build_t5_kwargs
        kwargs = build_t5_kwargs(
            local_repo=self.local_repo,
            t5_mode="fp16",
            is_flux2=False,
            dtype=None,
        )
        self.assertEqual(kwargs, {})

    def test_no_t5_dir_disables_encoder(self):
        from genbox.pipline_image.pipeline_flux import build_t5_kwargs
        local_repo_no_t5 = Path(self.tmp.name + "_notexist")
        local_repo_no_t5.mkdir()
        kwargs = build_t5_kwargs(
            local_repo=local_repo_no_t5,
            t5_mode="fp16",
            is_flux2=False,
            dtype=None,
        )
        self.assertIsNone(kwargs.get("text_encoder_2"))


# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_sdl.py tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSDLPipelineConfig(unittest.TestCase):
    def test_sd15_defaults(self):
        from genbox.pipline_image.pipeline_sdl import SDLPipelineConfig
        cfg = SDLPipelineConfig(model_id="sd15_base", architecture="sd15")
        self.assertEqual(cfg.width, 512)
        self.assertEqual(cfg.height, 512)
        self.assertAlmostEqual(cfg.guidance_scale, 7.5)

    def test_sdxl_defaults(self):
        from genbox.pipline_image.pipeline_sdl import SDLPipelineConfig
        cfg = SDLPipelineConfig(model_id="sdxl_base", architecture="sdxl")
        self.assertEqual(cfg.width, 1024)
        self.assertEqual(cfg.height, 1024)

    def test_sd35_defaults(self):
        from genbox.pipline_image.pipeline_sdl import SDLPipelineConfig
        cfg = SDLPipelineConfig(model_id="sd35_medium", architecture="sd35")
        self.assertEqual(cfg.width, 1024)
        self.assertEqual(cfg.guidance_scale, 4.5)

    def test_custom_values_override(self):
        from genbox.pipline_image.pipeline_sdl import SDLPipelineConfig
        cfg = SDLPipelineConfig(
            model_id="sd15_base", architecture="sd15",
            steps=50, width=768, height=512,
        )
        self.assertEqual(cfg.steps, 50)
        self.assertEqual(cfg.width, 768)


class TestSDLPipelineClassSelection(unittest.TestCase):
    """_select_sdl_pipeline_class returns the correct diffusers class name."""

    def test_sd15_returns_stable_diffusion_pipeline(self):
        from genbox.pipline_image.pipeline_sdl import _select_sdl_pipeline_class
        cls_name = _select_sdl_pipeline_class("sd15")
        self.assertEqual(cls_name, "StableDiffusionPipeline")

    def test_sdxl_returns_stable_diffusion_xl_pipeline(self):
        from genbox.pipline_image.pipeline_sdl import _select_sdl_pipeline_class
        cls_name = _select_sdl_pipeline_class("sdxl")
        self.assertEqual(cls_name, "StableDiffusionXLPipeline")

    def test_sd35_returns_stable_diffusion3_pipeline(self):
        from genbox.pipline_image.pipeline_sdl import _select_sdl_pipeline_class
        cls_name = _select_sdl_pipeline_class("sd35")
        self.assertEqual(cls_name, "StableDiffusion3Pipeline")

    def test_unknown_raises(self):
        from genbox.pipline_image.pipeline_sdl import _select_sdl_pipeline_class
        with self.assertRaises(ValueError):
            _select_sdl_pipeline_class("unknown_arch")


class TestSDLLocalPathResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_repo(self, arch, model_id):
        d = self.models_dir / arch / model_id
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_full_repo_found(self):
        from genbox.pipline_image.pipeline_sdl import _resolve_sdl_local_path
        self._make_repo("sdxl", "sdxl_base")
        entry = _entry(architecture="sdxl", id="sdxl_base", full_repo=True, quant="fp16")
        path = _resolve_sdl_local_path(entry, self.models_dir)
        self.assertTrue(path.exists())

    def test_missing_raises(self):
        from genbox.pipline_image.pipeline_sdl import _resolve_sdl_local_path
        entry = _entry(architecture="sdxl", id="missing", full_repo=True, quant="fp16")
        with self.assertRaises(FileNotFoundError):
            _resolve_sdl_local_path(entry, self.models_dir)

    def test_safetensors_single_file(self):
        from genbox.pipline_image.pipeline_sdl import _resolve_sdl_local_path
        arch_dir = self.models_dir / "sd15"
        arch_dir.mkdir(parents=True)
        f = arch_dir / "model.safetensors"
        f.write_bytes(b"\x00" * 8)
        entry = _entry(
            architecture="sd15", id="sd15_base",
            full_repo=False, quant="fp16",
            hf_filename="model.safetensors",
        )
        path = _resolve_sdl_local_path(entry, self.models_dir)
        self.assertEqual(path, f)


class TestSDLCallKwargs(unittest.TestCase):
    def test_sd15_negative_prompt_included(self):
        from genbox.pipline_image.pipeline_sdl import build_sdl_call_kwargs
        kwargs = build_sdl_call_kwargs(
            architecture="sd15",
            prompt="a cat", negative_prompt="ugly",
            width=512, height=512, steps=20,
            guidance_scale=7.5, generator=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "ugly")

    def test_sdxl_negative_prompt_included(self):
        from genbox.pipline_image.pipeline_sdl import build_sdl_call_kwargs
        kwargs = build_sdl_call_kwargs(
            architecture="sdxl",
            prompt="a cat", negative_prompt="blur",
            width=1024, height=1024, steps=30,
            guidance_scale=7.5, generator=None,
        )
        self.assertIn("negative_prompt", kwargs)

    def test_sd35_no_negative_prompt(self):
        from genbox.pipline_image.pipeline_sdl import build_sdl_call_kwargs
        kwargs = build_sdl_call_kwargs(
            architecture="sd35",
            prompt="a cat", negative_prompt="ugly",
            width=1024, height=1024, steps=28,
            guidance_scale=4.5, generator=None,
        )
        # SD3.5 handles negative_prompt internally — no explicit kwarg
        self.assertNotIn("negative_prompt", kwargs)

    def test_step_count_in_kwargs(self):
        from genbox.pipline_image.pipeline_sdl import build_sdl_call_kwargs
        kwargs = build_sdl_call_kwargs(
            architecture="sd15",
            prompt="test", negative_prompt="",
            width=512, height=512, steps=42,
            guidance_scale=7.0, generator=None,
        )
        self.assertEqual(kwargs["num_inference_steps"], 42)


class TestSDLOutputMeta(unittest.TestCase):
    def test_meta_has_architecture(self):
        from genbox.pipline_image.pipeline_sdl import build_sdl_output_meta
        meta = build_sdl_output_meta(
            architecture="sdxl",
            model_id="sdxl_base",
            prompt="test",
            negative_prompt="",
            width=1024, height=1024,
            steps=30, guidance_scale=7.5,
            seed=42, lora_specs=[], accel=[],
            sampler="default", elapsed_s=2.0,
            output_path=Path("/tmp/img.png"),
        )
        self.assertEqual(meta["architecture"], "sdxl")
        self.assertIn("pipeline", meta)
        self.assertIn("seed", meta)


# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_pony.py tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPonyPipelineConfig(unittest.TestCase):
    def test_pony_defaults(self):
        from genbox.pipline_image.pipeline_pony import PonyPipelineConfig
        cfg = PonyPipelineConfig(model_id="pony_xl")
        # Pony uses specific quality tags in prompts
        self.assertEqual(cfg.quality_tags, "score_9, score_8_up, score_7_up")
        self.assertEqual(cfg.negative_quality_tags,
                         "score_1, score_2, score_3, score_4")

    def test_custom_quality_tags(self):
        from genbox.pipline_image.pipeline_pony import PonyPipelineConfig
        cfg = PonyPipelineConfig(
            model_id="pony_xl",
            quality_tags="score_9",
            negative_quality_tags="score_1",
        )
        self.assertEqual(cfg.quality_tags, "score_9")

    def test_pony_width_height_defaults(self):
        from genbox.pipline_image.pipeline_pony import PonyPipelineConfig
        cfg = PonyPipelineConfig(model_id="pony_xl")
        self.assertEqual(cfg.width, 1024)
        self.assertEqual(cfg.height, 1024)

    def test_inherits_sdxl_guidance(self):
        from genbox.pipline_image.pipeline_pony import PonyPipelineConfig
        cfg = PonyPipelineConfig(model_id="pony_xl")
        self.assertAlmostEqual(cfg.guidance_scale, 7.0)


class TestPonyPromptInjection(unittest.TestCase):
    """Test that quality tags are prepended/appended correctly."""

    def test_quality_tags_prepended_to_prompt(self):
        from genbox.pipline_image.pipeline_pony import build_pony_prompt
        prompt = build_pony_prompt(
            user_prompt="a cute cat",
            quality_tags="score_9, score_8_up",
        )
        self.assertTrue(prompt.startswith("score_9, score_8_up"))
        self.assertIn("a cute cat", prompt)

    def test_empty_quality_tags_returns_prompt_unchanged(self):
        from genbox.pipline_image.pipeline_pony import build_pony_prompt
        prompt = build_pony_prompt("a cute cat", quality_tags="")
        self.assertEqual(prompt, "a cute cat")

    def test_negative_tags_prepended(self):
        from genbox.pipline_image.pipeline_pony import build_pony_negative_prompt
        neg = build_pony_negative_prompt(
            user_negative="blurry",
            negative_quality_tags="score_1, score_2",
        )
        self.assertTrue(neg.startswith("score_1, score_2"))
        self.assertIn("blurry", neg)

    def test_empty_user_negative_only_tags(self):
        from genbox.pipline_image.pipeline_pony import build_pony_negative_prompt
        neg = build_pony_negative_prompt("", negative_quality_tags="score_1")
        self.assertEqual(neg.strip(), "score_1")


class TestPonyCallKwargs(unittest.TestCase):
    def test_negative_prompt_included(self):
        from genbox.pipline_image.pipeline_pony import build_pony_call_kwargs
        kwargs = build_pony_call_kwargs(
            prompt="score_9, score_8_up, a cat",
            negative_prompt="score_1, blur",
            width=1024, height=1024,
            steps=30, guidance_scale=7.0,
            generator=None,
        )
        self.assertIn("negative_prompt", kwargs)
        self.assertEqual(kwargs["negative_prompt"], "score_1, blur")

    def test_resolution_in_kwargs(self):
        from genbox.pipline_image.pipeline_pony import build_pony_call_kwargs
        kwargs = build_pony_call_kwargs(
            prompt="score_9, cat",
            negative_prompt="score_1",
            width=896, height=768,
            steps=25, guidance_scale=7.0,
            generator=None,
        )
        self.assertEqual(kwargs["width"], 896)
        self.assertEqual(kwargs["height"], 768)


class TestPonyOutputMeta(unittest.TestCase):
    def test_meta_identifies_as_pony(self):
        from genbox.pipline_image.pipeline_pony import build_pony_output_meta
        meta = build_pony_output_meta(
            model_id="pony_xl",
            prompt="score_9, a cat",
            negative_prompt="score_1",
            width=1024, height=1024,
            steps=30, guidance_scale=7.0,
            seed=0, lora_specs=[], accel=[],
            sampler="default", elapsed_s=3.0,
            output_path=Path("/tmp/pony.png"),
            quality_tags="score_9",
            negative_quality_tags="score_1",
        )
        self.assertEqual(meta["pipeline"], "pony_text_to_image")
        self.assertIn("quality_tags", meta)
        self.assertIn("negative_quality_tags", meta)

    def test_meta_seed_stored(self):
        from genbox.pipline_image.pipeline_pony import build_pony_output_meta
        meta = build_pony_output_meta(
            model_id="pony_xl", prompt="p", negative_prompt="n",
            width=1024, height=1024, steps=1, guidance_scale=7.0,
            seed=999, lora_specs=[], accel=[], sampler="d",
            elapsed_s=0.1, output_path=Path("/tmp/x.png"),
            quality_tags="", negative_quality_tags="",
        )
        self.assertEqual(meta["seed"], 999)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-pipeline: accelerator wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestAcceleratorWiringContract(unittest.TestCase):
    """
    Verify that each pipeline module exposes apply_pipeline_accelerators()
    and that it delegates correctly to utils_image_pipeline.apply_accelerators.
    """

    def _mock_pipe(self):
        pipe = MagicMock()
        pipe.enable_model_cpu_offload = MagicMock()
        pipe.enable_sequential_cpu_offload = MagicMock()
        pipe.to = MagicMock()
        return pipe

    def test_flux_exposes_apply_pipeline_accelerators(self):
        from genbox.pipline_image.pipeline_flux import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_sdl_exposes_apply_pipeline_accelerators(self):
        from genbox.pipline_image.pipeline_sdl import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_pony_exposes_apply_pipeline_accelerators(self):
        from genbox.pipline_image.pipeline_pony import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_flux_apply_calls_model_offload_on_cuda(self):
        from genbox.pipline_image.pipeline_flux import apply_pipeline_accelerators
        pipe = self._mock_pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12, accel=[])
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_sdl_apply_calls_model_offload_on_cuda(self):
        from genbox.pipline_image.pipeline_sdl import apply_pipeline_accelerators
        pipe = self._mock_pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12, accel=[])
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_pony_apply_calls_model_offload_on_cuda(self):
        from genbox.pipline_image.pipeline_pony import apply_pipeline_accelerators
        pipe = self._mock_pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12, accel=[])
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_cpu_device_uses_pipe_to(self):
        from genbox.pipline_image.pipeline_flux import apply_pipeline_accelerators
        pipe = self._mock_pipe()
        apply_pipeline_accelerators(pipe, device="cpu", vram_gb=0, accel=[])
        pipe.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
