"""
TDD: pipeline_wan.py and pipeline_ltx.py
Data-transformation, config, routing, and contract tests.
No real diffusers/torch — all mocked.
Run: python -m unittest genbox.test_video_pipelines -v
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _entry(**kw):
    defaults = dict(
        id="wan_1_3b", name="WAN 2.1 1.3B", type="video",
        architecture="wan", quant="bf16",
        hf_filename="model_index.json",
        hf_pipeline_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        full_repo=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _ltx_entry(**kw):
    defaults = dict(
        id="ltx2_fp8", name="LTX 0.9.5", type="video",
        architecture="ltx", quant="bf16",
        hf_filename="model_index.json",
        hf_pipeline_repo="Lightricks/LTX-Video",
        full_repo=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_wan.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestWanPipelineConfig(unittest.TestCase):
    def test_t2v_defaults(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan_1_3b", mode="t2v")
        self.assertEqual(cfg.mode, "t2v")
        self.assertEqual(cfg.width, 832)
        self.assertEqual(cfg.height, 480)
        self.assertAlmostEqual(cfg.guidance_scale, 5.0)

    def test_wan22_720p_defaults(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan22_1_4b", mode="t2v")
        # wan22 defaults to 720p
        self.assertEqual(cfg.height, 720)
        self.assertEqual(cfg.width, 1280)

    def test_i2v_mode(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan_1_3b", mode="i2v")
        self.assertEqual(cfg.mode, "i2v")

    def test_frames_are_4n_plus_1(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan_1_3b", mode="t2v")
        self.assertEqual((cfg.frames - 1) % 4, 0)

    def test_explicit_overrides(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(
            model_id="wan_1_3b", mode="t2v",
            width=1280, height=720, steps=30, guidance_scale=6.0
        )
        self.assertEqual(cfg.width, 1280)
        self.assertEqual(cfg.height, 720)
        self.assertEqual(cfg.steps, 30)

    def test_loras_empty_by_default(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan_1_3b", mode="t2v")
        self.assertEqual(cfg.loras, [])

    def test_negative_prompt_default_empty(self):
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        cfg = WanPipelineConfig(model_id="wan_1_3b", mode="t2v")
        self.assertEqual(cfg.negative_prompt, "")


class TestWanLocalPathResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_repo(self, name):
        d = self.models_dir / "wan" / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_full_repo_found(self):
        from genbox.piplen_video.pipeline_wan import _resolve_wan_local_path
        self._make_repo("wan_1_3b")
        entry = _entry()
        path = _resolve_wan_local_path(entry, self.models_dir)
        self.assertTrue(path.exists())
        self.assertTrue((path / "model_index.json").exists())

    def test_missing_raises(self):
        from genbox.piplen_video.pipeline_wan import _resolve_wan_local_path
        entry = _entry(id="missing_model")
        with self.assertRaises(FileNotFoundError):
            _resolve_wan_local_path(entry, self.models_dir)

    def test_gguf_raises_value_error(self):
        from genbox.piplen_video.pipeline_wan import _resolve_wan_local_path
        entry = _entry(quant="gguf-q4", full_repo=False, hf_filename="wan.gguf")
        with self.assertRaises(ValueError):
            _resolve_wan_local_path(entry, self.models_dir)


class TestWanVaeKwargs(unittest.TestCase):
    """WAN VAE must always be float32 — critical correctness requirement."""

    def test_vae_dtype_is_float32(self):
        from genbox.piplen_video.pipeline_wan import build_wan_vae_kwargs
        import sys
        torch_mock = MagicMock()
        torch_mock.float32 = "float32_sentinel"
        torch_mock.bfloat16 = "bfloat16_sentinel"
        with patch.dict(sys.modules, {"torch": torch_mock}):
            kwargs = build_wan_vae_kwargs(local_path=Path("/tmp/model"), _torch=torch_mock)
        self.assertEqual(kwargs["torch_dtype"], "float32_sentinel")

    def test_vae_subfolder_set(self):
        from genbox.piplen_video.pipeline_wan import build_wan_vae_kwargs
        import sys
        torch_mock = MagicMock()
        torch_mock.float32 = "float32_sentinel"
        with patch.dict(sys.modules, {"torch": torch_mock}):
            kwargs = build_wan_vae_kwargs(local_path=Path("/tmp/model"), _torch=torch_mock)
        self.assertEqual(kwargs["subfolder"], "vae")


class TestWanSchedulerSetup(unittest.TestCase):
    def test_flow_shift_applied_for_480p(self):
        from genbox.piplen_video.pipeline_wan import build_wan_scheduler_kwargs
        kwargs = build_wan_scheduler_kwargs(height=480)
        self.assertAlmostEqual(kwargs["flow_shift"], 3.0)

    def test_flow_shift_applied_for_720p(self):
        from genbox.piplen_video.pipeline_wan import build_wan_scheduler_kwargs
        kwargs = build_wan_scheduler_kwargs(height=720)
        self.assertAlmostEqual(kwargs["flow_shift"], 5.0)


class TestWanCallKwargs(unittest.TestCase):
    def test_t2v_no_image(self):
        from genbox.piplen_video.pipeline_wan import build_wan_call_kwargs
        kwargs = build_wan_call_kwargs(
            mode="t2v", prompt="cat", negative_prompt="",
            width=832, height=480, frames=81, steps=50,
            guidance_scale=5.0, generator=None, image=None,
        )
        self.assertNotIn("image", kwargs)
        self.assertEqual(kwargs["prompt"], "cat")

    def test_i2v_includes_image(self):
        from genbox.piplen_video.pipeline_wan import build_wan_call_kwargs
        img_mock = MagicMock()
        kwargs = build_wan_call_kwargs(
            mode="i2v", prompt="cat", negative_prompt="",
            width=832, height=480, frames=81, steps=50,
            guidance_scale=5.0, generator=None, image=img_mock,
        )
        self.assertIs(kwargs["image"], img_mock)

    def test_negative_prompt_included_when_set(self):
        from genbox.piplen_video.pipeline_wan import build_wan_call_kwargs
        kwargs = build_wan_call_kwargs(
            mode="t2v", prompt="cat", negative_prompt="blurry",
            width=832, height=480, frames=81, steps=50,
            guidance_scale=5.0, generator=None, image=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "blurry")

    def test_frames_in_kwargs(self):
        from genbox.piplen_video.pipeline_wan import build_wan_call_kwargs
        kwargs = build_wan_call_kwargs(
            mode="t2v", prompt="cat", negative_prompt="",
            width=832, height=480, frames=81, steps=50,
            guidance_scale=5.0, generator=None, image=None,
        )
        self.assertEqual(kwargs["num_frames"], 81)


class TestWanOutputMeta(unittest.TestCase):
    def test_meta_contains_variant(self):
        from genbox.piplen_video.pipeline_wan import build_wan_output_meta
        meta = build_wan_output_meta(
            wan_variant="wan21_1_3b", model_id="wan_1_3b",
            mode="t2v", prompt="cat", negative_prompt="",
            width=832, height=480, frames=81, fps=16,
            steps=50, guidance_scale=5.0, seed=0,
            lora_specs=[], accel=[], sampler="default",
            elapsed_s=5.0, output_path=Path("/tmp/v.mp4"),
        )
        self.assertEqual(meta["wan_variant"], "wan21_1_3b")
        self.assertEqual(meta["mode"], "t2v")
        self.assertIn("frames", meta)

    def test_meta_pipeline_name(self):
        from genbox.piplen_video.pipeline_wan import build_wan_output_meta
        meta = build_wan_output_meta(
            wan_variant="wan22_a14b", model_id="m", mode="i2v",
            prompt="p", negative_prompt="",
            width=1280, height=720, frames=81, fps=24,
            steps=50, guidance_scale=5.0, seed=1,
            lora_specs=[], accel=[], sampler="d",
            elapsed_s=1.0, output_path=Path("/tmp/v.mp4"),
        )
        self.assertIn("wan", meta["pipeline"].lower())


class TestWanApplyPipelineAccelerators(unittest.TestCase):
    def _pipe(self):
        p = MagicMock()
        p.enable_model_cpu_offload = MagicMock()
        p.enable_sequential_cpu_offload = MagicMock()
        p.to = MagicMock()
        p.vae = MagicMock()
        p.vae.enable_tiling = MagicMock()
        return p

    def test_exposes_apply_pipeline_accelerators(self):
        from genbox.piplen_video.pipeline_wan import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_model_offload_on_cuda(self):
        from genbox.piplen_video.pipeline_wan import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12)
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_cpu_device_uses_pipe_to(self):
        from genbox.piplen_video.pipeline_wan import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cpu", vram_gb=0)
        pipe.to.assert_called_once_with("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_ltx.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLtxPipelineConfig(unittest.TestCase):
    def test_classic_t2v_defaults(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx2_fp8", variant="classic", mode="t2v")
        self.assertAlmostEqual(cfg.guidance_scale, 5.0)
        self.assertEqual(cfg.frames, 97)

    def test_distilled_guidance_1(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx23_fp8", variant="distilled_13b", mode="t2v")
        self.assertAlmostEqual(cfg.guidance_scale, 1.0)
        self.assertLessEqual(cfg.steps, 10)

    def test_ltx2_defaults(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx2_model", variant="ltx2", mode="t2v")
        self.assertAlmostEqual(cfg.guidance_scale, 4.0)

    def test_frames_8n_plus_1(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx2_fp8", variant="classic", mode="t2v")
        self.assertEqual((cfg.frames - 1) % 8, 0)

    def test_i2v_mode_stored(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx2_fp8", variant="classic", mode="i2v")
        self.assertEqual(cfg.mode, "i2v")

    def test_decode_timestep_default(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(model_id="ltx2_fp8", variant="classic", mode="t2v")
        self.assertAlmostEqual(cfg.decode_timestep, 0.05)

    def test_explicit_override_respected(self):
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        cfg = LtxPipelineConfig(
            model_id="ltx2_fp8", variant="classic", mode="t2v",
            width=1216, height=704, steps=30
        )
        self.assertEqual(cfg.width, 1216)
        self.assertEqual(cfg.steps, 30)


class TestLtxLocalPathResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_repo(self, name):
        d = self.models_dir / "ltx" / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return d

    def test_full_repo_found(self):
        from genbox.piplen_video.pipeline_ltx import _resolve_ltx_local_path
        self._make_repo("ltx2_fp8")
        entry = _ltx_entry()
        path = _resolve_ltx_local_path(entry, self.models_dir)
        self.assertTrue(path.exists())

    def test_missing_raises(self):
        from genbox.piplen_video.pipeline_ltx import _resolve_ltx_local_path
        entry = _ltx_entry(id="missing")
        with self.assertRaises(FileNotFoundError):
            _resolve_ltx_local_path(entry, self.models_dir)


class TestLtxCallKwargs(unittest.TestCase):
    def test_t2v_no_image(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_call_kwargs
        kwargs = build_ltx_call_kwargs(
            mode="t2v", variant="classic",
            prompt="cat", negative_prompt="blur",
            width=768, height=512, frames=97, fps=24,
            steps=50, guidance_scale=5.0,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            generator=None, image=None,
        )
        self.assertNotIn("image", kwargs)
        self.assertIn("decode_timestep", kwargs)

    def test_i2v_includes_image(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_call_kwargs
        img_mock = MagicMock()
        kwargs = build_ltx_call_kwargs(
            mode="i2v", variant="classic",
            prompt="cat", negative_prompt="blur",
            width=768, height=512, frames=97, fps=24,
            steps=50, guidance_scale=5.0,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            generator=None, image=img_mock,
        )
        self.assertIs(kwargs["image"], img_mock)

    def test_negative_prompt_included(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_call_kwargs
        kwargs = build_ltx_call_kwargs(
            mode="t2v", variant="classic",
            prompt="cat", negative_prompt="worst quality",
            width=768, height=512, frames=97, fps=24,
            steps=50, guidance_scale=5.0,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            generator=None, image=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "worst quality")

    def test_frame_rate_in_kwargs(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_call_kwargs
        kwargs = build_ltx_call_kwargs(
            mode="t2v", variant="classic",
            prompt="cat", negative_prompt="",
            width=768, height=512, frames=97, fps=30,
            steps=50, guidance_scale=5.0,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            generator=None, image=None,
        )
        self.assertEqual(kwargs.get("frame_rate"), 30)

    def test_distilled_low_guidance(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_call_kwargs
        kwargs = build_ltx_call_kwargs(
            mode="t2v", variant="distilled_13b",
            prompt="cat", negative_prompt="",
            width=768, height=512, frames=97, fps=24,
            steps=8, guidance_scale=1.0,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            generator=None, image=None,
        )
        self.assertAlmostEqual(kwargs["guidance_scale"], 1.0)


class TestLtxOutputMeta(unittest.TestCase):
    def test_meta_has_variant(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_output_meta
        meta = build_ltx_output_meta(
            ltx_variant="distilled_13b", model_id="ltx23_fp8",
            mode="t2v", prompt="cat", negative_prompt="blur",
            width=768, height=512, frames=97, fps=24,
            steps=8, guidance_scale=1.0, seed=42,
            lora_specs=[], accel=[], sampler="default",
            elapsed_s=3.0, output_path=Path("/tmp/v.mp4"),
        )
        self.assertEqual(meta["ltx_variant"], "distilled_13b")
        self.assertEqual(meta["mode"], "t2v")
        self.assertIn("frames", meta)

    def test_pipeline_name_contains_ltx(self):
        from genbox.piplen_video.pipeline_ltx import build_ltx_output_meta
        meta = build_ltx_output_meta(
            ltx_variant="classic", model_id="ltx2_fp8",
            mode="i2v", prompt="p", negative_prompt="",
            width=768, height=512, frames=97, fps=24,
            steps=50, guidance_scale=5.0, seed=0,
            lora_specs=[], accel=[], sampler="d",
            elapsed_s=1.0, output_path=Path("/tmp/v.mp4"),
        )
        self.assertIn("ltx", meta["pipeline"].lower())


class TestLtxApplyPipelineAccelerators(unittest.TestCase):
    def _pipe(self):
        p = MagicMock()
        p.enable_model_cpu_offload = MagicMock()
        p.enable_sequential_cpu_offload = MagicMock()
        p.to = MagicMock()
        p.vae = MagicMock()
        p.vae.enable_tiling = MagicMock()
        return p

    def test_exposes_apply_pipeline_accelerators(self):
        from genbox.piplen_video.pipeline_ltx import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_model_offload_on_cuda_mid_vram(self):
        from genbox.piplen_video.pipeline_ltx import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12)
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_cpu_device_to_cpu(self):
        from genbox.piplen_video.pipeline_ltx import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cpu", vram_gb=0)
        pipe.to.assert_called_once_with("cpu")

    def test_vae_tiling_opt_in(self):
        from genbox.piplen_video.pipeline_ltx import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(
            pipe, device="cuda", vram_gb=12, enable_vae_tiling=True
        )
        pipe.vae.enable_tiling.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
