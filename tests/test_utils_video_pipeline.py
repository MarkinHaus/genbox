"""
TDD: utils_video_pipeline.py
Data-transformation and contract tests — no real ML imports.
Run: python -m unittest genbox.test_utils_video_pipeline -v
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _entry(**kw):
    defaults = dict(
        id="ltx2_fp8", name="LTX 0.9.5", type="video",
        architecture="ltx", quant="bf16",
        hf_filename="model_index.json",
        hf_pipeline_repo="", full_repo=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ── Frame snapping ─────────────────────────────────────────────────────────────

class TestSnapFramesVideo(unittest.TestCase):
    def test_ltx_valid_passthrough(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(97, "ltx"), 97)    # 8*12+1

    def test_ltx_snaps_93_to_97(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(93, "ltx"), 97)

    def test_ltx_minimum_9(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(1, "ltx"), 9)

    def test_ltxv2_same_constraint_as_ltx(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        # LTX-2 same 8n+1 rule
        self.assertEqual(snap_frames(121, "ltx"), 121)  # 8*15+1

    def test_wan_valid_passthrough(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(81, "wan"), 81)    # 4*20+1

    def test_wan_snaps_80_to_81(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(80, "wan"), 81)

    def test_wan_minimum_5(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(1, "wan"), 5)

    def test_other_arch_unchanged(self):
        from genbox.utils.utils_video_pipeline import snap_frames
        self.assertEqual(snap_frames(30, "other"), 30)


# ── Flow-shift helpers ─────────────────────────────────────────────────────────

class TestWanFlowShift(unittest.TestCase):
    def test_720p_is_5(self):
        from genbox.utils.utils_video_pipeline import wan_flow_shift
        self.assertAlmostEqual(wan_flow_shift(720), 5.0)

    def test_1080p_is_5(self):
        from genbox.utils.utils_video_pipeline import wan_flow_shift
        self.assertAlmostEqual(wan_flow_shift(1080), 5.0)

    def test_480p_is_3(self):
        from genbox.utils.utils_video_pipeline import wan_flow_shift
        self.assertAlmostEqual(wan_flow_shift(480), 3.0)

    def test_boundary_exactly_720(self):
        from genbox.utils.utils_video_pipeline import wan_flow_shift
        self.assertAlmostEqual(wan_flow_shift(720), 5.0)


# ── Variant detection ──────────────────────────────────────────────────────────

class TestDetectLtxVariant(unittest.TestCase):
    def test_09_5_is_classic(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("Lightricks/LTX-Video", "ltx2_fp8"), "classic")

    def test_097_is_distilled_13b(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("Lightricks/LTX-Video-0.9.7-distilled", "ltx23_fp8"), "distilled_13b")

    def test_098_is_distilled_13b(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("Lightricks/LTX-Video-0.9.8-dev", "v098"), "distilled_13b")

    def test_ltx2_repo(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("Lightricks/LTX-2", "ltx2_model"), "ltx2")

    def test_id_fallback_distilled(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("", "ltx23_distilled"), "distilled_13b")

    def test_id_fallback_classic(self):
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        self.assertEqual(detect_ltx_variant("", "ltx2_fp8"), "classic")


class TestDetectWanVariant(unittest.TestCase):
    def test_1_3b(self):
        from genbox.utils.utils_video_pipeline import detect_wan_variant
        self.assertEqual(detect_wan_variant("wan_1_3b"), "wan21_1_3b")

    def test_14b_21(self):
        from genbox.utils.utils_video_pipeline import detect_wan_variant
        self.assertEqual(detect_wan_variant("wan21_14b_diffusers"), "wan21_14b")

    def test_22(self):
        from genbox.utils.utils_video_pipeline import detect_wan_variant
        self.assertEqual(detect_wan_variant("wan22_1_4b"), "wan22_a14b")

    def test_22_i2v(self):
        from genbox.utils.utils_video_pipeline import detect_wan_variant
        self.assertEqual(detect_wan_variant("wan22_i2v"), "wan22_a14b")


# ── LTX pipeline class selection ──────────────────────────────────────────────

class TestSelectLtxPipelineClass(unittest.TestCase):
    def test_classic_t2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("classic", mode="t2v")
        self.assertEqual(cls, "LTXPipeline")

    def test_classic_i2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("classic", mode="i2v")
        self.assertEqual(cls, "LTXImageToVideoPipeline")

    def test_distilled_13b_t2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("distilled_13b", mode="t2v")
        self.assertEqual(cls, "LTXConditionPipeline")

    def test_distilled_13b_i2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("distilled_13b", mode="i2v")
        self.assertEqual(cls, "LTXConditionPipeline")

    def test_ltx2_t2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("ltx2", mode="t2v")
        self.assertEqual(cls, "LTX2Pipeline")

    def test_ltx2_i2v(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        cls = select_ltx_pipeline_class("ltx2", mode="i2v")
        self.assertEqual(cls, "LTX2ImageToVideoPipeline")

    def test_unknown_raises(self):
        from genbox.utils.utils_video_pipeline import select_ltx_pipeline_class
        with self.assertRaises(ValueError):
            select_ltx_pipeline_class("unknown_variant", mode="t2v")


# ── WAN pipeline class selection ──────────────────────────────────────────────

class TestSelectWanPipelineClass(unittest.TestCase):
    def test_t2v_returns_wan_pipeline(self):
        from genbox.utils.utils_video_pipeline import select_wan_pipeline_class
        self.assertEqual(select_wan_pipeline_class("t2v"), "WanPipeline")

    def test_i2v_returns_wan_image_to_video(self):
        from genbox.utils.utils_video_pipeline import select_wan_pipeline_class
        self.assertEqual(select_wan_pipeline_class("i2v"), "WanImageToVideoPipeline")

    def test_unknown_mode_raises(self):
        from genbox.utils.utils_video_pipeline import select_wan_pipeline_class
        with self.assertRaises(ValueError):
            select_wan_pipeline_class("v2v_unsupported")


# ── LTX distilled guidance defaults ───────────────────────────────────────────

class TestLtxDistilledDefaults(unittest.TestCase):
    def test_classic_guidance_above_1(self):
        from genbox.utils.utils_video_pipeline import ltx_generation_defaults
        d = ltx_generation_defaults("classic")
        self.assertGreater(d["guidance_scale"], 1.0)

    def test_distilled_guidance_is_1(self):
        from genbox.utils.utils_video_pipeline import ltx_generation_defaults
        d = ltx_generation_defaults("distilled_13b")
        self.assertAlmostEqual(d["guidance_scale"], 1.0)

    def test_distilled_steps_low(self):
        from genbox.utils.utils_video_pipeline import ltx_generation_defaults
        d = ltx_generation_defaults("distilled_13b")
        self.assertLessEqual(d["steps"], 10)

    def test_ltx2_returns_dict(self):
        from genbox.utils.utils_video_pipeline import ltx_generation_defaults
        d = ltx_generation_defaults("ltx2")
        self.assertIn("guidance_scale", d)
        self.assertIn("frames", d)

    def test_classic_decode_timestep(self):
        from genbox.utils.utils_video_pipeline import ltx_generation_defaults
        d = ltx_generation_defaults("classic")
        self.assertIn("decode_timestep", d)
        self.assertAlmostEqual(d["decode_timestep"], 0.05)


# ── WAN generation defaults ────────────────────────────────────────────────────

class TestWanGenerationDefaults(unittest.TestCase):
    def test_1_3b_480p(self):
        from genbox.utils.utils_video_pipeline import wan_generation_defaults
        d = wan_generation_defaults("wan21_1_3b")
        self.assertEqual(d["height"], 480)
        self.assertEqual(d["width"], 832)
        self.assertAlmostEqual(d["guidance_scale"], 5.0)

    def test_14b_480p(self):
        from genbox.utils.utils_video_pipeline import wan_generation_defaults
        d = wan_generation_defaults("wan21_14b")
        self.assertEqual(d["height"], 480)

    def test_22_a14b_720p(self):
        from genbox.utils.utils_video_pipeline import wan_generation_defaults
        d = wan_generation_defaults("wan22_a14b")
        self.assertEqual(d["height"], 720)
        self.assertEqual(d["width"], 1280)

    def test_frames_4n_plus_1(self):
        from genbox.utils.utils_video_pipeline import wan_generation_defaults
        d = wan_generation_defaults("wan21_1_3b")
        # frames must satisfy 4n+1
        self.assertEqual((d["frames"] - 1) % 4, 0)


# ── Video output path ──────────────────────────────────────────────────────────

class TestBuildVideoOutputPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_custom_path_returned(self):
        from genbox.utils.utils_video_pipeline import build_video_output_path
        custom = Path(self.tmp.name) / "out.mp4"
        result = build_video_output_path("vid", "model", 0, outputs_dir=Path(self.tmp.name), custom=custom)
        self.assertEqual(result, custom)

    def test_auto_path_is_mp4(self):
        from genbox.utils.utils_video_pipeline import build_video_output_path
        result = build_video_output_path("vid", "wan_1_3b", 42, outputs_dir=Path(self.tmp.name))
        self.assertEqual(result.suffix, ".mp4")

    def test_filename_has_model_and_seed(self):
        from genbox.utils.utils_video_pipeline import build_video_output_path
        result = build_video_output_path("vid", "wan_1_3b", 77, outputs_dir=Path(self.tmp.name))
        self.assertIn("wan_1_3b", result.name)
        self.assertIn("77", result.name)

    def test_dir_created(self):
        from genbox.utils.utils_video_pipeline import build_video_output_path
        out_dir = Path(self.tmp.name) / "new_subdir"
        build_video_output_path("vid", "m", 1, outputs_dir=out_dir)
        self.assertTrue(out_dir.exists())


# ── Video output metadata ──────────────────────────────────────────────────────

class TestBuildVideoOutputMeta(unittest.TestCase):
    def _call(self, **kw):
        from genbox.utils.utils_video_pipeline import build_video_output_meta
        defaults = dict(
            pipeline_name="t2v", model_id="wan_1_3b",
            prompt="a cat", negative_prompt="",
            width=832, height=480, frames=81, fps=16,
            steps=50, guidance_scale=5.0,
            seed=42, lora_specs=[], accel=[], sampler="default",
            elapsed_s=10.0, output_path=Path("/tmp/v.mp4"),
        )
        defaults.update(kw)
        return build_video_output_meta(**defaults)

    def test_required_keys_present(self):
        meta = self._call()
        for k in ("pipeline", "model", "prompt", "width", "height",
                  "frames", "fps", "steps", "guidance_scale",
                  "seed", "loras", "elapsed_s", "timestamp", "output_path"):
            self.assertIn(k, meta, msg=f"Missing: {k}")

    def test_extra_dict_merged(self):
        meta = self._call(extra={"wan_variant": "wan21_1_3b"})
        self.assertEqual(meta["wan_variant"], "wan21_1_3b")

    def test_elapsed_rounded(self):
        meta = self._call(elapsed_s=12.34567)
        self.assertEqual(meta["elapsed_s"], 12.35)

    def test_lora_specs_normalized(self):
        meta = self._call(lora_specs=[("style.safetensors", 0.8)])
        self.assertEqual(len(meta["loras"]), 1)
        self.assertIn("path", meta["loras"][0])
        self.assertIn("weight", meta["loras"][0])


# ── Video frame saver ──────────────────────────────────────────────────────────

class TestSaveVideoFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_imageio_called_with_path(self):
        from genbox.utils.utils_video_pipeline import save_video_frames
        out = Path(self.tmp.name) / "test.mp4"
        imageio_mock = MagicMock()
        writer_mock = MagicMock()
        imageio_mock.get_writer.return_value.__enter__ = MagicMock(return_value=writer_mock)
        imageio_mock.get_writer.return_value.__exit__ = MagicMock(return_value=False)
        # imageio context manager usage
        ctx = imageio_mock.get_writer.return_value
        ctx.__enter__.return_value = writer_mock

        import sys
        with patch.dict(sys.modules, {"imageio": imageio_mock}):
            # PIL frames
            frame = MagicMock()
            frame.convert.return_value = MagicMock()
            frame.convert.return_value.__class__ = type("PIL", (), {"convert": lambda s, m: s})

            import numpy as np_real
            np_mock = MagicMock()
            np_mock.array.return_value = np_real.zeros((480, 832, 3), dtype=np_real.uint8)
            with patch.dict(sys.modules, {"numpy": np_mock}):
                save_video_frames([frame], out, fps=16)

        imageio_mock.get_writer.assert_called_once_with(
            str(out), fps=16, codec="libx264", quality=8
        )

    def test_missing_imageio_raises(self):
        from genbox.utils.utils_video_pipeline import save_video_frames
        import sys
        out = Path(self.tmp.name) / "test.mp4"
        with patch.dict(sys.modules, {"imageio": None}):
            with self.assertRaises((ImportError, TypeError)):
                save_video_frames([], out)


# ── Accelerator: video offload ─────────────────────────────────────────────────

class TestVideoAccelerators(unittest.TestCase):
    def _pipe(self):
        p = MagicMock()
        p.enable_model_cpu_offload = MagicMock()
        p.enable_sequential_cpu_offload = MagicMock()
        p.to = MagicMock()
        return p

    def test_cuda_low_vram_sequential(self):
        from genbox.utils.utils_video_pipeline import apply_video_accelerators
        pipe = self._pipe()
        apply_video_accelerators(pipe, device="cuda", vram_gb=6)
        pipe.enable_sequential_cpu_offload.assert_called_once()

    def test_cuda_mid_vram_model_offload(self):
        from genbox.utils.utils_video_pipeline import apply_video_accelerators
        pipe = self._pipe()
        apply_video_accelerators(pipe, device="cuda", vram_gb=12)
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_cpu_device_uses_pipe_to(self):
        from genbox.utils.utils_video_pipeline import apply_video_accelerators
        pipe = self._pipe()
        apply_video_accelerators(pipe, device="cpu", vram_gb=0)
        pipe.to.assert_called_once_with("cpu")

    def test_env_override_none_no_offload(self):
        from genbox.utils.utils_video_pipeline import apply_video_accelerators
        pipe = self._pipe()
        apply_video_accelerators(pipe, device="cuda", vram_gb=8, env_override="none")
        pipe.to.assert_called_once_with("cuda")

    def test_vae_tiling_attempted_when_requested(self):
        from genbox.utils.utils_video_pipeline import apply_video_accelerators
        pipe = self._pipe()
        pipe.vae = MagicMock()
        pipe.vae.enable_tiling = MagicMock()
        apply_video_accelerators(pipe, device="cuda", vram_gb=12, enable_vae_tiling=True)
        pipe.vae.enable_tiling.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
