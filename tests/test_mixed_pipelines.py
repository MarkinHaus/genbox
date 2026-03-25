"""
TDD: pipeline_img2img, pipeline_inpaint, pipeline_outpaint, pipeline_img2video
Data-transformation, config, routing, mask-logic, and contract tests.
No real diffusers/torch — all mocked where needed.
Run: python -m unittest genbox.test_mixed_pipelines -v
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _entry(**kw):
    d = dict(
        id="sdxl_base", name="SDXL 1.0", type="image",
        architecture="sdxl", quant="fp16",
        hf_filename="model_index.json", hf_pipeline_repo="",
        full_repo=True,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _ltx_entry(**kw):
    d = dict(
        id="ltx2_fp8", name="LTX 0.9.5", type="video",
        architecture="ltx", quant="bf16",
        hf_filename="model_index.json",
        hf_pipeline_repo="Lightricks/LTX-Video", full_repo=True,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _wan_entry(**kw):
    d = dict(
        id="wan_1_3b", name="WAN 1.3B", type="video",
        architecture="wan", quant="bf16",
        hf_filename="model_index.json",
        hf_pipeline_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", full_repo=True,
    )
    d.update(kw)
    return SimpleNamespace(**d)


# ══════════════════════════════════════════════════════════════════════════════
# pipeline_img2img.py
# ══════════════════════════════════════════════════════════════════════════════

class TestImg2ImgConfig(unittest.TestCase):
    def test_defaults(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="sdxl_base", architecture="sdxl")
        self.assertAlmostEqual(cfg.strength, 0.75)
        self.assertEqual(cfg.steps, 30)
        self.assertAlmostEqual(cfg.guidance_scale, 7.5)

    def test_flux_defaults(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="flux1_dev", architecture="flux")
        self.assertAlmostEqual(cfg.guidance_scale, 3.5)
        self.assertEqual(cfg.steps, 28)

    def test_sd15_defaults(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="sd15_base", architecture="sd15")
        self.assertEqual(cfg.width, 512)
        self.assertEqual(cfg.height, 512)

    def test_strength_clamped_to_01(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="m", architecture="sdxl", strength=1.5)
        self.assertLessEqual(cfg.strength, 1.0)

    def test_strength_minimum(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="m", architecture="sdxl", strength=-0.1)
        self.assertGreaterEqual(cfg.strength, 0.0)

    def test_loras_empty_by_default(self):
        from genbox.piplen_video.pipeline_img2img import Img2ImgConfig
        cfg = Img2ImgConfig(model_id="m", architecture="sdxl")
        self.assertEqual(cfg.loras, [])


class TestSelectImg2ImgPipelineClass(unittest.TestCase):
    def test_sd15(self):
        from genbox.piplen_video.pipeline_img2img import select_img2img_pipeline_class
        self.assertEqual(select_img2img_pipeline_class("sd15"),
                         "StableDiffusionImg2ImgPipeline")

    def test_sdxl(self):
        from genbox.piplen_video.pipeline_img2img import select_img2img_pipeline_class
        self.assertEqual(select_img2img_pipeline_class("sdxl"),
                         "StableDiffusionXLImg2ImgPipeline")

    def test_sd35(self):
        from genbox.piplen_video.pipeline_img2img import select_img2img_pipeline_class
        self.assertEqual(select_img2img_pipeline_class("sd35"),
                         "StableDiffusion3Img2ImgPipeline")

    def test_flux(self):
        from genbox.piplen_video.pipeline_img2img import select_img2img_pipeline_class
        self.assertEqual(select_img2img_pipeline_class("flux"),
                         "FluxImg2ImgPipeline")

    def test_unknown_raises(self):
        from genbox.piplen_video.pipeline_img2img import select_img2img_pipeline_class
        with self.assertRaises(ValueError):
            select_img2img_pipeline_class("unknown_arch")


class TestBuildImg2ImgCallKwargs(unittest.TestCase):
    def test_strength_included(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_call_kwargs
        img = MagicMock()
        kwargs = build_img2img_call_kwargs(
            architecture="sdxl", prompt="cat", negative_prompt="ugly",
            image=img, strength=0.8,
            steps=30, guidance_scale=7.5, generator=None,
        )
        self.assertAlmostEqual(kwargs["strength"], 0.8)
        self.assertIs(kwargs["image"], img)

    def test_flux_no_negative_prompt(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_call_kwargs
        kwargs = build_img2img_call_kwargs(
            architecture="flux", prompt="cat", negative_prompt="ugly",
            image=MagicMock(), strength=0.9,
            steps=28, guidance_scale=3.5, generator=None,
        )
        self.assertNotIn("negative_prompt", kwargs)

    def test_sdxl_includes_negative_prompt(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_call_kwargs
        kwargs = build_img2img_call_kwargs(
            architecture="sdxl", prompt="cat", negative_prompt="blur",
            image=MagicMock(), strength=0.7,
            steps=30, guidance_scale=7.5, generator=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "blur")

    def test_steps_passed_through(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_call_kwargs
        kwargs = build_img2img_call_kwargs(
            architecture="sd15", prompt="p", negative_prompt="n",
            image=MagicMock(), strength=0.6,
            steps=42, guidance_scale=7.5, generator=None,
        )
        self.assertEqual(kwargs["num_inference_steps"], 42)


class TestImg2ImgOutputMeta(unittest.TestCase):
    def test_required_keys(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_output_meta
        meta = build_img2img_output_meta(
            architecture="sdxl", model_id="sdxl_base",
            prompt="cat", negative_prompt="", input_image=Path("/tmp/in.png"),
            width=1024, height=1024, strength=0.75,
            steps=30, guidance_scale=7.5, seed=0,
            lora_specs=[], accel=[], sampler="default",
            elapsed_s=2.0, output_path=Path("/tmp/out.png"),
        )
        for k in ("pipeline", "model", "prompt", "strength", "input_image",
                  "seed", "elapsed_s"):
            self.assertIn(k, meta)

    def test_pipeline_name_is_img2img(self):
        from genbox.piplen_video.pipeline_img2img import build_img2img_output_meta
        meta = build_img2img_output_meta(
            architecture="flux", model_id="m", prompt="p", negative_prompt="",
            input_image=Path("/tmp/in.png"), width=1024, height=1024,
            strength=0.8, steps=28, guidance_scale=3.5, seed=0,
            lora_specs=[], accel=[], sampler="d",
            elapsed_s=1.0, output_path=Path("/tmp/out.png"),
        )
        self.assertIn("img2img", meta["pipeline"])


# ══════════════════════════════════════════════════════════════════════════════
# pipeline_inpaint.py — mask utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadMask(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_mask_png(self, name="mask.png"):
        """Create a real white-on-black PIL mask PNG."""
        try:
            from PIL import Image as PILImage
            img = PILImage.new("L", (64, 64), color=0)
            path = self.base / name
            img.save(str(path))
            return path
        except ImportError:
            self.skipTest("PIL not available")

    def test_path_returns_pil_image(self):
        from genbox.pipline_image.pipeline_inpaint import load_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        p = self._make_mask_png()
        result = load_mask(p, target_size=(64, 64))
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(result.mode, "L")

    def test_pil_image_passthrough_resized(self):
        from genbox.pipline_image.pipeline_inpaint import load_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        img = PILImage.new("RGB", (128, 128), color=255)
        result = load_mask(img, target_size=(64, 64))
        self.assertEqual(result.size, (64, 64))


class TestBlurMask(unittest.TestCase):
    def test_blur_radius_0_unchanged(self):
        from genbox.pipline_image.pipeline_inpaint import blur_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        mask = PILImage.new("L", (64, 64), color=255)
        result = blur_mask(mask, radius=0)
        self.assertEqual(result.size, mask.size)
        self.assertEqual(result.mode, "L")

    def test_blur_positive_radius(self):
        from genbox.pipline_image.pipeline_inpaint import blur_mask
        try:
            from PIL import Image as PILImage, ImageFilter
        except ImportError:
            self.skipTest("PIL not available")
        mask = PILImage.new("L", (64, 64), color=128)
        result = blur_mask(mask, radius=5)
        # Should return an L-mode image of same size
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(result.mode, "L")

    def test_returns_l_mode(self):
        from genbox.pipline_image.pipeline_inpaint import blur_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        mask = PILImage.new("RGB", (32, 32), color=(255, 0, 0))
        result = blur_mask(mask, radius=2)
        self.assertEqual(result.mode, "L")


class TestDilateMask(unittest.TestCase):
    def test_dilate_0_unchanged(self):
        from genbox.pipline_image.pipeline_inpaint import dilate_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        mask = PILImage.new("L", (32, 32), color=200)
        result = dilate_mask(mask, pixels=0)
        self.assertEqual(result.size, mask.size)

    def test_dilate_returns_l_mode(self):
        from genbox.pipline_image.pipeline_inpaint import dilate_mask
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        mask = PILImage.new("L", (32, 32), color=128)
        result = dilate_mask(mask, pixels=4)
        self.assertEqual(result.mode, "L")


class TestSelectInpaintPipelineClass(unittest.TestCase):
    def test_sd15(self):
        from genbox.pipline_image.pipeline_inpaint import select_inpaint_pipeline_class
        self.assertEqual(select_inpaint_pipeline_class("sd15"),
                         "StableDiffusionInpaintPipeline")

    def test_sdxl(self):
        from genbox.pipline_image.pipeline_inpaint import select_inpaint_pipeline_class
        self.assertEqual(select_inpaint_pipeline_class("sdxl"),
                         "StableDiffusionXLInpaintPipeline")

    def test_sd35(self):
        from genbox.pipline_image.pipeline_inpaint import select_inpaint_pipeline_class
        self.assertEqual(select_inpaint_pipeline_class("sd35"),
                         "StableDiffusion3InpaintPipeline")

    def test_flux(self):
        from genbox.pipline_image.pipeline_inpaint import select_inpaint_pipeline_class
        self.assertEqual(select_inpaint_pipeline_class("flux"),
                         "FluxInpaintPipeline")

    def test_unknown_raises(self):
        from genbox.pipline_image.pipeline_inpaint import select_inpaint_pipeline_class
        with self.assertRaises(ValueError):
            select_inpaint_pipeline_class("wan")


class TestInpaintConfig(unittest.TestCase):
    def test_defaults(self):
        from genbox.pipline_image.pipeline_inpaint import InpaintConfig
        cfg = InpaintConfig(model_id="sdxl_base", architecture="sdxl")
        self.assertAlmostEqual(cfg.strength, 0.99)
        self.assertEqual(cfg.blur_radius, 0)
        self.assertEqual(cfg.dilate_pixels, 0)

    def test_mask_mode_white_inpaint(self):
        from genbox.pipline_image.pipeline_inpaint import InpaintConfig
        cfg = InpaintConfig(model_id="m", architecture="sd15")
        # Default: white = inpaint, black = keep
        self.assertEqual(cfg.mask_mode, "white_inpaint")

    def test_blur_and_dilate_stored(self):
        from genbox.pipline_image.pipeline_inpaint import InpaintConfig
        cfg = InpaintConfig(
            model_id="m", architecture="sdxl",
            blur_radius=5, dilate_pixels=8,
        )
        self.assertEqual(cfg.blur_radius, 5)
        self.assertEqual(cfg.dilate_pixels, 8)


class TestBuildInpaintCallKwargs(unittest.TestCase):
    def test_mask_image_included(self):
        from genbox.pipline_image.pipeline_inpaint import build_inpaint_call_kwargs
        img = MagicMock()
        mask = MagicMock()
        kwargs = build_inpaint_call_kwargs(
            architecture="sdxl",
            prompt="cat", negative_prompt="",
            image=img, mask_image=mask,
            width=1024, height=1024, strength=0.99,
            steps=30, guidance_scale=7.5, generator=None,
        )
        self.assertIs(kwargs["mask_image"], mask)
        self.assertIs(kwargs["image"], img)

    def test_flux_no_negative_prompt(self):
        from genbox.pipline_image.pipeline_inpaint import build_inpaint_call_kwargs
        kwargs = build_inpaint_call_kwargs(
            architecture="flux",
            prompt="cat", negative_prompt="ugly",
            image=MagicMock(), mask_image=MagicMock(),
            width=1024, height=1024, strength=0.95,
            steps=28, guidance_scale=3.5, generator=None,
        )
        self.assertNotIn("negative_prompt", kwargs)


class TestInpaintOutputMeta(unittest.TestCase):
    def test_has_mask_info(self):
        from genbox.pipline_image.pipeline_inpaint import build_inpaint_output_meta
        meta = build_inpaint_output_meta(
            architecture="sdxl", model_id="sdxl_base",
            prompt="cat", negative_prompt="",
            input_image=Path("/tmp/in.png"), mask_image=Path("/tmp/mask.png"),
            width=1024, height=1024, strength=0.99,
            blur_radius=5, dilate_pixels=8, mask_mode="white_inpaint",
            steps=30, guidance_scale=7.5, seed=0,
            lora_specs=[], accel=[], sampler="default",
            elapsed_s=2.0, output_path=Path("/tmp/out.png"),
        )
        self.assertIn("mask_image", meta)
        self.assertEqual(meta["blur_radius"], 5)
        self.assertEqual(meta["dilate_pixels"], 8)
        self.assertEqual(meta["mask_mode"], "white_inpaint")
        self.assertIn("inpaint", meta["pipeline"])


# ══════════════════════════════════════════════════════════════════════════════
# pipeline_outpaint.py
# ══════════════════════════════════════════════════════════════════════════════

class TestOutpaintCanvasExpansion(unittest.TestCase):
    def test_expand_all_sides(self):
        from genbox.pipline_image.pipeline_outpaint import expand_canvas
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        img = PILImage.new("RGB", (512, 512), color=(128, 64, 32))
        result, mask = expand_canvas(
            img, left=64, right=64, top=64, bottom=64, fill_color=(0, 0, 0)
        )
        self.assertEqual(result.size, (640, 640))
        self.assertEqual(mask.size, (640, 640))

    def test_expand_right_only(self):
        from genbox.pipline_image.pipeline_outpaint import expand_canvas
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        img = PILImage.new("RGB", (512, 512))
        result, mask = expand_canvas(img, left=0, right=128, top=0, bottom=0)
        self.assertEqual(result.size, (640, 512))

    def test_mask_is_l_mode(self):
        from genbox.pipline_image.pipeline_outpaint import expand_canvas
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        img = PILImage.new("RGB", (64, 64))
        _, mask = expand_canvas(img, left=16, right=16, top=16, bottom=16)
        self.assertEqual(mask.mode, "L")

    def test_mask_white_in_expanded_region(self):
        from genbox.pipline_image.pipeline_outpaint import expand_canvas
        try:
            from PIL import Image as PILImage
            import numpy as np
        except ImportError:
            self.skipTest("PIL/numpy not available")
        img = PILImage.new("RGB", (64, 64), color=(100, 100, 100))
        _, mask = expand_canvas(img, left=0, right=32, top=0, bottom=0)
        arr = np.array(mask)
        # Right 32 pixels should be white (255 = inpaint)
        self.assertTrue((arr[:, 64:] == 255).all())
        # Original area should be black (0 = keep)
        self.assertTrue((arr[:, :64] == 0).all())

    def test_zero_expansion_unchanged(self):
        from genbox.pipline_image.pipeline_outpaint import expand_canvas
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest("PIL not available")
        img = PILImage.new("RGB", (128, 128))
        result, mask = expand_canvas(img, left=0, right=0, top=0, bottom=0)
        self.assertEqual(result.size, (128, 128))


class TestOutpaintConfig(unittest.TestCase):
    def test_defaults(self):
        from genbox.pipline_image.pipeline_outpaint import OutpaintConfig
        cfg = OutpaintConfig(model_id="sdxl_base", architecture="sdxl")
        self.assertEqual(cfg.left, 0)
        self.assertEqual(cfg.right, 0)
        self.assertEqual(cfg.top, 0)
        self.assertEqual(cfg.bottom, 0)
        self.assertEqual(cfg.feather_radius, 16)
        self.assertAlmostEqual(cfg.strength, 0.99)

    def test_expansion_values_stored(self):
        from genbox.pipline_image.pipeline_outpaint import OutpaintConfig
        cfg = OutpaintConfig(
            model_id="m", architecture="sdxl",
            left=64, right=128, top=32, bottom=0,
        )
        self.assertEqual(cfg.left, 64)
        self.assertEqual(cfg.right, 128)

    def test_total_expansion_property(self):
        from genbox.pipline_image.pipeline_outpaint import OutpaintConfig
        cfg = OutpaintConfig(
            model_id="m", architecture="sdxl",
            left=32, right=32, top=0, bottom=0,
        )
        self.assertEqual(cfg.total_horizontal, 64)
        self.assertEqual(cfg.total_vertical, 0)


class TestOutpaintMaskFeathering(unittest.TestCase):
    def test_feathered_mask_has_gradient(self):
        from genbox.pipline_image.pipeline_outpaint import feather_mask
        try:
            from PIL import Image as PILImage
            import numpy as np
        except ImportError:
            self.skipTest("PIL/numpy not available")
        # mask: right half = white (255), left half = black (0)
        mask = PILImage.new("L", (128, 64), color=0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([64, 0, 127, 63], fill=255)

        result = feather_mask(mask, radius=8)
        arr = np.array(result)
        # Should have intermediate values at the boundary
        mid_col = arr[:, 60:68]
        has_intermediate = ((mid_col > 0) & (mid_col < 255)).any()
        self.assertTrue(has_intermediate)

    def test_feather_radius_0_unchanged(self):
        from genbox.pipline_image.pipeline_outpaint import feather_mask
        try:
            from PIL import Image as PILImage
            import numpy as np
        except ImportError:
            self.skipTest("PIL/numpy not available")
        mask = PILImage.new("L", (64, 64), color=255)
        result = feather_mask(mask, radius=0)
        arr = np.array(result)
        self.assertTrue((arr == 255).all())


class TestOutpaintOutputMeta(unittest.TestCase):
    def test_has_expansion_info(self):
        from genbox.pipline_image.pipeline_outpaint import build_outpaint_output_meta
        meta = build_outpaint_output_meta(
            architecture="sdxl", model_id="sdxl_base",
            prompt="cat", negative_prompt="",
            input_image=Path("/tmp/in.png"),
            left=64, right=64, top=0, bottom=0,
            feather_radius=16,
            original_size=(512, 512), canvas_size=(640, 512),
            strength=0.99, steps=30, guidance_scale=7.5, seed=0,
            lora_specs=[], accel=[], sampler="default",
            elapsed_s=2.5, output_path=Path("/tmp/out.png"),
        )
        self.assertEqual(meta["expand_left"], 64)
        self.assertEqual(meta["expand_right"], 64)
        self.assertEqual(meta["original_size"], (512, 512))
        self.assertEqual(meta["canvas_size"], (640, 512))
        self.assertIn("outpaint", meta["pipeline"])


# ══════════════════════════════════════════════════════════════════════════════
# pipeline_img2video.py
# ══════════════════════════════════════════════════════════════════════════════

class TestImg2VideoConfig(unittest.TestCase):
    def test_wan_defaults(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        cfg = Img2VideoConfig(model_id="wan_1_3b", backend="wan")
        self.assertEqual(cfg.backend, "wan")
        self.assertEqual(cfg.width, 832)
        self.assertEqual(cfg.height, 480)

    def test_ltx_defaults(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        cfg = Img2VideoConfig(model_id="ltx2_fp8", backend="ltx")
        self.assertEqual(cfg.backend, "ltx")
        self.assertAlmostEqual(cfg.guidance_scale, 5.0)

    def test_frames_snap_wan(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        cfg = Img2VideoConfig(model_id="wan_1_3b", backend="wan", frames=80)
        # Should snap to 4n+1
        self.assertEqual((cfg.frames - 1) % 4, 0)

    def test_frames_snap_ltx(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        cfg = Img2VideoConfig(model_id="ltx2_fp8", backend="ltx", frames=96)
        # Should snap to 8n+1
        self.assertEqual((cfg.frames - 1) % 8, 0)

    def test_image_path_stored(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        p = Path("/tmp/frame.png")
        cfg = Img2VideoConfig(model_id="wan_1_3b", backend="wan", image=p)
        self.assertEqual(cfg.image, p)

    def test_unknown_backend_raises(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        with self.assertRaises(ValueError):
            Img2VideoConfig(model_id="m", backend="unsupported_backend")

    def test_loras_empty_default(self):
        from genbox.piplen_video.pipeline_img2video import Img2VideoConfig
        cfg = Img2VideoConfig(model_id="wan_1_3b", backend="wan")
        self.assertEqual(cfg.loras, [])


class TestImg2VideoBackendDetection(unittest.TestCase):
    def test_detect_wan_from_entry(self):
        from genbox.piplen_video.pipeline_img2video import detect_i2v_backend
        entry = _wan_entry()
        self.assertEqual(detect_i2v_backend(entry), "wan")

    def test_detect_ltx_from_entry(self):
        from genbox.piplen_video.pipeline_img2video import detect_i2v_backend
        entry = _ltx_entry()
        self.assertEqual(detect_i2v_backend(entry), "ltx")

    def test_unknown_architecture_raises(self):
        from genbox.piplen_video.pipeline_img2video import detect_i2v_backend
        entry = _entry(architecture="sdxl", type="image")
        with self.assertRaises(ValueError):
            detect_i2v_backend(entry)


class TestImg2VideoConfigFromEntry(unittest.TestCase):
    """build_i2v_config_from_entry creates the right sub-config."""

    def test_wan_config_returned(self):
        from genbox.piplen_video.pipeline_img2video import build_i2v_config_from_entry, Img2VideoConfig
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig
        entry = _wan_entry()
        cfg = build_i2v_config_from_entry(
            Img2VideoConfig(model_id="wan_1_3b", backend="wan"),
            entry=entry,
        )
        self.assertIsInstance(cfg, WanPipelineConfig)
        self.assertEqual(cfg.mode, "i2v")

    def test_ltx_config_returned(self):
        from genbox.piplen_video.pipeline_img2video import build_i2v_config_from_entry, Img2VideoConfig
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig
        entry = _ltx_entry()
        cfg = build_i2v_config_from_entry(
            Img2VideoConfig(model_id="ltx2_fp8", backend="ltx"),
            entry=entry,
        )
        self.assertIsInstance(cfg, LtxPipelineConfig)
        self.assertEqual(cfg.mode, "i2v")

    def test_wan_config_has_image(self):
        from genbox.piplen_video.pipeline_img2video import build_i2v_config_from_entry, Img2VideoConfig
        entry = _wan_entry()
        img_path = Path("/tmp/frame.png")
        i2v = Img2VideoConfig(model_id="wan_1_3b", backend="wan", image=img_path)
        cfg = build_i2v_config_from_entry(i2v, entry=entry)
        self.assertEqual(cfg.image, img_path)


class TestImg2VideoOutputMeta(unittest.TestCase):
    def test_meta_has_backend(self):
        from genbox.piplen_video.pipeline_img2video import build_i2v_output_meta
        meta = build_i2v_output_meta(
            backend="wan", model_id="wan_1_3b",
            prompt="a cat", negative_prompt="",
            input_image=Path("/tmp/frame.png"),
            width=832, height=480, frames=81, fps=16,
            steps=50, guidance_scale=5.0, seed=42,
            lora_specs=[], accel=[], elapsed_s=10.0,
            output_path=Path("/tmp/v.mp4"),
        )
        self.assertEqual(meta["backend"], "wan")
        self.assertIn("img2video", meta["pipeline"])
        import platform
        self.assertEqual(str(meta["input_image"]),  "\\tmp\\frame.png" if  platform.system() == "Windows" else "/tmp/frame.png")

    def test_ltx_meta(self):
        from genbox.piplen_video.pipeline_img2video import build_i2v_output_meta
        meta = build_i2v_output_meta(
            backend="ltx", model_id="ltx2_fp8",
            prompt="motion", negative_prompt="blur",
            input_image=Path("/tmp/start.png"),
            width=768, height=512, frames=97, fps=24,
            steps=50, guidance_scale=5.0, seed=0,
            lora_specs=[], accel=[], elapsed_s=5.0,
            output_path=Path("/tmp/v.mp4"),
        )
        self.assertEqual(meta["backend"], "ltx")


# ── Apply accelerators contract: all 4 new pipelines expose it ────────────────

class TestApplyAcceleratorsContract(unittest.TestCase):
    def _pipe(self):
        p = MagicMock()
        p.enable_model_cpu_offload = MagicMock()
        p.enable_sequential_cpu_offload = MagicMock()
        p.to = MagicMock()
        return p

    def test_img2img_exposes_apply_pipeline_accelerators(self):
        from genbox.piplen_video.pipeline_img2img import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_inpaint_exposes_apply_pipeline_accelerators(self):
        from genbox.pipline_image.pipeline_inpaint import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_outpaint_exposes_apply_pipeline_accelerators(self):
        from genbox.pipline_image.pipeline_outpaint import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_img2video_exposes_apply_pipeline_accelerators(self):
        from genbox.piplen_video.pipeline_img2video import apply_pipeline_accelerators
        self.assertTrue(callable(apply_pipeline_accelerators))

    def test_img2img_model_offload_cuda(self):
        from genbox.piplen_video.pipeline_img2img import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cuda", vram_gb=12)
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_inpaint_cpu_device(self):
        from genbox.pipline_image.pipeline_inpaint import apply_pipeline_accelerators
        pipe = self._pipe()
        apply_pipeline_accelerators(pipe, device="cpu", vram_gb=0)
        pipe.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
