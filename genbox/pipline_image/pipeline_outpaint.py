"""
genbox/pipeline_outpaint.py
Outpainting — extend an image beyond its original borders.

Workflow:
  1. expand_canvas: paste original onto larger canvas, create binary border mask
  2. feather_mask:  blur the mask seam for smooth transition
  3. delegate to pipeline_inpaint.inpaint() for fill generation

Supports all architectures available in pipeline_inpaint (flux, sd15, sdxl, sd35).
All mask operations are pure PIL — no torch at mask-generation time.

100 % offline after installation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from genbox.utils.utils_image_pipeline import (
    apply_loras_to_pipe,
    build_lora_adapter_list,
    build_output_meta,
    build_output_path,
    make_generator,
    resolve_device,
    resolve_dtype,
    resolve_seed,
    set_scheduler,
)
from genbox.pipline_image.pipeline_inpaint import (
    apply_pipeline_accelerators as _inpaint_accels,
    build_inpaint_call_kwargs,
    load_inpaint_pipe,
)

log = logging.getLogger("genbox.pipeline_outpaint")


# ── Canvas expansion (pure PIL) ────────────────────────────────────────────────

def expand_canvas(
    image: "PIL.Image.Image",
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple["PIL.Image.Image", "PIL.Image.Image"]:
    """
    Expand the image canvas by adding pixels on each side.

    Returns:
        (canvas, mask)
        canvas: RGB image with original pasted at offset, borders filled with fill_color
        mask:   L-mode image, white (255) = expanded region, black (0) = original area

    The mask uses the standard diffusers inpaint convention: white = regenerate.
    """
    from PIL import Image as PILImage  # type: ignore

    orig_w, orig_h = image.size
    new_w  = orig_w + left + right
    new_h  = orig_h + top + bottom

    canvas = PILImage.new("RGB", (new_w, new_h), color=fill_color)
    canvas.paste(image.convert("RGB"), (left, top))

    # Mask: white everywhere, then black over original
    mask = PILImage.new("L", (new_w, new_h), color=255)
    from PIL import ImageDraw  # type: ignore
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [left, top, left + orig_w - 1, top + orig_h - 1],
        fill=0,
    )

    return canvas, mask


def feather_mask(
    mask: "PIL.Image.Image",
    radius: float,
) -> "PIL.Image.Image":
    """
    Apply Gaussian blur to mask edges for smooth seam blending.
    radius=0 → unchanged.
    Returns "L" mode PIL image.
    """
    from PIL import ImageFilter  # type: ignore

    if radius <= 0:
        return mask.convert("L")

    return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class OutpaintConfig:
    model_id:        str
    architecture:    str
    prompt:          str   = ""
    negative_prompt: str   = ""
    input_image:     Optional[Union[str, Path]] = None
    # Expansion amounts in pixels
    left:            int   = 0
    right:           int   = 0
    top:             int   = 0
    bottom:          int   = 0
    # Mask feathering at the seam
    feather_radius:  float = 16.0
    # Inpaint strength
    strength:        float = 0.99
    steps:           int   = 30
    guidance_scale:  float = 7.5
    seed:            int   = -1
    sampler:         str   = "default"
    t5_mode:         str   = "fp16"
    loras:           list  = field(default_factory=list)
    accel:           list  = field(default_factory=list)
    output:          Optional[Union[str, Path]] = None

    @property
    def total_horizontal(self) -> int:
        return self.left + self.right

    @property
    def total_vertical(self) -> int:
        return self.top + self.bottom


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_outpaint_output_meta(
    architecture: str, model_id: str,
    prompt: str, negative_prompt: str,
    input_image: Union[str, Path],
    left: int, right: int, top: int, bottom: int,
    feather_radius: float,
    original_size: tuple[int, int],
    canvas_size: tuple[int, int],
    strength: float,
    steps: int, guidance_scale: float, seed: int,
    lora_specs: list, accel: list, sampler: str,
    elapsed_s: float, output_path: Path,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_output_meta(
        pipeline_name=f"outpaint_{architecture}",
        model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
        width=canvas_size[0], height=canvas_size[1], steps=steps,
        guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["input_image"]   = str(input_image)
    meta["expand_left"]   = left
    meta["expand_right"]  = right
    meta["expand_top"]    = top
    meta["expand_bottom"] = bottom
    meta["feather_radius"] = feather_radius
    meta["original_size"] = original_size
    meta["canvas_size"]   = canvas_size
    meta["strength"]      = strength
    meta["architecture"]  = architecture
    if extra:
        meta.update(extra)
    return meta


# ── Accelerators (proxy to inpaint) ───────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe, device: str, vram_gb: int = 16,
    accel: Optional[list] = None,
    has_quantized_encoders: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """Proxy to inpaint accelerator — outpaint reuses the inpaint pipeline."""
    _inpaint_accels(
        pipe, device=device, vram_gb=vram_gb, accel=accel,
        has_quantized_encoders=has_quantized_encoders,
        env_override=env_override,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def outpaint(
    cfg: OutpaintConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
    tracker=None,            # Optional[GenProgressTracker]
    enable_preview: bool = True,
    preview_interval: int = 5,
) -> dict:
    """
    Run outpainting — extend the image on any combination of sides.

    tracker: GenProgressTracker — forwarded to the inpaint call.

    Returns dict with keys: output_path, metadata, elapsed_s
    """
    import time
    from diffusers.utils import load_image  # type: ignore

    if cfg.input_image is None:
        raise ValueError("OutpaintConfig.input_image must be set")
    if cfg.total_horizontal == 0 and cfg.total_vertical == 0:
        raise ValueError("OutpaintConfig: at least one expansion direction must be > 0")

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    log.info(f"Outpaint | arch={cfg.architecture} model={cfg.model_id} "
             f"expand L{cfg.left} R{cfg.right} T{cfg.top} B{cfg.bottom} seed={seed}")

    if tracker is not None:
        tracker.set_stage("loading model")

    original  = load_image(str(cfg.input_image)).convert("RGB")
    orig_size = original.size

    canvas, raw_mask = expand_canvas(
        original,
        left=cfg.left, right=cfg.right,
        top=cfg.top, bottom=cfg.bottom,
    )
    mask        = feather_mask(raw_mask, radius=cfg.feather_radius)
    canvas_size = canvas.size

    pipe = load_inpaint_pipe(entry, models_dir, dtype, t5_mode=cfg.t5_mode)
    set_scheduler(pipe, cfg.architecture, cfg.sampler)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture=cfg.architecture)

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
        has_quantized_encoders=(cfg.t5_mode == "int8"),
    )

    # Callback wired via build_inpaint_call_kwargs — arch-aware
    from genbox.utils.utils_image_pipeline import (
        make_flux_step_callback, make_sdl_step_callback,
    )
    step_callback = None
    if tracker is not None:
        if cfg.architecture == "flux":
            step_callback = make_flux_step_callback(
                tracker=tracker,
                height=canvas_size[1],
                width=canvas_size[0],
                preview_interval=preview_interval,
                enable_preview=enable_preview,
            )
        else:
            step_callback = make_sdl_step_callback(
                tracker=tracker,
                preview_interval=preview_interval,
                enable_preview=enable_preview,
            )

    gen    = make_generator(seed, device)
    kwargs = build_inpaint_call_kwargs(
        architecture=cfg.architecture,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        image=canvas, mask_image=mask,
        width=canvas_size[0], height=canvas_size[1],
        strength=cfg.strength,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen,
        callback_on_step_end=step_callback,
    )

    if tracker is not None:
        tracker.set_stage("denoising")

    kwargs.pop("strength", None)  # not supported by all variants
    result   = pipe(**kwargs)
    image    = result.images[0]

    if tracker is not None:
        tracker.set_stage("saving")

    _out_dir = Path(outputs_dir) if outputs_dir else Path.cwd() / "genbox_outputs"
    out_path = build_output_path("outp", cfg.model_id, seed, "png",
                                 outputs_dir=_out_dir, custom=cfg.output)
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_outpaint_output_meta(
        architecture=cfg.architecture, model_id=cfg.model_id,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        input_image=cfg.input_image,
        left=cfg.left, right=cfg.right, top=cfg.top, bottom=cfg.bottom,
        feather_radius=cfg.feather_radius,
        original_size=orig_size, canvas_size=canvas_size,
        strength=cfg.strength,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=seed,
        lora_specs=cfg.loras, accel=cfg.accel, sampler=cfg.sampler,
        elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"Outpaint done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
