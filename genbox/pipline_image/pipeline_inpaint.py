"""
genbox/pipeline_inpaint.py
Inpainting for all supported architectures.

Supported backends:
  flux  → FluxInpaintPipeline
  sd15  → StableDiffusionInpaintPipeline
  sdxl  → StableDiffusionXLInpaintPipeline
  sd35  → StableDiffusion3InpaintPipeline

Mask convention (default): white (255) = inpaint, black (0) = keep original.
Can be inverted via mask_mode="black_inpaint".

Mask utilities (no-torch, pure PIL):
  load_mask(path_or_pil, target_size)   → PIL "L" image
  blur_mask(mask, radius)               → Gaussian-blurred edges for soft transitions
  dilate_mask(mask, pixels)             → expand mask region outward
  All helpers return PIL "L" mode images.

100 % offline after installation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from genbox.utils.utils_image_pipeline import (
    apply_accelerators,
    apply_loras_to_pipe,
    build_lora_adapter_list,
    build_output_meta,
    build_output_path,
    inject_compile,
    make_generator,
    resolve_device,
    resolve_dtype,
    resolve_offload_mode,
    resolve_seed,
    set_scheduler, make_flux_step_callback, make_sdl_step_callback,
)
from genbox.pipline_image.pipeline_flux import load_flux_pipe
from genbox.pipline_image.pipeline_sdl import _resolve_sdl_local_path

log = logging.getLogger("genbox.pipeline_inpaint")

# ── Pipeline class map ─────────────────────────────────────────────────────────

_INPAINT_CLASS_MAP: dict[str, str] = {
    "flux": "FluxInpaintPipeline",
    "sd15": "StableDiffusionInpaintPipeline",
    "sdxl": "StableDiffusionXLInpaintPipeline",
    "sd35": "StableDiffusion3InpaintPipeline",
}


def select_inpaint_pipeline_class(architecture: str) -> str:
    cls = _INPAINT_CLASS_MAP.get(architecture)
    if cls is None:
        raise ValueError(
            f"Inpainting not supported for architecture {architecture!r}. "
            f"Supported: {list(_INPAINT_CLASS_MAP)}"
        )
    return cls


# ── Mask utilities (pure PIL, no torch) ───────────────────────────────────────

def load_mask(
    mask: Union[str, Path, "PIL.Image.Image"],
    target_size: Optional[tuple[int, int]] = None,
) -> "PIL.Image.Image":
    """
    Load a mask image and convert to "L" (greyscale) mode.
    Optionally resize to target_size (w, h).
    Accepts: file path (str/Path) or PIL Image.
    """
    from PIL import Image as PILImage  # type: ignore

    if isinstance(mask, (str, Path)):
        img = PILImage.open(str(mask)).convert("L")
    else:
        img = mask.convert("L")

    if target_size is not None:
        img = img.resize(target_size, PILImage.LANCZOS)

    return img


def blur_mask(
    mask: "PIL.Image.Image",
    radius: float,
) -> "PIL.Image.Image":
    """
    Apply Gaussian blur to a mask for soft/feathered edges.
    radius=0 → no change.
    Returns "L" mode PIL image.
    """
    from PIL import Image as PILImage, ImageFilter  # type: ignore

    if hasattr(mask, "convert"):
        mask = mask.convert("L")
    else:
        mask = PILImage.fromarray(mask).convert("L")

    if radius <= 0:
        return mask

    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def dilate_mask(
    mask: "PIL.Image.Image",
    pixels: int,
) -> "PIL.Image.Image":
    """
    Dilate (expand) the white regions of a mask by `pixels`.
    pixels=0 → no change.
    Uses PIL MaxFilter as a simple morphological dilation.
    Returns "L" mode PIL image.
    """
    from PIL import Image as PILImage, ImageFilter  # type: ignore

    if hasattr(mask, "convert"):
        mask = mask.convert("L")
    else:
        mask = PILImage.fromarray(mask).convert("L")

    if pixels <= 0:
        return mask

    # MaxFilter size: odd, ≈ 2*pixels+1
    size = max(3, 2 * pixels + 1)
    # Ensure odd
    if size % 2 == 0:
        size += 1
    return mask.filter(ImageFilter.MaxFilter(size=size))


def prepare_mask(
    mask: Union[str, Path, "PIL.Image.Image"],
    target_size: tuple[int, int],
    blur_radius: float = 0,
    dilate_pixels: int = 0,
    mask_mode: str = "white_inpaint",
) -> "PIL.Image.Image":
    """
    Full mask pipeline: load → dilate → blur → optionally invert.

    mask_mode:
      "white_inpaint"  → white=inpaint, black=keep (diffusers default)
      "black_inpaint"  → invert: black=inpaint, white=keep
    """
    from PIL import ImageOps  # type: ignore

    m = load_mask(mask, target_size=target_size)

    if dilate_pixels > 0:
        m = dilate_mask(m, pixels=dilate_pixels)

    if blur_radius > 0:
        m = blur_mask(m, radius=blur_radius)

    if mask_mode == "black_inpaint":
        m = ImageOps.invert(m)

    return m


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class InpaintConfig:
    model_id:        str
    architecture:    str
    prompt:          str   = ""
    negative_prompt: str   = ""
    input_image:     Optional[Union[str, Path]] = None
    mask_image:      Optional[Union[str, Path]] = None
    width:           int   = 1024
    height:          int   = 1024
    strength:        float = 0.99
    steps:           int   = 30
    guidance_scale:  float = 7.5
    seed:            int   = -1
    sampler:         str   = "default"
    t5_mode:         str   = "fp16"
    # Mask processing
    blur_radius:     float = 0
    dilate_pixels:   int   = 0
    mask_mode:       str   = "white_inpaint"  # "white_inpaint" | "black_inpaint"
    loras:           list  = field(default_factory=list)
    accel:           list  = field(default_factory=list)
    output:          Optional[Union[str, Path]] = None


# ── Call kwargs ────────────────────────────────────────────────────────────────

def build_inpaint_call_kwargs(
    architecture: str,
    prompt: str,
    negative_prompt: str,
    image,
    mask_image,
    width: int,
    height: int,
    strength: float,
    steps: int,
    guidance_scale: float,
    generator,
    callback_on_step_end=None,
    callback_tensor_inputs: Optional[list] = None,
    extra: Optional[dict] = None,
) -> dict:
    kwargs: dict = dict(
        prompt              = prompt,
        image               = image,
        mask_image          = mask_image,
        width               = width,
        height              = height,
        strength            = strength,
        num_inference_steps = steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )
    if architecture != "flux" and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if callback_on_step_end is not None:
        kwargs["callback_on_step_end"] = callback_on_step_end
        kwargs["callback_on_step_end_tensor_inputs"] = (
            callback_tensor_inputs if callback_tensor_inputs is not None
            else ["latents"]
        )
    if extra:
        kwargs.update(extra)
    return kwargs


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_inpaint_output_meta(
    architecture: str, model_id: str,
    prompt: str, negative_prompt: str,
    input_image: Union[str, Path],
    mask_image: Union[str, Path],
    width: int, height: int, strength: float,
    blur_radius: float, dilate_pixels: int, mask_mode: str,
    steps: int, guidance_scale: float, seed: int,
    lora_specs: list, accel: list, sampler: str,
    elapsed_s: float, output_path: Path,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_output_meta(
        pipeline_name=f"inpaint_{architecture}",
        model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps,
        guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["strength"]      = strength
    meta["input_image"]   = str(input_image)
    meta["mask_image"]    = str(mask_image)
    meta["blur_radius"]   = blur_radius
    meta["dilate_pixels"] = dilate_pixels
    meta["mask_mode"]     = mask_mode
    meta["architecture"]  = architecture
    if extra:
        meta.update(extra)
    return meta


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_inpaint_pipe(entry, models_dir: Union[str, Path], dtype, t5_mode: str = "fp16"):
    """
    Load the inpaint variant of a model.
    FLUX: load T2I then re-wrap via AutoPipelineForInpainting.from_pipe()
    SDL:  load directly with arch-specific inpaint class.
    """
    import diffusers  # type: ignore

    arch = entry.architecture

    if arch == "flux":
        t2i_pipe = load_flux_pipe(entry, models_dir, dtype, t5_mode=t5_mode)
        return diffusers.AutoPipelineForInpainting.from_pipe(t2i_pipe)

    local_path = _resolve_sdl_local_path(entry, models_dir)
    cls_name   = select_inpaint_pipeline_class(arch)
    PipeClass  = getattr(diffusers, cls_name)

    load_kw: dict = dict(torch_dtype=dtype, local_files_only=True, safety_checker=None)
    if entry.full_repo:
        if arch == "sdxl":
            import torch as _t  # type: ignore
            load_kw["use_safetensors"] = True
            if dtype == _t.float16:
                load_kw["variant"] = "fp16"
        pipe = PipeClass.from_pretrained(str(local_path), **load_kw)
    else:
        load_kw.pop("safety_checker", None)
        pipe = PipeClass.from_single_file(str(local_path), **load_kw)

    return pipe


# ── Accelerators ──────────────────────────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe, device: str, vram_gb: int = 16,
    accel: Optional[list] = None,
    has_quantized_encoders: bool = False,
    env_override: Optional[str] = None,
) -> None:
    accel = accel or []
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    offload_mode = resolve_offload_mode(vram_gb, env_override, has_quantized_encoders)
    inject_compile(pipe, accel)
    apply_accelerators(pipe, device=device, offload_mode=offload_mode, accel=accel)



# ── Public API ─────────────────────────────────────────────────────────────────

def inpaint(
    cfg: InpaintConfig,
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
    Run inpainting.
    cfg.input_image and cfg.mask_image must be set.

    tracker: GenProgressTracker — live progress updates when provided.

    Returns dict with keys: output_path, metadata, elapsed_s
    """
    import time
    from diffusers.utils import load_image  # type: ignore

    if cfg.input_image is None or cfg.mask_image is None:
        raise ValueError("InpaintConfig: both input_image and mask_image must be set")

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    log.info(f"Inpaint | arch={cfg.architecture} model={cfg.model_id} "
             f"strength={cfg.strength} blur={cfg.blur_radius} "
             f"dilate={cfg.dilate_pixels} seed={seed}")

    if tracker is not None:
        tracker.set_stage("loading model")

    init_image = load_image(str(cfg.input_image)).convert("RGB")
    init_image = init_image.resize((cfg.width, cfg.height))

    mask = prepare_mask(
        cfg.mask_image,
        target_size=(cfg.width, cfg.height),
        blur_radius=cfg.blur_radius,
        dilate_pixels=cfg.dilate_pixels,
        mask_mode=cfg.mask_mode,
    )

    pipe = load_inpaint_pipe(entry, models_dir, dtype, t5_mode=cfg.t5_mode)
    set_scheduler(pipe, cfg.architecture, cfg.sampler)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture=cfg.architecture)

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
        has_quantized_encoders=(cfg.t5_mode == "int8"),
    )

    step_callback = None
    if tracker is not None:
        if cfg.architecture == "flux":
            step_callback = make_flux_step_callback(
                tracker=tracker,
                height=cfg.height,
                width=cfg.width,
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
        image=init_image, mask_image=mask,
        width=cfg.width, height=cfg.height, strength=cfg.strength,
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
    out_path = build_output_path("inp", cfg.model_id, seed, "png",
                                 outputs_dir=_out_dir, custom=cfg.output)
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_inpaint_output_meta(
        architecture=cfg.architecture, model_id=cfg.model_id,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        input_image=cfg.input_image, mask_image=cfg.mask_image,
        width=cfg.width, height=cfg.height, strength=cfg.strength,
        blur_radius=cfg.blur_radius, dilate_pixels=cfg.dilate_pixels,
        mask_mode=cfg.mask_mode,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=seed,
        lora_specs=cfg.loras, accel=cfg.accel, sampler=cfg.sampler,
        elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"Inpaint done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
