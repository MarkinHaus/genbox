"""
genbox/pipeline_img2img.py
Image-to-image generation for all supported architectures.

Supported backends (re-uses loaders from existing pipeline files):
  flux  → FluxImg2ImgPipeline
  sd15  → StableDiffusionImg2ImgPipeline
  sdxl  → StableDiffusionXLImg2ImgPipeline  (also Pony)
  sd35  → StableDiffusion3Img2ImgPipeline

Key parameter: strength (0.0–1.0)
  0.0 = keep original, 1.0 = ignore original completely
  Typical range: 0.5–0.85 for style transfer, 0.9–0.99 for strong changes

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
    set_scheduler,
)
from genbox.pipline_image.pipeline_flux import (
    load_flux_pipe, )
from genbox.pipline_image.pipeline_sdl import (
    _resolve_sdl_local_path,
)

log = logging.getLogger("genbox.pipeline_img2img")

# ── Architecture defaults ──────────────────────────────────────────────────────

_ARCH_DEFAULTS: dict[str, dict] = {
    "flux": {"width": 1024, "height": 1024, "steps": 28, "guidance_scale": 3.5},
    "sd15": {"width": 512,  "height": 512,  "steps": 20, "guidance_scale": 7.5},
    "sdxl": {"width": 1024, "height": 1024, "steps": 30, "guidance_scale": 7.5},
    "sd35": {"width": 1024, "height": 1024, "steps": 28, "guidance_scale": 4.5},
}

# ── Pipeline class map ─────────────────────────────────────────────────────────

_I2I_CLASS_MAP: dict[str, str] = {
    "flux": "FluxImg2ImgPipeline",
    "sd15": "StableDiffusionImg2ImgPipeline",
    "sdxl": "StableDiffusionXLImg2ImgPipeline",
    "sd35": "StableDiffusion3Img2ImgPipeline",
}


def select_img2img_pipeline_class(architecture: str) -> str:
    cls = _I2I_CLASS_MAP.get(architecture)
    if cls is None:
        raise ValueError(
            f"img2img not supported for architecture {architecture!r}. "
            f"Supported: {list(_I2I_CLASS_MAP)}"
        )
    return cls


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Img2ImgConfig:
    model_id:        str
    architecture:    str
    prompt:          str   = ""
    negative_prompt: str   = ""
    input_image:     Optional[Union[str, Path]] = None
    strength:        float = 0.75
    width:           int   = 0
    height:          int   = 0
    steps:           int   = 0
    guidance_scale:  float = 0.0
    seed:            int   = -1
    sampler:         str   = "default"
    t5_mode:         str   = "fp16"
    loras:           list  = field(default_factory=list)
    accel:           list  = field(default_factory=list)
    output:          Optional[Union[str, Path]] = None

    def __post_init__(self):
        d = _ARCH_DEFAULTS.get(self.architecture, _ARCH_DEFAULTS["sdxl"])
        if self.width         == 0:   self.width          = d["width"]
        if self.height        == 0:   self.height         = d["height"]
        if self.steps         == 0:   self.steps          = d["steps"]
        if self.guidance_scale == 0.0: self.guidance_scale = d["guidance_scale"]
        self.strength = max(0.0, min(1.0, self.strength))


# ── Call kwargs ────────────────────────────────────────────────────────────────

def build_img2img_call_kwargs(
    architecture: str,
    prompt: str,
    negative_prompt: str,
    image,
    strength: float,
    steps: int,
    guidance_scale: float,
    generator,
    extra: Optional[dict] = None,
) -> dict:
    kwargs: dict = dict(
        prompt              = prompt,
        image               = image,
        strength            = strength,
        num_inference_steps = steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )
    if architecture != "flux" and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if extra:
        kwargs.update(extra)
    return kwargs


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_img2img_output_meta(
    architecture: str, model_id: str,
    prompt: str, negative_prompt: str,
    input_image: Union[str, Path],
    width: int, height: int, strength: float,
    steps: int, guidance_scale: float, seed: int,
    lora_specs: list, accel: list, sampler: str,
    elapsed_s: float, output_path: Path,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_output_meta(
        pipeline_name=f"img2img_{architecture}",
        model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps,
        guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["strength"]    = strength
    meta["input_image"] = str(input_image)
    meta["architecture"] = architecture
    if extra:
        meta.update(extra)
    return meta


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_img2img_pipe(entry, models_dir: Union[str, Path], dtype, t5_mode: str = "fp16"):
    """
    Load the img2img variant of a model.

    Strategy:
      FLUX → load full/GGUF pipe then wrap via AutoPipelineForImage2Image.from_pipe()
      SDL  → load directly with the arch-specific img2img class
    """
    import diffusers  # type: ignore

    arch = entry.architecture

    if arch == "flux":
        # Load as T2I first, then re-wrap (shares weights, no extra VRAM)
        t2i_pipe = load_flux_pipe(entry, models_dir, dtype, t5_mode=t5_mode)
        pipe = diffusers.AutoPipelineForImage2Image.from_pipe(t2i_pipe)
        return pipe

    # SDL variants
    local_path = _resolve_sdl_local_path(entry, models_dir)
    cls_name   = select_img2img_pipeline_class(arch)
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
    apply_accelerators(pipe, device=device, offload_mode=offload_mode, accel=accel)
    inject_compile(pipe, accel)


# ── Public API ─────────────────────────────────────────────────────────────────

def image_to_image(
    cfg: Img2ImgConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
) -> dict:
    """
    Run image-to-image generation.
    cfg.input_image must be set to a valid image path.

    Returns dict with keys: output_path, metadata, elapsed_s
    """
    import time
    from diffusers.utils import load_image  # type: ignore

    if cfg.input_image is None:
        raise ValueError("Img2ImgConfig.input_image must be set")

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    log.info(f"I2I | arch={cfg.architecture} model={cfg.model_id} "
             f"strength={cfg.strength} seed={seed} device={device}")

    init_image = load_image(str(cfg.input_image)).convert("RGB")
    init_image = init_image.resize((cfg.width, cfg.height))

    pipe = load_img2img_pipe(entry, models_dir, dtype, t5_mode=cfg.t5_mode)
    set_scheduler(pipe, cfg.architecture, cfg.sampler)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture=cfg.architecture)

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
        has_quantized_encoders=(cfg.t5_mode == "int8"),
    )

    gen    = make_generator(seed, device)
    kwargs = build_img2img_call_kwargs(
        architecture=cfg.architecture,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        image=init_image, strength=cfg.strength,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen,
    )

    result   = pipe(**kwargs)
    image    = result.images[0]

    _out_dir = Path(outputs_dir) if outputs_dir else Path.cwd() / "genbox_outputs"
    out_path = build_output_path("i2i", cfg.model_id, seed, "png",
                                 outputs_dir=_out_dir, custom=cfg.output)
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_img2img_output_meta(
        architecture=cfg.architecture, model_id=cfg.model_id,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        input_image=cfg.input_image,
        width=cfg.width, height=cfg.height, strength=cfg.strength,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=seed,
        lora_specs=cfg.loras, accel=cfg.accel, sampler=cfg.sampler,
        elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"I2I done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
