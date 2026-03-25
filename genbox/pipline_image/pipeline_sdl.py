"""
genbox/pipeline_sdl.py
Unified SDL image pipeline: Stable Diffusion 1.5, SDXL, SD 3.5.

Supports:
  - Full diffusers-format repos for all three variants
  - Custom .safetensors single-file loading
  - Multi-LoRA via PEFT (set_adapters)
  - Full accelerator stack (offload + xformers + torch.compile)
  - Scheduler swapping (DPM++, Euler, DDIM, UniPC, …)

Pony variants are handled in pipeline_pony.py (SDXL subclass with rating tags).
100 % offline after installation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

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

log = logging.getLogger("genbox.pipeline_sdl")

# ── Architecture-specific defaults ────────────────────────────────────────────

_SDL_DEFAULTS: dict[str, dict] = {
    "sd15":  {"width": 512,  "height": 512,  "steps": 20, "guidance_scale": 7.5},
    "sdxl":  {"width": 1024, "height": 1024, "steps": 30, "guidance_scale": 7.5},
    "sd35":  {"width": 1024, "height": 1024, "steps": 28, "guidance_scale": 4.5},
}


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class SDLPipelineConfig:
    """
    Generation config for SD1.5 / SDXL / SD3.5 pipelines.
    Defaults are architecture-aware.
    """
    model_id:       str
    architecture:   str         # "sd15" | "sdxl" | "sd35"
    prompt:         str         = ""
    negative_prompt: str        = ""
    width:          int         = 0    # 0 → arch default
    height:         int         = 0    # 0 → arch default
    steps:          int         = 0    # 0 → arch default
    guidance_scale: float       = 0.0  # 0 → arch default
    seed:           int         = -1
    sampler:        str         = "default"
    loras:          list        = field(default_factory=list)
    accel:          list        = field(default_factory=list)
    output:         Optional[Union[str, Path]] = None

    def __post_init__(self):
        defaults = _SDL_DEFAULTS.get(self.architecture, _SDL_DEFAULTS["sdxl"])
        if self.width      == 0:    self.width          = defaults["width"]
        if self.height     == 0:    self.height         = defaults["height"]
        if self.steps      == 0:    self.steps          = defaults["steps"]
        if self.guidance_scale == 0.0: self.guidance_scale = defaults["guidance_scale"]


# ── Diffusers class selection ─────────────────────────────────────────────────

_SDL_PIPELINE_CLASSES: dict[str, str] = {
    "sd15":  "StableDiffusionPipeline",
    "sdxl":  "StableDiffusionXLPipeline",
    "sd35":  "StableDiffusion3Pipeline",
}


def _select_sdl_pipeline_class(architecture: str) -> str:
    """Return the diffusers pipeline class name for the given architecture."""
    cls = _SDL_PIPELINE_CLASSES.get(architecture)
    if cls is None:
        raise ValueError(
            f"Unknown SDL architecture: {architecture!r}. "
            f"Supported: {list(_SDL_PIPELINE_CLASSES)}"
        )
    return cls


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_sdl_local_path(
    entry, models_dir: Union[str, Path]
) -> tuple[Path, bool]:
    """
    Return (local_path, is_single_file).
    is_single_file=True → .safetensors / .ckpt → use from_single_file.
    is_single_file=False → full diffusers repo dir → use from_pretrained.
    Detection is filesystem-based, not entry.full_repo.
    """
    models_dir = Path(models_dir)
    arch_dir   = models_dir / entry.architecture

    if entry.full_repo:
        p = arch_dir / entry.id
        if not (p / "model_index.json").exists():
            raise FileNotFoundError(
                f"SDL model not found: {p}\n"
                "Download via Models panel."
            )
        return p, False

    p = arch_dir / Path(entry.hf_filename).name
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    # Filesystem-check statt entry.full_repo
    return p, not p.is_dir()


# ── Call kwargs builder ───────────────────────────────────────────────────────

def build_sdl_call_kwargs(
    architecture: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    generator,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build pipeline __call__ kwargs for SDL variants.

    SD3.5 does not accept explicit negative_prompt (handled internally).
    SD1.5 / SDXL include negative_prompt when non-empty.
    """
    kwargs: dict[str, Any] = dict(
        prompt              = prompt,
        width               = width,
        height              = height,
        num_inference_steps = steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )

    # SD3.5 handles negative_prompt internally — do not pass it
    if architecture != "sd35" and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if extra:
        kwargs.update(extra)

    return kwargs


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_sdl_output_meta(
    architecture: str,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    lora_specs: list,
    accel: list,
    sampler: str,
    elapsed_s: float,
    output_path: Path,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_output_meta(
        pipeline_name=f"sdl_{architecture}_text_to_image",
        model_id=model_id,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps,
        guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["architecture"] = architecture
    if extra:
        meta.update(extra)
    return meta


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_sdl_pipe(entry, models_dir: Union[str, Path], dtype):
    """
    Load an SDL pipeline from local storage.

    Handles full-repo and single-file (.safetensors) modes.
    Returns a diffusers pipeline (not yet moved to device).
    """
    import diffusers  # type: ignore

    local_path, is_single_file = _resolve_sdl_local_path(entry, models_dir)
    cls_name   = _select_sdl_pipeline_class(entry.architecture)
    PipeClass  = getattr(diffusers, cls_name)

    load_kwargs: dict = dict(torch_dtype=dtype, local_files_only=True)

    if not is_single_file:
        # ── Full-repo (from_pretrained) ─────────────────────────────────────
        load_kwargs["safety_checker"] = None
        if entry.architecture == "sdxl":
            load_kwargs["use_safetensors"] = True
            import torch as _torch
            if dtype == _torch.float16:
                load_kwargs["variant"] = "fp16"
        pipe = PipeClass.from_pretrained(str(local_path), **load_kwargs)
    else:
        # ── Single-file: custom .safetensors oder .ckpt ─────────────────────
        # from_single_file enthält den kompletten SDL-Checkpoint (kein Transformer-Swap nötig).
        # safety_checker und variant werden nicht übergeben — from_single_file ignoriert sie.
        log.info(f"Custom single-file: {local_path.name} (arch={entry.architecture})")
        pipe = PipeClass.from_single_file(str(local_path), **load_kwargs)

    # Optional: enable xformers if available (best-effort at load time)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe


# ── Accelerator entry point ────────────────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe,
    device: str,
    vram_gb: int = 16,
    accel: Optional[list] = None,
    has_quantized_encoders: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """Apply CPU offload + optional accelerators to pipe in-place."""
    accel = accel or []
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    offload_mode = resolve_offload_mode(vram_gb, env_override, has_quantized_encoders)
    apply_accelerators(pipe, device=device, offload_mode=offload_mode, accel=accel)
    inject_compile(pipe, accel)


# ── Public generation function ─────────────────────────────────────────────────

def text_to_image(
    cfg: SDLPipelineConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
):
    """
    Run SDL (SD1.5 / SDXL / SD3.5) text-to-image generation.

    Returns:
        dict with keys: output_path, metadata, elapsed_s
    """
    import time

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    log.info(
        f"SDL T2I | arch={cfg.architecture} model={cfg.model_id} "
        f"seed={seed} steps={cfg.steps} device={device}"
    )

    pipe = load_sdl_pipe(entry, models_dir, dtype)
    set_scheduler(pipe, cfg.architecture, cfg.sampler)

    # LoRAs
    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture=cfg.architecture)

    # Accelerators
    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
    )

    gen    = make_generator(seed, device)
    kwargs = build_sdl_call_kwargs(
        architecture=cfg.architecture,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen,
    )

    result   = pipe(**kwargs)
    image    = result.images[0]

    _out_dir = Path(outputs_dir) if outputs_dir else Path.cwd() / "genbox_outputs"
    out_path = build_output_path(
        "img", cfg.model_id, seed, "png",
        outputs_dir=_out_dir, custom=cfg.output,
    )
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_sdl_output_meta(
        architecture=cfg.architecture,
        model_id=cfg.model_id,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        seed=seed, lora_specs=cfg.loras, accel=cfg.accel,
        sampler=cfg.sampler, elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"SDL T2I done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
