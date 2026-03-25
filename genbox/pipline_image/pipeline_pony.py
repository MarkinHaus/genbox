"""
genbox/pipeline_pony.py
Pony Diffusion pipeline — SDXL-based models trained with quality/rating prompt tags.

Pony models (e.g. Pony Diffusion V6 XL) require specific prompt prefixes
(score_9, score_8_up, …) for best quality. This pipeline handles that
injection automatically while remaining fully compatible with SDXL infrastructure.

Supports:
  - All Pony-variant SDXL full-repo models
  - Custom .safetensors Pony fine-tunes
  - Multi-LoRA via PEFT
  - Full accelerator stack (offload + xformers + torch.compile)
  - Rating/quality tag auto-injection

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
    set_scheduler, make_sdl_step_callback,
)
from genbox.pipline_image.pipeline_sdl import (
    load_sdl_pipe,
    apply_pipeline_accelerators,
)

log = logging.getLogger("genbox.pipeline_pony")

# ── Pony quality / rating tag defaults ─────────────────────────────────────────

PONY_DEFAULT_QUALITY_TAGS     = "score_9, score_8_up, score_7_up"
PONY_DEFAULT_NEGATIVE_TAGS    = "score_1, score_2, score_3, score_4"


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class PonyPipelineConfig:
    """
    Generation config for Pony Diffusion (SDXL-based) pipelines.

    Quality/rating tags are automatically prepended to prompt and negative_prompt.
    """
    model_id:             str
    prompt:               str   = ""
    negative_prompt:      str   = ""
    width:                int   = 1024
    height:               int   = 1024
    steps:                int   = 30
    guidance_scale:       float = 7.0
    seed:                 int   = -1
    sampler:              str   = "default"
    quality_tags:         str   = PONY_DEFAULT_QUALITY_TAGS
    negative_quality_tags: str  = PONY_DEFAULT_NEGATIVE_TAGS
    loras:                list  = field(default_factory=list)
    accel:                list  = field(default_factory=list)
    output:               Optional[Union[str, Path]] = None


# ── Prompt tag injection ───────────────────────────────────────────────────────

def build_pony_prompt(user_prompt: str, quality_tags: str) -> str:
    """
    Prepend quality tags to the user prompt.

        quality_tags="score_9, score_8_up"
        user_prompt="a cute cat"
        → "score_9, score_8_up, a cute cat"

    Empty quality_tags → user_prompt unchanged.
    """
    if not quality_tags:
        return user_prompt
    if not user_prompt:
        return quality_tags
    return f"{quality_tags}, {user_prompt}"


def build_pony_negative_prompt(user_negative: str, negative_quality_tags: str) -> str:
    """
    Prepend negative quality tags to user's negative prompt.

        negative_quality_tags="score_1, score_2"
        user_negative="blurry"
        → "score_1, score_2, blurry"
    """
    if not negative_quality_tags:
        return user_negative
    if not user_negative:
        return negative_quality_tags
    return f"{negative_quality_tags}, {user_negative}"


# ── Call kwargs ────────────────────────────────────────────────────────────────

def build_pony_call_kwargs(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    generator,
    callback_on_step_end=None,
    callback_tensor_inputs: Optional[list] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build pipeline __call__ kwargs for Pony/SDXL inference.
    Always includes negative_prompt (Pony is guidance-heavy).
    callback_on_step_end: diffusers callback, fn(pipe, step, ts, cb_kwargs) → dict.
    """
    kwargs = dict(
        prompt              = prompt,
        negative_prompt     = negative_prompt,
        width               = width,
        height              = height,
        num_inference_steps = steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )
    if callback_on_step_end is not None:
        kwargs["callback_on_step_end"] = callback_on_step_end
        kwargs["callback_on_step_end_tensor_inputs"] = (
            callback_tensor_inputs if callback_tensor_inputs is not None
            else ["latents"]
        )
    if extra:
        kwargs.update(extra)
    return kwargs


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_pony_output_meta(
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
    quality_tags: str,
    negative_quality_tags: str,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_output_meta(
        pipeline_name="pony_text_to_image",
        model_id=model_id,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps,
        guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["quality_tags"]          = quality_tags
    meta["negative_quality_tags"] = negative_quality_tags
    meta["architecture"]          = "sdxl"
    if extra:
        meta.update(extra)
    return meta


# ── Public generation function ─────────────────────────────────────────────────

def text_to_image(
    cfg: PonyPipelineConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
    tracker=None,            # Optional[GenProgressTracker]
    enable_preview: bool = True,
    preview_interval: int = 5,
):
    """
    Run Pony Diffusion text-to-image generation.

    tracker: GenProgressTracker — live progress updates when provided.

    Returns:
        dict with keys: output_path, metadata, elapsed_s
    """
    import time

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    full_prompt   = build_pony_prompt(cfg.prompt, cfg.quality_tags)
    full_negative = build_pony_negative_prompt(cfg.negative_prompt, cfg.negative_quality_tags)

    log.info(
        f"Pony T2I | model={cfg.model_id} seed={seed} steps={cfg.steps} device={device}"
    )

    if tracker is not None:
        tracker.set_stage("loading model")

    pipe = load_sdl_pipe(entry, models_dir, dtype)
    set_scheduler(pipe, "sdxl", cfg.sampler)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture="sdxl")

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
    )

    step_callback = None
    if tracker is not None:
        step_callback = make_sdl_step_callback(
            tracker=tracker,
            preview_interval=preview_interval,
            enable_preview=enable_preview,
        )

    gen    = make_generator(seed, device)
    kwargs = build_pony_call_kwargs(
        prompt=full_prompt, negative_prompt=full_negative,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen,
        callback_on_step_end=step_callback,
    )

    if tracker is not None:
        tracker.set_stage("denoising")

    result   = pipe(**kwargs)
    image    = result.images[0]

    if tracker is not None:
        tracker.set_stage("saving")

    _out_dir = Path(outputs_dir) if outputs_dir else Path.cwd() / "genbox_outputs"
    out_path = build_output_path(
        "img", cfg.model_id, seed, "png",
        outputs_dir=_out_dir, custom=cfg.output,
    )
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_pony_output_meta(
        model_id=cfg.model_id,
        prompt=full_prompt, negative_prompt=full_negative,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        seed=seed, lora_specs=cfg.loras, accel=cfg.accel,
        sampler=cfg.sampler, elapsed_s=elapsed, output_path=out_path,
        quality_tags=cfg.quality_tags,
        negative_quality_tags=cfg.negative_quality_tags,
    )

    log.info(f"Pony T2I done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
