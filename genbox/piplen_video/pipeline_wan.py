"""
genbox/pipeline_wan.py
WAN video generation pipeline — Wan 2.1 and Wan 2.2.

Supports:
  - WAN 2.1 T2V-1.3B  (480p, 8GB VRAM)
  - WAN 2.1 T2V-14B   (480p/720p, 16GB VRAM)
  - WAN 2.2 T2V-A14B  (MoE, 720p, 24GB VRAM)
  - WAN 2.1/2.2 I2V   (Image-to-Video, all sizes)
  Modes: t2v (text-to-video) | i2v (image-to-video)

Critical WAN requirements (from official HF READMEs):
  • VAE MUST be loaded as torch.float32 — bfloat16 VAE → black / corrupted frames
  • CLIPVisionModel required for I2V (float32)
  • flow_shift: 3.0 for 480p, 5.0 for ≥720p
  • frames = 4n+1 (recommended)

GGUF is NOT supported for WAN (diffusers limitation).
100 % offline after model installation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from genbox.utils.utils_video_pipeline import (
    apply_video_accelerators,
    build_video_output_meta,
    build_video_output_path,
    detect_wan_variant,
    save_video_frames,
    select_wan_pipeline_class,
    snap_frames,
    wan_flow_shift,
    wan_generation_defaults,
)
from genbox.utils.utils_image_pipeline import (
    build_lora_adapter_list,
    apply_loras_to_pipe,
    make_generator,
    resolve_device,
    resolve_dtype,
    resolve_seed,
)

log = logging.getLogger("genbox.pipeline_wan")


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class WanPipelineConfig:
    """
    Generation config for WAN 2.1 / 2.2 (T2V and I2V).
    Defaults are auto-selected based on model_id variant detection.
    """
    model_id:         str
    mode:             str   = "t2v"     # "t2v" | "i2v"
    prompt:           str   = ""
    negative_prompt:  str   = ""
    width:            int   = 0         # 0 → variant default
    height:           int   = 0         # 0 → variant default
    frames:           int   = 0         # 0 → variant default
    fps:              int   = 0         # 0 → variant default
    steps:            int   = 0         # 0 → variant default
    guidance_scale:   float = 0.0       # 0 → variant default
    seed:             int   = -1
    sampler:          str   = "default"
    loras:            list  = field(default_factory=list)
    accel:            list  = field(default_factory=list)
    image:            Optional[Union[str, Path]] = None   # I2V start frame
    output:           Optional[Union[str, Path]] = None

    def __post_init__(self):
        variant = detect_wan_variant(self.model_id)
        d = wan_generation_defaults(variant)
        if self.width          == 0:   self.width          = d["width"]
        if self.height         == 0:   self.height         = d["height"]
        if self.frames         == 0:   self.frames         = d["frames"]
        if self.fps            == 0:   self.fps            = d["fps"]
        if self.steps          == 0:   self.steps          = d["steps"]
        if self.guidance_scale == 0.0: self.guidance_scale = d["guidance_scale"]
        # Snap frames to 4n+1
        self.frames = snap_frames(self.frames, "wan")


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_wan_local_path(entry, models_dir: Union[str, Path]) -> Path:
    """
    Resolve local path for a WAN model.

    WAN is always full diffusers-repo format.
    GGUF is explicitly unsupported (raises ValueError).
    Raises FileNotFoundError if model not installed.
    """
    models_dir = Path(models_dir)

    if "gguf" in getattr(entry, "quant", "").lower():
        raise ValueError(
            f"'{entry.name}' is a GGUF single-file. "
            "Diffusers-based WAN pipelines do not support GGUF.\n"
            "Use a full diffusers-format repo: wan_1_3b or wan21_14b_diffusers."
        )

    p = models_dir / "wan" / entry.id
    if not (p / "model_index.json").exists():
        raise FileNotFoundError(
            f"WAN model not found: {p}\n"
            "Download via Models panel."
        )
    return p


# ── VAE kwargs ─────────────────────────────────────────────────────────────────

def build_wan_vae_kwargs(
    local_path: Path,
    _torch=None,
) -> dict:
    """
    Build kwargs for AutoencoderKLWan.from_pretrained().

    WAN VAE MUST be float32. bfloat16/float16 → black/corrupted frames.
    This is enforced unconditionally — no override allowed.
    """
    if _torch is None:
        import torch as _torch  # type: ignore

    return {
        "subfolder":   "vae",
        "torch_dtype": _torch.float32,
        "local_files_only": True,
    }


# ── Scheduler setup ────────────────────────────────────────────────────────────

def build_wan_scheduler_kwargs(height: int) -> dict:
    """Return flow_shift kwargs dict for WAN UniPC scheduler."""
    return {"flow_shift": wan_flow_shift(height)}


def _apply_wan_scheduler(pipe, height: int) -> None:
    """Apply flow_shift to the WAN pipeline's scheduler in-place."""
    flow_shift = wan_flow_shift(height)
    try:
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # type: ignore
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift,
        )
        log.info(f"WAN scheduler: flow_shift={flow_shift} (height={height})")
    except Exception as e:
        log.warning(f"WAN scheduler setup failed (continuing with default): {e}")


# ── Call kwargs ────────────────────────────────────────────────────────────────

def build_wan_call_kwargs(
    mode: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    guidance_scale: float,
    generator,
    image=None,
    callback_on_step_end=None,
    callback_tensor_inputs: Optional[list] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build pipeline __call__ kwargs for WAN T2V or I2V.

    I2V: image must be a PIL Image (pre-loaded by caller).
    callback_on_step_end: diffusers callback, fn(pipe, step, ts, cb_kwargs) → dict.
    callback_tensor_inputs: defaults to ["latents"] when callback is set.
    """
    kwargs: dict = dict(
        prompt              = prompt,
        width               = width,
        height              = height,
        num_frames          = frames,
        num_inference_steps = steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )

    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if mode == "i2v" and image is not None:
        kwargs["image"] = image

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

def build_wan_output_meta(
    wan_variant: str,
    model_id: str,
    mode: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
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
    meta = build_video_output_meta(
        pipeline_name=f"wan_{mode}",
        model_id=model_id,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, frames=frames, fps=fps,
        steps=steps, guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["wan_variant"] = wan_variant
    meta["mode"]        = mode
    if extra:
        meta.update(extra)
    return meta


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_wan_pipe(
    entry,
    models_dir: Union[str, Path],
    dtype,
    mode: str = "t2v",
):
    """
    Load a WAN pipeline from local storage.

    VAE is always loaded as float32 (mandatory).
    I2V additionally loads CLIPVisionModel as float32.
    Returns a diffusers pipeline (not yet moved to device).
    """
    import torch        # type: ignore
    import diffusers    # type: ignore

    local_path = _resolve_wan_local_path(entry, models_dir)

    # VAE: always float32
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        str(local_path),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    cls_name  = select_wan_pipeline_class(mode)
    PipeClass = getattr(diffusers, cls_name)

    if mode == "i2v":
        # I2V needs CLIPVisionModel (float32)
        from transformers import CLIPVisionModel  # type: ignore
        image_encoder = CLIPVisionModel.from_pretrained(
            str(local_path),
            subfolder="image_encoder",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        pipe = PipeClass.from_pretrained(
            str(local_path),
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        pipe = PipeClass.from_pretrained(
            str(local_path),
            vae=vae,
            torch_dtype=dtype,
            local_files_only=True,
        )

    return pipe


# ── Accelerator entry point ────────────────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe,
    device: str,
    vram_gb: int = 16,
    accel: Optional[list] = None,
    enable_vae_tiling: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """Apply CPU offload + optional accelerators to a WAN pipeline in-place."""
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    apply_video_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=accel or [],
        enable_vae_tiling=enable_vae_tiling,
        env_override=env_override,
    )


# ── Public generation functions ────────────────────────────────────────────────

def generate(
    cfg: WanPipelineConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
    enable_vae_tiling: bool = False,
    tracker=None,             # Optional[GenProgressTracker] — Variante 1 + 3
    enable_noise_meter: bool = False,  # Variante 3
) -> dict:
    """
    Run WAN video generation (T2V or I2V).

    tracker: GenProgressTracker — live step/stage/noise updates when provided.
    enable_noise_meter: append latent std() per step to tracker.noise_std_history.

    Returns:
        dict with keys: output_path, metadata, elapsed_s
    """
    import time

    t0      = time.time()
    seed    = resolve_seed(cfg.seed)
    device  = resolve_device()
    dtype   = resolve_dtype(entry.quant)
    variant = detect_wan_variant(cfg.model_id)

    log.info(
        f"WAN {cfg.mode.upper()} | variant={variant} model={cfg.model_id} "
        f"seed={seed} frames={cfg.frames} {cfg.width}x{cfg.height} device={device}"
    )

    if tracker is not None:
        tracker.set_stage("loading model")

    pipe = load_wan_pipe(entry, models_dir, dtype, mode=cfg.mode)
    _apply_wan_scheduler(pipe, cfg.height)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture="wan")

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb,
        accel=cfg.accel, enable_vae_tiling=enable_vae_tiling,
    )

    step_callback = None
    if tracker is not None:
        from genbox.utils.utils_video_pipeline import make_video_step_callback
        step_callback = make_video_step_callback(
            tracker, enable_noise_meter=enable_noise_meter
        )

    gen = make_generator(seed, device)

    image = None
    if cfg.mode == "i2v" and cfg.image is not None:
        from diffusers.utils import load_image  # type: ignore
        image = load_image(str(cfg.image))

    frames_count = snap_frames(cfg.frames, "wan")

    kwargs = build_wan_call_kwargs(
        mode=cfg.mode,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height, frames=frames_count,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen, image=image,
        callback_on_step_end=step_callback,
    )

    if tracker is not None:
        tracker.set_stage("denoising")

    result = pipe(**kwargs)
    frames = getattr(result, "frames", None)
    if frames is not None:
        video_frames = frames
    else:
        video_frames = getattr(result, "videos", None)
    if video_frames is None:
        raise RuntimeError("WAN pipeline returned no frames — check diffusers version")
    video_frames = video_frames[0]

    if tracker is not None:
        tracker.set_stage("saving")

    _out_dir = Path(outputs_dir) if outputs_dir else Path.cwd() / "genbox_outputs"
    out_path = build_video_output_path(
        "vid", cfg.model_id, seed,
        outputs_dir=_out_dir, custom=cfg.output,
    )
    save_video_frames(video_frames, out_path, fps=cfg.fps)

    elapsed = time.time() - t0
    meta = build_wan_output_meta(
        wan_variant=variant, model_id=cfg.model_id, mode=cfg.mode,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height, frames=frames_count, fps=cfg.fps,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        seed=seed, lora_specs=cfg.loras, accel=cfg.accel,
        sampler=cfg.sampler, elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"WAN {cfg.mode.upper()} done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}


# Convenience aliases
def text_to_video(cfg: WanPipelineConfig, entry, models_dir, **kw):
    cfg.mode = "t2v"
    return generate(cfg, entry, models_dir, **kw)


def image_to_video(cfg: WanPipelineConfig, entry, models_dir, **kw):
    cfg.mode = "i2v"
    return generate(cfg, entry, models_dir, **kw)
