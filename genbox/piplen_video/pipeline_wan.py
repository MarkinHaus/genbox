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

def _resolve_wan_local_path(entry, models_dir: Union[str, Path]):
    """
    Resolve local path for a WAN model.
    Full diffusers repo  → returns directory (with model_index.json)
    GGUF single-file     → returns .gguf file path
    """
    models_dir = Path(models_dir)

    if entry.is_gguf():
        p = models_dir / "wan" / Path(entry.hf_filename).name
        if not p.exists():
            raise FileNotFoundError(
                f"WAN GGUF not found: {p}\nDownload via Models panel."
            )
        return p   # ← Path zur .gguf Datei

    p = models_dir / "wan" / entry.id
    if not (p / "model_index.json").exists():
        raise FileNotFoundError(
            f"WAN model not found: {p}\nDownload via Models panel."
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

def load_wan_pipe(entry, models_dir, dtype, mode: str = "t2v"):
    import torch
    import diffusers

    local_path = _resolve_wan_local_path(entry, models_dir)

    # ── GGUF: Transformer via from_single_file, Rest vom pipeline_repo ──────
    if entry.is_gguf():
        return _load_wan_pipe_gguf(entry, local_path, dtype, mode)

    # ── Full diffusers repo (bisheriger Code) ────────────────────────────────
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        str(local_path), subfolder="vae",
        torch_dtype=torch.float32, local_files_only=True,
    )
    cls_name  = select_wan_pipeline_class(mode)
    PipeClass = getattr(diffusers, cls_name)

    if mode == "i2v":
        from transformers import CLIPVisionModel
        image_encoder = CLIPVisionModel.from_pretrained(
            str(local_path), subfolder="image_encoder",
            torch_dtype=torch.float32, local_files_only=True,
        )
        return PipeClass.from_pretrained(
            str(local_path), vae=vae, image_encoder=image_encoder,
            torch_dtype=dtype, local_files_only=True,low_cpu_mem_usage=True
        )

    return PipeClass.from_pretrained(
        str(local_path), vae=vae,
        torch_dtype=dtype, local_files_only=True,low_cpu_mem_usage=True
    )


def _load_wan_pipe_gguf(entry, gguf_path: Path, dtype, mode: str):
    import torch
    import diffusers
    from genbox.config import cfg  # für models_dir

    pipeline_repo = entry.hf_pipeline_repo
    if not pipeline_repo:
        raise ValueError(
            f"GGUF entry '{entry.id}' hat kein hf_pipeline_repo. "
            "Kann VAE/Text-Encoder nicht laden."
        )

    # Shared config dir — identisch zu _shared_config_dir() in models.py
    safe = pipeline_repo.replace("/", "--")
    shared_dir = str(Path(cfg.models_dir) / "wan" / f"_shared_{safe}")

    log.info(f"WAN GGUF: transformer={gguf_path.name}, components={shared_dir}")

    transformer = diffusers.WanTransformer3DModel.from_single_file(
        str(gguf_path),
        config=shared_dir,  # ← Config vom passenden T2V-Repo
        subfolder="transformer",  # ← nicht vom I2V-Pipeline-Config
        quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )

    vae = diffusers.AutoencoderKLWan.from_pretrained(
        shared_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,   # ← offline
    )

    cls_name  = select_wan_pipeline_class(mode)
    PipeClass = getattr(diffusers, cls_name)

    if mode == "i2v":
        image_encoder_path = Path(shared_dir) / "image_encoder"

        if not image_encoder_path.exists():
            # Fallback: CLIP aus einem anderen WAN-Shard der bereits image_encoder hat
            wan_dir = Path(shared_dir).parent  # models_dir/wan/
            clip_source = None
            for candidate in wan_dir.glob("_shared_*"):
                if (candidate / "image_encoder" / "config.json").exists():
                    clip_source = str(candidate)
                    log.info(f"CLIP image_encoder nicht in {shared_dir} — "
                             f"verwende: {candidate.name}")
                    break

            if clip_source:
                from transformers import CLIPVisionModel
                image_encoder = CLIPVisionModel.from_pretrained(
                    clip_source,
                    subfolder="image_encoder",
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                return PipeClass.from_pretrained(
                    shared_dir,
                    transformer=transformer,
                    vae=vae,
                    image_encoder=image_encoder,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            else:
                log.warning(
                    "Kein image_encoder in shared_dir und kein WAN-Shard mit CLIP gefunden. "
                    "Lade ohne image_encoder — I2V-Konditionierung nicht verfügbar."
                )
                return PipeClass.from_pretrained(
                    shared_dir,
                    transformer=transformer,
                    vae=vae,
                    torch_dtype=dtype,
                    local_files_only=True,
                )

        from transformers import CLIPVisionModel
        image_encoder = CLIPVisionModel.from_pretrained(
            shared_dir,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        return PipeClass.from_pretrained(
            shared_dir,
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            local_files_only=True,
        )

    return PipeClass.from_pretrained(
        shared_dir,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
        local_files_only=True,
    )

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


def _apply_wan_teacache(pipe, accel: list, variant: str) -> None:
    """
    TeaCache für WAN — patcht pipe.transformer.forward mit Caching-Logik.
    Schwellenwert: 0.08 für 1.3B, 0.3 für 14B.
    Quelle: ali-vilab/TeaCache, TeaCache4Wan2.1
    """
    if "teacache" not in [a.lower() for a in accel]:
        return

    try:
        import torch

        # Schwellenwert je Modellgröße
        thresh = 0.08 if "1_3b" in variant or "1.3b" in variant else 0.3

        transformer = pipe.transformer
        original_forward = transformer.forward

        # State für den Cache
        cache_state = {
            "accumulated_rel_l1": 0.0,
            "prev_hidden": None,
            "cached_output": None,
            "step": 0,
        }

        def cached_forward(hidden_states, *args, **kwargs):
            if cache_state["prev_hidden"] is not None:
                # Relative L1 Distanz zum letzten Step
                diff = (hidden_states - cache_state["prev_hidden"]).abs().mean()
                norm = cache_state["prev_hidden"].abs().mean() + 1e-8
                rel_l1 = (diff / norm).item()
                cache_state["accumulated_rel_l1"] += rel_l1

                if (
                        cache_state["accumulated_rel_l1"] < thresh
                        and cache_state["cached_output"] is not None
                ):
                    cache_state["step"] += 1
                    return cache_state["cached_output"]

                cache_state["accumulated_rel_l1"] = 0.0  # reset nach Compute

            cache_state["prev_hidden"] = hidden_states.detach().clone()
            output = original_forward(hidden_states, *args, **kwargs)
            cache_state["cached_output"] = output
            cache_state["step"] += 1
            return output

        transformer.forward = cached_forward
        log.info(f"TeaCache enabled (WAN, thresh={thresh})")

    except Exception as e:
        log.warning(f"TeaCache apply failed: {e}")


def _apply_wan_compile(pipe, accel: list) -> None:
    if "compile" not in [a.lower() for a in accel]:
        return

    # compile + CPU-offload = inkompatibel (accelerate hooks wrappen mit
    # torch.compiler.disable → compile crasht). Nur kompilieren wenn
    # genug VRAM für offload=none vorhanden — User muss GENBOX_OFFLOAD=none setzen.
    offload_env = os.environ.get("GENBOX_OFFLOAD", "").lower()
    if offload_env != "none":
        log.warning(
            "torch.compile übersprungen — inkompatibel mit CPU-offload. "
            "Setze GENBOX_OFFLOAD=none um compile zu aktivieren (braucht >16GB VRAM)."
        )
        return

    try:
        import torch
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=False,   # fullgraph=True crasht mit accelerate
        )
        log.info("torch.compile applied to WAN transformer")
    except Exception as e:
        log.warning(f"torch.compile failed: {e}")


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

    _apply_wan_compile(pipe, cfg.accel)

    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb,
        accel=cfg.accel, enable_vae_tiling=enable_vae_tiling,
    )

    _apply_wan_teacache(pipe, cfg.accel, cfg.model_id)

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
