"""
genbox/pipeline_ltx.py
LTX-Video generation pipeline — all variants.

Variants covered:
  "classic"       — LTXV 0.9.1–0.9.5  (LTXPipeline / LTXImageToVideoPipeline)
  "distilled_13b" — LTXV 0.9.7 / 0.9.8 distilled 13B (LTXConditionPipeline,
                    optional LTXLatentUpsamplePipeline for 2-stage upscaling)
  "ltx2"          — LTX-2 (LTX2Pipeline / LTX2ImageToVideoPipeline,
                    joint audio-visual model)

Modes: t2v | i2v

Key LTX requirements:
  • frames = 8n+1 (min 9) for ALL LTX variants
  • classic / ltx2: guidance_scale ≈ 5.0
  • distilled: guidance_scale = 1.0, steps ≤ 10 (timestep-distilled)
  • decode_timestep = 0.05, image_cond_noise_scale = 0.025 (0.9.1+)

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
    detect_ltx_variant,
    ltx_generation_defaults,
    save_video_frames,
    select_ltx_pipeline_class,
    snap_frames,
)
from genbox.utils.utils_image_pipeline import (
    build_lora_adapter_list,
    apply_loras_to_pipe,
    make_generator,
    resolve_device,
    resolve_dtype,
    resolve_seed,
)

log = logging.getLogger("genbox.pipeline_ltx")


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class LtxPipelineConfig:
    """
    Generation config for LTX-Video (all variants).
    variant: "classic" | "distilled_13b" | "ltx2"
    mode:    "t2v" | "i2v"
    """
    model_id:              str
    variant:               str   = "classic"
    mode:                  str   = "t2v"
    prompt:                str   = ""
    negative_prompt:       str   = ""
    width:                 int   = 0
    height:                int   = 0
    frames:                int   = 0
    fps:                   int   = 0
    steps:                 int   = 0
    guidance_scale:        float = 0.0
    decode_timestep:       float = 0.0
    image_cond_noise_scale: float = 0.0
    seed:                  int   = -1
    sampler:               str   = "default"
    loras:                 list  = field(default_factory=list)
    accel:                 list  = field(default_factory=list)
    image:                 Optional[Union[str, Path]] = None
    output:                Optional[Union[str, Path]] = None

    def __post_init__(self):
        d = ltx_generation_defaults(self.variant)
        if self.width          == 0:   self.width          = d["width"]
        if self.height         == 0:   self.height         = d["height"]
        if self.frames         == 0:   self.frames         = d["frames"]
        if self.fps            == 0:   self.fps            = d["fps"]
        if self.steps          == 0:   self.steps          = d["steps"]
        if self.guidance_scale == 0.0: self.guidance_scale = d["guidance_scale"]
        if self.decode_timestep == 0.0:
            self.decode_timestep = d.get("decode_timestep", 0.05)
        if self.image_cond_noise_scale == 0.0:
            self.image_cond_noise_scale = d.get("image_cond_noise_scale", 0.025)
        # Snap frames to 8n+1
        self.frames = snap_frames(self.frames, "ltx")


# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_ltx_local_path(entry, models_dir: Union[str, Path]) -> Path:
    models_dir = Path(models_dir)

    if entry.is_gguf():
        p = models_dir / "ltx" / Path(entry.hf_filename).name
        if not p.exists():
            raise FileNotFoundError(
                f"LTX GGUF not found: {p}\nDownload via Models panel."
            )
        return p

    p = models_dir / "ltx" / entry.id
    if not (p / "model_index.json").exists():
        raise FileNotFoundError(
            f"LTX model not found: {p}\nDownload via Models panel."
        )
    return p


# ── Call kwargs ────────────────────────────────────────────────────────────────

def build_ltx_call_kwargs(
    mode: str,
    variant: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    decode_timestep: float,
    image_cond_noise_scale: float,
    generator,
    image=None,
    callback_on_step_end=None,
    callback_tensor_inputs: Optional[list] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build pipeline __call__ kwargs for LTX T2V or I2V.

    callback_on_step_end: diffusers callback, fn(pipe, step, ts, cb_kwargs) → dict.
    callback_tensor_inputs: defaults to ["latents"] when callback is set.
    Note: LTX latents in the callback are packed (B, seq_len, channels).
    std() for the noise meter works on any shape — no unpack needed.
    """
    kwargs: dict = dict(
        prompt                 = prompt,
        negative_prompt        = negative_prompt,
        width                  = width,
        height                 = height,
        num_frames             = frames,
        frame_rate             = fps,
        num_inference_steps    = steps,
        guidance_scale         = guidance_scale,
        decode_timestep        = decode_timestep,
        image_cond_noise_scale = image_cond_noise_scale,
        generator              = generator,
    )

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

def build_ltx_output_meta(
    ltx_variant: str,
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
        pipeline_name=f"ltx_{mode}",
        model_id=model_id,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, frames=frames, fps=fps,
        steps=steps, guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler=sampler,
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["ltx_variant"] = ltx_variant
    meta["mode"]        = mode
    if extra:
        meta.update(extra)
    return meta


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_ltx_pipe(entry, models_dir, dtype, variant: str, mode: str = "t2v"):
    import diffusers

    local_path = _resolve_ltx_local_path(entry, models_dir)

    # ── GGUF: Transformer via from_single_file, Rest vom pipeline_repo ──────
    if entry.is_gguf():
        return _load_ltx_pipe_gguf(entry, local_path, dtype, variant, mode)

    # ── Full diffusers repo (bisheriger Code) ────────────────────────────────
    cls_name = select_ltx_pipeline_class(variant, mode)

    if variant == "ltx2":
        try:
            from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2ImageToVideoPipeline
            PipeClass = LTX2ImageToVideoPipeline if mode == "i2v" else LTX2Pipeline
        except ImportError:
            PipeClass = getattr(diffusers, cls_name, None)
            if PipeClass is None:
                raise ImportError(
                    f"LTX-2 requires a recent diffusers version: pip install -U diffusers\n"
                    f"Looking for: {cls_name}"
                )
    else:
        PipeClass = getattr(diffusers, cls_name, None)
        if PipeClass is None:
            raise ImportError(
                f"diffusers class {cls_name!r} not found — update diffusers"
            )

    pipe = PipeClass.from_pretrained(
        str(local_path), torch_dtype=dtype, local_files_only=True,
    )
    try:
        pipe.vae.enable_tiling()
        log.info("LTX VAE tiling enabled")
    except Exception:
        pass
    return pipe


def _load_ltx_pipe_gguf(entry, gguf_path: Path, dtype, variant: str, mode: str):
    """
    LTX GGUF hybrid loader:
      - Transformer: LTXVideoTransformer3DModel.from_single_file(gguf) + GGUFQuantizationConfig
      - Rest:        from_pretrained(shared_dir, local_files_only=True)

    Benötigt: entry.hf_pipeline_repo (z.B. "Lightricks/LTX-Video-0.9.7-distilled")
    """
    import diffusers
    from genbox.config import cfg

    pipeline_repo = entry.hf_pipeline_repo
    if not pipeline_repo:
        raise ValueError(
            f"GGUF entry '{entry.id}' hat kein hf_pipeline_repo. "
            "Kann VAE/Text-Encoder nicht laden."
        )

    safe       = pipeline_repo.replace("/", "--")
    shared_dir = str(Path(cfg.models_dir) / "ltx" / f"_shared_{safe}")

    log.info(f"LTX GGUF: transformer={gguf_path.name}, components={shared_dir}")

    transformer = diffusers.LTXVideoTransformer3DModel.from_single_file(
        str(gguf_path),
        quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )

    cls_name  = select_ltx_pipeline_class(variant, mode)
    PipeClass = getattr(diffusers, cls_name, None)
    if PipeClass is None:
        raise ImportError(f"diffusers class {cls_name!r} not found — update diffusers")

    pipe = PipeClass.from_pretrained(
        shared_dir,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=True,
    )

    try:
        pipe.vae.enable_tiling()
        log.info("LTX GGUF VAE tiling enabled")
    except Exception:
        pass

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
    """Apply CPU offload + optional accelerators to an LTX pipeline in-place."""
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    apply_video_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=accel or [],
        enable_vae_tiling=enable_vae_tiling,
        env_override=env_override,
    )

def _apply_ltx_teacache(pipe, accel: list) -> None:
    """
    TeaCache für LTX-Video — nur wenn "teacache" in accel (UI-Checkbox).
    thresh=0.03 → lossless, thresh=0.05 → ~1.7x speedup (empfohlen).
    Quelle: ali-vilab/TeaCache, LTX-Video support seit 2024-12-30.
    """
    if "teacache" not in [a.lower() for a in accel]:
        return

    try:
        import torch

        thresh     = 0.05
        transformer = pipe.transformer
        original_forward = transformer.forward

        cache_state = {
            "accumulated_rel_l1": 0.0,
            "prev_hidden":        None,
            "cached_output":      None,
        }

        def cached_forward(hidden_states, *args, **kwargs):
            if cache_state["prev_hidden"] is not None:
                diff  = (hidden_states - cache_state["prev_hidden"]).abs().mean()
                norm  = cache_state["prev_hidden"].abs().mean() + 1e-8
                rel_l1 = (diff / norm).item()
                cache_state["accumulated_rel_l1"] += rel_l1

                if (
                    cache_state["accumulated_rel_l1"] < thresh
                    and cache_state["cached_output"] is not None
                ):
                    return cache_state["cached_output"]

                cache_state["accumulated_rel_l1"] = 0.0

            cache_state["prev_hidden"] = hidden_states.detach().clone()
            output = original_forward(hidden_states, *args, **kwargs)
            cache_state["cached_output"] = output
            return output

        transformer.forward = cached_forward
        log.info(f"TeaCache enabled (LTX, thresh={thresh})")

    except Exception as e:
        log.warning(f"TeaCache apply failed: {e}")


def _apply_ltx_compile(pipe, accel: list) -> None:
    if "compile" not in [a.lower() for a in accel]:
        return

    offload_env = os.environ.get("GENBOX_OFFLOAD", "").lower()
    if offload_env != "none":
        log.warning(
            "torch.compile übersprungen — inkompatibel mit CPU-offload. "
            "Setze GENBOX_OFFLOAD=none um compile zu aktivieren."
        )
        return

    try:
        import torch
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=False,
        )
        log.info("torch.compile applied to LTX transformer")
    except Exception as e:
        log.warning(f"torch.compile failed: {e}")

# ── Public generation function ─────────────────────────────────────────────────

def generate(
    cfg: LtxPipelineConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
    enable_vae_tiling: bool = True,
    tracker=None,             # Optional[GenProgressTracker] — Variante 1 + 3
    enable_noise_meter: bool = False,  # Variante 3
) -> dict:
    """
    Run LTX video generation (T2V or I2V, any variant).

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
    variant = cfg.variant

    if variant == "classic":
        variant = detect_ltx_variant(
            getattr(entry, "hf_pipeline_repo", ""),
            entry.id,
        )
        cfg.variant = variant

    frames  = snap_frames(cfg.frames, "ltx")

    log.info(
        f"LTX {cfg.mode.upper()} | variant={variant} model={cfg.model_id} "
        f"seed={seed} frames={frames} {cfg.width}x{cfg.height} device={device}"
    )

    if tracker is not None:
        tracker.set_stage("loading model")

    pipe = load_ltx_pipe(entry, models_dir, dtype, variant=variant, mode=cfg.mode)

    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture="ltx")

    _apply_ltx_compile(pipe, cfg.accel)
    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb,
        accel=cfg.accel, enable_vae_tiling=enable_vae_tiling,
    )

    _apply_ltx_teacache(pipe, cfg.accel)

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

    kwargs = build_ltx_call_kwargs(
        mode=cfg.mode, variant=variant,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height, frames=frames, fps=cfg.fps,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        decode_timestep=cfg.decode_timestep,
        image_cond_noise_scale=cfg.image_cond_noise_scale,
        generator=gen, image=image,
        callback_on_step_end=step_callback,
    )

    if tracker is not None:
        tracker.set_stage("denoising")

    result       = pipe(**kwargs)
    frames = getattr(result, "frames", None)
    if frames is not None:
        video_frames = frames
    else:
        video_frames = getattr(result, "videos", None)
    if video_frames is None:
        raise RuntimeError("LTX pipeline returned no frames — check diffusers version")
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
    meta = build_ltx_output_meta(
        ltx_variant=variant, model_id=cfg.model_id, mode=cfg.mode,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height, frames=frames, fps=cfg.fps,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        seed=seed, lora_specs=cfg.loras, accel=cfg.accel,
        sampler=cfg.sampler, elapsed_s=elapsed, output_path=out_path,
    )

    log.info(f"LTX {cfg.mode.upper()} done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}

# Convenience aliases
def text_to_video(cfg: LtxPipelineConfig, entry, models_dir, **kw):
    cfg.mode = "t2v"
    return generate(cfg, entry, models_dir, **kw)


def image_to_video(cfg: LtxPipelineConfig, entry, models_dir, **kw):
    cfg.mode = "i2v"
    return generate(cfg, entry, models_dir, **kw)
