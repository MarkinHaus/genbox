"""
genbox/utils_video_pipeline.py
Shared video pipeline infrastructure — used by pipeline_wan.py and pipeline_ltx.py.

Sections:
  A. Frame constraints (snap_frames, LTX 8n+1, WAN 4n+1)
  B. Variant detection  (LTX classic / distilled_13b / ltx2, WAN 2.1 / 2.2)
  C. Pipeline class selection (architecture × mode → diffusers class name)
  D. Generation defaults  (guidance_scale, steps, decode_timestep per variant)
  E. Accelerator section  (offload strategy + VAE tiling + torch.compile)
  F. Video frame saver    (imageio MP4 writer, PIL + tensor support)
  G. Output path + metadata builder

No ML imports at module level — all lazy.
100 % offline after model installation.
WAN VAE is always float32 — this is enforced by the loading helpers in pipeline_wan.py.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable

from genbox.utils.utils_image_pipeline import (
    resolve_offload_mode,
    apply_accelerators,
    inject_compile,
    _parse_lora_spec,
)

log = logging.getLogger("genbox.utils_video_pipeline")

LoraSpec = Union[str, tuple[str, float]]

# ──────────────────────────────────────────────────────────────────────────────
# A. Frame constraints
# ──────────────────────────────────────────────────────────────────────────────

# LTX: frames = 8n + 1, min 9 (covers all LTX variants incl. LTXV 0.9.7+ and LTX-2)
_LTX_BASE, _LTX_OFF, _LTX_MIN = 8, 1, 9
# WAN: frames = 4n + 1, min 5
_WAN_BASE, _WAN_OFF, _WAN_MIN = 4, 1, 5


def snap_frames(frames: int, architecture: str) -> int:
    """
    Snap frame count to architecture constraints.

    LTX / ltx2:  frames = 8n+1  (min 9)
    WAN:          frames = 4n+1  (min 5, recommended)
    Other:        unchanged.
    """
    if architecture in ("ltx", "ltx2"):
        frames = max(frames, _LTX_MIN)
        rem = (frames - _LTX_OFF) % _LTX_BASE
        if rem:
            frames += _LTX_BASE - rem
    elif architecture == "wan":
        frames = max(frames, _WAN_MIN)
        rem = (frames - _WAN_OFF) % _WAN_BASE
        if rem:
            frames += _WAN_BASE - rem
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# B. Variant detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_ltx_variant(hf_repo: str, model_id: str) -> str:
    """
    Classify an LTX model into one of three variants:
      "classic"       — LTXV 0.9.1 – 0.9.5  (LTXPipeline / LTXImageToVideoPipeline)
      "distilled_13b" — LTXV 0.9.7 / 0.9.8  (LTXConditionPipeline + upscaler)
      "ltx2"          — LTX-2                (LTX2Pipeline / LTX2ImageToVideoPipeline)

    Detection priority: hf_repo > model_id heuristic.
    """
    r = hf_repo.lower()
    i = model_id.lower()

    if "ltx-2" in r or "ltx_2" in r:
        return "ltx2"
    if "0.9.7" in r or "0.9.8" in r or "distilled" in r or "13b" in r:
        return "distilled_13b"

    # Fallback to model_id
    if "ltx2" in i and "23" not in i:   # "ltx2_fp8" but NOT "ltx23_fp8"
        # "ltx23" is our id for 0.9.7 distilled
        if "23" not in i:
            return "classic"
    if "ltx23" in i or "distilled" in i or "097" in i or "098" in i:
        return "distilled_13b"

    return "classic"


def detect_wan_variant(model_id: str) -> str:
    """
    Classify a WAN model into one of three variants:
      "wan21_1_3b"  — Wan 2.1 T2V-1.3B
      "wan21_14b"   — Wan 2.1 T2V-14B / I2V-14B
      "wan22_a14b"  — Wan 2.2 T2V/I2V A14B (MoE)
    """
    i = model_id.lower()
    if "2.2" in i or "22" in i:
        return "wan22_a14b"
    if "14b" in i:
        return "wan21_14b"
    return "wan21_1_3b"


# ──────────────────────────────────────────────────────────────────────────────
# C. Pipeline class selection
# ──────────────────────────────────────────────────────────────────────────────

_LTX_PIPELINE_MAP: dict[str, dict[str, str]] = {
    "classic":       {"t2v": "LTXPipeline",           "i2v": "LTXImageToVideoPipeline"},
    "distilled_13b": {"t2v": "LTXConditionPipeline",  "i2v": "LTXConditionPipeline"},
    "ltx2":          {"t2v": "LTX2Pipeline",           "i2v": "LTX2ImageToVideoPipeline"},
}

_WAN_PIPELINE_MAP: dict[str, str] = {
    "t2v": "WanPipeline",
    "i2v": "WanImageToVideoPipeline",
}


def select_ltx_pipeline_class(variant: str, mode: str) -> str:
    """Return diffusers class name for given LTX variant and mode ('t2v'|'i2v')."""
    entry = _LTX_PIPELINE_MAP.get(variant)
    if entry is None:
        raise ValueError(
            f"Unknown LTX variant: {variant!r}. "
            f"Supported: {list(_LTX_PIPELINE_MAP)}"
        )
    return entry.get(mode, entry["t2v"])


def select_wan_pipeline_class(mode: str) -> str:
    """Return diffusers class name for given WAN mode ('t2v'|'i2v')."""
    cls = _WAN_PIPELINE_MAP.get(mode)
    if cls is None:
        raise ValueError(
            f"Unknown WAN mode: {mode!r}. Supported: {list(_WAN_PIPELINE_MAP)}"
        )
    return cls


# ──────────────────────────────────────────────────────────────────────────────
# D. Generation defaults
# ──────────────────────────────────────────────────────────────────────────────

def ltx_generation_defaults(variant: str) -> dict:
    """
    Return recommended generation parameters for a given LTX variant.

    distilled_13b: guidance_scale=1.0, steps=4–8 (timestep-distilled)
    classic:       guidance_scale=5.0, steps=50, decode_timestep=0.05
    ltx2:          guidance_scale=4.0, steps=40
    """
    if variant == "distilled_13b":
        return {
            "guidance_scale":       1.0,
            "steps":                8,
            "frames":               97,   # 8*12+1
            "width":                768,
            "height":               512,
            "fps":                  24,
            "decode_timestep":      0.05,
            "image_cond_noise_scale": 0.025,
        }
    if variant == "ltx2":
        return {
            "guidance_scale":       4.0,
            "steps":                40,
            "frames":               121,  # 8*15+1
            "width":                768,
            "height":               512,
            "fps":                  24,
            "decode_timestep":      0.05,
            "image_cond_noise_scale": 0.025,
        }
    # classic (0.9.1 – 0.9.5)
    return {
        "guidance_scale":       5.0,
        "steps":                50,
        "frames":               97,
        "width":                768,
        "height":               512,
        "fps":                  24,
        "decode_timestep":      0.05,
        "image_cond_noise_scale": 0.025,
    }


def wan_generation_defaults(variant: str) -> dict:
    """
    Return recommended generation parameters for a given WAN variant.

    Sources: official Wan-AI HuggingFace READMEs.
    WAN VAE requires float32 — this is enforced by load helpers, not here.
    """
    if variant == "wan22_a14b":
        return {
            "width":          1280,
            "height":         720,
            "frames":         81,    # 4*20+1
            "fps":            24,
            "steps":          50,
            "guidance_scale": 5.0,
            "flow_shift":     5.0,   # 720p → 5.0
        }
    if variant == "wan21_14b":
        return {
            "width":          832,
            "height":         480,
            "frames":         81,
            "fps":            16,
            "steps":          50,
            "guidance_scale": 5.0,
            "flow_shift":     3.0,   # 480p → 3.0
        }
    # wan21_1_3b (default / fallback)
    return {
        "width":          832,
        "height":         480,
        "frames":         81,
        "fps":            16,
        "steps":          50,
        "guidance_scale": 5.0,
        "flow_shift":     3.0,
    }


def wan_flow_shift(height: int) -> float:
    """WAN flow_shift: 5.0 for ≥720p, 3.0 for 480p."""
    return 5.0 if height >= 720 else 3.0


# ──────────────────────────────────────────────────────────────────────────────
# E. Accelerator section
# ──────────────────────────────────────────────────────────────────────────────

def apply_video_accelerators(
    pipe,
    device: str,
    vram_gb: int = 16,
    accel: Optional[list] = None,
    enable_vae_tiling: bool = False,
    has_quantized_encoders: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """
    Apply CPU offload + optional accelerators to a video pipeline in-place.

    Video pipelines are typically larger than image pipelines, so the VRAM
    threshold for sequential offload is the same (≤8 GB) but VAE tiling is
    offered as an extra memory saver.

    enable_vae_tiling: call pipe.vae.enable_tiling() if the vae supports it.
    """
    accel = accel or []
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    offload_mode = resolve_offload_mode(vram_gb, env_override, has_quantized_encoders)
    apply_accelerators(pipe, device=device, offload_mode=offload_mode, accel=accel)
    inject_compile(pipe, accel)

    if enable_vae_tiling:
        try:
            pipe.vae.enable_tiling()
            log.info("VAE tiling enabled (reduces VRAM for large frames)")
        except Exception as e:
            log.warning(f"VAE tiling failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# E2. Video step callback (Variante 1 + optional Variante 3)
# ──────────────────────────────────────────────────────────────────────────────

def make_video_step_callback(
    tracker,                        # GenProgressTracker
    enable_noise_meter: bool = False,
) -> Callable:
    """
    Build a diffusers callback_on_step_end for video pipelines (WAN + LTX).

    Variante 1 — immer aktiv:
      tracker.set_step() pro Step → Fortschrittsbalken + ETA in der UI.

    Variante 3 — opt-in via enable_noise_meter:
      latents.float().std().item() → tracker.set_noise_std()
      Funktioniert auf packed (B, seq, C) und unpacked (B, C, F, H, W) Tensoren
      gleichermaßen — std() über alle Elemente ist formunabhängig.
      float() cast verhindert NaN bei bfloat16 auf manchen Plattformen.
      Kein VAE-Decode, kein VRAM-Overhead.

    Callback-Signatur (diffusers-Standard):
      fn(pipe, step_index: int, timestep: int, callback_kwargs: dict) → dict
    """
    def _callback(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
        try:
            tracker.set_step(step_index, stage="denoising")

            if enable_noise_meter:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    std = latents.float().std().item()
                    tracker.set_noise_std(std)
        except Exception:
            pass  # nie die Generation killen

        return {}

    return _callback

# ──────────────────────────────────────────────────────────────────────────────
# F. Video frame saver
# ──────────────────────────────────────────────────────────────────────────────

def save_video_frames(
    frames: list,
    out_path: Path,
    fps: int = 24,
) -> None:
    """
    Save a list of PIL images or numpy arrays as MP4.
    Uses imageio with libx264 codec. Cross-OS via imageio[ffmpeg].
    """
    try:
        import imageio  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        raise ImportError(
            "imageio[ffmpeg] is required for video saving: "
            "pip install 'imageio[ffmpeg]'"
        )

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        if hasattr(frame, "numpy"):          # torch.Tensor
            arr = frame.numpy()
        elif hasattr(frame, "convert"):      # PIL.Image
            arr = np.array(frame.convert("RGB"))
        else:
            arr = np.asarray(frame)
        writer.append_data(arr)
    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# G. Output path + metadata
# ──────────────────────────────────────────────────────────────────────────────

def build_video_output_path(
    kind: str,
    model_id: str,
    seed: int,
    outputs_dir: Union[str, Path],
    custom: Optional[Union[str, Path]] = None,
) -> Path:
    """Build a datestamped .mp4 output path under outputs_dir/<date>/."""
    if custom is not None:
        return Path(custom)

    outputs_dir = Path(outputs_dir)
    date_str    = datetime.now().strftime("%Y-%m-%d")
    out_dir     = outputs_dir / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob(f"{kind}_*"))
    idx      = len(existing) + 1
    fname    = f"{kind}_{idx:04d}_{model_id}_seed{seed}.mp4"
    return out_dir / fname


def build_video_output_meta(
    pipeline_name: str,
    model_id: str,
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
    """Build sidecar metadata dict for a video GenResult."""
    loras_meta = [
        {"path": p, "weight": w}
        for p, w in (_parse_lora_spec(s) for s in lora_specs)
    ]
    meta = {
        "pipeline":       pipeline_name,
        "model":          model_id,
        "prompt":         prompt,
        "negative_prompt": negative_prompt,
        "width":          width,
        "height":         height,
        "frames":         frames,
        "fps":            fps,
        "steps":          steps,
        "guidance_scale": guidance_scale,
        "seed":           seed,
        "loras":          loras_meta,
        "accel":          list(accel),
        "sampler":        sampler,
        "elapsed_s":      round(elapsed_s, 2),
        "timestamp":      datetime.now().isoformat(),
        "output_path":    str(output_path),
    }
    if extra:
        meta.update(extra)
    return meta
