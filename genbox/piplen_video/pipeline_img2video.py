"""
genbox/pipeline_img2video.py
Unified Image-to-Video dispatcher.

Routes to the correct backend (WAN or LTX) based on model architecture,
creating the appropriate sub-config and calling the backend's generate().

Supported backends:
  wan → pipeline_wan.generate()  (WAN 2.1 I2V-1.3B / 14B, WAN 2.2 I2V-A14B)
  ltx → pipeline_ltx.generate()  (LTXV classic, distilled 13B, LTX-2)

100 % offline after installation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from genbox.utils.utils_video_pipeline import (
    apply_video_accelerators,
    build_video_output_meta,
    detect_wan_variant,
    detect_ltx_variant,
    ltx_generation_defaults,
    snap_frames,
    wan_generation_defaults,
)
from genbox.piplen_video.pipeline_wan import WanPipelineConfig
from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig

log = logging.getLogger("genbox.pipeline_img2video")

_SUPPORTED_I2V_BACKENDS = {"wan", "ltx"}


# ── Backend detection ──────────────────────────────────────────────────────────

def detect_i2v_backend(entry) -> str:
    """
    Detect the correct I2V backend from a ModelEntry.
    Returns "wan" or "ltx". Raises ValueError for unsupported architectures.
    """
    arch = getattr(entry, "architecture", "")
    if arch == "wan":
        return "wan"
    if arch == "ltx":
        return "ltx"
    raise ValueError(
        f"Image-to-video not supported for architecture {arch!r}. "
        f"Supported: wan, ltx."
    )


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Img2VideoConfig:
    """
    Unified I2V config. Provides shared defaults and delegates to sub-configs.

    backend: "wan" | "ltx"
    All zero/None numeric values are filled from the backend's default table.
    """
    model_id:        str
    backend:         str
    prompt:          str   = ""
    negative_prompt: str   = ""
    image:           Optional[Union[str, Path]] = None
    width:           int   = 0
    height:          int   = 0
    frames:          int   = 0
    fps:             int   = 0
    steps:           int   = 0
    guidance_scale:  float = 0.0
    seed:            int   = -1
    sampler:         str   = "default"
    loras:           list  = field(default_factory=list)
    accel:           list  = field(default_factory=list)
    output:          Optional[Union[str, Path]] = None

    def __post_init__(self):
        if self.backend not in _SUPPORTED_I2V_BACKENDS:
            raise ValueError(
                f"Unsupported I2V backend: {self.backend!r}. "
                f"Supported: {sorted(_SUPPORTED_I2V_BACKENDS)}"
            )

        if self.backend == "wan":
            variant = detect_wan_variant(self.model_id)
            d = wan_generation_defaults(variant)
        else:
            # ltx: use classic defaults as baseline (caller should set variant)
            d = ltx_generation_defaults("classic")

        if self.width          == 0:   self.width          = d["width"]
        if self.height         == 0:   self.height         = d["height"]
        if self.frames         == 0:   self.frames         = d["frames"]
        if self.fps            == 0:   self.fps            = d["fps"]
        if self.steps          == 0:   self.steps          = d["steps"]
        if self.guidance_scale == 0.0: self.guidance_scale = d["guidance_scale"]

        # Snap frames to backend constraint
        arch = "wan" if self.backend == "wan" else "ltx"
        self.frames = snap_frames(self.frames, arch)


# ── Sub-config builder ─────────────────────────────────────────────────────────

def build_i2v_config_from_entry(
    cfg: Img2VideoConfig,
    entry,
) -> Union[WanPipelineConfig, LtxPipelineConfig]:
    """
    Translate an Img2VideoConfig into the backend-specific sub-config
    (WanPipelineConfig or LtxPipelineConfig), setting mode="i2v".
    """
    if cfg.backend == "wan":
        return WanPipelineConfig(
            model_id        = cfg.model_id,
            mode            = "i2v",
            prompt          = cfg.prompt,
            negative_prompt = cfg.negative_prompt,
            width           = cfg.width,
            height          = cfg.height,
            frames          = cfg.frames,
            fps             = cfg.fps,
            steps           = cfg.steps,
            guidance_scale  = cfg.guidance_scale,
            seed            = cfg.seed,
            sampler         = cfg.sampler,
            loras           = list(cfg.loras),
            accel           = list(cfg.accel),
            image           = cfg.image,
            output          = cfg.output,
        )
    else:
        # LTX: detect variant from entry
        variant = detect_ltx_variant(
            getattr(entry, "hf_pipeline_repo", ""),
            entry.id,
        )
        return LtxPipelineConfig(
            model_id        = cfg.model_id,
            variant         = variant,
            mode            = "i2v",
            prompt          = cfg.prompt,
            negative_prompt = cfg.negative_prompt,
            width           = cfg.width,
            height          = cfg.height,
            frames          = cfg.frames,
            fps             = cfg.fps,
            steps           = cfg.steps,
            guidance_scale  = cfg.guidance_scale,
            seed            = cfg.seed,
            sampler         = cfg.sampler,
            loras           = list(cfg.loras),
            accel           = list(cfg.accel),
            image           = cfg.image,
            output          = cfg.output,
        )


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_i2v_output_meta(
    backend: str,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    input_image: Union[str, Path],
    width: int, height: int, frames: int, fps: int,
    steps: int, guidance_scale: float, seed: int,
    lora_specs: list, accel: list,
    elapsed_s: float,
    output_path: Path,
    extra: Optional[dict] = None,
) -> dict:
    meta = build_video_output_meta(
        pipeline_name=f"img2video_{backend}",
        model_id=model_id,
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, frames=frames, fps=fps,
        steps=steps, guidance_scale=guidance_scale, seed=seed,
        lora_specs=lora_specs, accel=accel, sampler="default",
        elapsed_s=elapsed_s, output_path=output_path,
    )
    meta["backend"]      = backend
    meta["input_image"]  = str(input_image)
    if extra:
        meta.update(extra)
    return meta


# ── Accelerator entry point ────────────────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe, device: str, vram_gb: int = 16,
    accel: Optional[list] = None,
    enable_vae_tiling: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """Proxy to video accelerators."""
    apply_video_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=accel or [],
        enable_vae_tiling=enable_vae_tiling,
        env_override=env_override,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def image_to_video(
    cfg: Img2VideoConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
    enable_vae_tiling: bool = False,
) -> dict:
    """
    Run image-to-video generation.
    cfg.image must be set to a valid image path.

    Automatically routes to pipeline_wan or pipeline_ltx based on cfg.backend.

    Returns dict with keys: output_path, metadata, elapsed_s
    """
    if cfg.image is None:
        raise ValueError("Img2VideoConfig.image must be set to a start-frame path")

    backend = detect_i2v_backend(entry)  # validate entry matches config
    if backend != cfg.backend:
        log.warning(
            f"Entry architecture implies backend={backend!r} "
            f"but cfg.backend={cfg.backend!r}. Using entry-detected backend."
        )

    sub_cfg = build_i2v_config_from_entry(cfg, entry)

    if cfg.backend == "wan":
        from genbox.piplen_video.pipeline_wan import generate as _wan_gen
        return _wan_gen(
            sub_cfg, entry, models_dir,
            loras_dir=loras_dir, outputs_dir=outputs_dir,
            vram_gb=vram_gb, enable_vae_tiling=enable_vae_tiling,
        )
    else:
        from genbox.piplen_video.pipeline_ltx import generate as _ltx_gen
        return _ltx_gen(
            sub_cfg, entry, models_dir,
            loras_dir=loras_dir, outputs_dir=outputs_dir,
            vram_gb=vram_gb,
        )
