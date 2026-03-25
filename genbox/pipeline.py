"""
genbox/pipeline.py
Unified public API over all genbox sub-pipelines.

Public interface (used by CLI and UI):
  text_to_image(prompt, model, ...)
  image_to_image(prompt, input_image, ...)
  inpaint(prompt, input_image, mask_image, ...)
  outpaint(prompt, input_image, left, right, top, bottom, ...)
  text_to_video(prompt, model, ...)
  image_to_video(prompt, start_frame, model, ...)

Returns GenResult objects with .output_path, .metadata, .elapsed_s,
.save() and .remix().

Model resolution:
  - model=None  → config default (cfg.default_image_model / cfg.default_video_model)
  - model="id"  → REGISTRY lookup → ModelEntry
  - entry is passed to the correct sub-pipeline based on entry.architecture

Routing:
  image  flux           → pipeline_flux
  image  sd15/sdxl/sd35 → pipeline_sdl  (pony_xl → pipeline_pony)
  image  sdxl + "pony"  → pipeline_pony
  video  ltx            → pipeline_ltx
  video  wan            → pipeline_wan
  img2img               → pipeline_img2img  (all image archs)
  inpaint               → pipeline_inpaint  (all image archs)
  outpaint              → pipeline_outpaint (all image archs)
  img2video             → pipeline_img2video (wan + ltx)

100 % offline after model installation.
Heavy imports are lazy — only loaded when a pipeline is actually called.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import shutil

log = logging.getLogger("genbox.pipeline")

LoraSpec = Union[str, tuple[str, float]]

# ── GenResult ─────────────────────────────────────────────────────────────────

@dataclass
class GenResult:
    """Returned by every pipeline function. Holds output path + full metadata."""

    output_path: Path
    metadata:    dict  = field(default_factory=dict)
    elapsed_s:   float = 0.0

    def save(self, dest: Optional[Union[str, Path]] = None) -> Path:
        """
        Optionally copy output file to dest, then write sidecar .json.
        Returns final output path.
        """
        if dest is not None:
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.output_path, dest)
            self.output_path = dest

        meta_path = self.output_path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(self.metadata, indent=2, default=str),
            encoding="utf-8",
        )
        return self.output_path

    def remix(self, **overrides) -> dict:
        """
        Return a kwargs dict from this result's params + overrides.
        Pass directly to any pipeline function to remix.

            params = result.remix(seed=99, steps=40)
            new = pipeline.text_to_image(**params)
        """
        params = {
            k: v for k, v in self.metadata.items()
            if k not in ("timestamp", "elapsed_s", "output_path")
        }
        params.update(overrides)
        return params

    def __repr__(self):
        return (
            f"<GenResult {self.output_path.name}"
            f"  model={self.metadata.get('model')}"
            f"  seed={self.metadata.get('seed')}"
            f"  {self.elapsed_s:.1f}s>"
        )


# ── Config / directory helpers ────────────────────────────────────────────────

def _get_cfg():
    from genbox.config import cfg
    return cfg


def _models_dir() -> Path:
    cfg = _get_cfg()
    return cfg.models_dir if cfg else Path.home() / ".genbox" / "models"


def _loras_dir() -> Optional[Path]:
    cfg = _get_cfg()
    return cfg.loras_dir if cfg else None


def _outputs_dir() -> Path:
    cfg = _get_cfg()
    return cfg.outputs_dir if cfg else Path.cwd() / "genbox_outputs"


def _vram_gb() -> int:
    cfg = _get_cfg()
    return cfg.vram_gb if cfg else 16


def _default_accels() -> list[str]:
    cfg = _get_cfg()
    return cfg.active_accels if cfg else []


# ── Model resolution ──────────────────────────────────────────────────────────

def _resolve_model(model_id: Optional[str], kind: str = "image"):
    """
    Resolve model_id → (model_id, ModelEntry).
    Falls back to config defaults when model_id is None.
    """
    from genbox.models import get, REGISTRY

    cfg = _get_cfg()

    if model_id is None:
        if cfg is not None:
            model_id = (
                cfg.default_image_model if kind == "image"
                else cfg.default_video_model
            )
        else:
            model_id = "flux2_klein" if kind == "image" else "ltx2_fp8"

    return model_id, get(model_id)


def _is_pony(entry) -> bool:
    return "pony" in entry.id.lower() or "pony" in " ".join(entry.tags).lower()


# ── dict → GenResult adapter ──────────────────────────────────────────────────

def _wrap(result: dict) -> GenResult:
    """Convert sub-pipeline result dict to GenResult and write sidecar."""
    gr = GenResult(
        output_path=result["output_path"],
        metadata=result["metadata"],
        elapsed_s=result["elapsed_s"],
    )
    gr.save()
    return gr


# ══════════════════════════════════════════════════════════════════════════════
# TEXT → IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def text_to_image(
    prompt: str,
    model: Optional[str] = None,
    negative_prompt: str = "",
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    t5_mode: str = "fp16",
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
) -> GenResult:
    """
    Text → Image. Routes to pipeline_flux, pipeline_sdl, or pipeline_pony
    based on model architecture.
    """
    model_id, entry = _resolve_model(model, "image")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()
    loras = loras or []
    arch  = entry.architecture

    if arch == "flux":
        from genbox.pipline_image.pipeline_flux import FluxPipelineConfig, text_to_image as _gen
        cfg_obj = FluxPipelineConfig(
            model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
            width=width or 1024, height=height or 1024,
            steps=steps or 28, guidance_scale=guidance_scale or 3.5,
            seed=seed, t5_mode=t5_mode, sampler=sampler,
            loras=loras, accel=accel, output=output,
        )
        return _wrap(_gen(cfg_obj, entry, _models_dir(),
                         loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                         vram_gb=_vram_gb()))

    if arch in ("sd15", "sdxl", "sd35"):
        if _is_pony(entry):
            from genbox.pipline_image.pipeline_pony import PonyPipelineConfig, text_to_image as _gen
            cfg_obj = PonyPipelineConfig(
                model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                width=width or 1024, height=height or 1024,
                steps=steps or 30, guidance_scale=guidance_scale or 7.0,
                seed=seed, sampler=sampler, loras=loras, accel=accel, output=output,
            )
        else:
            from genbox.pipline_image.pipeline_sdl import SDLPipelineConfig, text_to_image as _gen
            cfg_obj = SDLPipelineConfig(
                model_id=model_id, architecture=arch,
                prompt=prompt, negative_prompt=negative_prompt,
                width=width or 0, height=height or 0,
                steps=steps or 0, guidance_scale=guidance_scale or 0.0,
                seed=seed, sampler=sampler, loras=loras, accel=accel, output=output,
            )
        return _wrap(_gen(cfg_obj, entry, _models_dir(),
                         loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                         vram_gb=_vram_gb()))

    raise ValueError(
        f"text_to_image: unsupported architecture {arch!r} for model {model_id!r}. "
        "Use an image model (flux, sd15, sdxl, sd35)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE → IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def image_to_image(
    prompt: str,
    input_image: Union[str, Path],
    model: Optional[str] = None,
    negative_prompt: str = "",
    strength: float = 0.75,
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    t5_mode: str = "fp16",
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
) -> GenResult:
    """Image → Image (img2img). Supports FLUX, SD1.5, SDXL, SD3.5."""
    model_id, entry = _resolve_model(model, "image")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()

    from genbox.piplen_video.pipeline_img2img import Img2ImgConfig, image_to_image as _gen
    cfg_obj = Img2ImgConfig(
        model_id=model_id, architecture=entry.architecture,
        prompt=prompt, negative_prompt=negative_prompt,
        input_image=Path(input_image), strength=strength,
        width=width or 0, height=height or 0,
        steps=steps or 0, guidance_scale=guidance_scale or 0.0,
        seed=seed, t5_mode=t5_mode, sampler=sampler,
        loras=loras or [], accel=accel, output=output,
    )
    return _wrap(_gen(cfg_obj, entry, _models_dir(),
                      loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                      vram_gb=_vram_gb()))


# ══════════════════════════════════════════════════════════════════════════════
# INPAINT
# ══════════════════════════════════════════════════════════════════════════════

def inpaint(
    prompt: str,
    input_image: Union[str, Path],
    mask_image: Union[str, Path],
    model: Optional[str] = None,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.99,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
    blur_radius: float = 0,
    dilate_pixels: int = 0,
    mask_mode: str = "white_inpaint",
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    t5_mode: str = "fp16",
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
) -> GenResult:
    """
    Inpainting — fill masked region guided by prompt.
    mask_mode: "white_inpaint" (white=fill) | "black_inpaint" (black=fill)
    blur_radius: soften mask edges (Gaussian blur radius in pixels)
    dilate_pixels: expand mask region outward
    """
    model_id, entry = _resolve_model(model, "image")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()

    from genbox.pipline_image.pipeline_inpaint import InpaintConfig, inpaint as _gen
    cfg_obj = InpaintConfig(
        model_id=model_id, architecture=entry.architecture,
        prompt=prompt, negative_prompt=negative_prompt,
        input_image=Path(input_image), mask_image=Path(mask_image),
        width=width, height=height, strength=strength,
        steps=steps, guidance_scale=guidance_scale,
        seed=seed, t5_mode=t5_mode, sampler=sampler,
        blur_radius=blur_radius, dilate_pixels=dilate_pixels,
        mask_mode=mask_mode,
        loras=loras or [], accel=accel, output=output,
    )
    return _wrap(_gen(cfg_obj, entry, _models_dir(),
                      loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                      vram_gb=_vram_gb()))


# ══════════════════════════════════════════════════════════════════════════════
# OUTPAINT
# ══════════════════════════════════════════════════════════════════════════════

def outpaint(
    prompt: str,
    input_image: Union[str, Path],
    model: Optional[str] = None,
    negative_prompt: str = "",
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    feather_radius: float = 16.0,
    strength: float = 0.99,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    t5_mode: str = "fp16",
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
) -> GenResult:
    """
    Outpainting — extend image beyond its borders.
    Specify expansion amounts in pixels: left, right, top, bottom.
    feather_radius: smooth seam blending (Gaussian radius).
    """
    model_id, entry = _resolve_model(model, "image")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()

    from genbox.pipline_image.pipeline_outpaint import OutpaintConfig, outpaint as _gen
    cfg_obj = OutpaintConfig(
        model_id=model_id, architecture=entry.architecture,
        prompt=prompt, negative_prompt=negative_prompt,
        input_image=Path(input_image),
        left=left, right=right, top=top, bottom=bottom,
        feather_radius=feather_radius, strength=strength,
        steps=steps, guidance_scale=guidance_scale,
        seed=seed, t5_mode=t5_mode, sampler=sampler,
        loras=loras or [], accel=accel, output=output,
    )
    return _wrap(_gen(cfg_obj, entry, _models_dir(),
                      loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                      vram_gb=_vram_gb()))


# ══════════════════════════════════════════════════════════════════════════════
# TEXT → VIDEO
# ══════════════════════════════════════════════════════════════════════════════

def text_to_video(
    prompt: str,
    model: Optional[str] = None,
    negative_prompt: str = "",
    width: int = 0,
    height: int = 0,
    frames: int = 0,
    fps: int = 0,
    steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
    enable_vae_tiling: bool = False,
) -> GenResult:
    """
    Text → Video. Routes to pipeline_wan or pipeline_ltx.
    """
    model_id, entry = _resolve_model(model, "video")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()
    arch  = entry.architecture

    if arch == "wan":
        from genbox.piplen_video.pipeline_wan import WanPipelineConfig, generate as _gen
        cfg_obj = WanPipelineConfig(
            model_id=model_id, mode="t2v",
            prompt=prompt, negative_prompt=negative_prompt,
            width=width or 0, height=height or 0,
            frames=frames or 0, fps=fps or 0,
            steps=steps or 0, guidance_scale=guidance_scale or 0.0,
            seed=seed, sampler=sampler, loras=loras or [], accel=accel,
            output=output,
        )
        return _wrap(_gen(cfg_obj, entry, _models_dir(),
                          loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                          vram_gb=_vram_gb(), enable_vae_tiling=enable_vae_tiling))

    if arch == "ltx":
        from genbox.piplen_video.pipeline_ltx import LtxPipelineConfig, generate as _gen
        from genbox.utils.utils_video_pipeline import detect_ltx_variant
        variant = detect_ltx_variant(entry.hf_pipeline_repo, entry.id)
        cfg_obj = LtxPipelineConfig(
            model_id=model_id, variant=variant, mode="t2v",
            prompt=prompt, negative_prompt=negative_prompt,
            width=width or 0, height=height or 0,
            frames=frames or 0, fps=fps or 0,
            steps=steps or 0, guidance_scale=guidance_scale or 0.0,
            seed=seed, sampler=sampler, loras=loras or [], accel=accel,
            output=output,
        )
        return _wrap(_gen(cfg_obj, entry, _models_dir(),
                          loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                          vram_gb=_vram_gb(), enable_vae_tiling=enable_vae_tiling))

    raise ValueError(
        f"text_to_video: unsupported architecture {arch!r} for {model_id!r}. "
        "Use a video model (ltx, wan)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE → VIDEO
# ══════════════════════════════════════════════════════════════════════════════

def image_to_video(
    prompt: str,
    start_frame: Union[str, Path],
    model: Optional[str] = None,
    end_frame: Optional[Union[str, Path]] = None,
    negative_prompt: str = "",
    width: int = 0,
    height: int = 0,
    frames: int = 0,
    fps: int = 0,
    steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    loras: Optional[list[LoraSpec]] = None,
    accel: Optional[list[str]] = None,
    sampler: str = "default",
    output: Optional[Union[str, Path]] = None,
    enable_vae_tiling: bool = False,
) -> GenResult:
    """
    Image → Video. Routes to pipeline_img2video (dispatches to wan / ltx).
    end_frame is supported for LTX-2 FLF (first-last-frame) mode.
    """
    model_id, entry = _resolve_model(model, "video")
    require_installed(model_id)
    accel = accel if accel is not None else _default_accels()
    arch  = entry.architecture

    from genbox.piplen_video.pipeline_img2video import Img2VideoConfig, image_to_video as _gen
    cfg_obj = Img2VideoConfig(
        model_id=model_id, backend=arch,
        prompt=prompt, negative_prompt=negative_prompt,
        image=Path(start_frame),
        width=width or 0, height=height or 0,
        frames=frames or 0, fps=fps or 0,
        steps=steps or 0, guidance_scale=guidance_scale or 0.0,
        seed=seed, sampler=sampler, loras=loras or [], accel=accel,
        output=output,
    )
    return _wrap(_gen(cfg_obj, entry, _models_dir(),
                      loras_dir=_loras_dir(), outputs_dir=_outputs_dir(),
                      vram_gb=_vram_gb(), enable_vae_tiling=enable_vae_tiling))


# ══════════════════════════════════════════════════════════════════════════════
# Model metadata helpers (public API for UI / scripts)
# ══════════════════════════════════════════════════════════════════════════════

def list_models(
    model_type: Optional[str] = None,
    installed_only: bool = False,
    max_vram: Optional[int] = None,
) -> list:
    """Return filtered ModelEntry list from REGISTRY."""
    from genbox.models import list_registry
    cfg = _get_cfg()
    vram = max_vram if max_vram is not None else (cfg.vram_gb if cfg else 999)
    return list_registry(model_type=model_type, max_vram=vram,
                         installed_only=installed_only)


def get_model_entry(model_id: str):
    """Return a single ModelEntry by ID."""
    from genbox.models import get
    return get(model_id)


def list_installed() -> list[dict]:
    """Return list of installed models as info dicts."""
    from genbox.models import list_local
    return list_local()


def is_installed(model_id: str) -> bool:
    """Return True if a model is installed locally."""
    from genbox.models import REGISTRY, _is_installed_entry, _discover_local_custom_models
    _discover_local_custom_models()
    if model_id not in REGISTRY:
        return False
    return _is_installed_entry(REGISTRY[model_id])


def require_installed(model_id: str) -> None:
    """
    Raise RuntimeError with a helpful message if the model is not installed.
    Call this at the top of each generation function before loading ML deps.
    """
    if not is_installed(model_id):
        raise RuntimeError(
            f"Model '{model_id}' is not installed.\n"
            f"Install it with:  genbox models download {model_id}\n"
            f"Or install all defaults:  genbox models install-defaults"
        )


def download(model_id: str, force: bool = False) -> Path:
    """Download a model from HuggingFace. Returns local path."""
    from genbox.models import get, download_model, _is_installed_entry
    entry = get(model_id)
    if not force and _is_installed_entry(entry):
        log.info(f"{model_id} already installed")
        cfg_local = _get_cfg()
        arch_dir = (_models_dir() / entry.architecture)
        if entry.full_repo:
            return arch_dir / entry.id
        return arch_dir / Path(entry.hf_filename).name
    return download_model(entry)


def install_defaults(profile: Optional[str] = None, dry_run: bool = False) -> list[str]:
    """Download all default models for the current VRAM profile."""
    from genbox.models import install_defaults as _install
    return _install(profile=profile, dry_run=dry_run)


def uninstall(model_id: str) -> bool:
    """Remove a locally installed model."""
    from genbox.models import uninstall_model
    return uninstall_model(model_id)


def write_lora_metadata(
    lora_path: Union[str, Path],
    architecture: str,
    trigger: str = "",
    description: str = "",
    preview_url: str = "",
) -> None:
    """Write sidecar .json for a LoRA file."""
    from genbox.utils.utils import write_lora_metadata as _write
    _write(Path(lora_path), architecture=architecture, trigger=trigger,
           description=description, preview_url=preview_url)


def write_model_metadata(
    model_path: Union[str, Path],
    description: str = "",
    preview_url: str = "",
    tags: Optional[list[str]] = None,
) -> None:
    """Write sidecar .json for a model file."""
    from genbox.utils.utils import write_model_metadata as _write
    _write(Path(model_path), description=description,
           preview_url=preview_url, tags=tags)


def register_custom(
    src: Union[str, Path],
    architecture: str,
    description: str = "",
    preview_url: str = "",
) -> dict:
    """Register a custom .safetensors or .gguf model file."""
    from genbox.models import register_custom_model
    return register_custom_model(Path(src), architecture=architecture,
                                 description=description, preview_url=preview_url)