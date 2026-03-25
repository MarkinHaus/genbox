"""
genbox/models.py
Model registry, HuggingFace download, VRAM-gating, LoRA management.
Compatible with genbox/utils.py (delegates list_loras / metadata to utils).

Registry covers (2025-03):
  IMAGE: FLUX.1 (schnell/dev GGUF Q4/Q8), FLUX.2 Klein (4B/9B fp8 + GGUF),
         Z-Image-Turbo, SD3.5-Medium, SD1.5, SDXL, SDXL-Turbo, Animagine XL,
         Pony Diffusion XL
  VIDEO: LTX-Video 0.9.5, LTX-Video 0.9.7-distilled 13B,
         LTX-Video 0.9.8-dev 13B, LTX-2,
         WAN 2.1 T2V 1.3B/14B, WAN 2.1 I2V 480P/720P, WAN 2.1 FLF2V 14B,
         WAN 2.2 T2V A14B, WAN 2.2 I2V A14B, WAN 2.2 TI2V 5B
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from genbox.config import cfg

# ── ModelEntry ────────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    id:               str        # registry key, e.g. "flux2_klein"
    name:             str        # display name
    type:             str        # "image" | "video"
    architecture:     str        # "flux" | "sd35" | "sd15" | "sdxl" | "ltx" | "wan"
    vram_min_gb:      int        # minimum VRAM to run
    hf_repo:          str        # HuggingFace repo id
    hf_filename:      str        # file within repo OR "model_index.json" for full-repos
    license:          str
    quant:            str        # "fp16" | "fp8" | "bf16" | "gguf-q8" | "gguf-q4"
    quality_stars:    int        # 1–5
    speed_stars:      int        # 1–5
    tags:             list[str]  = field(default_factory=list)
    notes:            str        = ""
    hf_pipeline_repo: str        = ""   # shared config repo for GGUF
    full_repo:        bool       = False # True → snapshot_download
    # I2V flag: True → model supports Image-to-Video natively
    supports_i2v:     bool       = False
    # Video mode: "t2v" | "i2v" | "flf2v" | "ti2v"
    video_mode:       str        = "t2v"

    @property
    def size_label(self) -> str:
        return self.hf_filename.split("/")[-1]

    def fits_vram(self, available_gb: int) -> bool:
        return available_gb >= self.vram_min_gb

    def stars(self, n: int) -> str:
        return "★" * n + "☆" * (5 - n)

    def is_gguf(self) -> bool:
        return "gguf" in self.quant.lower()


# ── Built-in REGISTRY ─────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelEntry] = {

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE — FLUX.2 Klein
    # ══════════════════════════════════════════════════════════════════════════
    "flux2_klein": ModelEntry(
        id="flux2_klein", name="FLUX.2 Klein 4B",
        type="image", architecture="flux", vram_min_gb=10,
        hf_repo="black-forest-labs/FLUX.2-klein-4B",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="fp8", quality_stars=5, speed_stars=4, full_repo=True,
        tags=["text2img", "img2img", "inpaint", "lora"],
        notes="Default image model. Full diffusers repo. T2I + I2I + Inpaint.",
    ),
    "flux2_klein9b": ModelEntry(
        id="flux2_klein9b", name="FLUX.2 Klein 9B",
        type="image", architecture="flux", vram_min_gb=12,
        hf_repo="black-forest-labs/FLUX.2-klein-9B",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="fp8", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["text2img", "img2img", "inpaint", "lora", "9b"],
        notes="9B variant — highest quality, Qwen3-8B encoder. 12GB+ VRAM.",
    ),
    "unsloth_flux2_9b_q8": ModelEntry(
        id="unsloth_flux2_9b_q8", name="FLUX.2 Klein 9B (Unsloth Q8)",
        type="image", architecture="flux", vram_min_gb=12,
        hf_repo="unsloth/FLUX.2-klein-9B-GGUF",
        hf_filename="flux-2-klein-9b-Q8_0.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.2-klein-9B",
        license="flux-non-commercial",
        quant="gguf-q8", quality_stars=5, speed_stars=4,
        tags=["text2img", "gguf", "9b"],
        notes="Flagship 9B in Q8_0. Qwen3-8B encoder. Min 12GB VRAM.",
    ),
    "unsloth_flux2_9b_q4": ModelEntry(
        id="unsloth_flux2_9b_q4", name="FLUX.2 Klein 9B (Unsloth Q4)",
        type="image", architecture="flux", vram_min_gb=8,
        hf_repo="unsloth/FLUX.2-klein-9B-GGUF",
        hf_filename="flux-2-klein-9b-Q4_K_S.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.2-klein-9B",
        license="flux-non-commercial",
        quant="gguf-q4", quality_stars=5, speed_stars=5,
        tags=["text2img", "gguf", "9b", "low-vram"],
        notes="9B quality at ~6GB. Q4_K_S quantization.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE — FLUX.1 GGUF
    # ══════════════════════════════════════════════════════════════════════════
    "flux1_schnell_q8": ModelEntry(
        id="flux1_schnell_q8", name="FLUX.1 Schnell (GGUF Q8)",
        type="image", architecture="flux", vram_min_gb=8,
        hf_repo="city96/FLUX.1-schnell-gguf",
        hf_filename="flux1-schnell-Q8_0.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-schnell",
        license="Apache 2.0",
        quant="gguf-q8", quality_stars=4, speed_stars=5,
        tags=["text2img", "fast", "gguf", "4-step"],
        notes="Fastest option. 4-step generation. Q8_0.",
    ),
    "flux1_schnell_q4": ModelEntry(
        id="flux1_schnell_q4", name="FLUX.1 Schnell (GGUF Q4)",
        type="image", architecture="flux", vram_min_gb=6,
        hf_repo="city96/FLUX.1-schnell-gguf",
        hf_filename="flux1-schnell-Q4_K_S.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-schnell",
        license="Apache 2.0",
        quant="gguf-q4", quality_stars=3, speed_stars=5,
        tags=["text2img", "fast", "gguf", "low-vram"],
        notes="6GB VRAM. Fastest on low-end hardware.",
    ),
    "flux1_dev_q8": ModelEntry(
        id="flux1_dev_q8", name="FLUX.1-dev (GGUF Q8)",
        type="image", architecture="flux", vram_min_gb=10,
        hf_repo="city96/FLUX.1-dev-gguf",
        hf_filename="flux1-dev-Q8_0.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-dev",
        license="flux-non-commercial",
        quant="gguf-q8", quality_stars=5, speed_stars=4,
        tags=["text2img", "gguf", "dev"],
        notes="FLUX.1-dev quality at Q8_0. Non-commercial.",
    ),
    "flux1_dev_q4": ModelEntry(
        id="flux1_dev_q4", name="FLUX.1-dev (GGUF Q4)",
        type="image", architecture="flux", vram_min_gb=8,
        hf_repo="city96/FLUX.1-dev-gguf",
        hf_filename="flux1-dev-Q4_K_S.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-dev",
        license="flux-non-commercial",
        quant="gguf-q4", quality_stars=5, speed_stars=5,
        tags=["text2img", "gguf", "dev", "low-vram"],
        notes="FLUX.1-dev quality at Q4_K_S. 8GB VRAM.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE — Other FLUX-class
    # ══════════════════════════════════════════════════════════════════════════
    "z_image_turbo": ModelEntry(
        id="z_image_turbo", name="Z-Image-Turbo 6B",
        type="image", architecture="flux", vram_min_gb=9,
        hf_repo="Tongyi-MAI/Z-Image-Turbo",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="fp8", quality_stars=5, speed_stars=5, full_repo=True,
        tags=["text2img", "fast", "lora"],
        notes="FLUX-class quality at lower footprint. Best speed/quality ratio.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE — SDL variants
    # ══════════════════════════════════════════════════════════════════════════
    "sd35_medium": ModelEntry(
        id="sd35_medium", name="SD 3.5 Medium 2B",
        type="image", architecture="sd35", vram_min_gb=6,
        hf_repo="stabilityai/stable-diffusion-3.5-medium",
        hf_filename="model_index.json",
        license="Stability Non-Commercial",
        quant="fp16", quality_stars=3, speed_stars=4, full_repo=True,
        tags=["text2img", "img2img", "inpaint", "lora", "controlnet"],
        notes="Largest LoRA + ControlNet ecosystem. 6GB VRAM.",
    ),
    "sd15_base": ModelEntry(
        id="sd15_base", name="Stable Diffusion 1.5 (Base)",
        type="image", architecture="sd15", vram_min_gb=4,
        hf_repo="runwayml/stable-diffusion-v1-5",
        hf_filename="model_index.json",
        license="CreativeML Open RAIL++-M",
        quant="fp16", quality_stars=3, speed_stars=4, full_repo=True,
        tags=["text2img", "img2img", "inpaint", "lora", "controlnet", "low-vram"],
        notes="Classic SD 1.5. Largest LoRA ecosystem. 4GB VRAM.",
    ),
    "sd15_realistic": ModelEntry(
        id="sd15_realistic", name="Realistic Vision V6.0 B1",
        type="image", architecture="sd15", vram_min_gb=4,
        hf_repo="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        hf_filename="model_index.json",
        license="CreativeML Open RAIL++-M",
        quant="fp16", quality_stars=4, speed_stars=4, full_repo=True,
        tags=["text2img", "img2img", "photorealistic", "lora"],
        notes="Photorealistic portrait/studio. No VAE bundled.",
    ),
    "sdxl_base": ModelEntry(
        id="sdxl_base", name="Stable Diffusion XL 1.0 (Base)",
        type="image", architecture="sdxl", vram_min_gb=8,
        hf_repo="stabilityai/stable-diffusion-xl-base-1.0",
        hf_filename="model_index.json",
        license="CreativeML Open RAIL++-M",
        quant="fp16", quality_stars=4, speed_stars=3, full_repo=True,
        tags=["text2img", "img2img", "inpaint", "lora", "high-res"],
        notes="SDXL base. Best quality/ecosystem balance. 8GB VRAM.",
    ),
    "sdxl_turbo": ModelEntry(
        id="sdxl_turbo", name="SDXL Turbo",
        type="image", architecture="sdxl", vram_min_gb=5,
        hf_repo="stabilityai/sdxl-turbo",
        hf_filename="model_index.json",
        license="Stability AI Community",
        quant="fp16", quality_stars=4, speed_stars=5, full_repo=True,
        tags=["text2img", "fast", "1-step", "lora"],
        notes="Fast SDXL — 1-4 steps. 5GB VRAM.",
    ),
    "animagine_xl": ModelEntry(
        id="animagine_xl", name="Animagine XL 4.0",
        type="image", architecture="sdxl", vram_min_gb=8,
        hf_repo="cagliostrolab/animagine-xl-4.0",
        hf_filename="model_index.json",
        license="CreativeML Open RAIL++-M",
        quant="fp16", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["text2img", "anime", "illustration", "lora"],
        notes="Best open anime illustration model.",
    ),
    "pony_xl": ModelEntry(
        id="pony_xl", name="Pony Diffusion v6 XL",
        type="image", architecture="sdxl", vram_min_gb=8,
        hf_repo="AingCreativeLab/pony-diffusion-v6-xl",
        hf_filename="model_index.json",
        license="pony LICENSE",
        quant="fp16", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["text2img", "anime", "character", "lora", "rating-tags", "pony"],
        notes="Pony Diffusion. Uses quality/rating prompt tags (score_9, score_8_up, …).",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO — LTX-Video
    # ══════════════════════════════════════════════════════════════════════════
    "ltx2_fp8": ModelEntry(
        id="ltx2_fp8", name="LTX-Video 0.9.5 (bf16)",
        type="video", architecture="ltx", vram_min_gb=10,
        hf_repo="Lightricks/LTX-Video",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=4, speed_stars=5, full_repo=True,
        tags=["text2video", "img2video", "lora", "30fps"],
        notes="Classic LTXV. frames=8n+1 (e.g. 25, 97). ~10GB VRAM.",
    ),
    "ltx23_fp8": ModelEntry(
        id="ltx23_fp8", name="LTX-Video 0.9.7 distilled 13B",
        type="video", architecture="ltx", vram_min_gb=12,
        hf_repo="Lightricks/LTX-Video-0.9.7-distilled",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=5, full_repo=True,
        tags=["text2video", "img2video", "distilled", "13b", "4-8step"],
        notes="13B distilled. guidance_scale=1.0, steps=4-8. Upscaler pipeline. 12GB VRAM.",
    ),
    "ltx_098_dev": ModelEntry(
        id="ltx_098_dev", name="LTX-Video 0.9.8-dev 13B",
        type="video", architecture="ltx", vram_min_gb=12,
        hf_repo="Lightricks/LTX-Video",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=5, full_repo=True,
        tags=["text2video", "img2video", "distilled", "13b", "long-video"],
        notes="0.9.8 dev — same repo as 0.9.5, latest checkpoint. Supports long videos.",
    ),
    "ltx2_model": ModelEntry(
        id="ltx2_model", name="LTX-2 (Joint Audio-Visual)",
        type="video", architecture="ltx", vram_min_gb=14,
        hf_repo="Lightricks/LTX-2",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=4, full_repo=True,
        tags=["text2video", "img2video", "audio", "2-stage"],
        notes="LTX-2: joint audio-visual. 2-stage (base + upscaler). 14GB+ VRAM.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO — WAN 2.1 T2V
    # ══════════════════════════════════════════════════════════════════════════
    "wan_1_3b": ModelEntry(
        id="wan_1_3b", name="WAN 2.1 T2V-1.3B (480p)",
        type="video", architecture="wan", vram_min_gb=8,
        hf_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=4, speed_stars=5, full_repo=True,
        tags=["text2video", "img2video", "low-vram", "480p"],
        notes="8GB VRAM. VAE=float32 mandatory. Optimal: 480p, 81 frames.",
    ),
    "wan21_14b_diffusers": ModelEntry(
        id="wan21_14b_diffusers", name="WAN 2.1 T2V-14B (480p/720p)",
        type="video", architecture="wan", vram_min_gb=16,
        hf_repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["text2video", "img2video", "14b", "480p", "720p"],
        notes="14B diffusers repo. 16GB+ with model_cpu_offload.",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO — WAN 2.1 I2V
    # ══════════════════════════════════════════════════════════════════════════
    "wan21_i2v_480p": ModelEntry(
        id="wan21_i2v_480p", name="WAN 2.1 I2V-14B 480P",
        type="video", architecture="wan", vram_min_gb=16,
        hf_repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["img2video", "i2v", "14b", "480p"],
        notes="Image-to-Video 14B at 480P. Needs CLIPVisionModel (float32) + VAE (float32).",
        supports_i2v=True, video_mode="i2v",
    ),
    "wan21_i2v_720p": ModelEntry(
        id="wan21_i2v_720p", name="WAN 2.1 I2V-14B 720P",
        type="video", architecture="wan", vram_min_gb=20,
        hf_repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=2, full_repo=True,
        tags=["img2video", "i2v", "14b", "720p"],
        notes="Image-to-Video 14B at 720P. 20GB+ VRAM recommended.",
        supports_i2v=True, video_mode="i2v",
    ),
    "wan21_flf2v_720p": ModelEntry(
        id="wan21_flf2v_720p", name="WAN 2.1 FLF2V-14B 720P (First+Last Frame)",
        type="video", architecture="wan", vram_min_gb=20,
        hf_repo="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=2, full_repo=True,
        tags=["img2video", "flf2v", "first-last-frame", "14b", "720p"],
        notes="First+Last-Frame conditioning. Set both start and end frame for interpolation.",
        supports_i2v=True, video_mode="flf2v",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO — WAN 2.2
    # ══════════════════════════════════════════════════════════════════════════
    "wan22_1_4b": ModelEntry(
        id="wan22_1_4b", name="WAN 2.2 T2V-A14B (MoE 720p)",
        type="video", architecture="wan", vram_min_gb=24,
        hf_repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=3, full_repo=True,
        tags=["text2video", "img2video", "moe", "a14b", "720p", "high-vram"],
        notes="WAN 2.2 MoE 14B×2. 24GB+ VRAM. Best open video quality.",
    ),
    "wan22_i2v_a14b": ModelEntry(
        id="wan22_i2v_a14b", name="WAN 2.2 I2V-A14B (MoE)",
        type="video", architecture="wan", vram_min_gb=24,
        hf_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=2, full_repo=True,
        tags=["img2video", "i2v", "moe", "a14b", "480p", "720p", "high-vram"],
        notes="WAN 2.2 I2V MoE. 480P + 720P. 24GB+ VRAM.",
        supports_i2v=True, video_mode="i2v",
    ),
    "wan22_ti2v_5b": ModelEntry(
        id="wan22_ti2v_5b", name="WAN 2.2 TI2V-5B (Text+Image→Video)",
        type="video", architecture="wan", vram_min_gb=12,
        hf_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        hf_filename="model_index.json", license="Apache 2.0",
        quant="bf16", quality_stars=5, speed_stars=4, full_repo=True,
        tags=["img2video", "ti2v", "5b", "720p", "consumer"],
        notes="TI2V 5B — T2V + I2V in one model. 720P@24fps. Runs on 4090 (12GB).",
        supports_i2v=True, video_mode="ti2v",
    ),
}


# ── DEFAULT BASE MODEL SETS per VRAM profile ──────────────────────────────────
# These are the recommended models to install for a fresh setup.

DEFAULT_MODELS: dict[str, list[str]] = {
    "8gb_low": [
        "flux1_schnell_q4",      # image — 6GB, fastest
        "wan_1_3b",              # video — 8GB, T2V + I2V
        "ltx2_fp8",              # video — 10GB but with offload works on 8GB
    ],
    "8gb_balanced": [
        "flux1_schnell_q8",      # image — 8GB
        "flux1_dev_q4",          # image — dev quality
        "wan_1_3b",              # video
        "ltx2_fp8",              # video
    ],
    "12gb_balanced": [
        "flux2_klein",           # image — default, fp8 full-repo
        "flux1_dev_q8",          # image — dev quality GGUF
        "sd15_base",             # image — SD1.5 for LoRA compat
        "ltx2_fp8",              # video — classic
        "ltx23_fp8",             # video — distilled 13B
        "wan_1_3b",              # video — T2V 1.3B
    ],
    "16gb_high": [
        "flux2_klein",
        "flux2_klein9b",
        "sdxl_base",
        "sd35_medium",
        "ltx23_fp8",
        "ltx2_fp8",
        "wan_1_3b",
        "wan21_14b_diffusers",
        "wan21_i2v_480p",
    ],
    "24gb_ultra": [
        "flux2_klein",
        "flux2_klein9b",
        "unsloth_flux2_9b_q8",
        "sdxl_base",
        "animagine_xl",
        "pony_xl",
        "sd35_medium",
        "ltx23_fp8",
        "ltx2_fp8",
        "ltx2_model",
        "wan_1_3b",
        "wan21_14b_diffusers",
        "wan21_i2v_480p",
        "wan21_i2v_720p",
        "wan22_1_4b",
        "wan22_i2v_a14b",
        "wan22_ti2v_5b",
    ],
}


def get_default_models(profile: Optional[str] = None) -> list[str]:
    """
    Return the list of recommended model IDs for a VRAM profile.
    Falls back to cfg.vram_profile if profile not given.
    """
    if profile is None:
        profile = cfg.vram_profile if cfg else "8gb_low"
    return DEFAULT_MODELS.get(profile, DEFAULT_MODELS["8gb_low"])


# ── Auto-discovery of local custom files ──────────────────────────────────────

_custom_discovered = False


def reset_discovery():
    """Call after a download so list_local() picks up new files immediately."""
    global _custom_discovered
    _custom_discovered = False


def _discover_local_custom_models():
    """Scan local model dirs for user-placed GGUF/safetensors not in REGISTRY."""
    global _custom_discovered
    if _custom_discovered or cfg is None:
        return
    _custom_discovered = True

    for arch in ("flux", "sd35", "sd15", "sdxl", "ltx", "wan"):
        arch_dir = cfg.models_dir / arch
        if not arch_dir.exists():
            continue
        for file in arch_dir.glob("*"):
            if file.is_dir() or file.suffix not in (".gguf", ".safetensors"):
                continue
            if any(e.hf_filename == file.name for e in REGISTRY.values()):
                continue

            model_type = "video" if arch in ("ltx", "wan") else "image"
            notes = ""
            if model_type == "video" and ".gguf" in file.name.lower():
                notes = "⚠ GGUF video: diffusers pipeline not supported. Use full diffusers repo."

            pipe_repo = _guess_pipe_repo(arch, file.name)

            fname = file.name.lower()
            if ".gguf" in fname:
                quant = "gguf-q4" if "q4" in fname else "gguf-q8"
            elif "fp8" in fname:
                quant = "fp8"
            else:
                quant = "fp16"

            entry = ModelEntry(
                id=f"local_{file.stem}",
                name=f"[{arch.upper()}] {file.stem[:28]}",
                type=model_type, architecture=arch, vram_min_gb=8,
                hf_repo="", hf_filename=file.name,
                license="Custom", quant=quant,
                quality_stars=0, speed_stars=0,
                tags=["local", "custom"],
                hf_pipeline_repo=pipe_repo, full_repo=False,
                notes=notes,
            )
            REGISTRY[entry.id] = entry


def _guess_pipe_repo(arch: str, filename: str) -> str:
    """Heuristic: guess the shared config repo for a custom GGUF."""
    fn = filename.lower()
    if arch == "flux":
        if "9b" in fn:
            return "black-forest-labs/FLUX.2-klein-9B"
        if "flux.2" in fn or "flux2" in fn or "flux-2" in fn:
            return "black-forest-labs/FLUX.2-klein-4B"
        if "dev" in fn:
            return "black-forest-labs/FLUX.1-dev"
        return "black-forest-labs/FLUX.1-schnell"
    if arch == "sd35":
        return "stabilityai/stable-diffusion-3.5-medium"
    return ""


# ── REGISTRY helpers ───────────────────────────────────────────────────────────

def get(model_id: str) -> ModelEntry:
    """Resolve model_id to ModelEntry. Triggers custom-file discovery."""
    _discover_local_custom_models()
    if model_id not in REGISTRY:
        raise KeyError(
            f"Unknown model: {model_id!r}. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[model_id]


def list_registry(
    model_type: Optional[str] = None,
    max_vram: Optional[int] = None,
    installed_only: bool = False,
    architecture: Optional[str] = None,
) -> list[ModelEntry]:
    """Filter registry by type, VRAM limit, architecture, and install state."""
    _discover_local_custom_models()
    entries = list(REGISTRY.values())
    if model_type:
        entries = [e for e in entries if e.type == model_type]
    if architecture:
        entries = [e for e in entries if e.architecture == architecture]
    if max_vram is not None:
        entries = [e for e in entries if e.vram_min_gb <= max_vram]
    if installed_only:
        entries = [e for e in entries if _is_installed_entry(e)]
    return entries


def _is_installed_entry(entry: ModelEntry) -> bool:
    if cfg is None:
        return False
    arch_dir = cfg.models_dir / entry.architecture

    if entry.full_repo:
        return (arch_dir / entry.id / "model_index.json").exists()

    fname = Path(entry.hf_filename).name
    if not (arch_dir / fname).exists():
        return False

    # GGUF image models also need a shared config dir
    if entry.is_gguf() and entry.type == "image":
        shared = _shared_config_dir(entry)
        return shared is not None and shared.exists() and any(shared.iterdir())

    return True


def _shared_config_dir(entry: ModelEntry) -> Optional[Path]:
    """Return shared pipeline-config dir for GGUF models (or None)."""
    if cfg is None or not entry.hf_pipeline_repo:
        return None
    safe = entry.hf_pipeline_repo.replace("/", "--")
    return cfg.models_dir / entry.architecture / f"_shared_{safe}"


# ── Local model listing ────────────────────────────────────────────────────────

def list_local(model_type: Optional[str] = None) -> list[dict]:
    """Return installed models as list of info dicts."""
    _discover_local_custom_models()
    if cfg is None:
        return []

    results = []
    seen_ids: set[str] = set()

    for entry in REGISTRY.values():
        if entry.id in seen_ids:
            continue
        if model_type and entry.type != model_type:
            continue
        if not _is_installed_entry(entry):
            continue

        seen_ids.add(entry.id)
        arch_dir = cfg.models_dir / entry.architecture

        if entry.full_repo:
            d = arch_dir / entry.id
            size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 3)
            path_str = str(d)
        else:
            f = arch_dir / Path(entry.hf_filename).name
            size_gb = f.stat().st_size / (1024 ** 3) if f.exists() else 0.0
            path_str = str(f)

        results.append({
            "id":           entry.id,
            "name":         entry.name,
            "path":         path_str,
            "size_gb":      round(size_gb, 2),
            "type":         entry.type,
            "architecture": entry.architecture,
            "quant":        entry.quant,
            "tags":         entry.tags,
        })

    return results


# ── LoRA management (delegates to utils) ──────────────────────────────────────

def list_loras(architecture: Optional[str] = None) -> list[dict]:
    """List all locally available LoRA .safetensors files."""
    if cfg is None:
        return []
    from genbox.utils.utils import list_loras as _list
    return _list(loras_dir=cfg.loras_dir, architecture=architecture)


def write_lora_metadata(
    lora_path: Path,
    architecture: str,
    trigger: str = "",
    description: str = "",
    preview_url: str = "",
) -> None:
    """Write sidecar .json metadata for a LoRA file."""
    from genbox.utils.utils import write_lora_metadata as _write
    _write(lora_path, architecture=architecture, trigger=trigger,
           description=description, preview_url=preview_url)


def register_custom_model(
    src: Path,
    architecture: str,
    description: str = "",
    preview_url: str = "",
) -> dict:
    """
    Copy a custom .safetensors / .gguf into the models dir and register it.
    Returns a registry-compatible dict.
    """
    if cfg is None:
        raise RuntimeError("genbox not configured — run `genbox setup` first.")
    from genbox.utils.utils import register_custom_file
    result = register_custom_file(
        src, architecture=architecture,
        models_dir=cfg.models_dir,
        description=description,
        preview_url=preview_url,
    )
    # Force re-discovery so it shows up immediately
    reset_discovery()
    return result


# ── HuggingFace search ─────────────────────────────────────────────────────────

HF_API = "https://huggingface.co/api"


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token and cfg is not None:
        token = getattr(cfg, "hf_token", "") or ""
    return token


def _hf_search(query: str, limit: int = 10) -> list[dict]:
    """Search HuggingFace model hub. Returns list of raw model dicts."""
    params = urlencode({"search": query, "limit": limit, "sort": "downloads", "direction": -1})
    url = f"{HF_API}/models?{params}"
    headers = {"User-Agent": "genbox/1.0"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except (URLError, HTTPError) as e:
        Log.err(f"HF search failed: {e}")
        return []


# ── Download ───────────────────────────────────────────────────────────────────

def download_model(entry: ModelEntry, dest_dir: Optional[Path] = None) -> Path:
    """
    Download a model from HuggingFace.
      full_repo   → snapshot_download  → models_dir/<arch>/<id>/
      single-file → hf_hub_download    → models_dir/<arch>/<filename>
      gguf image  → + shared pipeline config → models_dir/<arch>/_shared_<repo>/

    Returns path to downloaded file or directory.
    """
    if cfg is None:
        raise RuntimeError("genbox not configured — run `genbox setup` first.")

    if dest_dir is None:
        dest_dir = cfg.models_dir / entry.architecture
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not entry.fits_vram(cfg.vram_gb):
        Log.err(
            f"Warning: {entry.name} needs {entry.vram_min_gb}GB, "
            f"you have {cfg.vram_gb}GB. Proceeding anyway…"
        )

    token = _hf_token() or None

    # ── Full diffusers repo ────────────────────────────────────────────────────
    if entry.full_repo:
        from huggingface_hub import snapshot_download
        local_dir = dest_dir / entry.id
        Log.info(f"Downloading full repo: {entry.name}")
        Log.info(f"  repo : {entry.hf_repo}")
        Log.info(f"  to   : {local_dir}")
        try:
            result = snapshot_download(
                repo_id=entry.hf_repo,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
                ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
            )
            Log.ok(f"Downloaded: {result}")
            reset_discovery()
            return Path(result)
        except Exception as e:
            Log.err(f"Snapshot download failed: {e}")
            raise

    # ── Single-file (safetensors / gguf) ──────────────────────────────────────
    from huggingface_hub import hf_hub_download

    dest_file = dest_dir / Path(entry.hf_filename).name
    Log.info(f"Downloading: {entry.name}")
    Log.info(f"  repo : {entry.hf_repo}  /  {entry.hf_filename}")
    Log.info(f"  to   : {dest_file}")

    try:
        hf_hub_download(
            repo_id=entry.hf_repo,
            filename=entry.hf_filename,
            token=token,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        Log.ok(f"Downloaded: {dest_file}")
    except Exception as e:
        Log.err(f"Download failed: {e}")
        raise

    # ── GGUF image models: download shared pipeline config ─────────────────────
    if entry.is_gguf() and entry.type == "image" and entry.hf_pipeline_repo:
        from huggingface_hub import snapshot_download
        shared_dir = _shared_config_dir(entry)
        Log.info(f"Caching shared pipeline config: {entry.hf_pipeline_repo}")
        Log.info(f"  to: {shared_dir}")
        try:
            snapshot_download(
                repo_id=entry.hf_pipeline_repo,
                local_dir=str(shared_dir),
                local_dir_use_symlinks=False,
                token=token,
                ignore_patterns=[
                    "*.gguf", "*.safetensors", "*.bin",
                    "*.pt", "*.ot", "*.msgpack", "*.h5", "flax_*",
                ],
            )
            Log.ok(f"Config cached: {shared_dir}")
        except Exception as e:
            Log.err(f"Config cache failed (offline generation not possible): {e}")
            raise

    reset_discovery()
    return dest_file


def install_defaults(
    profile: Optional[str] = None,
    dry_run: bool = False,
) -> list[str]:
    """
    Download all base models for the current (or given) VRAM profile.
    Skips already-installed models.

    Returns list of model IDs that were (or would be) downloaded.
    """
    ids = get_default_models(profile)
    to_install = [mid for mid in ids if mid in REGISTRY and not _is_installed_entry(REGISTRY[mid])]

    if not to_install:
        Log.ok("All default models already installed.")
        return []

    Log.info(f"Installing {len(to_install)} base model(s) for profile "
             f"'{profile or (cfg.vram_profile if cfg else '8gb_low')}':")
    for mid in to_install:
        Log.info(f"  {mid}")

    if dry_run:
        return to_install

    failed = []
    for mid in to_install:
        entry = REGISTRY[mid]
        try:
            download_model(entry)
        except Exception as e:
            Log.err(f"Failed: {entry.name}: {e}")
            failed.append(mid)

    installed = [m for m in to_install if m not in failed]
    if installed:
        Log.ok(f"Installed: {installed}")
    if failed:
        Log.err(f"Failed: {failed}")

    return installed


# ── Uninstall ─────────────────────────────────────────────────────────────────

def uninstall_model(model_id: str) -> bool:
    """
    Remove a locally installed model.
    Deletes full-repo dir, single-file, GGUF config dir, and shared config.
    Returns True on success.
    """
    if cfg is None:
        Log.err("genbox not configured — run `genbox setup` first.")
        return False

    try:
        entry = get(model_id)
    except KeyError:
        Log.err(f"Unknown model: {model_id!r}")
        return False

    arch_dir = cfg.models_dir / entry.architecture
    success = False

    try:
        if entry.full_repo:
            d = arch_dir / entry.id
            if d.exists():
                Log.info(f"Removing: {d}")
                shutil.rmtree(d)
                success = True

        else:
            f = arch_dir / Path(entry.hf_filename).name
            if f.exists():
                Log.info(f"Removing: {f}")
                f.unlink()
                success = True

            # GGUF: also remove shared config
            shared = _shared_config_dir(entry)
            if shared and shared.exists():
                Log.info(f"Removing shared config: {shared}")
                shutil.rmtree(shared)

        if success:
            Log.ok(f"Uninstalled: {entry.name}")
            reset_discovery()
        else:
            Log.err(f"Not installed: {entry.name}")

    except Exception as e:
        Log.err(f"Uninstall failed: {e}")
        return False

    return success


# ── Progress bar ──────────────────────────────────────────────────────────────

def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {int(100 * done / total):3d}%"


# ── Logcat logger ─────────────────────────────────────────────────────────────

class Log:
    RESET = "\033[0m"
    GREY  = "\033[90m"
    WHITE = "\033[97m"
    CYAN  = "\033[96m"
    BLUE  = "\033[94m"
    RED   = "\033[91m"
    GREEN = "\033[92m"

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    @classmethod
    def info(cls, msg: str):
        print(f"{cls.GREY}[{cls._ts()}]{cls.RESET} {msg}")

    @classmethod
    def ok(cls, msg: str):
        print(f"{cls.GREY}[{cls._ts()}]{cls.RESET} {cls.GREEN}{msg}{cls.RESET}")

    @classmethod
    def accent(cls, msg: str):
        print(f"{cls.GREY}[{cls._ts()}]{cls.RESET} {cls.BLUE}{msg}{cls.RESET}")

    @classmethod
    def err(cls, msg: str):
        print(f"{cls.GREY}[{cls._ts()}]{cls.RESET} {cls.RED}{msg}{cls.RESET}",
              file=sys.stderr)

    @classmethod
    def row(cls, label: str, value: str, stars: str = ""):
        pad = max(0, 40 - len(label))
        line = (f"  {cls.WHITE}{label}{cls.RESET}"
                + " " * pad
                + f"{cls.CYAN}{value}{cls.RESET}")
        if stars:
            line += f"  {stars}"
        print(line)


# ── print helpers ─────────────────────────────────────────────────────────────

def print_registry(model_type: Optional[str] = None):
    vram = cfg.vram_gb if cfg else 99
    entries = list_registry(model_type=model_type)
    Log.info(f"Registry ({model_type or 'all'}) — VRAM: {vram}GB")
    print()
    for e in entries:
        compat = "" if e.fits_vram(vram) else "  [!] exceeds VRAM"
        Log.row(
            f"{e.name} [{e.quant}]",
            f"{e.vram_min_gb:2d}GB  {e.license:24s}",
            e.stars(e.quality_stars) + " Q  " + e.stars(e.speed_stars) + " S" + compat,
        )


def print_local():
    models = list_local()
    if not models:
        Log.info("No models installed.")
        return
    Log.info(f"Local models ({len(models)}):")
    for m in models:
        Log.row(f"{m['name'][:45]}", f"{m['architecture']:8s}  {m['size_gb']:5.1f} GB")