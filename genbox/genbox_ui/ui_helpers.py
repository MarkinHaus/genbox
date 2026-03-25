"""
genbox/ui_helpers.py
Pure helper functions for the UI — no Streamlit imports.
All functions are data-transformations, testable without a browser.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union


# ── Pipeline type list ────────────────────────────────────────────────────────

_PIPE_TYPES = [
    "Text → Image",
    "Image → Image",
    "Inpaint",
    "Outpaint",
    "Text → Video",
    "Image → Video",
]


def get_pipe_types() -> list[str]:
    """Return the ordered list of pipeline types shown in the UI.
    Multi-Image is intentionally absent (not implemented in new pipeline.py).
    """
    return list(_PIPE_TYPES)


# ── Pipeline name → UI mode mapping ──────────────────────────────────────────

def map_pipeline_to_mode(pipeline_name: str) -> str:
    """
    Map a metadata `pipeline` string to a UI pipe_type label.
    Used by the Remix button in the Library screen.
    """
    p = pipeline_name.lower()
    if "outpaint" in p:
        return "Outpaint"
    if "inpaint" in p:
        return "Inpaint"
    if "img2img" in p or "image_to_image" in p:
        return "Image → Image"
    if "img2video" in p or "i2v" in p or "image_to_video" in p:
        return "Image → Video"
    if "t2v" in p or "text_to_video" in p:
        return "Text → Video"
    if "text_to_image" in p or "t2i" in p or "pony" in p:
        return "Text → Image"
    return "Text → Image"


# ── Upload type detection ─────────────────────────────────────────────────────

_LORA_KEYWORDS = (
    "lora", "_lo_", "-lora-", "lora_",
    "_lora", "adapter", "dreambooth_lora",
)


def detect_upload_type(filename: str) -> str:
    """
    Auto-detect the type of an uploaded file from its name.
    Returns: "gguf" | "lora" | "model" | "unknown"
    """
    name_lower = filename.lower()
    ext = Path(filename).suffix.lower()

    if ext == ".gguf":
        return "gguf"

    if ext == ".safetensors":
        if any(kw in name_lower for kw in _LORA_KEYWORDS):
            return "lora"
        return "model"

    return "unknown"


# ── Architecture heuristic ────────────────────────────────────────────────────

def guess_arch_from_filename(filename: str) -> str:
    """
    Heuristic: infer the model architecture from a filename.
    Returns one of: "flux" | "sd15" | "sdxl" | "sd35" | "ltx" | "wan"
    Default fallback: "flux"
    """
    n = filename.lower()

    if any(x in n for x in ("flux", "f1-", "flux1", "flux2", "flux-1", "flux-2")):
        return "flux"
    if any(x in n for x in ("wan", "wan2", "wan21", "wan22")):
        return "wan"
    if any(x in n for x in ("ltx", "lightricks", "ltxv")):
        return "ltx"
    if any(x in n for x in ("pony", "animagine", "sdxl", "xl-base", "xl_base")):
        return "sdxl"
    if any(x in n for x in ("sd3", "sd35", "stable-diffusion-3")):
        return "sd35"
    if any(x in n for x in ("sd15", "sd1.5", "realistic_vision", "v1-5",
                             "dreamshaper", "deliberate", "revanimated")):
        return "sd15"

    return "flux"  # safest default


# ── Outputs loader ────────────────────────────────────────────────────────────

def load_outputs(outputs_dir: Union[str, Path]) -> list[dict]:
    """
    Scan outputs_dir for .json sidecar files (recursive).
    Returns list of metadata dicts, newest first.
    Each dict has additional keys: _meta_path, _file_path, _tag.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return []

    results = []
    for p in sorted(outputs_dir.rglob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        data["_meta_path"] = str(p)

        # Infer media file path
        is_video = "video" in data.get("pipeline", "").lower()
        media_ext = ".mp4" if is_video else ".png"
        data["_file_path"] = str(p.with_suffix(media_ext))

        # Tag = parent folder name (date-based or custom)
        data["_tag"] = p.parent.name

        results.append(data)

    return results


# ── Remix helper ──────────────────────────────────────────────────────────────

def build_remix_data(meta: dict) -> dict:
    """
    Map a generation metadata dict to session_state keys for the Generate screen.
    """
    return {
        "prompt":    meta.get("prompt", ""),
        "neg_prompt": meta.get("negative_prompt", ""),
        "sel_model": meta.get("model", ""),
        "seed":      meta.get("seed", -1),
        "steps":     meta.get("steps", 28),
        "pipe_type": map_pipeline_to_mode(meta.get("pipeline", "")),
    }


# ── Outpaint validation ───────────────────────────────────────────────────────

def validate_outpaint_expansion(
    left: int, right: int, top: int, bottom: int
) -> tuple[bool, str]:
    """
    Returns (valid, error_message).
    At least one side must have expansion > 0.
    """
    if left + right + top + bottom <= 0:
        return False, "At least one expansion direction (left/right/top/bottom) must be > 0."
    return True, ""


# ── LoRA label ────────────────────────────────────────────────────────────────

def format_lora_label(lo: dict) -> str:
    """
    Format a LoRA dict as a display label for multiselect.
    Example: "cinematic  [flux]  150MB  · trigger: cinematic style"
    """
    arch = lo.get("architecture", "?")
    size = lo.get("size_mb", 0)
    trigger = lo.get("trigger", "")
    badge = f"[{arch}]" if arch not in ("any", "?", "") else ""
    label = f"{lo['name']}  {badge}  {size:.0f}MB"
    if trigger:
        label += f"  · trigger: {trigger}"
    return label


# ── Default models for profile ────────────────────────────────────────────────

def get_install_defaults_for_profile(profile: str) -> list[str]:
    """
    Return the list of recommended model IDs for a VRAM profile.
    Delegates to models.get_default_models, with fallback.
    """
    try:
        from genbox.models import get_default_models
        return get_default_models(profile)
    except Exception:
        return ["flux1_schnell_q4", "ltx2_fp8", "wan_1_3b"]
