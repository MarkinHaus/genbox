"""
genbox/utils.py
Utility library for genbox — three sections:

  Section 1: LoRA / custom .safetensors / model file + metadata management
  Section 2: Image model utilities (unified SDL variants, FLUX 1/2, Pony, GGUF)
  Section 3: Video model utilities (LTX all variants, WAN all variants)

All pipelines remain 100 % offline after installation.
No external network calls here — only local filesystem operations.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger("genbox.utils")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — LoRA / custom .safetensors + model metadata management
# ──────────────────────────────────────────────────────────────────────────────

LoraSpec = Union[str, tuple[str, float]]

KNOWN_ARCHITECTURES = {"flux", "sd35", "sd15", "sdxl", "ltx", "wan"}


def parse_lora_spec(spec: LoraSpec) -> tuple[str, float]:
    """
    Normalize a LoRA spec to (path_str, weight).

        "style.safetensors"          → ("style.safetensors", 1.0)
        ("style.safetensors", 0.7)   → ("style.safetensors", 0.7)
        ["style.safetensors", 0.5]   → ("style.safetensors", 0.5)
    """
    if isinstance(spec, (list, tuple)):
        return str(spec[0]), float(spec[1])
    return str(spec), 1.0


# ── LoRA sidecar metadata ─────────────────────────────────────────────────────

def write_lora_metadata(
    lora_path: Union[str, Path],
    architecture: str,
    trigger: str = "",
    description: str = "",
    preview_url: str = "",
) -> None:
    """
    Write a sidecar .json next to a LoRA .safetensors file.

    Args:
        lora_path:    Path to .safetensors file
        architecture: e.g. "flux", "sd15", "sdxl", "sd35", "wan", "ltx"
        trigger:      Trigger word(s) for this LoRA
        description:  Human-readable description
        preview_url:  Optional URL to preview image (e.g. CivitAI or local path)
    """
    lora_path = Path(lora_path)
    meta: dict = {"architecture": architecture}
    if trigger:
        meta["trigger"] = trigger
    if description:
        meta["description"] = description
    if preview_url:
        meta["preview_url"] = preview_url

    sidecar = lora_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug(f"LoRA metadata written: {sidecar}")


def read_lora_metadata(lora_path: Union[str, Path]) -> dict:
    """
    Read sidecar .json for a LoRA file.
    Returns empty dict if sidecar does not exist or is malformed.
    """
    sidecar = Path(lora_path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Model sidecar metadata ────────────────────────────────────────────────────

def write_model_metadata(
    model_path: Union[str, Path],
    description: str = "",
    preview_url: str = "",
    tags: Optional[list[str]] = None,**extra
) -> None:
    """
    Write a sidecar .json next to a model file (.safetensors / .gguf).

    Args:
        model_path:  Path to model file
        description: Human-readable description / notes
        preview_url: Optional URL or local path to preview image
        tags:        Free-form tag list (e.g. ["portrait", "photorealistic"])
    """
    model_path = Path(model_path)
    meta: dict = {}
    if description:
        meta["description"] = description
    if preview_url:
        meta["preview_url"] = preview_url
    if tags:
        meta["tags"] = list(tags)

    meta.update({k: v for k, v in extra.items() if v is not None})

    sidecar = model_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug(f"Model metadata written: {sidecar}")


def read_model_metadata(model_path: Union[str, Path]) -> dict:
    """
    Read sidecar .json for a model file.
    Returns empty dict if not found or malformed.
    """
    sidecar = Path(model_path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── List LoRAs ────────────────────────────────────────────────────────────────

def list_loras(
    loras_dir: Union[str, Path],
    architecture: Optional[str] = None,
) -> list[dict]:
    """
    List all locally available LoRA .safetensors files under loras_dir.

    Returns a list of dicts:
        name, path, architecture, trigger, description, preview_url, size_mb

    Args:
        loras_dir:    Root directory to scan (recursively)
        architecture: Optional filter — only return LoRAs matching this arch
    """
    loras_dir = Path(loras_dir)
    results: list[dict] = []

    for path in sorted(loras_dir.rglob("*.safetensors")):
        meta = read_lora_metadata(path)

        arch = meta.get("architecture")
        if not arch:
            parent_name = path.parent.name
            arch = parent_name if parent_name in KNOWN_ARCHITECTURES else "any"

        if architecture and arch != architecture and arch != "any":
            continue

        results.append({
            "name":         path.stem,
            "path":         str(path),
            "architecture": arch,
            "trigger":      meta.get("trigger", ""),
            "description":  meta.get("description", ""),
            "preview_url":  meta.get("preview_url", ""),
            "size_mb":      round(path.stat().st_size / (1024 ** 2), 1),
        })

    return results


# ── Register custom model file ────────────────────────────────────────────────

def register_custom_file(
    src: Union[str, Path],
    architecture: str,
    models_dir: Union[str, Path],
    description: str = "",
    preview_url: str = "",
    tags: Optional[list[str]] = None,
    copy: bool = True,
) -> dict:
    """
    Register a custom .safetensors or .gguf file into genbox's local models dir.

    Copies the file into models_dir/<architecture>/<filename> (unless copy=False),
    writes a sidecar .json with description/preview_url/tags, and returns a
    lightweight registry-compatible dict that can be passed to pipeline functions.

    Args:
        src:          Source file path (.safetensors or .gguf)
        architecture: Target architecture: "flux", "sd15", "sdxl", "sd35", "ltx", "wan"
        models_dir:   Root models directory (cfg.models_dir)
        description:  Human-readable notes
        preview_url:  Optional preview image URL
        tags:         Free-form tag list
        copy:         If True, copy file; if False, use src in place (symlink-free)

    Returns:
        dict with keys: id, name, architecture, quant, path, description, preview_url
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    suffix = src.suffix.lower()
    if suffix not in (".safetensors", ".gguf"):
        raise ValueError(f"Unsupported file type: {suffix!r}. Must be .safetensors or .gguf")

    dest_dir = Path(models_dir) / architecture
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if copy and src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        log.info(f"Copied {src.name} → {dest}")
    else:
        dest = src

    # Detect quant from extension/name
    fname_lower = src.name.lower()
    if ".gguf" in fname_lower or suffix == ".gguf":
        quant = "gguf-q4" if "q4" in fname_lower else "gguf-q8"
    elif "fp8" in fname_lower:
        quant = "fp8"
    elif "fp16" in fname_lower:
        quant = "fp16"
    elif "bf16" in fname_lower:
        quant = "bf16"
    else:
        quant = "fp16"

    # Flux: Base-Modell aus Dateiname ableiten → hf_pipeline_repo für Loader
    hf_pipeline_repo: Optional[str] = None
    if architecture == "flux" and suffix == ".safetensors":
        base_key = infer_flux_base_from_stem(src.stem)
        if base_key:
            hf_pipeline_repo = FLUX_BASE_PIPELINE_REPOS.get(base_key)
            log.info(
                f"Custom FLUX safetensors '{src.name}': "
                f"auto-detected base → {base_key} ({hf_pipeline_repo})"
            )
        else:
            log.warning(
                f"Custom FLUX safetensors '{src.name}': base model not recognizable "
                "from filename. Set hf_pipeline_repo manually."
            )

    # SDL: Architektur aus Dateiname validieren / verfeinern
    inferred_arch: Optional[str] = None
    if architecture in ("sd15", "sdxl", "sd35") and suffix == ".safetensors":
        inferred_arch = infer_sdl_arch_from_stem(src.stem)
        if inferred_arch and inferred_arch != architecture:
            log.warning(
                f"Custom SDL safetensors '{src.name}': "
                f"declared arch={architecture!r} but filename suggests {inferred_arch!r}. "
                "Using declared arch — override 'architecture' arg if wrong."
            )
        elif inferred_arch:
            log.info(
                f"Custom SDL safetensors '{src.name}': "
                f"filename confirms arch={architecture!r}"
            )
        else:
            log.warning(
                f"Custom SDL safetensors '{src.name}': "
                "architecture not recognizable from filename — using declared arch."
            )

    write_model_metadata(
        dest,
        description=description,
        preview_url=preview_url,
        tags=tags,
        # Sidecar bekommt hf_pipeline_repo damit der Loader ihn lesen kann
        **({"hf_pipeline_repo": hf_pipeline_repo} if hf_pipeline_repo else {}),
    )

    model_id = f"local_{src.stem}"
    return {
        "id": model_id,
        "name": f"[{architecture.upper()}] {src.stem[:28]}",
        "architecture": architecture,
        "quant": quant,
        "path": str(dest),
        "description": description,
        "preview_url": preview_url,
        "tags": tags or ["local", "custom"],
        "hf_pipeline_repo": hf_pipeline_repo,  # ← Loader braucht das
        "inferred_arch": inferred_arch,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Image model utilities
# (unified SDL variants, FLUX 1/2, Pony, GGUF — all offline)
# ──────────────────────────────────────────────────────────────────────────────

# Mapping von Base-Model-Key → hf_pipeline_repo-ähnlicher Shared-Config-Name.
# Wird von register_custom_file genutzt, um hf_pipeline_repo automatisch zu setzen.
FLUX_BASE_PIPELINE_REPOS: dict[str, str] = {
    "flux1_dev":      "black-forest-labs/FLUX.1-dev",
    "flux1_schnell":  "black-forest-labs/FLUX.1-schnell",
    "flux2_4b":       "black-forest-labs/FLUX.2-klein-4B",
    "flux2_9b":       "black-forest-labs/FLUX.2-klein-9B",
    "flux2_b_4b":      "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2_b_9b":      "black-forest-labs/FLUX.2-klein-base-9B",
    "flux2_bf8_9b":     "black-forest-labs/FLUX.2-klein-base-9b-fp8 ",
}


def infer_flux_base_from_stem(stem: str) -> Optional[str]:
    """
    Heuristische Base-Modell-Erkennung aus einem Dateinamens-Stem.

    Priorität: FLUX.2-spezifische Tokens zuerst (spezifischer),
    dann FLUX.1 dev/schnell, dann generisch FLUX.1-dev als Fallback.

    Returns einen Key aus FLUX_BASE_PIPELINE_REPOS oder None.

    Beispiele:
        "my-model-flux2-9b-v1"   → "flux2_9b"
        "portrait-flux1-dev"     → "flux1_dev"
        "realvis-schnell-fp8"    → "flux1_schnell"
        "custom-flux-finetune"   → "flux1_dev"   (generischer Fallback)
    """
    n = stem.lower()
    # FLUX.2 / Klein — spezifischer, zuerst prüfen
    if any(k in n for k in ("flux2", "flux.2", "klein")):
        return "flux2_9b" if "9b" in n else "flux2_4b"
    # FLUX.1 Schnell
    if "schnell" in n:
        return "flux1_schnell"
    # FLUX.1 Dev oder generisch "flux"
    if any(k in n for k in ("flux1", "flux.1", "flux", "dev")):
        return "flux1_dev"
    return None

# SDL-Architektur-Erkennung aus Dateiname
SDL_ARCH_HINTS: dict[str, str] = {
    "sd35":  "sd35",
    "sd3.5": "sd35",
    "sd3":   "sd35",
    "sdxl":  "sdxl",
    "xl":    "sdxl",
    "pony":  "sdxl",   # Pony ist SDXL-basiert
    "animagine": "sdxl",
    "sd15":  "sd15",
    "sd1.5": "sd15",
    "v1-5":  "sd15",
    "v1_5":  "sd15",
}


def infer_sdl_arch_from_stem(stem: str) -> Optional[str]:
    """
    Heuristische Architektur-Erkennung für SDL-Modelle aus dem Dateinamens-Stem.

    Priorität: spezifischste Tokens zuerst (SD3.5 > SDXL > SD1.5).

    Beispiele:
        "realisticVision-sd15-v6"  → "sd15"
        "dreamshaper-xl-turbo"     → "sdxl"
        "ponyDiffusionV6XL"        → "sdxl"
        "sd3.5-large-turbo"        → "sd35"
        "unknown-model"            → None
    """
    n = stem.lower()
    # SD3.5 zuerst — spezifischster Token
    for token in ("sd35", "sd3.5", "sd3"):
        if token in n:
            return "sd35"
    # SDXL / Pony
    for token in ("sdxl", "pony", "animagine", "_xl", "-xl"):
        if token in n:
            return "sdxl"
    # SD1.5
    for token in ("sd15", "sd1.5", "v1-5", "v1_5"):
        if token in n:
            return "sd15"
    return None

# Architecture groups
FLUX_ARCHITECTURES    = {"flux"}
SDL_ARCHITECTURES     = {"sd15", "sdxl", "sd35"}   # all SD/unified variants
PONY_MODEL_IDS        = {"pony_xl"}                 # SDXL-based Pony models
IMAGE_ARCHITECTURES   = FLUX_ARCHITECTURES | SDL_ARCHITECTURES

# Scheduler map — public for external backends
SCHEDULER_MAP: dict[str, tuple[str, dict]] = {
    "FlowMatchEuler":     ("FlowMatchEulerDiscreteScheduler",    {}),
    "DPM++ 2M":           ("DPMSolverMultistepScheduler",         {"use_karras_sigmas": False}),
    "DPM++ 2M Karras":    ("DPMSolverMultistepScheduler",         {"use_karras_sigmas": True}),
    "Euler":              ("EulerDiscreteScheduler",              {}),
    "Euler A":            ("EulerAncestralDiscreteScheduler",     {}),
    "DDIM":               ("DDIMScheduler",                       {}),
    "UniPC":              ("UniPCMultistepScheduler",             {}),
    "UniPC (flow_shift)": ("UniPCMultistepScheduler",             {"_flow_shift": True}),
}

ARCH_SAMPLERS: dict[str, list[str]] = {
    "flux":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler"],
    "sd35":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM"],
    "sd15":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
    "sdxl":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
    "ltx":   ["FlowMatchEuler", "DPM++ 2M"],
    "wan":   ["UniPC (flow_shift)", "FlowMatchEuler", "DPM++ 2M"],
}


def get_recommended_sampler(architecture: str) -> str:
    """Return the recommended sampler for a given architecture."""
    samplers = ARCH_SAMPLERS.get(architecture, [])
    return samplers[0] if samplers else "default"


def is_gguf(quant: str) -> bool:
    """Return True if the quantization string denotes a GGUF model."""
    return "gguf" in quant.lower()


def is_flux2(model_id: str) -> bool:
    """Heuristic: True if model_id or name refers to FLUX.2 (klein)."""
    return "flux2" in model_id.lower() or "flux.2" in model_id.lower()


def is_pony_variant(model_id: str) -> bool:
    """True if this is a Pony-based model (SDXL fine-tune with pony/rating tags)."""
    return model_id in PONY_MODEL_IDS or "pony" in model_id.lower()


def detect_flux_variant_from_path(local_model_path: Union[str, Path]) -> tuple[str, str]:
    """
    Detect FLUX pipeline and transformer class names from local model files.

    Reads model_index.json (full-repo) or transformer/config.json (GGUF shared config).
    Returns (pipeline_class_name, transformer_class_name).
    Fallback: FLUX.1 classes.
    """
    p = Path(local_model_path)

    # Full-repo: model_index.json
    mi = p / "model_index.json"
    if mi.exists():
        try:
            data = json.loads(mi.read_text(encoding="utf-8"))
            pipe_cls = data.get("_class_name", "FluxPipeline")
            transformer_cls = (
                "Flux2Transformer2DModel" if "Flux2" in pipe_cls
                else "FluxTransformer2DModel"
            )
            return pipe_cls, transformer_cls
        except Exception:
            pass

    # GGUF: transformer/config.json
    tc = p / "transformer" / "config.json"
    if tc.exists():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            cls = data.get("_class_name", "FluxTransformer2DModel")
            pipe_cls = "Flux2KleinPipeline" if "Flux2" in cls else "FluxPipeline"
            return pipe_cls, cls
        except Exception:
            pass

    return "FluxPipeline", "FluxTransformer2DModel"


def get_image_model_local_path(
    entry,  # ModelEntry-like with .architecture, .full_repo, .id, .hf_filename, .quant, .hf_pipeline_repo
    models_dir: Union[str, Path],
) -> Path:
    """
    Resolve the local filesystem path for an image model.

    Returns the path to the model file or directory.
    Raises FileNotFoundError if not installed.
    """
    models_dir = Path(models_dir)
    arch_dir = models_dir / entry.architecture

    if entry.full_repo:
        p = arch_dir / entry.id
        if not (p / "model_index.json").exists():
            raise FileNotFoundError(
                f"Full-repo model not found: {p}\n"
                "Please download via Models panel."
            )
        return p

    if is_gguf(entry.quant):
        p = arch_dir / Path(entry.hf_filename).name
        if not p.exists():
            raise FileNotFoundError(f"GGUF file not found: {p}")
        return p

    # single safetensors
    p = arch_dir / Path(entry.hf_filename).name
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return p


def get_gguf_shared_config_dir(
    entry,
    models_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Return the shared config directory for a GGUF model (pipeline repo config).
    None if not applicable.
    """
    if not entry.hf_pipeline_repo:
        return None
    models_dir = Path(models_dir)
    safe_name = entry.hf_pipeline_repo.replace("/", "--")
    return models_dir / entry.architecture / f"_shared_{safe_name}"


def list_image_models_local(
    models_dir: Union[str, Path],
    architecture: Optional[str] = None,
    variant: Optional[str] = None,
) -> list[dict]:
    """
    List installed image models from the filesystem.

    Args:
        models_dir:   Root models directory
        architecture: Filter by architecture ("flux", "sd15", "sdxl", "sd35")
        variant:      Filter by variant string ("flux1", "flux2", "pony", "sdxl", "sd15", "sd35")

    Returns list of dicts: id, name, architecture, quant, path, size_gb, meta
    """
    models_dir = Path(models_dir)
    results: list[dict] = []

    for arch in (IMAGE_ARCHITECTURES if not architecture else {architecture}):
        arch_dir = models_dir / arch
        if not arch_dir.exists():
            continue

        # Full-repo dirs
        for d in arch_dir.iterdir():
            if d.is_dir() and (d / "model_index.json").exists():
                meta = read_model_metadata(d / "model_index.json")
                size_gb = sum(
                    f.stat().st_size for f in d.rglob("*") if f.is_file()
                ) / (1024 ** 3)
                entry_variant = _detect_variant(d.name, arch)
                if variant and entry_variant != variant:
                    continue
                results.append({
                    "id":           f"local_{d.name}",
                    "name":         d.name,
                    "architecture": arch,
                    "quant":        "fp16",
                    "path":         str(d),
                    "size_gb":      round(size_gb, 2),
                    "variant":      entry_variant,
                    "meta":         meta,
                })

        # Single-file models (.safetensors, .gguf)
        for f in arch_dir.glob("*.safetensors"):
            meta = read_model_metadata(f)
            size_gb = f.stat().st_size / (1024 ** 3)
            entry_variant = _detect_variant(f.stem, arch)
            if variant and entry_variant != variant:
                continue
            results.append({
                "id":           f"local_{f.stem}",
                "name":         f.stem,
                "architecture": arch,
                "quant":        "fp16",
                "path":         str(f),
                "size_gb":      round(size_gb, 2),
                "variant":      entry_variant,
                "meta":         meta,
            })

        for f in arch_dir.glob("*.gguf"):
            fname_lower = f.name.lower()
            quant = "gguf-q4" if "q4" in fname_lower else "gguf-q8"
            meta = read_model_metadata(f)
            size_gb = f.stat().st_size / (1024 ** 3)
            entry_variant = _detect_variant(f.stem, arch)
            if variant and entry_variant != variant:
                continue
            results.append({
                "id":           f"local_{f.stem}",
                "name":         f.stem,
                "architecture": arch,
                "quant":        quant,
                "path":         str(f),
                "size_gb":      round(size_gb, 2),
                "variant":      entry_variant,
                "meta":         meta,
            })

    return results


def _detect_variant(name_or_id: str, architecture: str) -> str:
    """Heuristic: return variant label for display/filtering."""
    n = name_or_id.lower()
    if architecture == "flux":
        if "flux2" in n or "flux.2" in n or "klein" in n:
            return "flux2"
        return "flux1"
    if architecture == "sdxl":
        if "pony" in n:
            return "pony"
        if "turbo" in n:
            return "sdxl_turbo"
        if "animagine" in n:
            return "animagine"
        return "sdxl"
    return architecture  # sd15, sd35 → identity


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Video model utilities (LTX all variants, WAN all variants)
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_ARCHITECTURES = {"ltx", "wan"}

# LTX frame constraint: frames = 8n+1, minimum 9
LTX_FRAME_BASE   = 8
LTX_FRAME_OFFSET = 1
LTX_FRAME_MIN    = 9

# WAN recommended: 4n+1, minimum 5
WAN_FRAME_BASE   = 4
WAN_FRAME_OFFSET = 1
WAN_FRAME_MIN    = 5


def snap_frames(frames: int, architecture: str) -> int:
    """
    Clamp frame count to architecture constraints.

    LTX: frames = 8n+1  (min 9)
    WAN: frames = 4n+1  (min 5, recommended)
    Other: returned unchanged.
    """
    if architecture == "ltx":
        frames = max(frames, LTX_FRAME_MIN)
        remainder = (frames - LTX_FRAME_OFFSET) % LTX_FRAME_BASE
        if remainder != 0:
            frames = frames + (LTX_FRAME_BASE - remainder)
    elif architecture == "wan":
        frames = max(frames, WAN_FRAME_MIN)
        remainder = (frames - WAN_FRAME_OFFSET) % WAN_FRAME_BASE
        if remainder != 0:
            frames = frames + (WAN_FRAME_BASE - remainder)
    return frames


def get_wan_flow_shift(height: int) -> float:
    """Return WAN flow_shift value based on output height."""
    return 5.0 if height >= 720 else 3.0


def get_video_model_local_path(
    entry,
    models_dir: Union[str, Path],
) -> Path:
    """
    Resolve local path for a video model (LTX or WAN).

    Both LTX and WAN are always full-repo diffusers format.
    GGUF is NOT supported for video pipelines (diffusers limitation).

    Raises FileNotFoundError if model not installed.
    Raises ValueError if GGUF is requested (unsupported).
    """
    models_dir = Path(models_dir)

    if is_gguf(getattr(entry, "quant", "")):
        raise ValueError(
            f"'{getattr(entry, 'name', entry)}' is a GGUF single-file. "
            "Diffusers-based video pipelines (WAN, LTX) do not support GGUF directly.\n"
            "Please use a full diffusers-format repo:\n"
            "  → WAN 1.3B:  wan_1_3b\n"
            "  → WAN 14B:   wan21_14b_diffusers"
        )

    arch_dir = models_dir / entry.architecture
    p = arch_dir / entry.id
    if not (p / "model_index.json").exists():
        raise FileNotFoundError(
            f"Video model not found: {p}\n"
            "Please download via Models panel."
        )
    return p


def list_video_models_local(
    models_dir: Union[str, Path],
    architecture: Optional[str] = None,
) -> list[dict]:
    """
    List installed video models (LTX and/or WAN variants) from the filesystem.

    Args:
        models_dir:   Root models directory
        architecture: Optional filter ("ltx" or "wan")

    Returns list of dicts: id, name, architecture, quant, path, size_gb, variant, meta
    """
    models_dir = Path(models_dir)
    results: list[dict] = []

    for arch in (VIDEO_ARCHITECTURES if not architecture else {architecture}):
        arch_dir = models_dir / arch
        if not arch_dir.exists():
            continue

        for d in arch_dir.iterdir():
            if not d.is_dir():
                continue
            if not (d / "model_index.json").exists():
                continue

            meta = read_model_metadata(d / "model_index.json")
            size_gb = sum(
                f.stat().st_size for f in d.rglob("*") if f.is_file()
            ) / (1024 ** 3)

            variant = _detect_video_variant(d.name, arch)

            results.append({
                "id":           f"local_{d.name}",
                "name":         d.name,
                "architecture": arch,
                "quant":        "bf16",
                "path":         str(d),
                "size_gb":      round(size_gb, 2),
                "variant":      variant,
                "meta":         meta,
            })

    return results


def _detect_video_variant(name: str, architecture: str) -> str:
    """Detect specific variant for LTX or WAN models."""
    n = name.lower()
    if architecture == "ltx":
        if "0.9.7" in n or "distilled" in n or "23" in n:
            return "ltx_distilled"
        return "ltx"
    if architecture == "wan":
        if "14b" in n:
            return "wan_14b"
        if "2.2" in n or "22" in n:
            return "wan22"
        if "1.3b" in n or "1_3b" in n:
            return "wan_1_3b"
        return "wan"
    return architecture


def get_video_generation_defaults(architecture: str, variant: str = "") -> dict:
    """
    Return sensible default generation parameters for a video architecture/variant.

    These match the official recommendations from the respective HF repos.
    """
    if architecture == "ltx":
        return {
            "width":          832,
            "height":         480,
            "frames":         97,     # 8*12+1
            "fps":            24,
            "steps":          50,
            "guidance_scale": 3.0,
        }
    if architecture == "wan":
        if "14b" in variant or "22" in variant:
            return {
                "width":          1280,
                "height":         720,
                "frames":         81,  # 4*20+1
                "fps":            24,
                "steps":          50,
                "guidance_scale": 5.0,
            }
        # 1.3B default
        return {
            "width":          832,
            "height":         480,
            "frames":         81,
            "fps":            16,
            "steps":          50,
            "guidance_scale": 5.0,
        }
    return {}