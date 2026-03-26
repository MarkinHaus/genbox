"""
genbox/utils_image_pipeline.py
Shared image pipeline infrastructure — used by pipeline_flux, pipeline_sdl, pipeline_pony.

Sections:
  A. Device / dtype resolution
  B. Seed resolution
  C. Accelerator section  (offload strategy + xformers + torch.compile)
  D. LoRA loader          (PEFT adapter stacking, dtype-safe cast)
  E. Scheduler setter
  F. Call kwargs builder  (arch-aware)
  G. Output path + metadata builder

No ML imports at module level — everything lazy.
100 % offline after model installation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, Callable

log = logging.getLogger("genbox.utils_image_pipeline")

LoraSpec = Union[str, tuple[str, float]]

# ──────────────────────────────────────────────────────────────────────────────
# A. Device / dtype resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(_torch=None) -> str:
    """Return best available device: 'cuda' > 'mps' > 'cpu'."""
    if _torch is None:
        import torch as _torch  # type: ignore
    if _torch.cuda.is_available():
        return "cuda"
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(quant: str, _torch=None):
    """Map quantization string to torch dtype."""
    if _torch is None:
        import torch as _torch  # type: ignore
    return {
        "fp8":     getattr(_torch, "bfloat16", _torch.float16),
        "fp16":    _torch.float16,
        "bf16":    _torch.bfloat16,
        "fp32":    _torch.float32,
        "gguf-q8": _torch.float16,
        "gguf-q4": _torch.float16,
    }.get(quant, _torch.float16)


# ──────────────────────────────────────────────────────────────────────────────
# B. Seed resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_seed(seed: int) -> int:
    """Return seed; negative → random."""
    return random.randint(0, 2**32 - 1) if seed < 0 else seed


def make_generator(seed: int, device: str, _torch=None):
    """Create a torch.Generator on the right device."""
    if _torch is None:
        import torch as _torch  # type: ignore
    gen_device = device if device == "cuda" else "cpu"
    return _torch.Generator(device=gen_device).manual_seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# C. Accelerator section
# ──────────────────────────────────────────────────────────────────────────────

# Boundary between "low VRAM" and "mid VRAM" for auto-offload selection
_LOW_VRAM_THRESHOLD_GB = 8


def resolve_offload_mode(
    vram_gb: int,
    env_override: Optional[str],
    has_quantized_encoders: bool = False,
) -> str:
    """
    Determine CPU offload mode.

    Priority:
      1. env_override (GENBOX_OFFLOAD env var value if set)
      2. has_quantized_encoders: int8 encoders crash under sequential → force model
      3. vram_gb threshold: ≤8 GB → sequential, >8 GB → model

    Returns: "sequential" | "model" | "none"
    """
    mode = (env_override or "").lower().strip() or None

    if mode is None:
        mode = "sequential" if vram_gb <= _LOW_VRAM_THRESHOLD_GB else "model"

    # int8-quantized encoders are incompatible with sequential offload
    if has_quantized_encoders and mode == "sequential":
        log.warning(
            "sequential offload incompatible with int8-quantized encoders "
            "(Int8Params cannot be moved to meta device). Falling back to model offload."
        )
        mode = "model"

    return mode


def apply_accelerators(pipe, device, offload_mode, accel=None):
    accel = accel or []
    if device != "cuda":
        pipe.to(device)
        print("running on", device, "no accelerators")
        return

    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_mode == "none":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    # ── xformers — nur für U-Net Architekturen (SD1.5, SDXL) ────────────────
    if "xformers" in accel:
        if hasattr(pipe, "unet"):   # DiTs haben kein .unet → Guard
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log.info("xformers enabled (U-Net)")
            except Exception as e:
                log.warning(f"xformers not available: {e}")
        else:
            log.debug("xformers skipped — not a U-Net architecture (DiT)")

    # ── SageAttention — wirklich anwenden via set_attn_processor ─────────────
    if "sageAttn" in accel:
        try:
            from sageattention import sageattn
            import torch

            _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

            def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, **kwargs):
                # Nur head_dim 64/128/256 → sageattn, alles andere → original
                # Kein Reshape: verändert Attention-Semantik und produziert Artefakte
                if query.shape[-1] in (64, 128, 256):
                    return sageattn(query, key, value,
                                    attn_mask=attn_mask, is_causal=is_causal)
                return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale)

            torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
            log.info("SageAttention: SDPA patched (64/128/256 → sage, Rest → original)")
        except ImportError:
            log.warning("sageAttn requested but not installed: pip install sageattention")
        except Exception as e:
            log.warning(f"SageAttention apply failed: {e}")


def inject_compile(pipe, accel: list[str]):
    if "compile" not in accel:
        return pipe

    offload_env = os.environ.get("GENBOX_OFFLOAD", "").lower()
    if offload_env != "none":
        log.warning(
            "torch.compile übersprungen — inkompatibel mit CPU-offload. "
            "Setze GENBOX_OFFLOAD=none."
        )
        return pipe

    try:
        import torch
        target = None
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            target = "transformer"
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead", fullgraph=False,
            )
        elif hasattr(pipe, "unet") and pipe.unet is not None:
            target = "unet"
            pipe.unet = torch.compile(
                pipe.unet, mode="reduce-overhead", fullgraph=False,
            )
        if target:
            log.info(f"torch.compile applied to {target}")
    except Exception as e:
        log.warning(f"torch.compile failed: {e}")
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# D. LoRA loader (PEFT-backed, dtype-safe)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_lora_spec(spec: LoraSpec) -> tuple[str, float]:
    if isinstance(spec, (list, tuple)):
        return str(spec[0]), float(spec[1])
    return str(spec), 1.0


def build_lora_adapter_list(
    loras: list[LoraSpec],
    loras_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """
    Resolve LoRA specs to absolute paths and build an adapter list.

    Each entry: {"path": Path, "weight": float, "adapter_name": str}
    Missing files are logged and excluded. Adapter names are guaranteed unique.
    """
    if not loras:
        return []

    loras_dir = Path(loras_dir) if loras_dir is not None else None
    result = []

    for i, spec in enumerate(loras):
        path_str, weight = _parse_lora_spec(spec)
        lora_path = Path(path_str)

        # Relative path → try loras_dir
        if not lora_path.is_absolute() and loras_dir is not None:
            candidate = loras_dir / path_str
            if candidate.exists():
                lora_path = candidate

        if not lora_path.exists():
            log.warning(f"LoRA not found: {lora_path} — skipping")
            continue

        result.append({
            "path":         lora_path,
            "weight":       weight,
            "adapter_name": f"lora_{i}",
        })

    return result


def _safe_cast_lora_params(pipe, target_dtype) -> None:
    """Cast only LoRA parameters (name contains 'lora') to target_dtype."""
    for component_name in ("transformer", "unet", "text_encoder", "text_encoder_2"):
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        for name, param in component.named_parameters():
            if "lora" in name.lower() and param.data.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)


def apply_loras_to_pipe(
    pipe,
    adapter_list: list[dict],
    architecture: str,
) -> None:
    """
    Load LoRA adapters into pipe via PEFT and activate them with correct weights.

    adapter_list: output of build_lora_adapter_list()
    Modifies pipe in-place. Returns None.

    Handles:
    - Single LoRA: load + set_adapters with weight
    - Multi-LoRA:  PEFT adapter stacking + set_adapters
    - Dtype fix:   casts LoRA float params to pipe.dtype (GGUF-safe)
    """
    if not adapter_list:
        return

    # Ensure PEFT is available
    try:
        import peft  # noqa
    except ImportError:
        raise ImportError("PEFT required for LoRA: pip install peft")

    loaded: list[str]  = []
    weights: list[float] = []

    for entry in adapter_list:
        adapter_name = entry["adapter_name"]
        lora_path    = entry["path"]
        weight       = entry["weight"]

        try:
            pipe.load_lora_weights(
                str(lora_path),
                adapter_name=adapter_name,
            )
        except Exception as e:
            log.warning(f"load_lora_weights failed for {lora_path.name}: {e}")
            continue

        # Verify registration in PEFT config
        registered: set[str] = set()
        try:
            registered = set(pipe.get_active_adapters())
        except Exception:
            pass
        try:
            registered |= set(pipe.transformer.peft_config.keys())
        except Exception:
            pass
        try:
            registered |= set(pipe.unet.peft_config.keys())
        except Exception:
            pass

        if adapter_name not in registered:
            log.warning(
                f"{lora_path.name}: loaded but adapter '{adapter_name}' not registered "
                f"(key mismatch for {architecture})"
            )
            continue

        loaded.append(adapter_name)
        weights.append(weight)
        log.info(f"LoRA loaded: {lora_path.name} as '{adapter_name}' (weight={weight})")

    if not loaded:
        log.warning("No LoRA adapters successfully loaded")
        return

    try:
        pipe.set_adapters(loaded, adapter_weights=weights)
        log.info(f"Active adapters: {list(zip(loaded, weights))}")

        # Cast LoRA float params to compute dtype (GGUF-safe: only touches lora_ params)
        import torch  # type: ignore
        target_dtype = getattr(pipe, "dtype", torch.bfloat16)
        _safe_cast_lora_params(pipe, target_dtype)
    except Exception as e:
        log.warning(f"set_adapters or dtype cast failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# E. Scheduler
# ──────────────────────────────────────────────────────────────────────────────

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

# Architectures that only tolerate FlowMatch schedulers
_FLOW_MATCH_ONLY = {"flux", "sd35"}

ARCH_SAMPLERS: dict[str, list[str]] = {
    "flux":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler"],
    "sd35":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM"],
    "sd15":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
    "sdxl":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
}


def set_scheduler(pipe, architecture: str, sampler: str, height: int = 0) -> None:
    """
    Set scheduler on pipe in-place.

    "default" → keep pipeline default (no swap).
    FLUX/SD35  → only FlowMatchEuler allowed; others silently skipped.
    Others     → lookup in SCHEDULER_MAP, from_config.
    """
    if sampler == "default":
        return

    if architecture in _FLOW_MATCH_ONLY:
        if sampler not in ("FlowMatchEuler", "default"):
            log.warning(
                f"Sampler '{sampler}' not compatible with {architecture} "
                f"(requires FlowMatch). Keeping default."
            )
        return

    info = SCHEDULER_MAP.get(sampler)
    if info is None:
        log.warning(f"Unknown sampler '{sampler}' — keeping default")
        return

    cls_name, raw_kwargs = info
    fc_kwargs = {k: v for k, v in raw_kwargs.items() if not k.startswith("_")}

    try:
        import diffusers as _d  # type: ignore
        sched_cls = getattr(_d, cls_name, None)
        if sched_cls is None:
            log.warning(f"diffusers does not have {cls_name!r} — update diffusers")
            return
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, **fc_kwargs)
        log.info(f"Scheduler → {cls_name} {fc_kwargs or ''}")
    except Exception as e:
        log.warning(f"Scheduler swap failed ({cls_name}): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# F. Call kwargs builder (architecture-aware)
# ──────────────────────────────────────────────────────────────────────────────

def build_call_kwargs(
    architecture: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    generator,
    t5_mode: str = "fp16",
    callback_on_step_end=None,
    callback_tensor_inputs: Optional[list] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build the kwargs dict for pipe(**kwargs).

    Architecture-specific rules:
      flux:  no negative_prompt; width/height snapped to 16; t5_mode=none → max_sequence_length=77
      sd35:  no negative_prompt (handled by arch internally)
      sd15/sdxl/pony: negative_prompt included

    callback_on_step_end:
      diffusers callback_on_step_end function.
      Signature: fn(pipe, step_index, timestep, callback_kwargs) → dict
      Only injected when not None.
    callback_tensor_inputs:
      List of tensor names to expose in callback_kwargs.
      Defaults to ["latents"] when callback is set.
      Must be a subset of pipe._callback_tensor_inputs.
    """
    # Snap resolution for transformer-based models
    if architecture in ("flux", "sd35"):
        width  = max((width  // 16) * 16, 256)
        height = max((height // 16) * 16, 256)

    kwargs: dict[str, Any] = dict(
        prompt               = prompt,
        width                = width,
        height               = height,
        num_inference_steps  = steps,
        guidance_scale       = guidance_scale,
        generator            = generator,
    )

    # Negative prompt
    if architecture not in ("flux",) and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    # T5 mode
    if architecture == "flux" and t5_mode == "none":
        kwargs["prompt_2"]            = None
        kwargs["max_sequence_length"] = 77

    # Live-progress callback
    if callback_on_step_end is not None:
        kwargs["callback_on_step_end"] = callback_on_step_end
        kwargs["callback_on_step_end_tensor_inputs"] = (
            callback_tensor_inputs if callback_tensor_inputs is not None
            else ["latents"]
        )

    if extra:
        kwargs.update(extra)

    return kwargs


# ──────────────────────────────────────────────────────────────────────────────
# F2. FLUX-specific step callback (packed-latent aware)
# ──────────────────────────────────────────────────────────────────────────────

def make_flux_step_callback(
    tracker,                        # GenProgressTracker
    height: int,
    width: int,
    preview_interval: int = 5,
    preview_dir: Optional[Path] = None,
    enable_preview: bool = True,
) -> Callable:
    """
    Build a diffusers callback_on_step_end for FLUX / FLUX.2 pipelines.

    Key difference vs the generic make_step_callback in gen_progress.py:
      FLUX keeps latents in *packed* form during denoising:
        shape (batch, seq_len, channels)  — NOT (batch, C, H, W)
      Before calling the VAE decoder, they must be unpacked via
        pipe._unpack_latents(latents, height, width, vae_scale_factor).
      This function handles that transparently.

    Usage (in text_to_image):
        cb = make_flux_step_callback(tracker, cfg.height, cfg.width,
                                     preview_interval=cfg.preview_interval,
                                     enable_preview=cfg.enable_preview)
        kwargs = build_call_kwargs(..., callback_on_step_end=cb)
        pipe(**kwargs)

    Returns a callback with diffusers signature:
        fn(pipe, step_index: int, timestep: int, callback_kwargs: dict) → dict
    """
    import tempfile as _tmpmod
    _preview_dir = preview_dir or Path(_tmpmod.mkdtemp(prefix="genbox_preview_"))

    def _unpack_latents(latents, pipe) -> "torch.Tensor":  # type: ignore[name-defined]
        """
        Unpack FLUX packed latents to (B, C, H, W) for VAE decode.
        FLUX packs latents as (B, seq, C) inside the denoising loop.
        Uses pipe._unpack_latents — a staticmethod present on all FLUX pipeline variants.
        Falls back silently if unavailable (e.g. future API change).
        """
        if latents.ndim == 3 and hasattr(pipe, "_unpack_latents"):
            try:
                return pipe._unpack_latents(
                    latents,
                    height,
                    width,
                    getattr(pipe, "vae_scale_factor", 8),
                )
            except Exception:
                pass
        return latents

    def _callback(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
        try:
            tracker.set_step(step_index, stage="denoising")

            if (
                enable_preview
                and preview_interval > 0
                and step_index > 0
                and step_index % preview_interval == 0
            ):
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    from genbox.utils.gen_progress import decode_latents_to_preview
                    unpacked = _unpack_latents(latents, pipe)
                    path = decode_latents_to_preview(
                        unpacked, pipe, _preview_dir, step_index
                    )
                    if path is not None:
                        tracker.set_preview(path)
        except Exception:
            pass  # never kill generation over a progress update

        return {}

    return _callback

# ──────────────────────────────────────────────────────────────────────────────
# F3. SDL-specific step callback (standard spatial latents)
# ──────────────────────────────────────────────────────────────────────────────

def make_sdl_step_callback(
    tracker,                        # GenProgressTracker
    preview_interval: int = 5,
    preview_dir: Optional[Path] = None,
    enable_preview: bool = True,
) -> Callable:
    """
    Build a diffusers callback_on_step_end for SDL pipelines
    (SD1.5, SDXL, SD3.5, Pony).

    SDL latents are already in (B, C, H, W) spatial form — no unpacking needed.
    decode_latents_to_preview is called directly.

    Returns a callback with diffusers signature:
        fn(pipe, step_index: int, timestep: int, callback_kwargs: dict) → dict
    """
    import tempfile as _tmpmod
    _preview_dir = preview_dir or Path(_tmpmod.mkdtemp(prefix="genbox_preview_"))

    def _callback(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
        try:
            tracker.set_step(step_index, stage="denoising")

            if (
                enable_preview
                and preview_interval > 0
                and step_index > 0
                and step_index % preview_interval == 0
            ):
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    from genbox.utils.gen_progress import decode_latents_to_preview
                    path = decode_latents_to_preview(
                        latents, pipe, _preview_dir, step_index
                    )
                    if path is not None:
                        tracker.set_preview(path)
        except Exception:
            pass  # never kill generation

        return {}

    return _callback
# ──────────────────────────────────────────────────────────────────────────────
# G. Output path + metadata
# ──────────────────────────────────────────────────────────────────────────────

def build_output_path(
    kind: str,
    model_id: str,
    seed: int,
    ext: str,
    outputs_dir: Union[str, Path],
    custom: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Build a datestamped output path inside outputs_dir/<date>/.
    If custom is given, return it unchanged.
    """
    if custom is not None:
        return Path(custom)

    outputs_dir = Path(outputs_dir)
    date_str    = datetime.now().strftime("%Y-%m-%d")
    out_dir     = outputs_dir / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob(f"{kind}_*"))
    idx      = len(existing) + 1
    fname    = f"{kind}_{idx:04d}_{model_id}_seed{seed}.{ext}"
    return out_dir / fname


def build_output_meta(
    pipeline_name: str,
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
    extra: Optional[dict] = None,
) -> dict:
    """
    Build the sidecar metadata dict for a GenResult.
    lora_specs: list of LoraSpec entries (raw, will be normalized).
    """
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
