"""
genbox/pipeline_flux.py
FLUX.1 and FLUX.2 (Klein) image generation pipeline.

Supports:
  - Full diffusers-format repos (FLUX.1-dev, FLUX.1-schnell, FLUX.2-klein-4B/9B)
  - GGUF single-file (Q4/Q8) with shared pipeline config
  - T5 / Qwen3 text encoder handling (fp16 | int8 | none)
  - Multi-LoRA via PEFT
  - Full accelerator stack (offload + xformers + torch.compile)

All paths are local-only. 100 % offline after model installation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from genbox.utils.utils_image_pipeline import (
    apply_accelerators,
    apply_loras_to_pipe,
    build_lora_adapter_list,
    build_output_meta,
    build_output_path,
    build_call_kwargs,
    inject_compile,
    make_generator,
    resolve_device,
    resolve_dtype,
    resolve_offload_mode,
    resolve_seed,
    set_scheduler,
)

log = logging.getLogger("genbox.pipeline_flux")


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class FluxPipelineConfig:
    """
    All parameters for a FLUX text-to-image generation run.
    Covers FLUX.1 (schnell/dev) and FLUX.2 (klein 4B/9B).
    """
    model_id:       str
    prompt:         str       = ""
    negative_prompt: str      = ""   # ignored for FLUX — kept for API symmetry
    width:          int       = 1024
    height:         int       = 1024
    steps:          int       = 28
    guidance_scale: float     = 3.5
    seed:           int       = -1
    t5_mode:        str       = "fp16"  # "fp16" | "int8" | "none"
    sampler:        str       = "default"
    loras:          list      = field(default_factory=list)
    accel:          list      = field(default_factory=list)
    output:         Optional[Union[str, Path]] = None


# ── Path helpers ───────────────────────────────────────────────────────────────

def _resolve_flux_local_path(
    entry,
    models_dir: Union[str, Path],
) -> tuple[Path, bool]:
    """
    Return (local_path, is_gguf) for a FLUX model entry.

    full_repo  → models_dir/flux/<id>/   (must contain model_index.json)
    gguf       → models_dir/flux/<filename>.gguf
    Raises FileNotFoundError if not installed.
    """
    models_dir = Path(models_dir)
    arch_dir   = models_dir / "flux"

    if entry.full_repo:
        p = arch_dir / entry.id
        if not (p / "model_index.json").exists():
            raise FileNotFoundError(
                f"FLUX model not found: {p}\n"
                "Download via Models panel."
            )
        return p, False

    # GGUF single file
    # Single-file: GGUF oder custom .safetensors — per Suffix unterscheiden
    p = arch_dir / Path(entry.hf_filename).name
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return p, p.suffix.lower() == ".gguf"


def _flux_gguf_shared_config(
    entry,
    models_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Return the shared pipeline config dir for GGUF models.
    None if no hf_pipeline_repo set.
    """
    if not entry.hf_pipeline_repo:
        return None
    safe_name = entry.hf_pipeline_repo.replace("/", "--")
    return Path(models_dir) / "flux" / f"_shared_{safe_name}"


# ── T5 / Qwen encoder kwargs ───────────────────────────────────────────────────

def build_t5_kwargs(
    local_repo: Path,
    t5_mode: str,
    is_flux2: bool,
    dtype,
) -> dict:
    """
    Build text_encoder_2 kwargs for from_pretrained / from_single_file.

    FLUX.2 klein uses Qwen3 — diffusers handles it from the repo directly.
    FLUX.1 uses T5-XXL (optional, expensive).

    Returns a dict to be **unpacked into from_pretrained().
    """
    # FLUX.2: Qwen3 — no manual T5 handling needed
    if is_flux2:
        return {}

    has_t5 = (local_repo / "text_encoder_2").exists()

    if not has_t5 or t5_mode == "none":
        log.info("T5 disabled")
        return {"text_encoder_2": None, "tokenizer_2": None}

    if t5_mode == "int8":
        try:
            from transformers import T5EncoderModel, BitsAndBytesConfig  # type: ignore
            t5 = T5EncoderModel.from_pretrained(
                str(local_repo),
                subfolder="text_encoder_2",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=dtype,
                local_files_only=True,
            )
            log.info("T5 int8 loaded")
            return {"text_encoder_2": t5}
        except Exception as e:
            log.warning(f"T5 int8 failed, falling back to fp16: {e}")

    log.info("T5 fp16")
    return {}  # fp16 → diffusers loads from repo automatically


# ── Detect FLUX variant from local repo ───────────────────────────────────────

def _detect_flux_classes_from_repo(local_repo: Path) -> tuple[str, str]:
    """
    Read model_index.json or transformer/config.json → (pipeline_class, transformer_class).
    Fallback to FLUX.1 classes.
    """
    import json

    mi = local_repo / "model_index.json"
    if mi.exists():
        try:
            data = json.loads(mi.read_text(encoding="utf-8"))
            pipe_cls = data.get("_class_name", "FluxPipeline")
            tr_cls   = "Flux2Transformer2DModel" if "Flux2" in pipe_cls else "FluxTransformer2DModel"
            return pipe_cls, tr_cls
        except Exception:
            pass

    tc = local_repo / "transformer" / "config.json"
    if tc.exists():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            tr_cls   = data.get("_class_name", "FluxTransformer2DModel")
            pipe_cls = "Flux2KleinPipeline" if "Flux2" in tr_cls else "FluxPipeline"
            return pipe_cls, tr_cls
        except Exception:
            pass

    return "FluxPipeline", "FluxTransformer2DModel"


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_flux_pipe(entry, models_dir: Union[str, Path], dtype, t5_mode: str = "fp16"):
    """
    Load a FLUX pipeline (FLUX.1 or FLUX.2) from local storage.

    Handles:
      - full diffusers-repo (fp8/bf16)
      - GGUF single-file (Q4/Q8) with shared config
      - T5 int8 / Qwen3 encoder
    Returns a diffusers pipeline (not yet moved to device).
    """
    import diffusers  # type: ignore

    local_path, _is_gguf = _resolve_flux_local_path(entry, models_dir)
    pipe_cls_name, tr_cls_name = _detect_flux_classes_from_repo(
        local_path if not _is_gguf else
        (_flux_gguf_shared_config(entry, models_dir) or local_path)
    )
    is_flux2 = "Flux2" in tr_cls_name

    PipelineClass    = getattr(diffusers, pipe_cls_name,    diffusers.FluxPipeline)
    TransformerClass = getattr(diffusers, tr_cls_name, diffusers.FluxTransformer2DModel)

    if _is_gguf:
        # ── GGUF path (unverändert) ─────────────────────────────────────────
        shared_config = _flux_gguf_shared_config(entry, models_dir)
        if shared_config is None or not shared_config.exists():
            raise FileNotFoundError(
                f"Shared GGUF config missing for {entry.name}.\n"
                f"Expected: {shared_config}\n"
                "Download via Models → 'Download Shared Config'."
            )
        transformer_config_dir = str(shared_config / "transformer")
        try:
            import gguf  # noqa
        except ImportError:
            raise ImportError("GGUF support requires: pip install 'gguf>=0.10.0'")

        transformer = TransformerClass.from_single_file(
            str(local_path),
            config=transformer_config_dir,
            quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            local_files_only=True,
        )
        t5_kw = build_t5_kwargs(shared_config, t5_mode, is_flux2, dtype)
        pipe = PipelineClass.from_pretrained(
            str(shared_config),
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
            **t5_kw,
        )

    elif local_path.is_dir():
        # ── Full-repo path (unverändert) ────────────────────────────────────
        t5_kw = build_t5_kwargs(local_path, t5_mode, is_flux2, dtype)
        pipe = PipelineClass.from_pretrained(
            str(local_path),
            torch_dtype=dtype,
            local_files_only=True,
            **t5_kw,
        )

    else:
        # ── Custom .safetensors — Transformer-swap mit shared config ────────
        # Identisch zu GGUF, aber ohne GGUFQuantizationConfig.
        # Setzt entry.hf_pipeline_repo auf das Base-Modell (z.B. "flux1_dev").
        shared_config = _flux_gguf_shared_config(entry, models_dir)
        if shared_config is None or not shared_config.exists():
            raise FileNotFoundError(
                f"Custom .safetensors '{local_path.name}' requires a shared pipeline "
                f"config for base model '{getattr(entry, 'hf_pipeline_repo', '?')}'.\n"
                f"Expected: {shared_config}\n"
                "Set entry.hf_pipeline_repo to the correct base model and download "
                "its shared config via Models → 'Download Shared Config'."
            )
        transformer_config_dir = str(shared_config / "transformer")
        transformer = TransformerClass.from_single_file(
            str(local_path),
            config=transformer_config_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
        log.info(f"Custom safetensors loaded: {local_path.name} (base: {shared_config.name})")
        t5_kw = build_t5_kwargs(shared_config, t5_mode, is_flux2, dtype)
        pipe = PipelineClass.from_pretrained(
            str(shared_config),
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
            **t5_kw,
        )

    # Clamp tokenizer for FLUX.1 with very large model_max_length
    if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
        max_len = getattr(pipe.tokenizer, "model_max_length", 0)
        if isinstance(max_len, int) and max_len > 1024:
            pipe.tokenizer.model_max_length = 512
            log.info(f"Tokenizer model_max_length clamped {max_len} → 512")

    return pipe


# ── Accelerator entry point ────────────────────────────────────────────────────

def apply_pipeline_accelerators(
    pipe,
    device: str,
    vram_gb: int = 16,
    accel: Optional[list] = None,
    has_quantized_encoders: bool = False,
    env_override: Optional[str] = None,
) -> None:
    """
    Apply CPU offload strategy + optional accelerators (xformers, compile, …).
    Modifies pipe in-place.
    """
    accel = accel or []
    env_override = env_override or os.environ.get("GENBOX_OFFLOAD", "").lower() or None
    offload_mode = resolve_offload_mode(vram_gb, env_override, has_quantized_encoders)
    apply_accelerators(pipe, device=device, offload_mode=offload_mode, accel=accel)
    inject_compile(pipe, accel)


# ── Public generation function ─────────────────────────────────────────────────

def text_to_image(
    cfg: FluxPipelineConfig,
    entry,
    models_dir: Union[str, Path],
    loras_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
    vram_gb: int = 16,
):
    """
    Run FLUX text-to-image generation.

    Args:
        cfg:        FluxPipelineConfig
        entry:      ModelEntry (from genbox.models.REGISTRY)
        models_dir: root models directory
        loras_dir:  root LoRA directory (for relative LoRA paths)
        outputs_dir: root outputs directory
        vram_gb:    available VRAM for offload decisions

    Returns:
        dict with keys: output_path, metadata, elapsed_s
    """
    import time
    from pathlib import Path as _Path

    t0     = time.time()
    seed   = resolve_seed(cfg.seed)
    device = resolve_device()
    dtype  = resolve_dtype(entry.quant)

    log.info(f"FLUX T2I | model={cfg.model_id} seed={seed} steps={cfg.steps} device={device}")

    pipe = load_flux_pipe(entry, models_dir, dtype, t5_mode=cfg.t5_mode)
    set_scheduler(pipe, "flux", cfg.sampler)

    # LoRAs
    adapter_list = build_lora_adapter_list(cfg.loras, loras_dir=loras_dir)
    apply_loras_to_pipe(pipe, adapter_list, architecture="flux")

    # Accelerators
    apply_pipeline_accelerators(
        pipe, device=device, vram_gb=vram_gb, accel=cfg.accel,
        has_quantized_encoders=(cfg.t5_mode == "int8"),
    )

    gen = make_generator(seed, device)
    kwargs = build_call_kwargs(
        architecture="flux",
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=cfg.width, height=cfg.height,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=gen, t5_mode=cfg.t5_mode,
    )

    result    = pipe(**kwargs)
    image     = result.images[0]

    _out_dir  = _Path(outputs_dir) if outputs_dir else _Path.cwd() / "genbox_outputs"
    out_path  = build_output_path("img", cfg.model_id, seed, "png",
                                  outputs_dir=_out_dir, custom=cfg.output)
    image.save(str(out_path))

    elapsed = time.time() - t0
    meta    = build_output_meta(
        pipeline_name="flux_text_to_image",
        model_id=cfg.model_id,
        prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
        width=kwargs["width"], height=kwargs["height"],
        steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        seed=seed, lora_specs=cfg.loras, accel=cfg.accel,
        sampler=cfg.sampler, elapsed_s=elapsed,
        output_path=out_path,
    )

    log.info(f"FLUX T2I done → {out_path.name} ({elapsed:.1f}s)")
    return {"output_path": out_path, "metadata": meta, "elapsed_s": elapsed}
