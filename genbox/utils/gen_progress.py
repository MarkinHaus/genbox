"""
genbox/gen_progress.py
Live generation progress tracking for the Streamlit UI.

Components:
  GenProgressTracker  — thread-safe state container
  make_step_callback  — diffusers callback_on_step_end factory
  decode_latents_to_preview — VAE decode → preview PNG (best-effort)
  GenRunner           — runs generation fn in a background thread
  format_step_label   — human-readable step/ETA string for UI

All ML work is optional and guarded — a decoding failure never crashes
the generation. The tracker always has a valid snapshot.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from genbox.pipeline import GenResult

# Optional PIL import — guarded so non-PIL environments still work
try:
    from PIL import Image as PIL_Image  # type: ignore
except ImportError:
    PIL_Image = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# GenProgressTracker
# ══════════════════════════════════════════════════════════════════════════════

class GenProgressTracker:
    """
    Thread-safe progress state for a single generation run.

    Fields in snapshot():
      step          — current denoising step (0-based)
      total         — total steps
      stage         — human label: "idle" | "loading model" | "denoising" | "saving" | "done"
      done          — True when generation completed successfully
      error         — True when generation failed
      error_msg     — exception message if error
      preview_path  — Path to latest latent preview PNG, or None
    """

    def __init__(self, total_steps: int):
        self._lock = threading.Lock()
        self._state: dict = {
            "step":         0,
            "total":        total_steps,
            "stage":        "idle",
            "done":         False,
            "error":        False,
            "error_msg":    "",
            "preview_path": None,
            "noise_std_history": [],
        }
        self._start_time: float = time.monotonic()
        # (monotonic_time, step) pairs for ETA estimation
        self._step_times: list[tuple[float, int]] = []

    def snapshot(self) -> dict:
        """Return a shallow copy of the current state. Thread-safe."""
        with self._lock:
            return dict(self._state)

    def fraction(self) -> float:
        """Return progress as 0.0–1.0."""
        with self._lock:
            total = self._state["total"]
            if total <= 0:
                return 0.0
            return min(1.0, self._state["step"] / total)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def eta_seconds(self) -> Optional[float]:
        """Estimate remaining seconds using linear regression on step times."""
        with self._lock:
            pairs = list(self._step_times)
            total = self._state["total"]
            step  = self._state["step"]

        if len(pairs) < 2 or step <= 0 or total <= step:
            return None

        # simple linear: steps/second rate from last few samples
        recent = pairs[-min(8, len(pairs)):]
        t0, s0 = recent[0]
        t1, s1 = recent[-1]
        if s1 <= s0:
            return None
        rate = (s1 - s0) / max(t1 - t0, 1e-6)  # steps/sec
        remaining = total - step
        return remaining / rate if rate > 0 else None

    def set_step(self, step: int, stage: str = "denoising"):
        with self._lock:
            self._state["step"]  = step
            self._state["stage"] = stage
            self._step_times.append((time.monotonic(), step))
            # Keep last 32 samples
            if len(self._step_times) > 32:
                self._step_times = self._step_times[-32:]

    def set_stage(self, stage: str):
        with self._lock:
            self._state["stage"] = stage

    def set_preview(self, path: Optional[Path]):
        with self._lock:
            self._state["preview_path"] = path

    def mark_done(self):
        with self._lock:
            self._state["step"]  = self._state["total"]
            self._state["done"]  = True
            self._state["stage"] = "done"

    def mark_error(self, msg: str):
        with self._lock:
            self._state["error"]     = True
            self._state["error_msg"] = msg
            self._state["stage"]     = "error"

    def set_noise_std(self, std: float):
        """
        Append a latent noise std sample to the history (Variante 3).
        Called from the video step callback after each denoising step.
        Thread-safe. Values are rounded to 4 decimal places.
        """
        with self._lock:
            self._state["noise_std_history"].append(round(std, 4))


# ══════════════════════════════════════════════════════════════════════════════
# Latent preview decoder (best-effort, never raises)
# ══════════════════════════════════════════════════════════════════════════════

def decode_latents_to_preview(
    latents,
    pipe,
    out_dir: Path,
    step: int,
) -> Optional[Path]:
    """
    Decode a latent tensor to a preview PNG using the pipeline VAE.

    Returns the Path of the saved PNG, or None on any failure.
    This function must never raise — exceptions are silently swallowed.
    """
    if PIL_Image is None:
        return None

    try:
        import torch  # type: ignore
        import numpy as np  # type: ignore

        vae    = pipe.vae
        factor = getattr(vae.config, "scaling_factor", 0.18215)

        with torch.no_grad():
            scaled   = latents / factor
            decoded  = vae.decode(scaled)
            sample   = decoded.sample

            # Normalize to 0–255
            arr = (sample / 2 + 0.5).clamp(0, 1)
            # Take first batch item: (C, H, W) → (H, W, C)
            arr = arr[0].permute(1, 2, 0).cpu().float().numpy()
            arr = (arr * 255).round().astype(np.uint8)

        img  = PIL_Image.fromarray(arr)
        # Downscale preview for speed (max 512px wide)
        if img.width > 512:
            ratio = 512 / img.width
            img = img.resize(
                (512, int(img.height * ratio)), PIL_Image.LANCZOS
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"preview_step{step:04d}.png"
        img.save(str(out_path))
        return out_path

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# make_step_callback
# ══════════════════════════════════════════════════════════════════════════════

def make_step_callback(
    tracker: GenProgressTracker,
    preview_interval: int = 5,
    decode_fn: Optional[Callable] = None,
    preview_dir: Optional[Path] = None,
) -> Callable:
    """
    Factory: return a diffusers callback_on_step_end function.

    The callback:
      1. Updates tracker.step and stage
      2. Every `preview_interval` steps, calls decode_fn(latents, pipe, step_index)
         if provided, and updates tracker.preview_path

    Signature matches diffusers callback_on_step_end:
      fn(pipe, step_index: int, timestep: int, callback_kwargs: dict) → dict
      First arg is the pipeline instance (self of the pipeline).

    decode_fn contract: (latents, pipe, step_index: int) -> Optional[Path]
      Use a lambda or functools.partial to pre-bind architecture-specific
      params (e.g. height/width for FLUX unpacking).
      For FLUX use make_flux_step_callback() from utils_image_pipeline instead.

    The callback NEVER raises — any internal error is silently swallowed
    so generation can continue unaffected.
    """
    _preview_dir = preview_dir or Path(
        __import__("tempfile").mkdtemp(prefix="genbox_preview_")
    )

    def _callback(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
        try:
            tracker.set_step(step_index, stage="denoising")

            if (
                decode_fn is not None
                and preview_interval > 0
                and step_index > 0
                and step_index % preview_interval == 0
            ):
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    # decode_fn receives (latents, pipe, step_index)
                    path = decode_fn(latents, pipe, step_index)
                    if path is not None:
                        tracker.set_preview(path)
        except Exception:
            pass  # never kill generation over a progress update

        return {}

    return _callback


# ══════════════════════════════════════════════════════════════════════════════
# GenRunner — background thread wrapper
# ══════════════════════════════════════════════════════════════════════════════

class GenRunner(threading.Thread):
    """
    Run a generation function in a daemon thread.

    Usage:
        tracker = GenProgressTracker(total_steps=28)
        runner  = GenRunner(fn=lambda t: pipeline.text_to_image(...), tracker=tracker)
        runner.start()
        # poll tracker.snapshot() from main thread
        runner.join()
        if runner.exception:
            handle_error(runner.exception)
        else:
            use(runner.result)
    """

    def __init__(self, fn: Callable, tracker: GenProgressTracker):
        super().__init__(daemon=True)
        self._fn        = fn
        self._tracker   = tracker
        self.result:    Optional[GenResult] = None
        self.exception: Optional[Exception] = None

    def run(self):
        try:
            self.result = self._fn(self._tracker)
            self._tracker.mark_done()
        except Exception as e:
            self.exception = e
            self._tracker.mark_error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Stage / ETA label formatting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_eta(seconds: Optional[float]) -> str:
    """Format ETA as '1m 5s', '30s', or '…'."""
    if seconds is None:
        return "…"
    s = int(seconds)
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def format_step_label(
    step: int,
    total: int,
    stage: str,
    eta: Optional[float],
) -> str:
    """
    Build a human-readable one-liner for the UI progress area.

    Examples:
      "loading model …"
      "denoising  5 / 28  ·  ETA 12s"
      "done  28 / 28  ✓"
    """
    stage_l = stage.lower()

    if "done" in stage_l:
        return f"done  {total} / {total}  ✓"

    if "error" in stage_l:
        return "error — see logs"

    if "load" in stage_l or stage_l in ("idle",):
        return f"{stage} …"

    if "saving" in stage_l:
        return "saving output …"

    eta_str = _fmt_eta(eta)
    return f"{stage}  {step} / {total}  ·  ETA {eta_str}"
