"""
genbox/ui_gen_progress.py
Streamlit progress widget utilities — wraps gen_progress.py for UI display.

These functions ARE allowed to import streamlit. They are NOT tested via
unittest (Streamlit widgets can't run headlessly). They are thin wrappers
that delegate all logic to gen_progress.py.

Public surface:
  render_progress(tracker, placeholder_bar, placeholder_label, placeholder_preview)
  run_with_progress(fn, tracker, poll_interval_ms, ...) → dict or raises
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import streamlit as st

from genbox.utils.gen_progress import (
    GenProgressTracker,
    GenRunner,
    format_step_label,
)


def render_progress(
    tracker: GenProgressTracker,
    ph_bar,       # st.empty() placeholder for progress bar
    ph_label,     # st.empty() placeholder for step label
    ph_preview,   # st.empty() placeholder for preview image
    show_preview: bool = True,
) -> None:
    """
    Update Streamlit placeholders from the current tracker snapshot.
    Call this in a polling loop; it never blocks.
    """
    snap    = tracker.snapshot()
    frac    = tracker.fraction()
    eta     = tracker.eta_seconds()
    label   = format_step_label(snap["step"], snap["total"], snap["stage"], eta)

    # ── progress bar ──────────────────────────────────────────────────────────
    ph_bar.progress(frac, text=None)

    # ── step label ────────────────────────────────────────────────────────────
    color = "#2dd4a0" if snap["done"] else "#ff4a4a" if snap["error"] else "#4a9eff"
    ph_label.markdown(
        f'<span style="font-size:11px;color:{color};">{label}</span>',
        unsafe_allow_html=True,
    )

    # ── intermediate preview ──────────────────────────────────────────────────
    if show_preview and snap["preview_path"] is not None:
        p = Path(snap["preview_path"])
        if p.exists():
            try:
                ph_preview.image(
                    str(p),
                    caption=f"Preview — step {snap['step']} / {snap['total']}",
                    width='stretch',
                )
            except Exception:
                pass  # preview display is purely cosmetic — never crash


def run_with_progress(
    fn: Callable,
    total_steps: int,
    poll_interval_s: float = 0.4,
    show_preview: bool = True,
    preview_interval: int = 5,
) -> dict:
    """
    Run `fn(tracker)` in a background thread while showing live progress.

    `fn` receives the GenProgressTracker and must return a result dict
    (output_path, metadata, elapsed_s) on success, or raise on failure.

    Returns the result dict.
    Raises the original exception if generation failed.

    Usage in screen_generate():
        result_dict = run_with_progress(
            fn=lambda t: pipeline.text_to_image(
                ...,
                callback_on_step_end=make_step_callback(t, preview_interval=5,
                    decode_fn=decode_latents_to_preview),
            ),
            total_steps=cfg_obj.steps,
        )
    """
    tracker = GenProgressTracker(total_steps=total_steps)

    # ── UI placeholders ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-label">Progress</div>',
        unsafe_allow_html=True,
    )
    ph_bar     = st.empty()
    ph_label   = st.empty()
    ph_preview = st.empty()

    # Initial render
    render_progress(tracker, ph_bar, ph_label, ph_preview, show_preview)

    # ── Launch background thread ──────────────────────────────────────────────
    runner = GenRunner(fn=fn, tracker=tracker)
    runner.start()

    # ── Poll until done ────────────────────────────────────────────────────────
    while runner.is_alive():
        render_progress(tracker, ph_bar, ph_label, ph_preview, show_preview)
        time.sleep(poll_interval_s)

    runner.join()
    # Final render
    render_progress(tracker, ph_bar, ph_label, ph_preview, show_preview)

    if runner.exception is not None:
        raise runner.exception

    return runner.result


def make_logcat(max_lines: int = 20):
    """
    Return a (placeholder, log_fn) pair.
    log_fn(msg, kind="") appends a line to the logcat display.

    kind: "" | "ok" | "accent" | "err"
    """
    ph     = st.empty()
    lines  = []

    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    def log(msg: str, kind: str = ""):
        css_cls = {"ok": "ok", "accent": "accent", "err": "err"}.get(kind, "")
        cls_attr = f' class="{css_cls}"' if css_cls else ""
        lines.append(
            f'<span class="ts">[{_ts()}]</span> '
            f'<span{cls_attr}>{msg}</span><br>'
        )
        ph.markdown(
            f'<div class="logcat">{"".join(lines[-max_lines:])}</div>',
            unsafe_allow_html=True,
        )

    return ph, log
