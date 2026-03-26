"""
genbox/ui.py
Streamlit UI — 4 screens: Generate · Library · Models · Pipeline
Design: terminal-dark, #0d0d0d bg, #4a9eff accent, JetBrains Mono
Run: genbox ui   OR   streamlit run genbox/ui.py

Generation progress:
  - Live step counter + ETA via GenProgressTracker
  - Intermediate latent previews every N steps (VAE decode, best-effort)
  - Stage labels: loading model → denoising → saving
  - GenRunner runs generation in background thread; UI polls at 400ms
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st

from genbox.pipeline import GenResult

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genbox.config import cfg
from genbox import models as model_lib
from genbox.models import REGISTRY, list_registry, list_local, list_loras
from genbox.genbox_ui.ui_helpers import (
    get_pipe_types, detect_upload_type,
    guess_arch_from_filename, load_outputs, build_remix_data,
    validate_outpaint_expansion, format_lora_label, get_install_defaults_for_profile,
)
from genbox.utils.gen_progress import (
    GenProgressTracker, GenRunner, make_step_callback,
    decode_latents_to_preview, format_step_label,
)
from genbox.genbox_ui.ui_gen_progress import make_logcat, render_progress

# ── constants ──────────────────────────────────────────────────────────────────

_UI_SAMPLERS: dict[str, list[str]] = {
    "flux":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler"],
    "sd35":  ["FlowMatchEuler", "DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM"],
    "sd15":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
    "sdxl":  ["DPM++ 2M", "DPM++ 2M Karras", "Euler", "Euler A", "DDIM", "UniPC"],
    "ltx":   ["FlowMatchEuler", "DPM++ 2M"],
    "wan":   ["UniPC (flow_shift)", "FlowMatchEuler", "DPM++ 2M"],
}
_SAMPLER_HINTS = {
    "FlowMatchEuler":     "Standard Flow-Matching. Empfohlen für FLUX & LTX.",
    "DPM++ 2M":           "Schnelle Konvergenz. Weniger Steps nötig.",
    "DPM++ 2M Karras":    "DPM++ mit Karras-Sigma — oft schärfer bei 15–25 Steps.",
    "Euler":              "Klassisch stabil. Gut für hohe Step-Zahlen.",
    "Euler A":            "Ancestral — mehr Variation pro Seed.",
    "DDIM":               "Deterministisch & reproduzierbar.",
    "UniPC":              "WAN Default ohne flow_shift.",
    "UniPC (flow_shift)": "WAN empfohlen: flow_shift 3.0 (480p) / 5.0 (720p).",
}
_QUICK_TAGS = ["draft", "final", "portrait", "landscape", "wip", "upscale"]

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="genbox",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "genbox — local AI generation"},
)

# ── CSS ────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stApp"] {
    background: #0d0d0d !important; color: #c8c8c8 !important;
    font-family: 'JetBrains Mono','Fira Code','Consolas',monospace !important; font-size: 13px;
}
[data-testid="stSidebar"] { background: #111111 !important; border-right: 1px solid #2a2a2a; }
[data-testid="stSidebar"] * { font-family: inherit !important; }
h1,h2,h3 { font-family: inherit !important; font-weight: 500 !important;
    letter-spacing: 0.04em; color: #e0e0e0 !important; }
h1 { font-size: 15px !important; color: #4a9eff !important; }
h2 { font-size: 13px !important; color: #c8c8c8 !important; border-bottom: 1px solid #2a2a2a; padding-bottom: 6px; }
h3 { font-size: 12px !important; color: #6b6b6b !important; }
input, textarea, [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
    background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 2px !important; color: #c8c8c8 !important;
    font-family: inherit !important; font-size: 13px !important; }
input:focus, textarea:focus { border-color: #4a9eff !important; box-shadow: 0 0 0 1px #4a9eff22 !important; }
[data-testid="stSelectbox"] > div > div { background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 2px !important; color: #c8c8c8 !important; font-family: inherit !important; }
[data-testid="stMetric"] { background: #141414; border: 1px solid #2a2a2a; border-radius: 2px; padding: 10px 14px; }
[data-testid="stMetricValue"] { color: #4a9eff !important; font-size: 18px !important; }
[data-testid="stMetricLabel"] { color: #6b6b6b !important; font-size: 11px !important; }
hr { border-color: #2a2a2a !important; margin: 16px 0 !important; }
code, pre { background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 2px !important; color: #2dd4a0 !important; font-family: inherit !important; font-size: 12px !important; }
[data-testid="stExpander"] { background: #141414 !important; border: 1px solid #2a2a2a !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { color: #6b6b6b !important; font-size: 12px !important; }
[data-testid="stProgress"] > div > div { background: #4a9eff !important; }
[data-testid="stProgress"] { background: #1a1a1a !important; }
[data-testid="stAlert"] { background: #141414 !important; border-radius: 2px !important;
    border-left: 2px solid #4a9eff !important; color: #c8c8c8 !important; }
[data-testid="stImage"] img { border: 1px solid #2a2a2a; border-radius: 2px; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #4a9eff; }
.stButton > button { background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 2px !important; color: #c8c8c8 !important; font-family: inherit !important;
    font-size: 12px !important; letter-spacing: 0.06em; transition: border-color .15s, color .15s; padding: 6px 16px !important; }
.stButton > button:hover { border-color: #4a9eff !important; color: #4a9eff !important; background: #0d0d0d !important; }
.stButton > button:active { background: #4a9eff22 !important; }
[data-testid="stRadio"] label { color: #6b6b6b !important; font-family: inherit !important;
    font-size: 12px !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stRadio"] label:hover { color: #c8c8c8 !important; }
[data-testid="stSidebarCollapseButton"] span, [data-testid="stSidebarCollapseButton"] svg { display: none !important; }
[data-testid="stSidebarCollapseButton"]::after { content: "‹"; color: #3a3a3a; font-size: 14px; }
[data-testid="stSidebarCollapseButton"][aria-expanded="false"]::after { content: "›"; }
.logcat { background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 2px;
    padding: 12px 16px; font-size: 11px; line-height: 1.7; max-height: 320px; overflow-y: auto; color: #6b6b6b; }
.logcat .ts { color: #3a3a3a; }
.logcat .ok { color: #2dd4a0; }
.logcat .accent { color: #4a9eff; }
.logcat .err { color: #ff4a4a; }
.gen-card { background: #141414; border: 1px solid #2a2a2a; border-radius: 2px;
    padding: 12px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }
.gen-card:hover { border-color: #4a9eff; }
.gen-card .prompt { color: #c8c8c8; font-size: 11px; line-height: 1.5; }
.gen-card .meta { color: #6b6b6b; font-size: 10px; margin-top: 6px; }
.vram-ok  { color: #2dd4a0 !important; }
.vram-warn { color: #ffaa00 !important; }
.vram-err  { color: #ff4a4a !important; }
.lora-badge { display: inline-block; background: #1a1a1a; border: 1px solid #3a3a3a;
    border-radius: 2px; padding: 1px 6px; font-size: 10px; color: #ff8c42; margin-left: 6px; }
.lora-path-hint { background: #0d0d0d; border: 1px dashed #2a2a2a; border-radius: 2px;
    padding: 10px 12px; font-size: 10px; color: #3a3a3a; line-height: 1.8; }
.tag { display: inline-block; background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 2px; padding: 1px 6px; font-size: 10px; color: #6b6b6b; margin: 2px 2px 0 0; }
.section-label { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a4a4a; margin-bottom: 8px; margin-top: 16px; }
.drop-zone { background: #0d0d0d; border: 1px dashed #3a3a3a; border-radius: 2px;
    padding: 20px; text-align: center; color: #4a4a4a; font-size: 11px; }
.drop-zone:hover { border-color: #4a9eff; color: #4a9eff; }
/* Progress area */
.progress-stage { font-size: 11px; color: #4a9eff; margin: 4px 0; }
.preview-label  { font-size: 10px; color: #6b6b6b; margin-bottom: 4px; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def _logline(msg: str, kind: str = "") -> str:
    css = {"ok": "ok", "accent": "accent", "err": "err"}.get(kind, "")
    cls = f' class="{css}"' if css else ""
    return f'<span class="ts">[{_ts()}]</span> <span{cls}>{msg}</span><br>'

def _vram_color(entry) -> str:
    if cfg is None: return "vram-ok"
    if entry.vram_min_gb <= cfg.vram_gb: return "vram-ok"
    if entry.vram_min_gb <= cfg.vram_gb + 2: return "vram-warn"
    return "vram-err"

def _stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)

def _is_installed(entry) -> bool:
    return model_lib._is_installed_entry(entry)

def _xor_crypt(data: bytes, key: str = "genbox_inkognito_secret") -> bytes:
    key_bytes = key.encode()
    k = len(key_bytes)
    return bytes(b ^ key_bytes[i % k] for i, b in enumerate(data))

def go_generate():
    st.session_state["_nav_pending"] = "Generate"

def _run_pipeline_code(code: str) -> tuple[str, str]:
    full = f"""
import sys
sys.path.insert(0, {str(_ROOT)!r})
from genbox import pipeline
{code}
"""
    try:
        r = __import__("subprocess").run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=600,
        )
        return r.stdout, r.stderr
    except __import__("subprocess").TimeoutExpired:
        return "", "Timeout after 600s"
    except Exception as e:
        return "", str(e)


# ── sidebar nav ────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("## genbox")
        if cfg:
            vram = cfg.vram_gb
            color = "vram-ok" if vram >= 10 else "vram-warn"
            st.markdown(
                f'<span class="{color}" style="font-size:11px;">'
                f'{vram}GB VRAM &nbsp;·&nbsp; {cfg.vram_profile}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="vram-err" style="font-size:11px;">not configured</span>',
                        unsafe_allow_html=True)
        st.markdown("---")

        if "_nav_pending" in st.session_state:
            _nav_target = st.session_state.pop("_nav_pending")
            st.session_state.pop("nav_radio", None)
            st.session_state["nav_radio"] = _nav_target

        screen = st.radio(
            "Navigation", ["Generate", "Library", "Models", "Pipeline"],
            label_visibility="collapsed", key="nav_radio",
        )
        st.markdown("---")
        has_token = bool(os.environ.get("HF_TOKEN") or (cfg and getattr(cfg, "hf_token", "")))
        st.markdown(
            '<span style="font-size:10px;color:#2dd4a0;">HF token ✓</span>' if has_token
            else '<span style="font-size:10px;color:#4a4a4a;">no HF token</span>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown('<span style="font-size:10px;color:#3a3a3a;">v0.2.0</span>',
                    unsafe_allow_html=True)
    return screen


# ══════════════════════════════════════════════════════════════════════════════
# Screen 1 — Generate
# ══════════════════════════════════════════════════════════════════════════════

def screen_generate():
    st.markdown("## Generate")

    # Handle pending remix
    if st.session_state.pop("_remix_pending", False):
        data = st.session_state.pop("_remix_data", {})
        for k in ("prompt", "neg_prompt", "sel_model", "steps", "guidance", "seed", "pipe_type"):
            st.session_state.pop(k, None)
        for k, v in data.items():
            st.session_state[k] = v

    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        # ── pipeline type (no Multi-Image) ────────────────────────────────────
        st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
        pipe_type = st.radio(
            "Pipeline Type", get_pipe_types(),
            horizontal=True, label_visibility="collapsed", key="pipe_type",
        )
        st.markdown("")

        # ── prompt ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Prompt</div>', unsafe_allow_html=True)
        prompt = st.text_area("Prompt", height=90, placeholder="describe the output…",
                              key="prompt", label_visibility="collapsed")
        neg_prompt = st.text_input("Negative prompt", placeholder="optional negative prompt",
                                   key="neg_prompt", label_visibility="collapsed")

        # ── input image ───────────────────────────────────────────────────────
        needs_input = pipe_type in ("Image → Image", "Inpaint", "Outpaint", "Image → Video")
        uploaded = None
        if needs_input:
            st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Input image", type=["png", "jpg", "jpeg", "webp"],
                key="single_img", label_visibility="collapsed",
            )

            if uploaded:
                try:
                    from PIL import Image as PILImage
                    img_obj = PILImage.open(uploaded).convert("RGB")
                    orig_w, orig_h = img_obj.size
                    snap_w = max(256, (orig_w // 16) * 16)
                    snap_h = max(256, (orig_h // 16) * 16)
                    c_prev, c_sync = st.columns([1, 1])
                    with c_prev:
                        st.image(img_obj, caption=f"{orig_w}×{orig_h}", width='stretch')
                    with c_sync:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button(f"Sync {snap_w}×{snap_h}", key="btn_sync_res"):
                            for k in ("width", "height", "vid_w", "vid_h"):
                                if "vid" in k:
                                    st.session_state[k] = snap_w if "w" in k else snap_h
                                else:
                                    st.session_state[k] = snap_w if k == "width" else snap_h
                            st.rerun()
                except Exception:
                    pass

            # ── Inpaint mask upload ────────────────────────────────────────────
            if pipe_type == "Inpaint":
                st.markdown('<div class="section-label">Mask (white = fill)</div>',
                            unsafe_allow_html=True)
                mask_upload = st.file_uploader(
                    "Mask image", type=["png", "jpg", "jpeg"],
                    key="mask_img", label_visibility="collapsed",
                )
                with st.expander("Mask settings"):
                    blur_r  = st.slider("Blur radius (feather)",  0, 30, 0, key="blur_r",
                                        help="Gaussian blur on mask edges for soft transitions")
                    dilate  = st.slider("Dilate (expand mask)",   0, 30, 0, key="dilate_px",
                                        help="Expand mask outward by N pixels")
                    mask_mode = st.radio("Convention",
                                         ["white_inpaint", "black_inpaint"],
                                         horizontal=True, key="mask_mode",
                                         help="white_inpaint = white area gets filled")
                if mask_upload:
                    try:
                        from PIL import Image as PILImage
                        mask_prev = PILImage.open(mask_upload).convert("L")
                        st.image(mask_prev, caption="Mask preview", width='stretch')
                    except Exception:
                        pass

            # ── Outpaint expansion ─────────────────────────────────────────────
            if pipe_type == "Outpaint":
                st.markdown('<div class="section-label">Canvas Expansion (px)</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1: expand_left   = st.number_input("Left",   0, 2048, 0, 64, key="exp_l")
                with c2: expand_right  = st.number_input("Right",  0, 2048, 0, 64, key="exp_r")
                with c3: expand_top    = st.number_input("Top",    0, 2048, 0, 64, key="exp_t")
                with c4: expand_bottom = st.number_input("Bottom", 0, 2048, 0, 64, key="exp_b")
                feather = st.slider("Feather radius", 0, 64, 16, key="feather",
                                    help="Smooth the seam between original and generated area")

            # ── Image→Video end frame ─────────────────────────────────────────
            if pipe_type == "Image → Video":
                end_frame = st.file_uploader(
                    "End frame (optional, LTX FLF mode)",
                    type=["png", "jpg"], key="end_frame",
                    label_visibility="collapsed",
                )

        # ── model ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
        is_video   = "Video" in pipe_type
        model_type = "video" if is_video else "image"

        avail_models = [e for e in list_registry(model_type=model_type) if _is_installed(e)]
        if not avail_models:
            st.warning(f"No {model_type} models installed. Go to **Models** → download one first.")
            return

        model_labels = {
            e.id: f"{e.name}  ({e.quant})  ·  {e.vram_min_gb}GB"
            for e in avail_models
        }
        default_id = ((cfg.default_video_model if is_video else cfg.default_image_model)
                      if cfg else avail_models[0].id)
        if default_id not in model_labels:
            default_id = avail_models[0].id

        selected_model = st.selectbox(
            "Model", list(model_labels.keys()),
            format_func=lambda x: model_labels[x],
            index=list(model_labels.keys()).index(default_id),
            key="sel_model", label_visibility="collapsed",
        )

        # ── sampler ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Sampler</div>', unsafe_allow_html=True)
        sel_entry    = REGISTRY.get(selected_model)
        _arch_key    = sel_entry.architecture if sel_entry else "flux"
        sampler_opts = _UI_SAMPLERS.get(_arch_key, ["FlowMatchEuler"])
        selected_sampler = st.selectbox(
            "Sampler", sampler_opts, index=0, key="sel_sampler", label_visibility="collapsed",
        )
        st.markdown(
            f'<span style="font-size:10px;color:#4a4a4a;">'
            f'{_SAMPLER_HINTS.get(selected_sampler, "")}</span>',
            unsafe_allow_html=True,
        )

        # ── loras ─────────────────────────────────────────────────────────────
        arch = REGISTRY[selected_model].architecture if selected_model else None
        local_loras = list_loras(architecture=arch)
        n_lora = len(local_loras)
        lora_label_html = (
            f'<div class="section-label">LoRA '
            f'<span class="lora-badge">{n_lora}</span></div>'
            if n_lora else '<div class="section-label">LoRA</div>'
        )
        st.markdown(lora_label_html, unsafe_allow_html=True)

        lora_paths: list[tuple[str, float]] = []
        if not local_loras:
            lora_dir = str(cfg.loras_dir) if cfg else "loras/"
            st.markdown(
                f'<div class="lora-path-hint">'
                f'No LoRAs found.<br>'
                f'Place .safetensors in: <span style="color:#4a9eff;">{lora_dir}/</span><br>'
                f'Or in arch subfolders: <span style="color:#4a9eff;">{lora_dir}/flux/  {lora_dir}/wan/</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            lora_display = {format_lora_label(lo): lo["path"] for lo in local_loras}
            sel_loras = st.multiselect(
                "LoRA", list(lora_display.keys()),
                key="sel_loras", label_visibility="collapsed",
                placeholder="Select LoRAs…",
            )
            if sel_loras:
                st.markdown('<div class="section-label" style="margin-top:6px;">LoRA Weights</div>',
                            unsafe_allow_html=True)
                for lora_label in sel_loras:
                    short = lora_label.split("  ")[0]
                    weight = st.slider(short, 0.0, 2.0, 1.0, 0.05, key=f"lora_w_{lora_label}")
                    lora_paths.append((lora_display[lora_label], weight))
            if arch == "wan" and sel_loras:
                st.markdown(
                    '<div style="color:#ff8c42;font-size:10px;margin-top:4px;">'
                    '⚠ WAN LoRA: diffusers support is experimental.</div>',
                    unsafe_allow_html=True,
                )

    with col_right:
        # ── params ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)

        if not is_video:
            w = st.select_slider("Width",  [512,640,768,896,1024,1152,1280,1408,1536], 1024, key="width")
            h = st.select_slider("Height", [512,640,768,896,1024,1152,1280,1408,1536], 1024, key="height")
        else:
            w = st.select_slider("Width",  [512,640,768,832,960,1024,1280], 832, key="vid_w")
            h = st.select_slider("Height", [320,480,512,576,704,720],   480, key="vid_h")
            n_frames = st.slider("Frames", 5, 201, 81, step=4, key="frames",
                                 help="WAN: 4n+1 (e.g. 81)  ·  LTX: 8n+1 (e.g. 97)")
            vid_fps  = st.slider("FPS", 8, 30, 24, key="fps")

        steps    = st.slider("Steps",    1, 100, 28, key="steps")
        guidance = st.slider("Guidance", 1.0, 15.0, 3.5, step=0.5, key="guidance")
        seed     = st.number_input("Seed  (-1 = random)", value=-1, step=1, key="seed")

        if pipe_type in ("Image → Image", "Inpaint", "Outpaint"):
            strength = st.slider("Strength", 0.1, 1.0, 0.99 if pipe_type == "Inpaint" else 0.75,
                                 step=0.05, key="strength")

        # ── T5 encoder (FLUX only) ─────────────────────────────────────────────
        is_flux = sel_entry and sel_entry.architecture == "flux"
        if is_flux and not is_video:
            st.markdown('<div class="section-label">T5 Text Encoder</div>', unsafe_allow_html=True)
            t5_mode = st.radio("T5 Mode", ["fp16", "int8"], index=0, horizontal=True,
                               key="t5_mode", label_visibility="collapsed",
                               help="fp16: best quality  ·  int8: ~50% less VRAM (bitsandbytes)")
            _t5_hints = {
                "fp16": '<span style="font-size:10px;color:#6b6b6b;">Standard quality</span>',
                "int8": '<span style="font-size:10px;color:#2dd4a0;">Low-VRAM — needs bitsandbytes</span>',
            }
            st.markdown(_t5_hints.get(t5_mode, ""), unsafe_allow_html=True)
        else:
            t5_mode = "fp16"

        # ── preview interval ──────────────────────────────────────────────────
        st.markdown('<div class="section-label">Live Preview</div>', unsafe_allow_html=True)
        show_preview = st.checkbox("Show intermediate previews", value=True, key="show_preview",
                                   help="Decode VAE latents every N steps (requires a bit more VRAM)")
        if show_preview:
            preview_interval = st.slider("Preview every N steps", 1, 20, 5,
                                         key="preview_interval",
                                         help="Lower = more previews but slightly slower")
        else:
            preview_interval = 0

        # ── accelerators ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Accelerators</div>', unsafe_allow_html=True)
        default_accels = cfg.active_accels if cfg else []
        use_sage = st.checkbox("SageAttention2++", value="sageAttn" in default_accels, key="use_sage")
        use_tea  = st.checkbox("TeaCache",          value="teacache"  in default_accels, key="use_tea")
        use_xformers  = st.checkbox("xformers Only for sdl1.5 and xl type",          value="xformers"  in default_accels, key="use_xformers")
        use_comp = st.checkbox("torch.compile",     value="compile"   in default_accels, key="use_comp")
        accel = (["sageAttn"] if use_sage else []) + (["teacache"] if use_tea else []) + (["compile"] if use_comp else [])+ (["xformers"] if use_xformers else [])

        # ── output ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        custom_out  = st.text_input("Output path", placeholder="auto",
                                    key="custom_out", label_visibility="collapsed")

        # Quick-select tag pills + free-text input
        st.markdown('<span style="font-size:10px;color:#4a4a4a;">Tag:</span>',
                    unsafe_allow_html=True)
        tag_cols = st.columns(len(_QUICK_TAGS))
        for i, qt in enumerate(_QUICK_TAGS):
            with tag_cols[i]:
                if st.button(qt, key=f"qtag_{qt}"):
                    st.session_state["output_tag"] = qt
                    st.rerun()
        output_tag = st.text_input("Tag / Folder", key="output_tag",
                                   placeholder="e.g. portraits",
                                   label_visibility="collapsed")

        st.markdown("")
        do_gen = st.button("  RUN GENERATION  ", key="btn_gen", width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    # Generation execution with live progress
    # ══════════════════════════════════════════════════════════════════════════
    if do_gen:
        if not prompt.strip():
            st.error("Prompt is empty.")
            return

        # ── Validate outpaint before starting ─────────────────────────────────
        if pipe_type == "Outpaint":
            ok, msg = validate_outpaint_expansion(
                expand_left, expand_right, expand_top, expand_bottom
            )
            if not ok:
                st.error(msg)
                return

        # ── Logcat header ──────────────────────────────────────────────────────
        _ph_log, log = make_logcat(max_lines=25)
        log(f"pipeline: {pipe_type}", "accent")
        log(f"model: {selected_model}  steps: {steps}  seed: {seed}")
        log(f"accel: {', '.join(accel) or 'none'}")
        if lora_paths:
            log(f"loras: {[Path(p).name for p, _ in lora_paths]}")

        # ── Progress area ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Progress</div>', unsafe_allow_html=True)
        ph_bar = st.empty()

        # Step counter + ETA in einer Zeile
        sc_left, sc_mid, sc_right = st.columns([2, 3, 2], gap="small")
        with sc_left:
            ph_step = st.empty()  # "12 / 28"
        with sc_mid:
            ph_label = st.empty()  # "denoising · ETA 14s"
        with sc_right:
            ph_stage = st.empty()  # "elapsed: 8.3s"

        # Preview (Bild) | Noise-Meter (Video)
        pv_left, pv_right = st.columns([2, 1], gap="medium")
        with pv_left:
            ph_preview = st.empty()
        with pv_right:
            ph_noise = st.empty()  # Variante 3 — nur bei Video sichtbar

        try:
            from genbox import pipeline
            tmp_dir = __import__("pathlib").Path(
                __import__("tempfile").mkdtemp(prefix="genbox_")
            )

            # ── Save uploaded files to temp ─────────────────────────────────
            input_path = None
            mask_path  = None
            end_path   = None

            if uploaded:
                input_path = tmp_dir / uploaded.name
                input_path.write_bytes(uploaded.getvalue())

            if pipe_type == "Inpaint":
                mu = st.session_state.get("mask_img") or locals().get("mask_upload")
                if mu is not None:
                    mask_path = tmp_dir / mu.name
                    mask_path.write_bytes(mu.getvalue())

            if pipe_type == "Image → Video":
                ef = st.session_state.get("end_frame")
                if ef is not None:
                    end_path = tmp_dir / ef.name
                    end_path.write_bytes(ef.getvalue())

            # ── Build tracker + callback factory ────────────────────────────
            tracker = GenProgressTracker(total_steps=steps)
            preview_tmp = tmp_dir / "previews"

            def _decode_fn(latents, pipe_obj):
                return decode_latents_to_preview(latents, pipe_obj, preview_tmp, step=tracker.snapshot()["step"])

            def _make_cb(t):
                return make_step_callback(
                    t,
                    preview_interval=preview_interval if show_preview else 0,
                    decode_fn=_decode_fn if show_preview else None,
                    preview_dir=preview_tmp,
                )

            # ── Generation function (runs in thread) ─────────────────────────
            # ── Build tracker ────────────────────────────────────────────────
            tracker = GenProgressTracker(total_steps=steps)
            preview_tmp = tmp_dir / "previews"

            # ── Generation function (runs in thread) ─────────────────────────
            def gen_fn(t: GenProgressTracker):
                kwargs = dict(
                    prompt=prompt, model=selected_model,
                    steps=steps, guidance_scale=guidance,
                    seed=int(seed), loras=lora_paths,
                    accel=accel, sampler=selected_sampler,
                    output=custom_out.strip() or None,
                    tracker=t,
                    enable_preview=show_preview,
                    preview_interval=preview_interval if show_preview else 0,
                )

                if pipe_type in ("Text → Image", "Image → Image"):
                    kwargs["t5_mode"] = t5_mode

                if pipe_type == "Text → Image":
                    kwargs.update(width=w, height=h, negative_prompt=neg_prompt)
                    return pipeline.text_to_image(**kwargs)

                elif pipe_type == "Image → Image":
                    if not input_path:
                        raise ValueError("Input image required for Image → Image")
                    kwargs.update(input_image=input_path, strength=strength,
                                  width=w, height=h)
                    return pipeline.image_to_image(**kwargs)

                elif pipe_type == "Inpaint":
                    if not input_path:
                        raise ValueError("Input image required for Inpaint")
                    if not mask_path:
                        raise ValueError("Mask image required for Inpaint — upload a mask file")
                    kwargs.update(
                        input_image=input_path, mask_image=mask_path,
                        width=w, height=h, strength=strength,
                        blur_radius=blur_r, dilate_pixels=dilate,
                        mask_mode=mask_mode,
                    )
                    return pipeline.inpaint(**kwargs)

                elif pipe_type == "Outpaint":
                    if not input_path:
                        raise ValueError("Input image required for Outpaint")
                    kwargs.update(
                        input_image=input_path,
                        left=int(expand_left), right=int(expand_right),
                        top=int(expand_top), bottom=int(expand_bottom),
                        feather_radius=float(feather), strength=strength,
                    )
                    return pipeline.outpaint(**kwargs)

                elif pipe_type == "Text → Video":
                    kwargs.update(
                        width=w, height=h, frames=n_frames,
                        fps=vid_fps, negative_prompt=neg_prompt,
                        enable_noise_meter=kwargs.pop("enable_preview"),
                    )
                    kwargs.pop("preview_interval")
                    return pipeline.text_to_video(**kwargs)

                elif pipe_type == "Image → Video":
                    if not input_path:
                        raise ValueError("Start frame required for Image → Video")
                    kwargs.update(
                        start_frame=input_path, end_frame=end_path,
                        width=w, height=h, frames=n_frames,
                        fps=vid_fps, negative_prompt=neg_prompt,
                        enable_noise_meter=kwargs.pop("enable_preview"),
                    )
                    kwargs.pop("preview_interval")
                    return pipeline.image_to_video(**kwargs)

                raise ValueError(f"Unknown pipeline type: {pipe_type}")

            # ── Run in thread + poll progress ────────────────────────────────
            log("starting generation …", "accent")
            runner = GenRunner(fn=gen_fn, tracker=tracker)
            runner.start()

            def _render_step(snap, frac, eta):
                """Render step counter, progress bar, label, elapsed."""
                step = snap["step"]
                total = snap["total"]
                stage = snap["stage"]
                done = snap["done"]
                err = snap["error"]

                ph_bar.progress(min(frac, 1.0))

                # Step counter — large, accent color
                bar_color = "#2dd4a0" if done else "#ff4a4a" if err else "#4a9eff"
                ph_step.markdown(
                    f'<span style="font-size:20px;font-weight:500;'
                    f'color:{bar_color};letter-spacing:0.04em;">'
                    f'{step}&thinsp;/&thinsp;{total}</span>',
                    unsafe_allow_html=True,
                )

                # ETA label
                eta_str = (
                    f"{int(eta) // 60}m {int(eta) % 60}s" if eta and eta >= 60
                    else f"{int(eta)}s" if eta else "…"
                )
                ph_label.markdown(
                    f'<span style="font-size:11px;color:#6b6b6b;">'
                    f'{stage}&nbsp;&nbsp;·&nbsp;&nbsp;ETA {eta_str}</span>',
                    unsafe_allow_html=True,
                )

                # Elapsed
                ph_stage.markdown(
                    f'<span style="font-size:10px;color:#4a4a4a;">'
                    f'elapsed: {tracker.elapsed_seconds():.1f}s</span>',
                    unsafe_allow_html=True,
                )

            def _render_noise(snap):
                """
                Render latent std sparkline (Variante 3 — video only).
                Pure SVG, no external lib. Normalized polyline, descending curve.
                """
                history = snap.get("noise_std_history", [])
                if len(history) < 2:
                    return

                w_px, h_px = 160, 48
                pad = 4
                min_v = min(history)
                max_v = max(history)
                rng = max(max_v - min_v, 1e-6)

                n = len(history)
                pts = []
                for i, v in enumerate(history):
                    x = pad + (w_px - 2 * pad) * i / max(n - 1, 1)
                    y = h_px - pad - (h_px - 2 * pad) * (v - min_v) / rng
                    pts.append(f"{x:.1f},{y:.1f}")

                polyline = " ".join(pts)
                last_std = history[-1]

                svg = (
                    f'<svg viewBox="0 0 {w_px} {h_px}" '
                    f'xmlns="http://www.w3.org/2000/svg" '
                    f'style="width:100%;max-width:{w_px}px;display:block;">'
                    f'<rect width="{w_px}" height="{h_px}" fill="#0a0a0a" rx="2"/>'
                    f'<polyline points="{polyline}" '
                    f'fill="none" stroke="#4a9eff" stroke-width="1.5" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>'
                    f'<text x="{w_px - pad}" y="{h_px - pad}" '
                    f'text-anchor="end" fill="#6b6b6b" '
                    f'font-size="8" font-family="JetBrains Mono,monospace">'
                    f'σ {last_std:.3f}</text>'
                    f'</svg>'
                )
                ph_noise.markdown(
                    f'<div style="margin-top:4px;">'
                    f'<span style="font-size:9px;color:#3a3a3a;'
                    f'letter-spacing:0.1em;text-transform:uppercase;">'
                    f'noise σ</span>'
                    f'{svg}</div>',
                    unsafe_allow_html=True,
                )

            while runner.is_alive():
                snap = tracker.snapshot()
                frac = tracker.fraction()
                eta = tracker.eta_seconds()
                _render_step(snap, frac, eta)
                if show_preview and snap["preview_path"]:
                    p = Path(snap["preview_path"])
                    if p.exists():
                        try:
                            ph_preview.image(
                                str(p),
                                caption=f"Step {snap['step']} / {snap['total']}",
                                width='stretch',
                            )
                        except Exception:
                            pass
                if is_video:
                    _render_noise(snap)
                __import__("time").sleep(0.4)

            runner.join()

            # Final render
            snap = tracker.snapshot()
            frac = tracker.fraction()
            _render_step(snap, frac, None)
            if is_video:
                _render_noise(snap)
            render_progress(tracker, ph_bar, ph_label, ph_preview, show_preview)

            if runner.exception:
                raise runner.exception

            result: GenResult = runner.result

            # Final progress update
            snap  = tracker.snapshot()
            frac  = tracker.fraction()
            label = format_step_label(snap["step"], snap["total"], snap["stage"], None)
            ph_bar.progress(min(frac, 1.0))
            color = "#2dd4a0" if snap["done"] else "#ff4a4a"
            ph_label.markdown(
                f'<span class="progress-stage" style="color:{color};">{label}</span>',
                unsafe_allow_html=True,
            )

            if runner.exception:
                raise runner.exception

            result: GenResult = runner.result

            # ── Tag / move output ──────────────────────────────────────────────
            tag = (output_tag or "").strip()
            output_path = Path(result.output_path)
            metadata    = result.metadata
            elapsed     = result.elapsed_s

            if tag and cfg and output_path.exists():
                target_dir = cfg.outputs_dir / tag
                target_dir.mkdir(parents=True, exist_ok=True)
                new_path = target_dir / output_path.name
                shutil.move(str(output_path), str(new_path))
                orig_json = output_path.with_suffix(".json")
                if orig_json.exists():
                    shutil.move(str(orig_json), str(target_dir / orig_json.name))
                output_path = new_path

                if tag.lower().startswith('_'):
                    raw = output_path.read_bytes()
                    output_path.write_bytes(_xor_crypt(raw))

            log(f"done in {elapsed:.1f}s  →  {output_path.name}", "ok")
            shutil.rmtree(tmp_dir, ignore_errors=True)

            # ── Result display ─────────────────────────────────────────────────
            st.markdown("---")
            col_r1, col_r2 = st.columns([1, 1])
            with col_r1:
                if output_path.exists():
                    try:
                        raw = output_path.read_bytes()
                        if tag.lower().startswith("_") if tag else False:
                            raw = _xor_crypt(raw)
                        if output_path.suffix == ".png":
                            st.image(raw, width='stretch')
                        else:
                            st.video(raw)
                    except Exception as e:
                        st.error(f"Preview error: {e}")
            with col_r2:
                st.markdown("**Result**")
                st.markdown(
                    f'<div class="gen-card">'
                    f'<div class="prompt">{(prompt or "")[:80]}</div>'
                    f'<div class="meta">'
                    f'model: {selected_model}  ·  seed: {metadata.get("seed","?")}  ·  '
                    f'{elapsed:.1f}s'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Full metadata"):
                    st.json(metadata)

        except Exception as e:
            log(f"ERROR: {e}", "err")
            ph_bar.progress(0.0)
            ph_label.markdown(
                f'<span class="progress-stage" style="color:#ff4a4a;">error — {e}</span>',
                unsafe_allow_html=True,
            )
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# Screen 2 — Library
# ══════════════════════════════════════════════════════════════════════════════

def screen_library():
    st.markdown("## Library")
    if cfg is None:
        st.warning("genbox not configured. Run `genbox setup`.")
        return

    outputs = load_outputs(cfg.outputs_dir)
    tags = sorted(set(o.get("_tag", "default") for o in outputs))

    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])
    with col_f1: filter_type  = st.selectbox("Type",  ["all","image","video"], key="lib_type")
    with col_f2: filter_model = st.selectbox("Model", ["all"] + sorted(REGISTRY.keys()), key="lib_model")
    with col_f3: filter_tag   = st.selectbox("Tag",   ["all"] + [t for t in tags if t != "default"], key="lib_tag")
    with col_f4:
        st.markdown(""); st.markdown("")
        st.button("Refresh", key="lib_refresh")

    if filter_type != "all":
        outputs = [o for o in outputs if filter_type in o.get("pipeline", "")]
    if filter_model != "all":
        outputs = [o for o in outputs if o.get("model") == filter_model]
    if filter_tag == "all":
        outputs = [o for o in outputs if not o.get("_tag", "").lower().startswith('_')]
    else:
        outputs = [o for o in outputs if o.get("_tag") == filter_tag]

    st.markdown(f'<span style="font-size:11px;color:#6b6b6b;">{len(outputs)} items</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    if not outputs:
        st.markdown('<span style="color:#3a3a3a;">no outputs match the filters.</span>',
                    unsafe_allow_html=True)
        return

    COLS = 3
    for row_items in [outputs[i:i+COLS] for i in range(0, len(outputs), COLS)]:
        cols = st.columns(COLS, gap="small")
        for col, item in zip(cols, row_items):
            with col:
                fpath = Path(item.get("_file_path", ""))
                media_bytes = None
                if fpath.exists():
                    try:
                        media_bytes = fpath.read_bytes()
                        if item.get("_encrypted"):
                            media_bytes = _xor_crypt(media_bytes)
                    except Exception:
                        pass

                if media_bytes:
                    if fpath.suffix == ".png":
                        st.image(media_bytes, width='stretch')
                    else:
                        st.video(media_bytes)
                else:
                    st.markdown(
                        f'<div style="background:#141414;border:1px solid #2a2a2a;height:120px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'color:#3a3a3a;font-size:10px;">'
                        f'{fpath.name or "file missing"}</div>',
                        unsafe_allow_html=True,
                    )

                prompt_short = (item.get("prompt", "") or "")[:55]
                model_id = item.get("model", "?")
                seed     = item.get("seed", "?")
                elapsed  = item.get("elapsed_s", "")
                ts       = (item.get("timestamp", "") or "")[:10]
                st.markdown(
                    f'<div class="gen-card">'
                    f'<div class="prompt">{"🔒 " if item.get("_encrypted") else ""}'
                    f'{prompt_short}{"…" if len(item.get("prompt",""))>55 else ""}</div>'
                    f'<div class="meta">{model_id} · seed {seed} · {elapsed}s · {ts}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    uid = f"{item.get('seed','')}_{ts}_{fpath.name}"
                    if st.button("Remix", key=f"remix_{uid}"):
                        st.session_state["_remix_data"]    = build_remix_data(item)
                        st.session_state["_remix_pending"] = True
                        go_generate()
                        st.rerun()
                with c2:
                    if media_bytes:
                        st.download_button(
                            "Save", data=media_bytes,
                            file_name=fpath.name,
                            key=f"dl_{fpath.name}_{ts}",
                            mime="image/png" if fpath.suffix==".png" else "video/mp4",
                        )


# ══════════════════════════════════════════════════════════════════════════════
# Screen 3 — Models
# ══════════════════════════════════════════════════════════════════════════════

def screen_models():
    st.markdown("## Models")
    vram = cfg.vram_gb if cfg else 0
    col_a, col_b = st.columns([3, 1], gap="medium")

    with col_a:
        # ── Install defaults ───────────────────────────────────────────────────
        st.markdown("### Quick Install")
        profile_opts = ["8gb_low", "8gb_balanced", "12gb_balanced", "16gb_high", "24gb_ultra"]
        cur_profile  = cfg.vram_profile if cfg else "8gb_low"
        sel_profile  = st.selectbox("VRAM profile", profile_opts,
                                    index=profile_opts.index(cur_profile)
                                    if cur_profile in profile_opts else 0,
                                    key="install_profile")
        default_ids = get_install_defaults_for_profile(sel_profile)
        already = [mid for mid in default_ids if mid in REGISTRY and _is_installed(REGISTRY[mid])]
        todo    = [mid for mid in default_ids if mid in REGISTRY and not _is_installed(REGISTRY[mid])]

        if todo:
            st.markdown(
                f'<span style="font-size:11px;color:#6b6b6b;">'
                f'To install ({len(todo)}): {", ".join(todo[:6])}'
                f'{"…" if len(todo)>6 else ""}</span>',
                unsafe_allow_html=True,
            )
            c_dry, c_go = st.columns([1, 1])
            with c_dry:
                if st.button("Preview (dry run)", key="btn_dry_install"):
                    for mid in todo:
                        e = REGISTRY[mid]
                        st.markdown(f'<span style="font-size:11px;color:#4a9eff;">○ {mid}  '
                                    f'({e.vram_min_gb}GB  {e.quant})</span>', unsafe_allow_html=True)
            with c_go:
                if st.button("Install all", key="btn_install_defaults"):
                    if cfg is None:
                        st.error("Run genbox setup first.")
                    else:
                        log_ph = st.empty()
                        log_lines: list[str] = []
                        def _l(msg, kind=""):
                            log_lines.append(_logline(msg, kind))
                            log_ph.markdown(f'<div class="logcat">{"".join(log_lines[-20:])}</div>',
                                            unsafe_allow_html=True)
                        for mid in todo:
                            entry = REGISTRY[mid]
                            _l(f"downloading {entry.name} …", "accent")
                            try:
                                model_lib.download_model(entry)
                                _l(f"✓ {entry.name}", "ok")
                            except Exception as ex:
                                _l(f"✗ {entry.name}: {ex}", "err")
                        st.rerun()
        else:
            st.success(f"All {len(default_ids)} base models for {sel_profile} installed ✓")

        st.markdown("---")

        # ── Registry browser ───────────────────────────────────────────────────
        st.markdown("### Registry")
        filter_mtype = st.radio("Filter", ["all", "image", "video"],
                                horizontal=True, key="reg_type", label_visibility="collapsed")
        entries = list_registry(model_type=None if filter_mtype=="all" else filter_mtype)

        for i, e in enumerate(entries):
            vc       = _vram_color(e)
            compat   = "" if e.fits_vram(vram) else "  ⚠ exceeds VRAM"
            inst_dot = "● " if _is_installed(e) else "○ "
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in e.tags)

            with st.expander(f"{inst_dot}{e.name}  [{e.quant}]  {_stars(e.quality_stars)} Q  {_stars(e.speed_stars)} S{compat}"):
                st.markdown(
                    f'<span class="{vc}">{e.vram_min_gb}GB min</span>'
                    f'&nbsp;&nbsp;<span style="color:#6b6b6b;">{e.license}</span>'
                    f'<br><span style="color:#6b6b6b;font-size:10px;">{e.hf_repo}</span>'
                    f'<br><br>{tags_html}'
                    f'<br><span style="color:#4a4a4a;font-size:10px;">{e.notes}</span>',
                    unsafe_allow_html=True,
                )

                from genbox.models import _shared_config_dir
                is_gguf_e    = e.is_gguf() if hasattr(e, "is_gguf") else ("gguf" in e.quant)
                is_inst      = _is_installed(e)
                shared_dir   = _shared_config_dir(e)
                shared_ok    = bool(shared_dir and shared_dir.exists() and any(shared_dir.iterdir()))

                if is_gguf_e and cfg:
                    gguf_ok = (cfg.models_dir / e.architecture / Path(e.hf_filename).name).exists()
                else:
                    gguf_ok = is_inst

                if e.hf_repo == "" and (not is_gguf_e or shared_ok):
                    btn_label, btn_dis = "✓ Local Custom Ready", True
                elif e.hf_repo == "" and is_gguf_e and not shared_ok:
                    btn_label, btn_dis = "↓ Download Shared Config", False
                elif is_inst and (not is_gguf_e or shared_ok):
                    btn_label, btn_dis = f"✓ Installed", True
                elif gguf_ok and is_gguf_e and not shared_ok:
                    btn_label, btn_dis = "⚠ Repair: Download Shared Config", False
                else:
                    btn_label, btn_dis = f"↓ Download  {e.name}", False

                if is_gguf_e:
                    sz = {"gguf-q8": "~11.8GB", "gguf-q4": "~6GB"}.get(e.quant, "")
                    st.markdown(
                        f'<span style="font-size:10px;color:#6b6b6b;">'
                        f'GGUF: {sz}  ·  '
                        f'{"✓ shared cached" if shared_ok else "~9.5GB shared (T5+VAE, once)"}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

                if st.button(btn_label, key=f"dl_{e.id}_{i}", disabled=btn_dis):
                    if cfg is None:
                        st.error("Run genbox setup first.")
                    else:
                        log_ph = st.empty()
                        log_lines = []
                        def _l2(msg, kind=""):
                            log_lines.append(_logline(msg, kind))
                            log_ph.markdown(f'<div class="logcat">{"".join(log_lines)}</div>',
                                            unsafe_allow_html=True)
                        with st.spinner("…"):
                            try:
                                model_lib.download_model(e)
                                _l2(f"✓ {e.name} ready", "ok")
                                model_lib.reset_discovery()
                                st.rerun()
                            except Exception as ex:
                                _l2(str(ex), "err")

                # Healing: detect missing VAE weights in shared config
                if shared_ok and e.hf_pipeline_repo:
                    vae_dir = shared_dir / "vae"
                    vae_has_weights = vae_dir.exists() and any(
                        (vae_dir / f).exists()
                        for f in ("diffusion_pytorch_model.bin",
                                  "diffusion_pytorch_model.safetensors")
                    )
                    if not vae_has_weights:
                        st.markdown(
                            '<span style="font-size:10px;color:#ff6b35;">'
                            '⚠ VAE weights missing — generation will fail</span>',
                            unsafe_allow_html=True,
                        )
                        if st.button("🔧 Heal: download missing weights",
                                     key=f"heal_{e.id}_{i}"):
                            if cfg is None:
                                st.error("Run genbox setup first.")
                            else:
                                log_ph = st.empty()
                                log_lines: list[str] = []

                                def _lh(msg, kind=""):
                                    log_lines.append(_logline(msg, kind))
                                    log_ph.markdown(
                                        f'<div class="logcat">{"".join(log_lines)}</div>',
                                        unsafe_allow_html=True,
                                    )

                                with st.spinner("Healing …"):
                                    try:
                                        model_lib.heal_model(e)
                                        _lh("✓ Heal complete", "ok")
                                        st.rerun()
                                    except Exception as ex:
                                        _lh(str(ex), "err")

        # ── HF search ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Search HuggingFace")
        search_q = st.text_input("Query", placeholder="flux quantized / wan 2.2 / ltx-video …",
                                 key="hf_search", label_visibility="collapsed")
        if st.button("Search", key="btn_hf_search") and search_q:
            with st.spinner("Querying HuggingFace…"):
                results = model_lib._hf_search(search_q, limit=15)
            st.session_state["hf_results"] = results

        for r in st.session_state.get("hf_results", []):
            repo     = r.get("id", "")
            dls      = r.get("downloads", 0)
            tag_html = "".join(f'<span class="tag">{t}</span>' for t in r.get("tags", [])[:6])
            with st.expander(f"{repo}  ↓{dls:,}"):
                st.markdown(tag_html or '<span style="color:#3a3a3a;">no tags</span>',
                            unsafe_allow_html=True)
                dl_mode = st.radio("Mode", ["Full Repo", "Single File (GGUF/safetensors)"],
                                   key=f"dlm_{repo}", horizontal=True)
                if dl_mode == "Single File (GGUF/safetensors)":
                    c_fetch, c_sel, c_arch, c_base = st.columns([1, 2, 1, 2])
                    with c_fetch:
                        if st.button("Fetch files", key=f"fbtn_{repo}"):
                            try:
                                from huggingface_hub import list_repo_files
                                st.session_state[f"files_{repo}"] = [
                                    f for f in list_repo_files(repo)
                                    if f.endswith((".safetensors",".gguf",".bin",".pt"))
                                ]
                            except Exception as e:
                                st.error(f"Fetch failed: {e}")
                    repo_files = st.session_state.get(f"files_{repo}", [])
                    if repo_files:
                        with c_sel:
                            st.selectbox("File", repo_files, key=f"self_{repo}", label_visibility="collapsed")
                        with c_arch:
                            st.selectbox("Arch", ["flux","sd15","sdxl","sd35","ltx","wan"],
                                         key=f"arch_{repo}")
                        with c_base:
                            st.selectbox("Base (T5/VAE)",
                                         ["black-forest-labs/FLUX.1-schnell",
                                          "black-forest-labs/FLUX.2-klein-4B",
                                          "black-forest-labs/FLUX.2-klein-9B",
                                          "stabilityai/stable-diffusion-3.5-medium",
                                          "stabilityai/stable-diffusion-xl-base-1.0",
                                          "runwayml/stable-diffusion-v1-5",
                                          "Lightricks/LTX-Video",
                                          "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                                          "Wan-AI/Wan2.1-T2V-14B-Diffusers"],
                                         key=f"base_{repo}")

                if st.button("↓ Download", key=f"dl_hf_{repo.replace('/','__')}"):
                    if cfg is None:
                        st.error("Run genbox setup first.")
                    else:
                        log_ph = st.empty(); log_lines = []
                        def _hfl(msg, kind=""):
                            log_lines.append(_logline(msg, kind))
                            log_ph.markdown(f'<div class="logcat">{"".join(log_lines)}</div>', unsafe_allow_html=True)
                        try:
                            from huggingface_hub import snapshot_download, hf_hub_download
                            token = getattr(cfg,"hf_token",None) or None
                            if dl_mode == "Single File (GGUF/safetensors)":
                                sel_file    = st.session_state.get(f"self_{repo}")
                                target_arch = st.session_state.get(f"arch_{repo}", "flux")
                                base_repo   = st.session_state.get(f"base_{repo}")
                                if not sel_file:
                                    st.error("Fetch and select a file first.")
                                else:
                                    dest = cfg.models_dir / target_arch
                                    dest.mkdir(parents=True, exist_ok=True)
                                    _hfl(f"Downloading {sel_file} …", "accent")
                                    hf_hub_download(repo_id=repo, filename=sel_file,
                                                    local_dir=str(dest), local_dir_use_symlinks=False, token=token)
                                    _hfl(f"Saved ✓", "ok")
                                    if base_repo and sel_file.endswith((".gguf",".safetensors")):
                                        safe = base_repo.replace("/","--")
                                        cfg_dir = cfg.models_dir / target_arch / f"_shared_{safe}"
                                        cfg_dir.mkdir(parents=True, exist_ok=True)
                                        _hfl(f"Caching shared config: {base_repo} …", "accent")
                                        snapshot_download(repo_id=base_repo, local_dir=str(cfg_dir),
                                                          local_dir_use_symlinks=False, token=token,
                                                          ignore_patterns=["*.gguf",
                                                                           "transformer/*.safetensors",
                                                                           "transformer/*.bin",
                                                                           "*.pt", "*.ot", "*.msgpack", "*.h5",
                                                                           "flax_*"])
                                        _hfl("Shared config cached ✓", "ok")
                                    model_lib.reset_discovery()
                                    st.success("Done!")
                            else:
                                safe = repo.replace("/","--")
                                dest = cfg.models_dir / "flux" / safe
                                dest.mkdir(parents=True, exist_ok=True)
                                _hfl(f"Downloading full repo {repo} …", "accent")
                                snapshot_download(repo_id=repo, local_dir=str(dest),
                                                  local_dir_use_symlinks=False, token=token,
                                                  ignore_patterns=["*.msgpack","*.h5","flax_*"])
                                _hfl(f"Saved → {dest}", "ok")
                                model_lib.reset_discovery()
                                st.success("Done!")
                        except Exception as ex:
                            st.error(str(ex))

        # ── Drag+Drop custom file zone ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Add Custom File")
        st.markdown(
            '<div class="drop-zone">Drop .gguf · .safetensors · LoRA here (or use uploader below)</div>',
            unsafe_allow_html=True,
        )
        custom_upload = st.file_uploader(
            "Custom file", type=["gguf","safetensors"],
            key="custom_file_upload", label_visibility="collapsed",
        )
        if custom_upload:
            detected = detect_upload_type(custom_upload.name)
            guessed_arch = guess_arch_from_filename(custom_upload.name)

            c_type, c_arch2 = st.columns([1, 1])
            with c_type:
                file_type = st.selectbox("File type",
                                         ["model", "lora", "gguf"],
                                         index=["model","lora","gguf"].index(
                                             "gguf" if detected=="gguf" else
                                             "lora" if detected=="lora" else "model"),
                                         key="custom_type")
            with c_arch2:
                file_arch = st.selectbox("Architecture",
                                         ["flux","sd15","sdxl","sd35","ltx","wan"],
                                         index=["flux","sd15","sdxl","sd35","ltx","wan"].index(
                                             guessed_arch if guessed_arch in ["flux","sd15","sdxl","sd35","ltx","wan"] else "flux"),
                                         key="custom_arch")

            file_desc    = st.text_input("Description (optional)", key="custom_desc")
            file_preview = st.text_input("Preview URL (optional)", key="custom_preview_url")

            if file_type == "lora":
                file_trigger = st.text_input("Trigger word(s)", key="custom_trigger",
                                             placeholder="e.g. cinematic style")
            else:
                file_trigger = ""

            if st.button("Register file", key="btn_register_custom"):
                if cfg is None:
                    st.error("Run genbox setup first.")
                else:
                    try:
                        tmp = Path(tempfile.mkdtemp(prefix="genbox_cu_"))
                        src = tmp / custom_upload.name
                        src.write_bytes(custom_upload.getvalue())

                        if file_type == "lora":
                            dest_dir = cfg.loras_dir / file_arch
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest = dest_dir / custom_upload.name
                            shutil.copy2(src, dest)
                            from genbox.utils import write_lora_metadata
                            write_lora_metadata(
                                dest, architecture=file_arch,
                                trigger=file_trigger,
                                description=file_desc,
                                preview_url=file_preview,
                            )
                            st.success(f"LoRA registered: {dest}")
                        else:
                            from genbox.models import register_custom_model
                            result = register_custom_model(
                                src, architecture=file_arch,
                                description=file_desc, preview_url=file_preview,
                            )
                            st.success(f"Model registered: {result['id']}")
                            model_lib.reset_discovery()
                        shutil.rmtree(tmp, ignore_errors=True)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    with col_b:
        # ── Installed models ───────────────────────────────────────────────────
        st.markdown("### Installed")
        local = list_local()
        if not local:
            st.markdown('<span style="color:#3a3a3a;font-size:11px;">none</span>',
                        unsafe_allow_html=True)
        else:
            for i_, m in enumerate(local):
                m_id = m.get("id") or next(
                    (e.id for e in REGISTRY.values() if e.name == m["name"]), None)
                if not m_id:
                    continue
                c_info, c_act = st.columns([5, 1])
                with c_info:
                    st.markdown(
                        f'<div class="gen-card" style="margin-bottom:4px;">'
                        f'<div class="prompt">{m["name"]}</div>'
                        f'<div class="meta">{m["architecture"]} · {m.get("quant","fp16")} · {m["size_gb"]:.1f}GB</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_act:
                    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"del_{m_id}_{i_}", help="Uninstall"):
                        st.session_state[f"confirm_{m_id}"] = not st.session_state.get(f"confirm_{m_id}", False)
                        st.rerun()

                if st.session_state.get(f"confirm_{m_id}"):
                    with st.container():
                        st.warning(f"Uninstall **{m['name']}**?")
                        cy, cn = st.columns(2)
                        if cy.button("Yes", key=f"yes_{m_id}"):
                            model_lib.uninstall_model(m_id)
                            st.session_state.pop(f"confirm_{m_id}", None)
                            st.rerun()
                        if cn.button("Cancel", key=f"no_{m_id}"):
                            st.session_state.pop(f"confirm_{m_id}", None)
                            st.rerun()

        st.markdown("---")

        # ── LoRAs ──────────────────────────────────────────────────────────────
        st.markdown("### LoRAs")
        loras = list_loras()
        if not loras:
            st.markdown(
                f'<div class="lora-path-hint">Place .safetensors in<br>'
                f'<span style="color:#4a9eff;">{cfg.loras_dir if cfg else "loras/"}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            for j, lo in enumerate(loras):
                lora_path = Path(lo["path"])
                # Preview image from metadata
                preview_url = lo.get("preview_url", "")
                desc        = lo.get("description", "")

                with st.expander(f"{lo['name']}  [{lo['architecture']}]  {lo['size_mb']:.0f}MB"):
                    if preview_url:
                        if preview_url.startswith("http"):
                            st.markdown(
                                f'<img src="{preview_url}" style="max-width:100%;border:1px solid #2a2a2a;border-radius:2px;margin-bottom:8px;">',
                                unsafe_allow_html=True,
                            )
                        elif Path(preview_url).exists():
                            st.image(preview_url, width='stretch')

                    if desc:
                        st.markdown(f'<span style="font-size:10px;color:#6b6b6b;">{desc}</span>',
                                    unsafe_allow_html=True)
                    if lo.get("trigger"):
                        st.markdown(
                            f'<span class="lora-badge">trigger: {lo["trigger"]}</span>',
                            unsafe_allow_html=True,
                        )

                    # Edit metadata inline
                    with st.expander("Edit metadata"):
                        new_arch    = st.selectbox("Architecture",
                                                    ["flux","sd15","sdxl","sd35","ltx","wan","any"],
                                                    index=["flux","sd15","sdxl","sd35","ltx","wan","any"].index(
                                                        lo["architecture"] if lo["architecture"] in
                                                        ["flux","sd15","sdxl","sd35","ltx","wan","any"] else "any"),
                                                    key=f"lo_arch_{j}")
                        new_trigger = st.text_input("Trigger", value=lo.get("trigger",""),
                                                     key=f"lo_trig_{j}")
                        new_desc    = st.text_input("Description", value=lo.get("description",""),
                                                     key=f"lo_desc_{j}")
                        new_prev    = st.text_input("Preview URL", value=lo.get("preview_url",""),
                                                     key=f"lo_prev_{j}")
                        if st.button("Save metadata", key=f"lo_save_{j}"):
                            try:
                                from genbox.utils import write_lora_metadata
                                write_lora_metadata(
                                    lora_path, architecture=new_arch,
                                    trigger=new_trigger, description=new_desc,
                                    preview_url=new_prev,
                                )
                                st.success("Metadata saved ✓")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    # Delete LoRA
                    if st.button("🗑️ Delete LoRA", key=f"lo_del_{j}"):
                        st.session_state[f"lo_confirm_{j}"] = True
                        st.rerun()
                    if st.session_state.get(f"lo_confirm_{j}"):
                        st.warning(f"Delete **{lo['name']}**?")
                        c_ly, c_ln = st.columns(2)
                        if c_ly.button("Yes, delete", key=f"lo_yes_{j}"):
                            try:
                                lora_path.unlink(missing_ok=True)
                                sidecar = lora_path.with_suffix(".json")
                                if sidecar.exists(): sidecar.unlink()
                                st.session_state.pop(f"lo_confirm_{j}", None)
                                st.success("Deleted")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                        if c_ln.button("Cancel", key=f"lo_no_{j}"):
                            st.session_state.pop(f"lo_confirm_{j}", None)
                            st.rerun()

        st.markdown("---")
        st.markdown("### VRAM")
        st.metric("Available", f"{vram} GB")
        st.metric("Profile", cfg.vram_profile if cfg else "—")
        compatible = sum(1 for e in REGISTRY.values() if e.fits_vram(vram))
        st.metric("Compatible models", compatible)


# ══════════════════════════════════════════════════════════════════════════════
# Screen 4 — Pipeline
# ══════════════════════════════════════════════════════════════════════════════

_TEMPLATES = {
    "Text → Image (FLUX.2 Klein)": """\
result = pipeline.text_to_image(
    prompt="a brutalist tower at dusk, fog, cinematic",
    model="flux2_klein",
    steps=28, seed=42,
    accel=["sageAttn", "teacache"],
)
print(result)
""",
    "Text → Image (SDXL)": """\
result = pipeline.text_to_image(
    prompt="portrait of a samurai, volumetric lighting, 8k",
    model="sdxl_base",
    steps=30, guidance_scale=7.5, seed=42,
)
print(result)
""",
    "Text → Image (Pony)": """\
result = pipeline.text_to_image(
    prompt="a cat on a surfboard",
    model="pony_xl",
    steps=30, guidance_scale=7.0, seed=42,
    # quality_tags prepended automatically by pipeline_pony
)
print(result)
""",
    "Image → Image": """\
result = pipeline.image_to_image(
    prompt="transform into oil painting",
    input_image="/path/to/input.png",
    model="flux2_klein",
    strength=0.75, seed=42,
)
print(result)
""",
    "Inpaint": """\
result = pipeline.inpaint(
    prompt="replace with foggy mountain",
    input_image="/path/to/input.png",
    mask_image="/path/to/mask.png",   # white = fill, black = keep
    model="sdxl_base",
    blur_radius=5, dilate_pixels=8,
    seed=42,
)
print(result)
""",
    "Outpaint": """\
result = pipeline.outpaint(
    prompt="extend the scene with a forest",
    input_image="/path/to/input.png",
    model="sdxl_base",
    right=256, bottom=128,
    feather_radius=20,
    seed=42,
)
print(result)
""",
    "Text → Video (WAN 1.3B)": """\
result = pipeline.text_to_video(
    prompt="slow drone shot over a midnight city",
    model="wan_1_3b",
    frames=81, fps=16, seed=42,
    accel=["sageAttn"],
)
print(result)
""",
    "Text → Video (LTX distilled)": """\
result = pipeline.text_to_video(
    prompt="cinematic pullback through a forest",
    model="ltx23_fp8",   # 0.9.7 distilled 13B
    frames=97, fps=24, seed=42,
    # guidance_scale=1.0 and steps=8 set automatically
)
print(result)
""",
    "Image → Video": """\
result = pipeline.image_to_video(
    prompt="camera slowly pulls back",
    start_frame="/path/to/start.png",
    model="ltx2_fp8",
    frames=97, fps=24, seed=42,
)
print(result)
""",
    "Batch variations": """\
for seed in [42, 1337, 99999, 7]:
    r = pipeline.text_to_image(
        prompt="abstract geometric form in fog",
        model="flux2_klein",
        seed=seed, steps=20,
    )
    print(f"seed {seed} →", r.output_path.name)
""",
    "Remix from sidecar": """\
import json
from pathlib import Path

meta = json.loads(Path("/path/to/output.json").read_text())
params = {k: v for k, v in meta.items()
          if k not in ("timestamp","elapsed_s","output_path","loras","accel")}
params["seed"] = 9999   # change seed
result = pipeline.text_to_image(**params)
print(result)
""",
}


def screen_pipeline():
    st.markdown("## Pipeline")

    if st.session_state.pop("_pipe_clear_pending", False):
        st.session_state.pop("pipeline_code", None)
        st.session_state.pop("_active_tpl", None)

    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        st.markdown("### Editor")

        template = st.selectbox(
            "Template", ["(blank)"] + list(_TEMPLATES.keys()), key="tpl_sel",
        )

        default_code = _TEMPLATES.get(template, "") if template != "(blank)" else ""

        if st.session_state.get("_active_tpl") != template:
            st.session_state["_active_tpl"] = template
            st.session_state["pipeline_code"] = default_code

        if "pipeline_code" not in st.session_state:
            st.session_state["pipeline_code"] = default_code

        code = st.text_area(
            "Pipeline code",
            height=380,
            key="pipeline_code",
            label_visibility="collapsed",
            placeholder="from genbox import pipeline\n\nresult = pipeline.text_to_image(...)\nprint(result)",
        )

        c1, c2, _ = st.columns([1, 1, 2])
        with c1: run   = st.button("RUN",   key="btn_run_pipe",   width='stretch')
        with c2: clear = st.button("Clear", key="btn_clear_pipe", width='stretch')
        if clear:
            st.session_state["_pipe_clear_pending"] = True
            st.rerun()

        if run:
            if not code.strip():
                st.error("Nothing to run.")
            else:
                with st.spinner("Running…"):
                    stdout, stderr = _run_pipeline_code(code)
                st.session_state["pipe_stdout"] = stdout
                st.session_state["pipe_stderr"] = stderr

    with col_r:
        st.markdown("### Output")
        stdout = st.session_state.get("pipe_stdout", "")
        stderr = st.session_state.get("pipe_stderr", "")

        if stdout or stderr:
            log_html = ""
            for line in stdout.splitlines():
                kind = "ok" if "done" in line.lower() or "GenResult" in line else ""
                log_html += _logline(line, kind)
            for line in stderr.splitlines():
                log_html += _logline(line, "err")
            st.markdown(f'<div class="logcat" style="max-height:420px">{log_html}</div>',
                        unsafe_allow_html=True)

            for line in stdout.splitlines():
                for ext in [".png", ".mp4", ".jpg"]:
                    if ext in line:
                        for part in line.split():
                            p = Path(part.strip("→ "))
                            if p.exists():
                                if ext == ".png": st.image(str(p))
                                else:             st.video(str(p))
                                break
        else:
            st.markdown('<span style="color:#3a3a3a;font-size:11px;">output appears here after RUN</span>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### API reference")
        for sig, ret in [
            ("pipeline.text_to_image(prompt, model, steps, seed, loras, accel)",  "→ GenResult"),
            ("pipeline.image_to_image(prompt, input_image, strength, …)",          "→ GenResult"),
            ("pipeline.inpaint(prompt, input_image, mask_image, blur_radius, …)",  "→ GenResult"),
            ("pipeline.outpaint(prompt, input_image, left, right, top, bottom, …)","→ GenResult"),
            ("pipeline.text_to_video(prompt, frames, fps, …)",                     "→ GenResult"),
            ("pipeline.image_to_video(prompt, start_frame, …)",                    "→ GenResult"),
            ("result.save(dest=None)",                                              "→ Path"),
            ("result.remix(**overrides)",                                           "→ dict"),
        ]:
            st.markdown(
                f'<code style="display:block;margin-bottom:4px;">{sig}</code>'
                f'<span style="color:#2dd4a0;font-size:10px;margin-left:8px;">{ret}</span>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if cfg is None:
        st.error("genbox is not configured. Run `genbox setup` in a terminal first.")
        st.code("pip install -e .[image,video,ui]\ngenbox setup", language="bash")
        return

    screen = _sidebar()
    if screen == "Generate":  screen_generate()
    elif screen == "Library": screen_library()
    elif screen == "Models":  screen_models()
    elif screen == "Pipeline":screen_pipeline()


if __name__ == "__main__":
    main()
