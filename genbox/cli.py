"""
genbox/cli.py
Command-line interface.  Entrypoint: genbox <command> [options]

Commands
  setup          first-run wizard
  info           config, installed models, accelerators
  gen            generate image or video (all modes)
  models         list / local / search / download / install-defaults / uninstall
  loras          list / tag LoRA weights
  ui             launch Streamlit UI
  run            run a pipeline Python script
  test           run test suite

Cross-OS: pure Python, no shell-isms, pathlib throughout.
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
import io as _io
if sys.stdout.encoding and "cp" in sys.stdout.encoding.lower():
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── ANSI colours ─────────────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()

def _c(text, code): return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"
def grey(t):   return _c(t, "90")
def white(t):  return _c(t, "97")
def blue(t):   return _c(t, "94")
def cyan(t):   return _c(t, "96")
def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def dim(t):    return _c(t, "2")
def yellow(t): return _c(t, "93")

def _sep():        print(grey("─" * 52))
def _ok(msg):      print(f"  {green('✓')}  {msg}")
def _err(msg):     print(f"  {red('✗')}  {msg}", file=sys.stderr)
def _info(msg):    print(f"  {grey('·')}  {grey(msg)}")
def _warn(msg):    print(f"  {yellow('!')}  {yellow(msg)}")

def _header(title):
    print()
    print(blue(f"  {title}"))
    _sep()

def _row(label, value, value_color=None):
    vc = value_color or cyan
    pad = max(1, 28 - len(label))
    print(f"  {white(label)}" + " " * pad + vc(str(value)))


# ── Guards ────────────────────────────────────────────────────────────────────

def _require_config():
    from genbox.config import cfg
    if cfg is None:
        _err("genbox is not configured.")
        print(f"  Run: {blue('genbox setup')}\n")
        sys.exit(1)
    return cfg


def _require_pipeline():
    """Import genbox.pipeline — also validates core deps installed."""
    try:
        import genbox.pipeline as pipeline
        return pipeline
    except ImportError as e:
        _err(f"Pipeline dependencies not installed: {e}")
        print(f"  Run: {blue('pip install torch diffusers transformers accelerate peft')}\n")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# setup
# ══════════════════════════════════════════════════════════════════════════════

def cmd_setup(args):
    from genbox.config import cfg as current_cfg, run_setup
    if current_cfg is not None and not getattr(args, "force", False):
        print()
        _ok(f"Already configured at {current_cfg.home}")
        _row("VRAM",    f"{current_cfg.vram_gb} GB")
        _row("Profile", current_cfg.vram_profile)
        print()
        print(f"  Use {blue('genbox setup --force')} to reconfigure.\n")
        return
    run_setup(force=getattr(args, "force", False))


# ══════════════════════════════════════════════════════════════════════════════
# info
# ══════════════════════════════════════════════════════════════════════════════

def cmd_info(args):
    _header("genbox info")
    cfg = _require_config()

    _row("Home",        str(cfg.home))
    _row("VRAM",        f"{cfg.vram_gb} GB")
    _row("Profile",     cfg.vram_profile)
    _row("Image model", cfg.default_image_model)
    _row("Video model", cfg.default_video_model)
    _row("Accels",      ", ".join(cfg.active_accels) or "none")

    print()
    _header("Accelerators")
    for name, pkg in [
        ("SageAttention2++", "sageattention"),
        ("TeaCache",         "teacache"),
    ]:
        try:
            __import__(pkg)
            _ok(f"{name}")
        except ImportError:
            _info(f"{name} not installed  (pip install {pkg})")
    try:
        import torch
        _ok(f"torch {torch.__version__}  {dim('(torch.compile available)')}")
    except ImportError:
        _info("torch not installed")

    print()
    _header("Installed models")
    from genbox.models import list_local
    local = list_local()
    if local:
        for m in local:
            _row(f"{m['name'][:32]}", f"{m['architecture']:8s}  {m['size_gb']:.1f} GB")
    else:
        _info("none — run: genbox models install-defaults")

    print()
    _header("LoRAs")
    from genbox.models import list_loras
    loras = list_loras()
    if loras:
        for lo in loras:
            t = f"  trigger: {lo['trigger']}" if lo.get("trigger") else ""
            d = f"  {dim(lo['description'][:40])}" if lo.get("description") else ""
            _row(lo["name"][:32], f"{lo['architecture']:8s}  {lo['size_mb']:.0f} MB{t}{d}")
    else:
        _info(f"none — place .safetensors in {cfg.loras_dir}")

    print()
    _header("Registry summary")
    from genbox.models import REGISTRY
    img = sum(1 for e in REGISTRY.values() if e.type == "image")
    vid = sum(1 for e in REGISTRY.values() if e.type == "video")
    _row("Image models", str(img))
    _row("Video models", str(vid))
    _row("Total", str(len(REGISTRY)))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# gen  (all modes)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_gen(args):
    cfg = _require_config()
    pipeline = _require_pipeline()

    accel = args.accel or cfg.active_accels
    if accel == ["none"]:
        accel = []

    loras  = args.lora or []
    output = Path(args.output) if getattr(args, "output", None) else None
    mode   = getattr(args, "mode", "t2i")

    _header(f"Generation  [{mode}]")
    _row("Prompt",  args.prompt[:58] + ("…" if len(args.prompt) > 58 else ""))
    _row("Model",   args.model or "(config default)")
    _row("Mode",    mode)
    _row("Seed",    str(args.seed))
    _row("Accel",   ", ".join(accel) or "none")
    if loras:
        _row("LoRAs", ", ".join(Path(p).name for p in loras))
    print()

    common = dict(
        model           = args.model or None,
        negative_prompt = getattr(args, "negative_prompt", "") or "",
        steps           = args.steps,
        guidance_scale  = args.guidance,
        seed            = args.seed,
        loras           = loras,
        accel           = accel,
        output          = output,
    )

    try:
        # ── IMAGE modes ───────────────────────────────────────────────────────
        if mode == "t2i":
            result = pipeline.text_to_image(
                prompt=args.prompt, width=args.width, height=args.height, **common
            )

        elif mode == "i2i":
            if not getattr(args, "input", None):
                _err("--input required for i2i mode"); sys.exit(1)
            result = pipeline.image_to_image(
                prompt=args.prompt,
                input_image=Path(args.input),
                strength=getattr(args, "strength", 0.75),
                width=args.width, height=args.height,
                **common,
            )

        elif mode == "inpaint":
            if not getattr(args, "input", None):
                _err("--input required for inpaint mode"); sys.exit(1)
            if not getattr(args, "mask", None):
                _err("--mask required for inpaint mode"); sys.exit(1)
            result = pipeline.inpaint(
                prompt=args.prompt,
                input_image=Path(args.input),
                mask_image=Path(args.mask),
                width=args.width, height=args.height,
                strength=getattr(args, "strength", 0.99),
                blur_radius=getattr(args, "blur_radius", 0),
                dilate_pixels=getattr(args, "dilate", 0),
                mask_mode=getattr(args, "mask_mode", "white_inpaint"),
                **common,
            )

        elif mode == "outpaint":
            if not getattr(args, "input", None):
                _err("--input required for outpaint mode"); sys.exit(1)
            total = (getattr(args, "expand_left", 0) + getattr(args, "expand_right", 0)
                     + getattr(args, "expand_top", 0) + getattr(args, "expand_bottom", 0))
            if total == 0:
                _err("At least one --expand-left/right/top/bottom must be > 0")
                sys.exit(1)
            result = pipeline.outpaint(
                prompt=args.prompt,
                input_image=Path(args.input),
                left=getattr(args, "expand_left", 0),
                right=getattr(args, "expand_right", 0),
                top=getattr(args, "expand_top", 0),
                bottom=getattr(args, "expand_bottom", 0),
                feather_radius=getattr(args, "feather_radius", 16.0),
                strength=getattr(args, "strength", 0.99),
                **common,
            )

        # ── VIDEO modes ───────────────────────────────────────────────────────
        elif mode == "t2v":
            result = pipeline.text_to_video(
                prompt=args.prompt,
                frames=getattr(args, "frames", 0),
                fps=getattr(args, "fps", 0),
                **common,
            )

        elif mode == "i2v":
            if not getattr(args, "input", None):
                _err("--input required for i2v mode"); sys.exit(1)
            result = pipeline.image_to_video(
                prompt=args.prompt,
                start_frame=Path(args.input),
                end_frame=Path(args.end_frame) if getattr(args, "end_frame", None) else None,
                frames=getattr(args, "frames", 0),
                fps=getattr(args, "fps", 0),
                **common,
            )

        else:
            _err(f"Unknown mode: {mode!r}")
            sys.exit(1)

        _ok(f"Done in {result.elapsed_s:.1f}s")
        _row("Output", str(result.output_path), green)
        _row("Seed",   str(result.metadata.get("seed", "?")))
        print()
        hint = (f"genbox gen -p \"{args.prompt[:38]}\" "
                f"--mode {mode} "
                f"--seed {result.metadata.get('seed', 0)}")
        print(f"  {dim('Remix:  ' + hint)}")
        print()

    except Exception as e:
        _err(str(e))
        if getattr(args, "verbose", False):
            import traceback; traceback.print_exc()
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# models
# ══════════════════════════════════════════════════════════════════════════════

def cmd_models(args):
    from genbox import models as mlib
    action = args.action or "list"

    if action == "list":
        cfg = _require_config()
        _header("Model registry")
        entries = mlib.list_registry(
            model_type=args.type or None,
            max_vram=None if getattr(args, "all", False) else cfg.vram_gb,
        )
        if not entries:
            _info("No models match current filter.")
            return
        for e in entries:
            compat  = green("✓") if e.fits_vram(cfg.vram_gb) else red("✗")
            inst    = cyan("●") if mlib._is_installed_entry(e) else grey("○")
            q = "★" * e.quality_stars + "☆" * (5 - e.quality_stars)
            s = "★" * e.speed_stars  + "☆" * (5 - e.speed_stars)
            print(
                f"  {compat} {inst} {white(e.id):38s}"
                f"  {cyan(str(e.vram_min_gb)+'GB'):7s}"
                f"  {grey(e.quant):12s}"
                f"  Q:{q}  S:{s}"
                f"  {dim(e.license)}"
            )
        print()
        installed_count = sum(1 for e in entries if mlib._is_installed_entry(e))
        _info(f"{len(entries)} model(s)  ·  {installed_count} installed  "
              f"·  ● installed  ○ not installed  ·  --all to show VRAM-incompatible")

    elif action == "local":
        _header("Installed models")
        local = mlib.list_local(model_type=args.type or None)
        if not local:
            _info("None installed.  Run: genbox models install-defaults")
            return
        for m in local:
            _row(m["name"][:32], f"{m['architecture']:8s}  {m['size_gb']:.1f} GB")
        print()
        _info(f"{len(local)} model(s) installed")

    elif action == "search":
        if not args.query:
            _err("Provide query:  genbox models search <query>")
            sys.exit(1)
        query = " ".join(args.query)
        _header(f"HuggingFace: {query!r}")
        results = mlib._hf_search(query, limit=getattr(args, "limit", 15))
        if not results:
            _info("No results.")
            return
        for r in results:
            repo = r.get("id", "")
            dls  = r.get("downloads", 0)
            tags = " ".join(r.get("tags", [])[:4])
            print(f"  {cyan(repo):55s}  {grey('↓')}{white(f'{dls:>8,}'):12s}  {dim(tags)}")
        print()
        _info(f"{len(results)} results  ·  --limit N for more")

    elif action == "download":
        cfg = _require_config()
        model_id = getattr(args, "model_id", None) or (args.query[0] if args.query else None)
        if not model_id:
            _err("Specify model ID:  genbox models download <model_id>")
            sys.exit(1)
        try:
            entry = mlib.get(model_id)
        except KeyError:
            _err(f"Unknown model: {model_id!r}")
            _info(f"Run: genbox models list  to see available models")
            sys.exit(1)

        _header(f"Download: {entry.name}")
        _row("Repo",    entry.hf_repo)
        _row("File",    entry.hf_filename)
        _row("VRAM",    f"{entry.vram_min_gb} GB min")
        _row("License", entry.license)
        print()

        if not entry.fits_vram(cfg.vram_gb) and not getattr(args, "force", False):
            _err(f"Requires {entry.vram_min_gb}GB, you have {cfg.vram_gb}GB.")
            print(f"  Pass {blue('--force')} to download anyway.\n")
            sys.exit(1)

        try:
            mlib.download_model(entry)
            _ok(f"Done: {entry.name}")
        except Exception as e:
            _err(f"Download failed: {e}")
            sys.exit(1)

    elif action == "install-defaults":
        cfg = _require_config()
        profile = getattr(args, "profile", None) or cfg.vram_profile
        dry_run = getattr(args, "dry_run", False)

        _header(f"Install default models  [profile: {profile}]")
        from genbox.models import get_default_models, _is_installed_entry, REGISTRY
        ids = get_default_models(profile)
        _info(f"Base set for {profile}: {ids}")
        print()

        if dry_run:
            _info("Dry run — no files will be downloaded:")
            for mid in ids:
                if mid in REGISTRY:
                    e = REGISTRY[mid]
                    installed = "●" if _is_installed_entry(e) else "○"
                    print(f"  {installed} {white(mid):38s}  {cyan(e.hf_repo)}")
            return

        installed = mlib.install_defaults(profile=profile)
        if installed:
            _ok(f"Installed: {installed}")
        else:
            _ok("All base models already installed.")

    elif action == "uninstall":
        model_id = getattr(args, "model_id", None) or (args.query[0] if args.query else None)
        if not model_id:
            _err("Specify model ID:  genbox models uninstall <model_id>")
            sys.exit(1)
        _header(f"Uninstall: {model_id}")
        if mlib.uninstall_model(model_id):
            _ok(f"Uninstalled: {model_id}")
        else:
            _err(f"Failed to uninstall: {model_id}")
            sys.exit(1)

    else:
        _err(f"Unknown action: {action!r}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# loras
# ══════════════════════════════════════════════════════════════════════════════

def cmd_loras(args):
    cfg = _require_config()
    action = getattr(args, "action", "list") or "list"

    if action == "list":
        _header("LoRA library")
        from genbox.models import list_loras
        loras = list_loras(architecture=getattr(args, "arch", None) or None)
        if not loras:
            _info(f"No LoRAs in {cfg.loras_dir}")
            _info("Place .safetensors files there, optionally with a .json sidecar for metadata.")
            print()
            return
        for lo in loras:
            t = f"  trigger: {lo['trigger']}" if lo.get("trigger") else ""
            d = f"  {dim(lo.get('description', '')[:40])}" if lo.get("description") else ""
            p = f"  {dim(lo.get('preview_url', ''))}" if lo.get("preview_url") else ""
            _row(lo["name"][:32], f"{lo['architecture']:8s}  {lo['size_mb']:.0f} MB{t}{d}{p}")
        print()
        _info(f"{len(loras)} LoRA(s)  ·  --arch flux|sd15|sdxl|sd35|ltx|wan to filter")

    elif action == "tag":
        # genbox loras tag <path> --arch flux --trigger "my_style" --desc "…" --preview url
        lora_path_str = (args.query[0] if args.query else None)
        if not lora_path_str:
            _err("Specify LoRA file path:  genbox loras tag <path.safetensors> --arch <arch>")
            sys.exit(1)
        lora_path = Path(lora_path_str)
        if not lora_path.exists():
            _err(f"File not found: {lora_path}")
            sys.exit(1)
        arch = getattr(args, "arch", None)
        if not arch:
            _err("--arch required for tag")
            sys.exit(1)
        from genbox.models import write_lora_metadata
        write_lora_metadata(
            lora_path,
            architecture=arch,
            trigger=getattr(args, "trigger", "") or "",
            description=getattr(args, "desc", "") or "",
            preview_url=getattr(args, "preview", "") or "",
        )
        _ok(f"Metadata written for {lora_path.name}")
    else:
        _err(f"Unknown loras action: {action!r}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ui
# ══════════════════════════════════════════════════════════════════════════════

def cmd_ui(args):
    cfg = _require_config()
    try:
        import streamlit  # noqa
    except ImportError:
        _info("Installing streamlit…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "--quiet"])

    import subprocess
    ui_path = Path(__file__).parent / "genbox_ui" / "ui.py"
    port = str(getattr(args, "port", None) or cfg.get("ui", "port", default=8501))
    env = os.environ.copy()
    env["GENBOX_HOME"] = str(cfg.home)
    print()
    _ok(f"UI → {blue('http://localhost:' + port)}")
    _info("Ctrl+C to stop.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path),
         "--server.port", port,
         "--server.headless", "true",
         "--server.maxUploadSize", "9999999",
         "--browser.gatherUsageStats", "false"],
        env=env,
    )


# ══════════════════════════════════════════════════════════════════════════════
# run
# ══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    _require_config()
    script = Path(args.script)
    if not script.exists():
        _err(f"Script not found: {script}")
        sys.exit(1)
    _header(f"Run: {script.name}")
    root = Path(__file__).parent.parent
    import subprocess
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{pp}" if pp else str(root)
    ret = subprocess.run([sys.executable, str(script)], env=env)
    sys.exit(ret.returncode)


# ══════════════════════════════════════════════════════════════════════════════
# test
# ══════════════════════════════════════════════════════════════════════════════

def cmd_test(args):
    import subprocess
    root = Path(__file__).parent.parent
    test_args = getattr(args, "tests", None) or ["tests"]
    cmd = [sys.executable, "-m", "unittest"] + test_args
    if getattr(args, "verbose", False):
        cmd.append("-v")
    ret = subprocess.run(cmd, cwd=str(root))
    sys.exit(ret.returncode)


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="genbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version="genbox 0.2.0")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # ── setup ─────────────────────────────────────────────────────────────────
    s = sub.add_parser("setup", help="first-run wizard")
    s.add_argument("--force", action="store_true")

    # ── info ──────────────────────────────────────────────────────────────────
    sub.add_parser("info", help="config, models, accelerators")

    # ── gen ───────────────────────────────────────────────────────────────────
    g = sub.add_parser(
        "gen", help="generate image or video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
          Modes:
            t2i      text → image  (default)
            i2i      image → image  (--input required)
            inpaint  inpainting     (--input + --mask required)
            outpaint outpainting    (--input + --expand-* required)
            t2v      text → video
            i2v      image → video  (--input required)

          Examples:
            genbox gen -p "midnight city" --mode t2i
            genbox gen -p "oil painting style" --mode i2i --input photo.png --strength 0.7
            genbox gen -p "remove person" --mode inpaint --input img.png --mask mask.png
            genbox gen -p "wider shot" --mode outpaint --input img.png --expand-left 256
            genbox gen -p "cinematic pullback" --mode t2v --model wan_1_3b --frames 81
            genbox gen -p "cat walking" --mode i2v --input frame.png --model ltx2_fp8
        """),
    )
    g.add_argument("--prompt",          "-p", required=True)
    g.add_argument("--mode",            "-M",
                   choices=["t2i", "i2i", "inpaint", "outpaint", "t2v", "i2v"],
                   default="t2i")
    g.add_argument("--model",           "-m", default=None)
    g.add_argument("--input",           "-i", default=None,
                   help="input image path (i2i / inpaint / outpaint / i2v)")
    g.add_argument("--mask",            default=None,
                   help="mask image (inpaint — white=fill, black=keep)")
    g.add_argument("--end-frame",       default=None, dest="end_frame",
                   help="optional end frame for i2v (LTX FLF mode)")
    g.add_argument("--negative-prompt", "-n", default="", dest="negative_prompt")
    g.add_argument("--width",           type=int, default=0)
    g.add_argument("--height",          type=int, default=0)
    g.add_argument("--steps",           type=int, default=0)
    g.add_argument("--guidance",        type=float, default=0.0)
    g.add_argument("--seed",            type=int, default=-1)
    g.add_argument("--strength",        type=float, default=0.75,
                   help="denoising strength 0.0–1.0 (i2i/inpaint/outpaint)")
    g.add_argument("--blur-radius",     type=float, default=0, dest="blur_radius",
                   help="Gaussian blur on inpaint mask (pixels)")
    g.add_argument("--dilate",          type=int, default=0,
                   help="dilate inpaint mask outward (pixels)")
    g.add_argument("--mask-mode",       default="white_inpaint", dest="mask_mode",
                   choices=["white_inpaint", "black_inpaint"],
                   help="mask convention (default: white=fill)")
    g.add_argument("--expand-left",     type=int, default=0, dest="expand_left")
    g.add_argument("--expand-right",    type=int, default=0, dest="expand_right")
    g.add_argument("--expand-top",      type=int, default=0, dest="expand_top")
    g.add_argument("--expand-bottom",   type=int, default=0, dest="expand_bottom")
    g.add_argument("--feather-radius",  type=float, default=16.0, dest="feather_radius",
                   help="outpaint seam feather (Gaussian radius)")
    g.add_argument("--frames",          type=int, default=0,
                   help="video frame count (0 = model default)")
    g.add_argument("--fps",             type=int, default=0,
                   help="video fps (0 = model default)")
    g.add_argument("--lora",            action="append", default=[], metavar="PATH")
    g.add_argument("--accel",           nargs="+", default=None,
                   choices=["sageAttn", "teacache", "xformers", "compile", "none"])
    g.add_argument("--output",          "-o", default=None)
    g.add_argument("--verbose",         "-v", action="store_true")

    # ── models ────────────────────────────────────────────────────────────────
    m = sub.add_parser("models", help="list / search / download / install-defaults / uninstall")
    m.add_argument("action", nargs="?", default="list",
                   choices=["list", "local", "search", "download",
                            "install-defaults", "uninstall"])
    m.add_argument("query", nargs="*")
    m.add_argument("--model-id",  default=None,   dest="model_id")
    m.add_argument("--type",      choices=["image", "video"], default=None)
    m.add_argument("--limit",     type=int, default=15)
    m.add_argument("--all",       action="store_true",
                   help="show VRAM-incompatible models too")
    m.add_argument("--force",     action="store_true",
                   help="download even if VRAM too low")
    m.add_argument("--profile",   default=None,
                   choices=["8gb_low", "8gb_balanced", "12gb_balanced",
                            "16gb_high", "24gb_ultra"],
                   help="VRAM profile for install-defaults")
    m.add_argument("--dry-run",   action="store_true", dest="dry_run",
                   help="show what would be installed without downloading")

    # ── loras ─────────────────────────────────────────────────────────────────
    lo = sub.add_parser("loras", help="list / tag LoRA weights")
    lo.add_argument("action", nargs="?", default="list",
                    choices=["list", "tag"])
    lo.add_argument("query", nargs="*")
    lo.add_argument("--arch",    choices=["flux", "sd15", "sdxl", "sd35", "ltx", "wan"],
                    default=None)
    lo.add_argument("--trigger", default="", help="trigger word(s)")
    lo.add_argument("--desc",    default="", help="description")
    lo.add_argument("--preview", default="", help="preview image URL or path")

    # ── ui ────────────────────────────────────────────────────────────────────
    u = sub.add_parser("ui", help="launch Streamlit UI")
    u.add_argument("--port", type=int, default=None)

    # ── run ───────────────────────────────────────────────────────────────────
    r = sub.add_parser("run", help="run a pipeline script")
    r.add_argument("script")

    # ── test ──────────────────────────────────────────────────────────────────
    t = sub.add_parser("test", help="run test suite")
    t.add_argument("tests", nargs="*", default=None)
    t.add_argument("--verbose", "-v", action="store_true")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "setup":   cmd_setup,
    "info":    cmd_info,
    "gen":     cmd_gen,
    "models":  cmd_models,
    "loras":   cmd_loras,
    "ui":      cmd_ui,
    "run":     cmd_run,
    "test":    cmd_test,
}


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        print()
        print(f"  {blue('genbox')}  {grey('0.2.0')}")
        print()
        rows = [
            ("setup  ",          "first-run wizard"),
            ("info   ",          "config, models, accelerators"),
            ("gen    ",          "generate: t2i i2i inpaint outpaint t2v i2v"),
            ("models ",          "list / search / download / install-defaults / uninstall"),
            ("loras  ",          "list / tag LoRA weights"),
            ("ui     ",          "launch Streamlit UI"),
            ("run    ",          "run a pipeline script"),
            ("test   ",          "run test suite"),
        ]
        for cmd, desc in rows:
            print(f"  {white(cmd):14s}{grey(desc)}")
        print()
        print(f"  {dim('genbox <command> --help  for details')}")
        print()
        return

    # convenience: genbox models download flux2_klein  (no --model-id)
    if args.command == "models" and getattr(args, "action", "") in ("download", "uninstall"):
        if not getattr(args, "model_id", None) and getattr(args, "query", None):
            args.model_id = args.query[0]

    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()