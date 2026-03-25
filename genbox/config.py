"""
genbox/config.py
Handles: GENBOX_HOME resolution, TOML config r/w, VRAM detection, first-run wizard.
Cross-OS: Windows (%APPDATA%), Linux/macOS (~/.genbox).
"""
import io
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib  # stdlib Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None

# Windows: suppress huggingface_hub symlink warning (no Developer Mode needed)
if platform.system() == "Windows":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ── Default config schema ──────────────────────────────────────────────────────

DEFAULTS: dict = {
    "genbox": {
        "home": "",           # set during setup
        "vram_gb": 0,         # auto-detected
        "vram_profile": "",   # 8gb_low | 12gb_balanced | 16gb_high | 24gb_ultra
        "first_run_done": False,
    },
    "paths": {
        "models": "",
        "outputs": "",
        "loras": "",
        "cache": "",
    },
    "defaults": {
        "image_model": "flux2_klein",
        "video_model": "ltx2_fp8",
        "steps": 28,
        "seed": -1,           # -1 = random
        "accel": ["sageAttn", "teacache"],
    },
    "ui": {
        "theme": "dark",
        "port": 8501,
    },
}

VRAM_PROFILES = {
    (0, 7):   "8gb_low",
    (7, 11):  "8gb_balanced",
    (11, 15): "12gb_balanced",
    (15, 23): "16gb_high",
    (23, 999): "24gb_ultra",
}


# ── Home directory resolution ──────────────────────────────────────────────────

def _default_home() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home()
    return base / ".genbox"


def _config_file(home: Path) -> Path:
    return home / "config.toml"


# ── VRAM detection ─────────────────────────────────────────────────────────────

def detect_vram() -> int:
    """Returns VRAM in GB. Tries torch → nvidia-smi → 0 (CPU fallback)."""
    # 1. torch (most accurate, no subprocess)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory // (1024 ** 3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 0  # Apple Silicon unified — report 0, handle separately
    except ImportError:
        pass

    # 2. nvidia-smi (works without torch)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip().split("\n")[0]
        return int(out) // 1024  # MiB → GiB
    except Exception:
        pass

    # 3. ROCm (AMD)
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        import json
        data = json.loads(out)
        vram_bytes = list(data.values())[0].get("VRAM Total Memory (B)", 0)
        return int(vram_bytes) // (1024 ** 3)
    except Exception:
        pass

    return 0  # CPU-only


def _vram_profile(vram_gb: int) -> str:
    for (lo, hi), name in VRAM_PROFILES.items():
        if lo <= vram_gb < hi:
            return name
    return "8gb_low"


# ── TOML helpers ───────────────────────────────────────────────────────────────

def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    if tomllib is None:
        _die("Missing dep: pip install tomli (Python < 3.11)")
    with open(path, "rb") as f:
        return tomllib.loads(f.read().decode())


def _save_toml(data: dict, path: Path) -> None:
    if tomli_w is None:
        # manual minimal TOML writer for zero-dep bootstrap
        _write_toml_manual(data, path)
        return
    with open(path, "wb") as f:
        f.write(tomli_w.dumps(data).encode())


def _write_toml_manual(data: dict, path: Path) -> None:
    """Minimal TOML writer — handles str/int/bool/list/nested dict."""
    lines = []

    def _val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            # Proper TOML basic-string escaping: backslash → \\, quote → \"
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(v, list):
            return "[" + ", ".join(_val(i) for i in v) + "]"
        return str(v)

    def _section(d: dict, prefix: str = ""):
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"\n[{prefix}{k}]")
                _section(v, "")
            else:
                lines.append(f"{k} = {_val(v)}")

    _section(data)
    path.write_text("\n".join(lines), encoding="utf-8")


def _die(msg: str):
    print(f"\n  [error] {msg}", file=sys.stderr)
    sys.exit(1)


# ── Setup wizard ───────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str) -> str:
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"  {prompt} ({hint}): ").strip().lower()
        if not val:
            return default
        return val.startswith("y")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)



# ── CUDA / GPU detection ───────────────────────────────────────────────────────

# Compute capability → max safe PyTorch CUDA backend
# sm < 7.5: cu128 dropped support; use cu126 max
# sm >= 7.5: cu128 fine (Turing, Ampere, Ada, Blackwell)
_SM_TO_MAX_BACKEND: list[tuple[int, str]] = [
    (50, "cpu"),    # sm < 5.0: too old for any modern PyTorch
    (50, "cu126"),  # sm 5.0–7.4: cu128 dropped sm < 7.5
    (75, "cu128"),  # sm 7.5+ (Turing+): full cu128 support
]

# PyTorch CUDA index URLs
_TORCH_INDEXES: dict[str, str] = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cpu":   "https://download.pytorch.org/whl/cpu",
}


def _run_cmd(args: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a subprocess, return (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(
            args,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1, ""


def _detect_nvidia_gpu() -> dict:
    """
    Detect NVIDIA GPU via nvidia-smi.
    Returns dict with keys: name, driver_version, cuda_driver_version,
    compute_major, compute_minor, vram_mb.
    Empty dict if no NVIDIA GPU found.
    """
    rc, out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0 or not out:
        return {}

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return {}

    # Use first GPU
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 4:
        return {}

    name, driver, compute_cap, vram_mb = parts[0], parts[1], parts[2], parts[3]
    try:
        major, minor = (int(x) for x in compute_cap.split("."))
    except (ValueError, AttributeError):
        major, minor = 0, 0

    # cuda_driver_version from nvidia-smi top line
    rc2, smi_out = _run_cmd(["nvidia-smi"])
    cuda_driver = ""
    for line in smi_out.splitlines():
        if "CUDA Version" in line:
            parts2 = line.split("CUDA Version:")
            if len(parts2) > 1:
                cuda_driver = parts2[1].strip().split()[0].strip("|").strip()
            break

    return {
        "name": name,
        "driver_version": driver,
        "cuda_driver_version": cuda_driver,
        "compute_major": major,
        "compute_minor": minor,
        "vram_mb": int(vram_mb) if vram_mb.isdigit() else 0,
        "all_gpus": lines,
    }


def _detect_existing_torch() -> dict:
    """Detect installed torch and its CUDA variant. Returns {} if not installed."""
    try:
        import torch  # noqa
        version = torch.__version__
        cuda_ver = getattr(torch.version, "cuda", None) or ""
        cuda_available = torch.cuda.is_available()
        device_name = ""
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        # extract backend tag e.g. "2.7.0+cu128" → "cu128"
        backend_tag = ""
        if "+" in version:
            backend_tag = version.split("+")[1]
        return {
            "version": version,
            "backend_tag": backend_tag,
            "cuda_version": cuda_ver,
            "cuda_available": cuda_available,
            "device_name": device_name,
        }
    except ImportError:
        return {}


def _sm_to_max_backend(major: int, minor: int) -> str:
    """Map CUDA compute capability to max safe PyTorch backend string."""
    sm = major * 10 + minor
    if sm < 50:
        return "cpu"   # Maxwell minimum for modern PyTorch
    if sm < 75:
        return "cu126"  # cu128 dropped sm < 7.5 (Turing)
    return "cu128"


def _find_uv() -> Optional[str]:
    """Find uv executable. Returns path or None."""
    import shutil
    found = shutil.which("uv")
    if found:
        return found
    # common install locations
    candidates = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path(os.environ.get("APPDATA", "")) / ".." / "Local" / "uv" / "bin" / "uv.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "uv" / "bin" / "uv.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _install_uv() -> Optional[str]:
    """Install uv via the official installer script. Returns uv path or None."""
    print("  Installing uv...")
    system = platform.system()
    if system == "Windows":
        rc, out = _run_cmd([
            "powershell", "-c",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ], timeout=120)
    else:
        rc, out = _run_cmd([
            "sh", "-c",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ], timeout=120)

    if rc != 0:
        print(f"  uv install failed:\n{out}")
        return None

    # Refresh PATH lookup after install
    uv = _find_uv()
    if uv:
        print(f"  uv installed → {uv}")
    return uv


def _uninstall_torch_uv(uv: str) -> bool:
    """Uninstall torch, torchvision, torchaudio via uv."""
    print("  Uninstalling existing torch...")
    rc, out = _run_cmd(
        [uv, "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
        timeout=60,
    )
    if rc == 0:
        print("  Uninstalled OK")
        return True
    # try without -y (older uv)
    rc, out = _run_cmd(
        [uv, "pip", "uninstall", "torch", "torchvision", "torchaudio"],
        timeout=60,
    )
    return rc == 0


def _install_torch_uv(uv: str, backend: str) -> bool:
    """
    Install torch + torchvision + torchaudio via uv with explicit CUDA backend.
    Uses --torch-backend=auto when backend='auto', otherwise explicit index-url.
    Streams output live.
    """
    packages = ["torch", "torchvision", "torchaudio"]

    if backend == "auto":
        cmd = [uv, "pip", "install"] + packages + ["--torch-backend=auto"]
        print(f"  Installing PyTorch (auto-detect CUDA)...")
    elif backend == "cpu":
        cmd = [uv, "pip", "install"] + packages + [
            "--index-url", _TORCH_INDEXES["cpu"],
        ]
        print("  Installing PyTorch (CPU-only)...")
    else:
        index_url = _TORCH_INDEXES.get(backend)
        if not index_url:
            print(f"  Unknown backend: {backend!r}")
            return False
        cmd = [uv, "pip", "install"] + packages + [
            "--index-url", index_url,
        ]
        print(f"  Installing PyTorch ({backend})...")

    print(f"  Command: {' '.join(cmd)}\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(f"    {line}", end="")
        proc.wait()
        success = proc.returncode == 0
        if success:
            print("\n  PyTorch installed successfully.")
        else:
            print(f"\n  Installation failed (exit {proc.returncode}).")
        return success
    except Exception as e:
        print(f"  Install error: {e}")
        return False


def _verify_torch_cuda() -> bool:
    """Quick verification: import torch, check cuda.is_available()."""
    rc, out = _run_cmd([
        sys.executable, "-c",
        "import torch; print('torch:', torch.__version__); "
        "print('cuda:', torch.cuda.is_available()); "
        "print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    ], timeout=30)
    if rc == 0:
        for line in out.splitlines():
            print(f"    {line}")
        return "cuda: True" in out
    print(f"  Verification failed: {out[:200]}")
    return False


def _setup_torch_interactive() -> Optional[str]:
    """
    Interactive PyTorch + CUDA setup.
    Returns chosen backend tag (e.g. 'cu128') or None if skipped.
    """
    print("\n" + "─" * 40)
    print("  PyTorch + CUDA Setup")
    print("─" * 40)

    # 1. Detect GPU
    gpu = _detect_nvidia_gpu()
    if gpu:
        sm = f"sm_{gpu['compute_major']}{gpu['compute_minor']}"
        max_backend = _sm_to_max_backend(gpu["compute_major"], gpu["compute_minor"])
        print(f"\n  GPU detected:")
        print(f"    {gpu['name']}")
        print(f"    Driver: {gpu['driver_version']}")
        if gpu["cuda_driver_version"]:
            print(f"    CUDA driver: {gpu['cuda_driver_version']}")
        print(f"    Compute capability: {gpu['compute_major']}.{gpu['compute_minor']} ({sm})")
        print(f"    VRAM: {gpu['vram_mb'] // 1024}GB")
        print(f"    Max compatible PyTorch backend: {max_backend}")
        if len(gpu["all_gpus"]) > 1:
            print(f"    ({len(gpu['all_gpus'])} GPUs total — using first)")
    else:
        max_backend = "cpu"
        print("\n  No NVIDIA GPU detected (CPU-only or AMD/Intel).")

    # 2. Detect existing torch
    existing = _detect_existing_torch()
    if existing:
        print(f"\n  Existing torch: {existing['version']}")
        print(f"    CUDA available: {existing['cuda_available']}")
        if existing["device_name"]:
            print(f"    Device: {existing['device_name']}")

        # Check if already correct
        if existing["cuda_available"] and gpu:
            print("\n  torch is already installed and CUDA works.")
            if not _ask_yn("  Reinstall/upgrade anyway?", default=False):
                return existing.get("backend_tag") or max_backend
    else:
        print("\n  torch is not installed.")

    # 3. Find or install uv
    uv = _find_uv()
    if not uv:
        print("\n  uv not found.")
        if _ask_yn("  Install uv now? (recommended)", default=True):
            uv = _install_uv()
        if not uv:
            print("  Skipping torch install. Install uv manually:")
            print("    Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
            print("    Linux:   curl -LsSf https://astral.sh/uv/install.sh | sh")
            return None
    else:
        rc, uv_ver = _run_cmd([uv, "--version"])
        print(f"\n  uv: {uv_ver}")

    # 4. Choose backend
    print(f"\n  Available PyTorch backends:")
    options = []
    if gpu:
        options.append(("auto",  "auto (uv detects CUDA automatically — recommended)"))
        if max_backend == "cu128":
            options.append(("cu128", "cu128 — CUDA 12.8 (PyTorch 2.7, latest)"))
        options.append(("cu126", "cu126 — CUDA 12.6 (PyTorch 2.7, stable)"))
        options.append(("cu118", "cu118 — CUDA 11.8 (older, wide compatibility)"))
    options.append(("cpu", "cpu  — CPU only (no GPU acceleration)"))
    options.append(("skip", "skip — do not install torch now"))

    for i, (tag, label) in enumerate(options, 1):
        marker = "  ←  recommended" if i == 1 else ""
        print(f"    [{i}] {label}{marker}")

    while True:
        choice = input(f"\n  Choose [1–{len(options)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            chosen_tag, chosen_label = options[int(choice) - 1]
            break
        print(f"  Enter a number between 1 and {len(options)}.")

    if chosen_tag == "skip":
        print("  Skipped torch installation.")
        return None

    # 5. Warn if user picks higher backend than GPU supports
    if gpu and chosen_tag not in ("auto", "cpu", "skip"):
        sm_val = gpu["compute_major"] * 10 + gpu["compute_minor"]
        if chosen_tag == "cu128" and sm_val < 75:
            print(f"\n  ⚠  Warning: {gpu['name']} has compute capability "
                  f"{gpu['compute_major']}.{gpu['compute_minor']} (sm_{sm_val}).")
            print(f"     cu128 requires sm_75+. Your GPU will silently fail at runtime.")
            print(f"     Recommended: cu126")
            if not _ask_yn("  Continue with cu128 anyway?", default=False):
                chosen_tag = "cu126"
                print("  Switched to cu126.")

    # 6. Uninstall old torch if present
    if existing:
        if _ask_yn(f"  Uninstall existing torch {existing['version']} first?", default=True):
            _uninstall_torch_uv(uv)

    # 7. Install
    success = _install_torch_uv(uv, chosen_tag)

    # 8. Verify
    if success:
        print("\n  Verifying installation...")
        _verify_torch_cuda()

    # 9. Optional: CUDA Toolkit für SageAttention/triton
    if success and gpu:
        _check_cuda_toolkit_optional()

        return chosen_tag if chosen_tag != "auto" else max_backend
# Minimaler CUDA Toolkit für SageAttention (nur nvcc + cudart, kein Driver)
# CUDA 12.8.1 Windows installer — nur toolkit components, kein Display-Driver
_CUDA_TOOLKIT_URL = (
    "https://developer.download.nvidia.com/compute/"
    "cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe"
)
# Gleiche Version für CUDA 12.6
_CUDA_126_TOOLKIT_URL = (
    "https://developer.download.nvidia.com/compute/"
    "cuda/12.6.3/local_installers/cuda_12.6.3_561.17_windows.exe"
)


def _check_nvcc() -> str:
    """Prüft ob nvcc verfügbar ist. Gibt Version zurück oder ''."""
    import shutil
    nvcc = shutil.which("nvcc")
    if not nvcc:
        # Häufige Windows-Pfade
        for candidate in [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe"),
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"),
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"),
        ]:
            if candidate.exists():
                nvcc = str(candidate)
                break
    if not nvcc:
        return ""
    rc, out = _run_cmd([nvcc, "--version"])
    if rc == 0:
        # "release 12.8, V12.8.61" → "12.8"
        for line in out.splitlines():
            if "release" in line:
                try:
                    return line.split("release")[1].split(",")[0].strip()
                except Exception:
                    pass
    return "found"


def _check_cuda_toolkit_optional():
    """
    Prüft ob nvcc vorhanden ist (nötig für SageAttention/triton).
    Bietet an den CUDA Toolkit zu installieren wenn nicht vorhanden.

    WICHTIG: Installiert NUR toolkit-Komponenten (nvcc, cudart etc.)
    KEIN Display-Driver — der ist bereits via NVIDIA-Treiber vorhanden.
    """
    nvcc_ver = _check_nvcc()
    if nvcc_ver:
        print(f"\n  nvcc: {nvcc_ver}  ✓ (SageAttention kann kompilieren)")
        return

    print("\n  nvcc nicht gefunden.")
    print("  nvcc wird für SageAttention und triton-Kompilierung benötigt.")
    print("  Ohne nvcc laufen SageAttention-Kernel nicht.")
    print("\n  Optionen:")
    print("    [1] CUDA Toolkit 12.8 jetzt herunterladen und installieren (silent, kein Driver)")
    print("    [2] CUDA Toolkit 12.6 jetzt herunterladen und installieren (stable)")
    print("    [3] Überspringen — SageAttention manuell installieren")
    print()
    print("  Download-Größe: ~3GB. Dauert je nach Verbindung 5–20 Minuten.")

    choice = input("  Wahl [1/2/3]: ").strip()
    if choice not in ("1", "2"):
        print("  Übersprungen. Installier den CUDA Toolkit manuell:")
        print("    https://developer.nvidia.com/cuda-downloads")
        return

    url = _CUDA_TOOLKIT_URL if choice == "1" else _CUDA_126_TOOLKIT_URL
    ver = "12.8" if choice == "1" else "12.6"
    installer = Path(os.environ.get("TEMP", Path.home())) / f"cuda_{ver}_installer.exe"

    print(f"\n  Lade CUDA Toolkit {ver} herunter...")
    print(f"  → {url}")
    print(f"  → {installer}")

    # Download mit curl (auf Windows 10+ vorhanden)
    rc, out = _run_cmd(
        ["curl.exe", "-L", "-o", str(installer), url],
        timeout=1800,  # 30min für ~3GB
    )
    if rc != 0 or not installer.exists():
        print(f"  Download fehlgeschlagen: {out[:200]}")
        print("  Manuell: https://developer.nvidia.com/cuda-downloads")
        return

    print(f"\n  Installiere CUDA Toolkit {ver} (silent, nur toolkit, kein Driver)...")
    print("  Das dauert 2–5 Minuten. Bitte warten...")

    # Silent install: nur toolkit-Komponenten, KEIN Display-Driver
    # -n = kein Auto-Reboot, -s = silent
    # Explizit KEINEN display_driver installieren
    ver_nodot = ver.replace(".", "")
    toolkit_components = [
        f"cuda_profiler_api_{ver}",
        f"cudart_{ver}",
        f"nvcc_{ver}",
        f"nvdisasm_{ver}",
        f"nvfatbin_{ver}",
        f"nvjitlink_{ver}",
        f"nvrtc_{ver}",
        f"nvrtc_dev_{ver}",
        f"nvtx_{ver}",
        f"thrust_{ver}",
        f"cublas_{ver}",
        f"cublas_dev_{ver}",
    ]

    install_cmd = [str(installer), "-n", "-s"] + toolkit_components
    rc, out = _run_cmd(install_cmd, timeout=600)

    if rc == 0:
        nvcc_after = _check_nvcc()
        if nvcc_after:
            print(f"  CUDA Toolkit {ver} installiert ✓  (nvcc: {nvcc_after})")
            print("  Starte das Terminal neu damit PATH aktualisiert wird.")
        else:
            print(f"  Installer fertig aber nvcc noch nicht im PATH.")
            print(f"  Füge manuell hinzu: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}\\bin")
    else:
        print(f"  Installation fehlgeschlagen (exit {rc}): {out[:300]}")
        print("  Manuell: https://developer.nvidia.com/cuda-downloads")

    # Installer aufräumen
    try:
        installer.unlink()
    except Exception:
        pass

# ── Setup wizard ───────────────────────────────────────────────────────────────

def run_setup(force: bool = False) -> "Config":
    """Interactive first-run wizard. Writes config.toml and returns loaded Config."""
    suggested_home = str(_default_home())

    print("\n  genbox setup\n" + "─" * 40)
    home_str = _ask("Installation directory", suggested_home)
    home = Path(home_str).expanduser().resolve()

    # GPU + VRAM
    vram = detect_vram()
    profile = _vram_profile(vram)
    gpu_label = f"{vram}GB detected" if vram > 0 else "CPU / not detected"
    print(f"\n  GPU VRAM: {gpu_label}  →  profile: {profile}")

    dirs = {
        "models":  str(home / "models"),
        "outputs": str(home / "outputs"),
        "loras":   str(home / "loras"),
        "cache":   str(home / "cache"),
    }
    print("\n  Directory layout:")
    for name, p in dirs.items():
        print(f"    {name:8s}  {p}")

    ok = _ask_yn("\n  Confirm and create directories", default=True)
    if not ok:
        print("  Aborted.")
        sys.exit(0)

    for p in dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    # HuggingFace token
    existing_token = os.environ.get("HF_TOKEN", "")
    print("\n  HuggingFace token (optional — removes rate-limit warnings).")
    print("  Get yours at: https://huggingface.co/settings/tokens")
    hf_token = _ask("HF_TOKEN", existing_token if existing_token else "leave empty to skip").strip()
    if hf_token.lower() in ("leave empty to skip", ""):
        hf_token = ""

    # PyTorch + CUDA interactive install
    if _ask_yn("\n  Set up PyTorch + CUDA now?", default=True):
        torch_backend = _setup_torch_interactive()
    else:
        torch_backend = None

    cfg_data = {
        "genbox": {
            "home": str(home),
            "vram_gb": vram,
            "vram_profile": profile,
            "first_run_done": True,
        },
        "paths": dirs,
        "defaults": DEFAULTS["defaults"],
        "ui": DEFAULTS["ui"],
        "hf": {"token": hf_token},
        "accelerators": {"active": []},
        "torch": {"backend": torch_backend or ""},
    }

    config_path = _config_file(home)
    _save_toml(cfg_data, config_path)
    print(f"\n  Config written → {config_path}")
    if hf_token:
        print("  HF token saved.")
    print("\n  Setup complete. Run `genbox ui` to start.\n")
    return Config(cfg_data, home)


# ── Config object ──────────────────────────────────────────────────────────────


class Config:
    """Thin wrapper around the TOML dict with typed accessors."""

    def __init__(self, data: dict, home: Path):
        self._data = data
        self.home = home

    # path shortcuts
    @property
    def models_dir(self) -> Path:
        return Path(self._data["paths"]["models"])

    @property
    def outputs_dir(self) -> Path:
        return Path(self._data["paths"]["outputs"])

    @property
    def loras_dir(self) -> Path:
        return Path(self._data["paths"]["loras"])

    @property
    def cache_dir(self) -> Path:
        return Path(self._data["paths"]["cache"])

    # hardware
    @property
    def vram_gb(self) -> int:
        return self._data["genbox"].get("vram_gb", 0)

    @property
    def vram_profile(self) -> str:
        return self._data["genbox"].get("vram_profile", "8gb_low")

    # defaults
    @property
    def default_image_model(self) -> str:
        return self._data["defaults"].get("image_model", "flux2_klein")

    @property
    def default_video_model(self) -> str:
        return self._data["defaults"].get("video_model", "ltx23_fp8")

    @property
    def active_accels(self) -> list[str]:
        return self._data.get("accelerators", {}).get("active", [])

    @property
    def hf_token(self) -> str:
        """HuggingFace token — from config or HF_TOKEN env var."""
        return (
            os.environ.get("HF_TOKEN")
            or self._data.get("hf", {}).get("token", "")
        )

    def get(self, *keys, default=None):
        """Dot-path accessor: cfg.get('ui', 'port', default=8501)"""
        d = self._data
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d

    def __repr__(self):
        return f"<Config home={self.home} vram={self.vram_gb}GB profile={self.vram_profile}>"


# ── Module-level singleton ─────────────────────────────────────────────────────

def _load_or_none() -> Optional["Config"]:
    """Load config from default location if it exists, else None."""
    home = _default_home()
    cfg_path = _config_file(home)
    if not cfg_path.exists():
        # also check GENBOX_HOME env override
        env = os.environ.get("GENBOX_HOME")
        if env:
            cfg_path = _config_file(Path(env))
            home = Path(env)
    if cfg_path.exists():
        data = _load_toml(cfg_path)
        # Ensure symlink warning stays suppressed across all entry points
        if platform.system() == "Windows":
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        return Config(data, home)
    return None


cfg: Optional[Config] = _load_or_none()