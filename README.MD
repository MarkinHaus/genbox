# genbox

Local AI image and video generation. Runs on consumer hardware (8–24 GB VRAM).

```bash
genbox setup
genbox models install-defaults
genbox gen -p "a brutalist tower at dusk"
genbox ui
```

---

## Requirements

- Python 3.10+
- CUDA GPU with 8 GB+ VRAM (CPU fallback available, slow)
- NVIDIA driver 525+ or ROCm 5.x for AMD

---

## Installation

```bash
git clone https://github.com/MarkinHaus/genbox
cd genbox
pip install -e .                    # core only
pip install -e ".[image]"           # + image generation
pip install -e ".[image,video]"     # + video generation
pip install -e ".[all]"             # + UI
```

First run:

```bash
genbox setup                        # wizard: detects VRAM, writes ~/.genbox/config.toml
genbox models install-defaults      # download base models for your VRAM profile
genbox ui                           # http://localhost:8501
```

---

## File structure

```
genbox/

  cli.py                   command-line interface
  config.py                setup wizard, VRAM detection, TOML config
  models.py                registry (28 models), HuggingFace downloader, LoRA manager
  pipeline.py              unified public API
  
  [pipline_image](genbox/pipline_image)
  pipeline_flux.py         FLUX.1 / FLUX.2 Klein (fp8 full-repo + GGUF Q4/Q8)
  pipeline_sdl.py          SD 1.5 / SDXL / SD 3.5
  pipeline_pony.py         Pony Diffusion XL (SDXL + quality-tag injection)
  pipeline_img2img.py      image-to-image (all image archs)
  pipeline_inpaint.py      inpainting with mask utilities (blur, dilate)
  pipeline_outpaint.py     outpainting (canvas expansion + seam feathering)
  [piplen_video](genbox/piplen_video)
  pipeline_wan.py          WAN 2.1 / 2.2 T2V + I2V (VAE=float32 enforced)
  pipeline_ltx.py          LTX-Video classic / distilled 13B / LTX-2
  pipeline_img2video.py    image-to-video dispatcher (WAN + LTX)
  
  [utils](genbox/utils)
  utils.py                 LoRA / model file + metadata helpers
  utils_image_pipeline.py  shared image pipeline (device, dtype, LoRA, scheduler, accel)
  utils_video_pipeline.py  shared video pipeline (frame-snap, variant detection)
  gen_progress.py          thread-safe generation progress tracker
  
  [genbox_ui](genbox/genbox_ui)
  ui_gen_progress.py       Streamlit progress widgets
  ui_helpers.py            pure UI helper functions (no Streamlit import)
  ui.py                    Streamlit UI — 4 screens

tests/
  test_cli.py              CLI parser + dispatch + pipeline guards (78 tests, 4 slow)

genbox/ (test files)
  test_utils_s1.py                  16 tests
  test_utils_s2s3.py                51 tests
  test_utils_image_pipeline.py      45 tests
  test_pipelines.py                 50 tests
  test_utils_video_pipeline.py      56 tests
  test_video_pipelines.py           43 tests
  test_mixed_pipelines.py           67 tests
  test_ui_helpers.py                39 tests
  test_gen_progress.py              33 tests
```

---

## CLI

```
genbox setup
genbox setup --force

genbox info

genbox gen -p "..."                   default: --mode t2i
genbox gen -p "..." --mode t2i        text → image
genbox gen -p "..." --mode i2i        image → image   --input required
genbox gen -p "..." --mode inpaint    inpaint          --input + --mask required
genbox gen -p "..." --mode outpaint   outpaint         --input + --expand-* required
genbox gen -p "..." --mode t2v        text → video
genbox gen -p "..." --mode i2v        image → video    --input required

genbox models list
genbox models list --type video
genbox models list --all
genbox models local
genbox models search "flux gguf"
genbox models download flux2_klein
genbox models install-defaults
genbox models install-defaults --profile 12gb_balanced --dry-run
genbox models uninstall flux2_klein

genbox loras
genbox loras list --arch flux
genbox loras tag style.safetensors --arch flux --trigger "cinematic" --desc "…" --preview "url"

genbox ui
genbox ui --port 9000

genbox run myscript.py

genbox test
genbox test tests.test_cli --verbose
```

### gen flags

| Flag | Default | Description |
|---|---|---|
| `--prompt / -p` | required | |
| `--mode / -M` | `t2i` | `t2i` `i2i` `inpaint` `outpaint` `t2v` `i2v` |
| `--model / -m` | config default | registry model ID |
| `--input / -i` | — | source image |
| `--mask` | — | mask image (inpaint) |
| `--end-frame` | — | end frame (i2v) |
| `--negative-prompt / -n` | `""` | |
| `--width` | 0 | 0 = model default |
| `--height` | 0 | 0 = model default |
| `--steps` | 0 | 0 = model default |
| `--guidance` | 0.0 | 0 = model default |
| `--seed` | -1 | -1 = random |
| `--strength` | 0.75 | i2i / inpaint / outpaint |
| `--blur-radius` | 0 | Gaussian blur on inpaint mask (px) |
| `--dilate` | 0 | expand inpaint mask outward (px) |
| `--mask-mode` | `white_inpaint` | `white_inpaint` or `black_inpaint` |
| `--expand-left/right/top/bottom` | 0 | outpaint expansion per side (px) |
| `--feather-radius` | 16.0 | outpaint seam feathering |
| `--frames` | 0 | 0 = model default |
| `--fps` | 0 | 0 = model default |
| `--lora` | — | path, repeat for multiple |
| `--accel` | config | `sageAttn` `teacache` `xformers` `compile` `none` |
| `--output / -o` | auto | |
| `--verbose / -v` | off | full traceback on error |

---

## Python API

```python
from genbox import pipeline

# text → image
result = pipeline.text_to_image(
    prompt="a brutalist tower at dusk, fog, cinematic",
    model="flux2_klein",          # None = config default
    negative_prompt="",
    width=0,                      # 0 = model default
    height=0,
    steps=0,
    guidance_scale=0.0,
    seed=-1,                      # -1 = random
    loras=None,                   # list of paths or (path, weight) tuples
    accel=None,                   # list: ["sageAttn", "teacache", "xformers", "compile"]
    t5_mode="fp16",               # "fp16" or "int8" (FLUX only)
    sampler="default",
    output=None,
)
print(result)                     # <GenResult img_…_flux2_klein_seed42.png  2.3s>
result.save()                     # write .json sidecar next to output file
params = result.remix(seed=99)    # dict of original params + overrides
pipeline.text_to_image(**params)

# image → image
pipeline.image_to_image(
    prompt="…",
    input_image="photo.png",      # required
    strength=0.75,
    # … same optional params as text_to_image
)

# inpaint
pipeline.inpaint(
    prompt="…",
    input_image="photo.png",      # required
    mask_image="mask.png",        # required — white=fill, black=keep
    width=1024, height=1024,
    strength=0.99,
    blur_radius=0,                # Gaussian blur on mask edges (px)
    dilate_pixels=0,              # expand mask outward (px)
    mask_mode="white_inpaint",    # or "black_inpaint"
)

# outpaint
pipeline.outpaint(
    prompt="…",
    input_image="photo.png",      # required
    left=0, right=256, top=0, bottom=0,
    feather_radius=16.0,
    strength=0.99,
)

# text → video
pipeline.text_to_video(
    prompt="…",
    model="wan_1_3b",
    frames=0, fps=0,              # 0 = model default
    enable_vae_tiling=False,
)

# image → video
pipeline.image_to_video(
    prompt="…",
    start_frame="start.png",      # required
    end_frame=None,               # optional (LTX FLF mode)
    frames=0, fps=0,
    enable_vae_tiling=False,
)
```

Model management:

```python
pipeline.is_installed("flux2_klein")               # → bool
pipeline.require_installed("flux2_klein")           # raises RuntimeError if not installed
pipeline.download("flux2_klein", force=False)       # → Path
pipeline.install_defaults(profile=None, dry_run=False)  # → list[str] installed IDs
pipeline.uninstall("flux2_klein")                   # → bool
pipeline.list_models(model_type=None, installed_only=False, max_vram=None)  # → list[ModelEntry]
pipeline.list_installed()                           # → list[dict]
pipeline.get_model_entry("flux2_klein")             # → ModelEntry
```

Metadata:

```python
pipeline.write_lora_metadata(
    "cinematic.safetensors",
    architecture="flux",
    trigger="",
    description="",
    preview_url="",
)
pipeline.write_model_metadata(
    "my_model.safetensors",
    description="",
    preview_url="",
    tags=None,
)
pipeline.register_custom("my_model.safetensors", architecture="sdxl")
```

`GenResult` fields:

```python
result.output_path    # Path
result.metadata       # dict: prompt, seed, model, steps, loras, accel, elapsed_s, …
result.elapsed_s      # float
result.save(dest=None)   # → Path, writes .json sidecar
result.remix(**kw)       # → dict for re-use as kwargs
```

---

## Models

### Image

| ID | VRAM | Quant | Name |
|---|---|---|---|
| `flux2_klein` | 10 GB | fp8 | FLUX.2 Klein 4B |
| `flux2_klein9b` | 12 GB | fp8 | FLUX.2 Klein 9B |
| `unsloth_flux2_9b_q8` | 12 GB | gguf-q8 | FLUX.2 Klein 9B (Unsloth Q8) |
| `unsloth_flux2_9b_q4` | 8 GB | gguf-q4 | FLUX.2 Klein 9B (Unsloth Q4) |
| `flux1_schnell_q8` | 8 GB | gguf-q8 | FLUX.1 Schnell (GGUF Q8) |
| `flux1_schnell_q4` | 6 GB | gguf-q4 | FLUX.1 Schnell (GGUF Q4) |
| `flux1_dev_q8` | 10 GB | gguf-q8 | FLUX.1-dev (GGUF Q8) |
| `flux1_dev_q4` | 8 GB | gguf-q4 | FLUX.1-dev (GGUF Q4) |
| `z_image_turbo` | 9 GB | fp8 | Z-Image-Turbo 6B |
| `sd35_medium` | 6 GB | fp16 | SD 3.5 Medium 2B |
| `sd15_base` | 4 GB | fp16 | Stable Diffusion 1.5 (Base) |
| `sd15_realistic` | 4 GB | fp16 | Realistic Vision V6.0 B1 |
| `sdxl_base` | 8 GB | fp16 | Stable Diffusion XL 1.0 (Base) |
| `sdxl_turbo` | 5 GB | fp16 | SDXL Turbo |
| `animagine_xl` | 8 GB | fp16 | Animagine XL 4.0 |
| `pony_xl` | 8 GB | fp16 | Pony Diffusion v6 XL |

### Video

| ID | VRAM | Name |
|---|---|---|
| `ltx2_fp8` | 10 GB | LTX-Video 0.9.5 (bf16) |
| `ltx23_fp8` | 12 GB | LTX-Video 0.9.7 distilled 13B |
| `ltx_098_dev` | 12 GB | LTX-Video 0.9.8-dev 13B |
| `ltx2_model` | 14 GB | LTX-2 (Joint Audio-Visual) |
| `wan_1_3b` | 8 GB | WAN 2.1 T2V-1.3B (480p) |
| `wan21_14b_diffusers` | 16 GB | WAN 2.1 T2V-14B (480p/720p) |
| `wan21_i2v_480p` | 16 GB | WAN 2.1 I2V-14B 480P |
| `wan21_i2v_720p` | 20 GB | WAN 2.1 I2V-14B 720P |
| `wan21_flf2v_720p` | 20 GB | WAN 2.1 FLF2V-14B 720P (First+Last Frame) |
| `wan22_1_4b` | 24 GB | WAN 2.2 T2V-A14B (MoE 720p) |
| `wan22_i2v_a14b` | 24 GB | WAN 2.2 I2V-A14B (MoE) |
| `wan22_ti2v_5b` | 12 GB | WAN 2.2 TI2V-5B (Text+Image→Video) |

### Default install sets (`genbox models install-defaults`)

| Profile | Model IDs |
|---|---|
| `8gb_low` | `flux1_schnell_q4` `wan_1_3b` `ltx2_fp8` |
| `8gb_balanced` | `flux1_schnell_q8` `flux1_dev_q4` `wan_1_3b` `ltx2_fp8` |
| `12gb_balanced` | `flux2_klein` `flux1_dev_q8` `sd15_base` `ltx2_fp8` `ltx23_fp8` `wan_1_3b` |
| `16gb_high` | + `flux2_klein9b` `sdxl_base` `sd35_medium` `wan21_14b_diffusers` `wan21_i2v_480p` |
| `24gb_ultra` | all of the above + `unsloth_flux2_9b_q8` `animagine_xl` `pony_xl` `ltx2_model` `wan21_i2v_720p` `wan22_1_4b` `wan22_i2v_a14b` `wan22_ti2v_5b` |

---

## LoRAs

Place `.safetensors` in `~/.genbox/loras/` or in arch subfolders (`loras/flux/`, `loras/wan/`, …).

```bash
genbox loras tag cinematic.safetensors --arch flux --trigger "cinematic style"
genbox gen -p "…" --lora cinematic.safetensors
genbox gen -p "…" --lora a.safetensors --lora b.safetensors
```

Sidecar `.json` written by `genbox loras tag` or `pipeline.write_lora_metadata()`:

```json
{
  "architecture": "flux",
  "trigger": "cinematic style",
  "description": "",
  "preview_url": ""
}
```

---

## Generation progress

`gen_progress.py` — thread-safe, no Streamlit dependency:

```python
from genbox.utils.gen_progress import (
    GenProgressTracker,         # state container
    make_step_callback,         # generic diffusers callback factory
    decode_latents_to_preview,  # VAE decode → PNG, returns None on failure
    GenRunner,                  # threading.Thread subclass
    format_step_label,          # → "denoising 5 / 28 · ETA 12s"
)
```

### Tracker creation

```python
tracker = GenProgressTracker(total_steps=28)
```

### Passing tracker to a pipeline

All generation functions accept `tracker=` directly:

```python
from genbox import pipeline

result = pipeline.text_to_image(
    prompt="...", model="flux1_dev",
    tracker=tracker,
    enable_preview=True,
    preview_interval=5,
)

result = pipeline.text_to_video(
    prompt="...", model="wan_1_3b",
    tracker=tracker,
    enable_noise_meter=True,   # video only — Variante 3
)
```

For **image pipelines** (`text_to_image`, `image_to_image`, `inpaint`, `outpaint`) the tracker receives step updates and optional latent preview images decoded via the VAE.

For **video pipelines** (`text_to_video`, `image_to_video`) the tracker receives step updates and optionally per-step latent noise statistics (`noise_std_history`) — no VAE decode is performed.

### Architecture-aware callbacks (used internally)

The correct callback is selected automatically inside each pipeline based on architecture. For custom use:

```python
from genbox.utils.utils_image_pipeline import (
    make_flux_step_callback,  # FLUX / FLUX.2 — handles packed latents (B, seq, C)
    make_sdl_step_callback,   # SD1.5 / SDXL / SD3.5 / Pony — spatial latents (B, C, H, W)
)
from genbox.utils.utils_video_pipeline import (
    make_video_step_callback, # WAN + LTX — Variante 1 + optional Variante 3
)

# FLUX example
cb = make_flux_step_callback(
    tracker,
    height=1024, width=1024,
    preview_interval=5,
    enable_preview=True,
)

# SDL example
cb = make_sdl_step_callback(
    tracker,
    preview_interval=5,
    enable_preview=True,
)

# Video example (step counter + noise meter, no VAE decode)
cb = make_video_step_callback(
    tracker,
    enable_noise_meter=True,
)

# pass cb as:
pipe(
    ...,
    callback_on_step_end=cb,
    callback_on_step_end_tensor_inputs=["latents"],
)
```

> **Note on `make_step_callback`:** The generic factory in `gen_progress.py` is kept for custom use. Its `decode_fn` must accept `(latents, pipe, step_index: int) → Optional[Path]`. For FLUX and LTX the latents in the callback are in packed form `(B, seq_len, channels)` — use the arch-specific factories above instead.

### Streamlit UI integration

```python
from genbox.genbox_ui.ui_gen_progress import run_with_progress, render_progress

# one-liner: launches thread, polls, renders bar + label + preview
result_dict = run_with_progress(
    fn=lambda tracker: pipeline.text_to_image(
        prompt="...", model="flux1_dev",
        tracker=tracker,
        enable_preview=True,
        preview_interval=5,
    ),
    total_steps=28,
    show_preview=True,
    poll_interval_s=0.4,
)

# or manual polling loop
runner = GenRunner(fn=gen_fn, tracker=tracker)
runner.start()
while runner.is_alive():
    render_progress(tracker, ph_bar, ph_label, ph_preview)
    time.sleep(0.4)
runner.join()
if runner.exception:
    raise runner.exception
result = runner.result
```

### `GenProgressTracker` public API

| Method | Description |
|---|---|
| `snapshot() → dict` | Thread-safe state copy. Keys: `step`, `total`, `stage`, `done`, `error`, `error_msg`, `preview_path`, `noise_std_history` |
| `fraction() → float` | `0.0–1.0` |
| `elapsed_seconds() → float` | Wall time since creation |
| `eta_seconds() → float \| None` | Linear estimate from last 8 step samples |
| `set_step(step, stage="denoising")` | Called by callback on each denoising step |
| `set_stage(stage)` | `"idle"` · `"loading model"` · `"denoising"` · `"saving"` · `"done"` · `"error"` |
| `set_preview(path)` | Path to latest decoded preview PNG |
| `set_noise_std(std: float)` | Append latent std sample — video pipelines, Variante 3 |
| `mark_done()` | Sets `step=total`, `stage="done"`, `done=True` |
| `mark_error(msg)` | Sets `error=True`, `error_msg=msg` |

### `noise_std_history` (video — Variante 3)

When `enable_noise_meter=True` is passed to a video pipeline, `tracker.snapshot()["noise_std_history"]` is populated with one `float` per denoising step. The values decrease monotonically from ~`1.0` (pure noise) toward ~`0.05–0.15` (converged). A flat or rising curve indicates a generation issue.

```python
snap = tracker.snapshot()
history = snap["noise_std_history"]   # e.g. [0.98, 0.81, 0.62, 0.44, ...]
```
---

## Accelerators

| Key | Package |
|---|---|
| `sageAttn` | `pip install sageattention` |
| `teacache` | `pip install git+https://github.com/ali-vilab/TeaCache` |
| `xformers` | `pip install xformers` |
| `compile` | built-in (torch ≥ 2.0) |

```toml
# ~/.genbox/config.toml
[accelerators]
active = ["sageAttn", "teacache"]
```

---

## Output structure

```
~/.genbox/outputs/
  2026-03-22/
    img_0001_flux2_klein_seed42.png
    img_0001_flux2_klein_seed42.json
    vid_0003_wan_1_3b_seed7.mp4
    vid_0003_wan_1_3b_seed7.json
```

---

## Testing

```bash
# all 478 fast tests (no GPU required)
genbox test

# specific file, verbose
genbox test tests.test_cli --verbose
genbox test genbox.test_pipelines --verbose

# directly
python -m unittest \
  genbox.test_utils_s1 genbox.test_utils_s2s3 \
  genbox.test_utils_image_pipeline genbox.test_pipelines \
  genbox.test_utils_video_pipeline genbox.test_video_pipelines \
  genbox.test_mixed_pipelines genbox.test_ui_helpers genbox.test_gen_progress \
  tests.test_cli -v

# slow integration tests (requires configured genbox installation)
GENBOX_SLOW_TESTS=1 python -m unittest tests.test_cli -v
# or
python -m unittest tests.test_cli --slow -v
```

Slow tests (4 total, `@slow` decorator in `tests/test_cli.py`) are skipped by default.
They test `genbox info`, `genbox models list --all`, `genbox loras`, and `genbox gen` against a real installation.

---

## Config reference

`~/.genbox/config.toml`:

```toml
[genbox]
home           = "/home/user/.genbox"
vram_gb        = 12
vram_profile   = "12gb_balanced"
first_run_done = true

[paths]
models  = "/home/user/.genbox/models"
outputs = "/home/user/.genbox/outputs"
loras   = "/home/user/.genbox/loras"
cache   = "/home/user/.genbox/cache"

[defaults]
image_model = "flux2_klein"
video_model = "ltx2_fp8"
steps       = 28
seed        = -1
accel       = ["sageAttn", "teacache"]

[ui]
port = 8501

[accelerators]
active = ["sageAttn", "teacache"]

[hf]
token = ""
```

Environment variables:

```bash
GENBOX_HOME=/mnt/fast/genbox genbox ui
GENBOX_OFFLOAD=none       genbox gen -p "…"   # no CPU offload
GENBOX_OFFLOAD=sequential genbox gen -p "…"   # sequential CPU offload
GENBOX_SLOW_TESTS=1       python -m unittest tests.test_cli
```

---

## License

Apache 2.0. Model licenses vary — check each model's HuggingFace page before commercial use.
`flux1_dev` and `flux2_klein` variants are non-commercial. `pony_xl` has its own license.