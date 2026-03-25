"""
TDD: utils_image_pipeline.py
Tests cover: data transforms, validation logic, builder functions.
No syntax tests — real logic and contract tests.
Run: python -m unittest genbox.test_utils_image_pipeline -v
"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _entry(**kw):
    defaults = dict(
        id="flux1_schnell_q8", name="FLUX.1 Schnell Q8", type="image",
        architecture="flux", quant="gguf-q8",
        hf_filename="flux1-schnell-Q8_0.gguf",
        hf_pipeline_repo="black-forest-labs/FLUX.1-schnell",
        full_repo=False,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ── Device / dtype resolution ──────────────────────────────────────────────────

class TestResolveDevice(unittest.TestCase):
    def test_returns_cpu_when_no_cuda(self):
        from genbox.utils.utils_image_pipeline import resolve_device
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = False
        result = resolve_device(_torch=torch_mock)
        self.assertEqual(result, "cpu")

    def test_returns_cuda_when_available(self):
        from genbox.utils.utils_image_pipeline import resolve_device
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        result = resolve_device(_torch=torch_mock)
        self.assertEqual(result, "cuda")

    def test_returns_mps_when_no_cuda(self):
        from genbox.utils.utils_image_pipeline import resolve_device
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = True
        result = resolve_device(_torch=torch_mock)
        self.assertEqual(result, "mps")


class TestResolveDtype(unittest.TestCase):
    def test_fp16_maps_to_float16(self):
        from genbox.utils.utils_image_pipeline import resolve_dtype
        torch_mock = MagicMock()
        torch_mock.float16 = "float16_sentinel"
        result = resolve_dtype("fp16", _torch=torch_mock)
        self.assertEqual(result, "float16_sentinel")

    def test_bf16_maps_to_bfloat16(self):
        from genbox.utils.utils_image_pipeline import resolve_dtype
        torch_mock = MagicMock()
        torch_mock.bfloat16 = "bfloat16_sentinel"
        result = resolve_dtype("bf16", _torch=torch_mock)
        self.assertEqual(result, "bfloat16_sentinel")

    def test_gguf_q8_maps_to_float16(self):
        from genbox.utils.utils_image_pipeline import resolve_dtype
        torch_mock = MagicMock()
        torch_mock.float16 = "float16_sentinel"
        result = resolve_dtype("gguf-q8", _torch=torch_mock)
        self.assertEqual(result, "float16_sentinel")

    def test_unknown_quant_defaults_to_float16(self):
        from genbox.utils.utils_image_pipeline import resolve_dtype
        torch_mock = MagicMock()
        torch_mock.float16 = "float16_sentinel"
        result = resolve_dtype("unknown", _torch=torch_mock)
        self.assertEqual(result, "float16_sentinel")


class TestResolveSeed(unittest.TestCase):
    def test_negative_returns_random(self):
        from genbox.utils.utils_image_pipeline import resolve_seed
        seeds = {resolve_seed(-1) for _ in range(10)}
        self.assertGreater(len(seeds), 1)  # at least some variance

    def test_positive_returns_same(self):
        from genbox.utils.utils_image_pipeline import resolve_seed
        self.assertEqual(resolve_seed(42), 42)
        self.assertEqual(resolve_seed(0), 0)


# ── Accelerator section ─────────────────────────────────────────────────────────

class TestResolveOffloadMode(unittest.TestCase):
    def test_sequential_for_low_vram(self):
        from genbox.utils.utils_image_pipeline import resolve_offload_mode
        mode = resolve_offload_mode(vram_gb=6, env_override=None)
        self.assertEqual(mode, "sequential")

    def test_model_for_mid_vram(self):
        from genbox.utils.utils_image_pipeline import resolve_offload_mode
        mode = resolve_offload_mode(vram_gb=12, env_override=None)
        self.assertEqual(mode, "model")

    def test_env_override_none_means_no_offload(self):
        from genbox.utils.utils_image_pipeline import resolve_offload_mode
        mode = resolve_offload_mode(vram_gb=24, env_override="none")
        self.assertEqual(mode, "none")

    def test_env_override_sequential_forces_sequential(self):
        from genbox.utils.utils_image_pipeline import resolve_offload_mode
        mode = resolve_offload_mode(vram_gb=24, env_override="sequential")
        self.assertEqual(mode, "sequential")

    def test_quantized_encoders_prevent_sequential(self):
        from genbox.utils.utils_image_pipeline import resolve_offload_mode
        mode = resolve_offload_mode(vram_gb=6, env_override="sequential",
                                   has_quantized_encoders=True)
        self.assertEqual(mode, "model")


class TestApplyAccelerators(unittest.TestCase):
    """apply_accelerators modifies pipe in-place. We check call routing."""

    def _make_pipe(self):
        pipe = MagicMock()
        pipe.enable_model_cpu_offload = MagicMock()
        pipe.enable_sequential_cpu_offload = MagicMock()
        pipe.to = MagicMock()
        return pipe

    def test_model_offload_called(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        apply_accelerators(pipe, device="cuda", offload_mode="model")
        pipe.enable_model_cpu_offload.assert_called_once()

    def test_sequential_offload_called(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        apply_accelerators(pipe, device="cuda", offload_mode="sequential")
        pipe.enable_sequential_cpu_offload.assert_called_once()

    def test_no_offload_calls_to_cuda(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        apply_accelerators(pipe, device="cuda", offload_mode="none")
        pipe.to.assert_called_once_with("cuda")

    def test_cpu_device_calls_to_cpu(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        apply_accelerators(pipe, device="cpu", offload_mode="model")
        pipe.to.assert_called_once_with("cpu")

    def test_xformers_attempted(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        pipe.enable_xformers_memory_efficient_attention = MagicMock()
        apply_accelerators(pipe, device="cuda", offload_mode="none", accel=["xformers"])
        pipe.enable_xformers_memory_efficient_attention.assert_called_once()

    def test_xformers_failure_does_not_raise(self):
        from genbox.utils.utils_image_pipeline import apply_accelerators
        pipe = self._make_pipe()
        pipe.enable_xformers_memory_efficient_attention = MagicMock(
            side_effect=RuntimeError("no xformers"))
        # must not raise
        apply_accelerators(pipe, device="cuda", offload_mode="none", accel=["xformers"])


# ── LoRA loading ───────────────────────────────────────────────────────────────

class TestBuildLoraAdapterList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_lora(self, name):
        p = self.base / name
        p.write_bytes(b"\x00" * 8)
        return p

    def test_empty_loras_returns_empty(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        result = build_lora_adapter_list([], loras_dir=self.base)
        self.assertEqual(result, [])

    def test_absolute_path_preserved(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        lora = self._make_lora("style.safetensors")
        result = build_lora_adapter_list([str(lora)], loras_dir=self.base)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], lora)
        self.assertAlmostEqual(result[0]["weight"], 1.0)

    def test_tuple_spec_preserves_weight(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        lora = self._make_lora("style.safetensors")
        result = build_lora_adapter_list([(str(lora), 0.7)], loras_dir=self.base)
        self.assertAlmostEqual(result[0]["weight"], 0.7)

    def test_relative_path_resolved_via_loras_dir(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        lora = self._make_lora("style.safetensors")
        result = build_lora_adapter_list(["style.safetensors"], loras_dir=self.base)
        self.assertEqual(result[0]["path"], lora)

    def test_missing_file_excluded_with_warning(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        result = build_lora_adapter_list(
            ["nonexistent.safetensors"], loras_dir=self.base
        )
        self.assertEqual(result, [])

    def test_adapter_names_are_unique(self):
        from genbox.utils.utils_image_pipeline import build_lora_adapter_list
        a = self._make_lora("a.safetensors")
        b = self._make_lora("b.safetensors")
        result = build_lora_adapter_list([str(a), str(b)], loras_dir=self.base)
        names = [r["adapter_name"] for r in result]
        self.assertEqual(len(names), len(set(names)))


class TestApplyLorasToAdapter(unittest.TestCase):
    """Test the state-machine that calls load_lora_weights + set_adapters."""

    def _make_pipe(self, registered_adapters=None):
        pipe = MagicMock()
        pipe.load_lora_weights = MagicMock()
        registered = registered_adapters or []
        # Simulate PEFT peft_config
        pipe.transformer = MagicMock()
        pipe.transformer.peft_config = {a: MagicMock() for a in registered}
        pipe.get_active_adapters = MagicMock(return_value=registered)
        pipe.set_adapters = MagicMock()
        pipe.dtype = MagicMock()
        return pipe

    def test_empty_loras_skips_all(self):
        from genbox.utils.utils_image_pipeline import apply_loras_to_pipe
        pipe = self._make_pipe()
        apply_loras_to_pipe(pipe, adapter_list=[], architecture="flux")
        pipe.load_lora_weights.assert_not_called()

    def test_load_called_for_each_adapter(self):
        import sys
        peft_mock = MagicMock()
        with patch.dict(sys.modules, {"peft": peft_mock}):
            from genbox.utils.utils_image_pipeline import apply_loras_to_pipe
            pipe = self._make_pipe(registered_adapters=["lora_0", "lora_1"])
            adapters = [
                {"path": Path("/tmp/a.safetensors"), "weight": 1.0, "adapter_name": "lora_0"},
                {"path": Path("/tmp/b.safetensors"), "weight": 0.5, "adapter_name": "lora_1"},
            ]
            apply_loras_to_pipe(pipe, adapter_list=adapters, architecture="flux")
            self.assertEqual(pipe.load_lora_weights.call_count, 2)

    def test_set_adapters_called_with_weights(self):
        import sys
        peft_mock = MagicMock()
        with patch.dict(sys.modules, {"peft": peft_mock}):
            from genbox.utils.utils_image_pipeline import apply_loras_to_pipe
            pipe = self._make_pipe(registered_adapters=["lora_0"])
            adapters = [
                {"path": Path("/tmp/a.safetensors"), "weight": 0.8, "adapter_name": "lora_0"},
            ]
            apply_loras_to_pipe(pipe, adapter_list=adapters, architecture="flux")
            call_args = pipe.set_adapters.call_args
            self.assertIn(0.8, call_args[1].get("adapter_weights",
                           call_args[0][1] if len(call_args[0]) > 1 else []))


# ── Scheduler setting ──────────────────────────────────────────────────────────

class TestSchedulerMap(unittest.TestCase):
    def test_scheduler_map_contains_flux_options(self):
        from genbox.utils.utils_image_pipeline import SCHEDULER_MAP
        self.assertIn("FlowMatchEuler", SCHEDULER_MAP)

    def test_scheduler_map_contains_sdxl_options(self):
        from genbox.utils.utils_image_pipeline import SCHEDULER_MAP
        self.assertIn("DPM++ 2M Karras", SCHEDULER_MAP)
        self.assertIn("Euler A", SCHEDULER_MAP)

    def test_all_entries_have_class_name_and_kwargs(self):
        from genbox.utils.utils_image_pipeline import SCHEDULER_MAP
        for key, (cls_name, kwargs) in SCHEDULER_MAP.items():
            self.assertIsInstance(cls_name, str, msg=f"Key {key}: class name not str")
            self.assertIsInstance(kwargs, dict, msg=f"Key {key}: kwargs not dict")


class TestBuildCallKwargs(unittest.TestCase):
    """build_call_kwargs transforms params into correct pipeline __call__ kwargs."""

    def test_flux_no_negative_prompt(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        kwargs = build_call_kwargs(
            architecture="flux",
            prompt="test", negative_prompt="bad",
            width=1024, height=1024, steps=28,
            guidance_scale=3.5, generator=None,
        )
        self.assertNotIn("negative_prompt", kwargs)

    def test_sdxl_includes_negative_prompt(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        kwargs = build_call_kwargs(
            architecture="sdxl",
            prompt="test", negative_prompt="bad",
            width=1024, height=1024, steps=28,
            guidance_scale=7.5, generator=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "bad")

    def test_sd15_includes_negative_prompt(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        kwargs = build_call_kwargs(
            architecture="sd15",
            prompt="test", negative_prompt="ugly",
            width=512, height=512, steps=20,
            guidance_scale=7.5, generator=None,
        )
        self.assertIn("negative_prompt", kwargs)

    def test_flux_width_height_snapped_to_16(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        kwargs = build_call_kwargs(
            architecture="flux",
            prompt="test", negative_prompt="",
            width=1023, height=1025, steps=28,
            guidance_scale=3.5, generator=None,
        )
        self.assertEqual(kwargs["width"] % 16, 0)
        self.assertEqual(kwargs["height"] % 16, 0)

    def test_t5_none_mode_adds_max_sequence_length(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        kwargs = build_call_kwargs(
            architecture="flux",
            prompt="test", negative_prompt="",
            width=1024, height=1024, steps=28,
            guidance_scale=3.5, generator=None,
            t5_mode="none",
        )
        self.assertEqual(kwargs.get("max_sequence_length"), 77)

    def test_generator_included(self):
        from genbox.utils.utils_image_pipeline import build_call_kwargs
        gen_mock = MagicMock()
        kwargs = build_call_kwargs(
            architecture="sdxl",
            prompt="test", negative_prompt="",
            width=1024, height=1024, steps=28,
            guidance_scale=7.5, generator=gen_mock,
        )
        self.assertIs(kwargs["generator"], gen_mock)


# ── Output path + metadata ─────────────────────────────────────────────────────

class TestBuildOutputMeta(unittest.TestCase):
    def test_contains_required_keys(self):
        from genbox.utils.utils_image_pipeline import build_output_meta
        meta = build_output_meta(
            pipeline_name="text_to_image",
            model_id="flux1_dev",
            prompt="a cat",
            negative_prompt="",
            width=1024, height=1024,
            steps=28, guidance_scale=3.5,
            seed=42, lora_specs=[],
            accel=[], sampler="default",
            elapsed_s=1.5,
            output_path=Path("/tmp/img.png"),
        )
        for key in ("pipeline", "model", "prompt", "width", "height",
                    "steps", "guidance_scale", "seed", "loras",
                    "accel", "sampler", "elapsed_s", "timestamp", "output_path"):
            self.assertIn(key, meta, msg=f"Missing key: {key}")

    def test_lora_specs_normalized(self):
        from genbox.utils.utils_image_pipeline import build_output_meta
        meta = build_output_meta(
            pipeline_name="text_to_image",
            model_id="flux1_dev",
            prompt="a cat",
            negative_prompt="",
            width=1024, height=1024,
            steps=28, guidance_scale=3.5,
            seed=42,
            lora_specs=[("style.safetensors", 0.8)],
            accel=[], sampler="default",
            elapsed_s=1.5,
            output_path=Path("/tmp/img.png"),
        )
        self.assertEqual(len(meta["loras"]), 1)
        self.assertIn("path", meta["loras"][0])
        self.assertIn("weight", meta["loras"][0])

    def test_elapsed_rounded(self):
        from genbox.utils.utils_image_pipeline import build_output_meta
        meta = build_output_meta(
            pipeline_name="t2i", model_id="m",
            prompt="p", negative_prompt="", width=1024, height=1024,
            steps=1, guidance_scale=1.0, seed=0, lora_specs=[],
            accel=[], sampler="d", elapsed_s=3.14159,
            output_path=Path("/tmp/x.png"),
        )
        self.assertEqual(meta["elapsed_s"], round(3.14159, 2))


class TestBuildOutputPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_custom_path_returned_as_is(self):
        from genbox.utils.utils_image_pipeline import build_output_path
        custom = Path(self.tmp.name) / "out.png"
        result = build_output_path("img", "model", 42, "png",
                                   outputs_dir=Path(self.tmp.name),
                                   custom=custom)
        self.assertEqual(result, custom)

    def test_auto_path_created_in_date_subdir(self):
        from genbox.utils.utils_image_pipeline import build_output_path
        result = build_output_path("img", "flux1", 42, "png",
                                   outputs_dir=Path(self.tmp.name))
        # Should be in a date subdir
        self.assertEqual(len(result.parts) - len(Path(self.tmp.name).parts), 2)

    def test_filename_contains_model_and_seed(self):
        from genbox.utils.utils_image_pipeline import build_output_path
        result = build_output_path("img", "flux1", 99, "png",
                                   outputs_dir=Path(self.tmp.name))
        self.assertIn("flux1", result.name)
        self.assertIn("99", result.name)

    def test_dir_created_automatically(self):
        from genbox.utils.utils_image_pipeline import build_output_path
        out_dir = Path(self.tmp.name) / "subdir"
        build_output_path("img", "model", 1, "png", outputs_dir=out_dir)
        self.assertTrue(out_dir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
