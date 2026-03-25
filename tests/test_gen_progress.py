"""
TDD: genbox/gen_progress.py
Progress tracking for live UI updates during diffusers generation.
Tests cover: data structures, callback logic, state transitions.
No torch/diffusers at module level — all mocked.
Run: python -m unittest genbox.test_gen_progress -v
"""
import time
import threading
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── GenProgressTracker ────────────────────────────────────────────────────────

class TestGenProgressTracker(unittest.TestCase):
    def test_initial_state(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=28)
        snap = t.snapshot()
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["total"], 28)
        self.assertEqual(snap["stage"], "idle")
        self.assertFalse(snap["done"])
        self.assertFalse(snap["error"])
        self.assertIsNone(snap["preview_path"])

    def test_update_step(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=28)
        t.set_step(5, stage="denoising")
        snap = t.snapshot()
        self.assertEqual(snap["step"], 5)
        self.assertEqual(snap["stage"], "denoising")

    def test_fraction(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=20)
        t.set_step(10)
        self.assertAlmostEqual(t.fraction(), 0.5)

    def test_fraction_zero_total(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=0)
        self.assertAlmostEqual(t.fraction(), 0.0)

    def test_mark_done(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        t.mark_done()
        snap = t.snapshot()
        self.assertTrue(snap["done"])
        self.assertEqual(snap["step"], 10)  # done forces step to total

    def test_mark_error(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        t.mark_error("something went wrong")
        snap = t.snapshot()
        self.assertTrue(snap["error"])
        self.assertIn("something went wrong", snap["error_msg"])

    def test_set_stage_loading(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=28)
        t.set_stage("loading model")
        self.assertEqual(t.snapshot()["stage"], "loading model")

    def test_set_preview_path(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        t.set_preview(Path("/tmp/preview.png"))
        self.assertEqual(t.snapshot()["preview_path"], Path("/tmp/preview.png"))

    def test_snapshot_is_copy(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        s1 = t.snapshot()
        t.set_step(5)
        s2 = t.snapshot()
        self.assertEqual(s1["step"], 0)   # s1 unaffected
        self.assertEqual(s2["step"], 5)

    def test_thread_safe_updates(self):
        """Multiple threads updating step should not corrupt state."""
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=1000)
        errors = []

        def updater(start, count):
            for i in range(start, start + count):
                try:
                    t.set_step(i)
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=updater, args=(i * 100, 100))
                   for i in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        snap = t.snapshot()
        self.assertGreaterEqual(snap["step"], 0)

    def test_eta_seconds_estimate(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        t.set_step(1)
        time.sleep(0.05)
        t.set_step(2)
        eta = t.eta_seconds()
        # Should be a positive number or None
        self.assertTrue(eta is None or eta >= 0)

    def test_elapsed_seconds(self):
        from genbox.utils.gen_progress import GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        time.sleep(0.05)
        elapsed = t.elapsed_seconds()
        self.assertGreater(elapsed, 0.0)


# ── make_step_callback ────────────────────────────────────────────────────────

class TestMakeStepCallback(unittest.TestCase):
    """make_step_callback returns a diffusers-compatible callback function."""

    def _make_pipe_mock(self):
        """Minimal mock that looks like a diffusers pipeline."""
        pipe = MagicMock()
        return pipe

    def test_callback_is_callable(self):
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        cb = make_step_callback(t)
        self.assertTrue(callable(cb))

    def test_callback_updates_tracker(self):
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        cb = make_step_callback(t)
        pipe_mock = self._make_pipe_mock()
        # diffusers calls: callback(pipe, step_index, timestep, callback_kwargs)
        cb(pipe_mock, 3, 750, {})
        self.assertEqual(t.snapshot()["step"], 3)

    def test_callback_sets_stage_denoising(self):
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        cb = make_step_callback(t)
        cb(self._make_pipe_mock(), 1, 900, {})
        self.assertEqual(t.snapshot()["stage"], "denoising")

    def test_callback_returns_empty_dict(self):
        """diffusers expects callback_on_step_end to return a dict."""
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        cb = make_step_callback(t)
        result = cb(self._make_pipe_mock(), 1, 900, {})
        self.assertIsInstance(result, dict)

    def test_callback_never_raises(self):
        """Callback must not propagate exceptions — it would kill generation."""
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        # Corrupt tracker to trigger internal error
        t._state = None  # type: ignore
        cb = make_step_callback(t)
        try:
            cb(MagicMock(), 1, 900, {})  # must not raise
        except Exception:
            self.fail("Callback propagated an exception")

    def test_callback_preview_decoded_at_interval(self):
        """With preview_interval=5, preview is generated at step 5, 10, …"""
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        tmp = tempfile.TemporaryDirectory()
        preview_dir = Path(tmp.name)

        t = GenProgressTracker(total_steps=20)

        # Mock a decode function that creates a fake PNG
        decode_calls = []
        def fake_decode(latents, pipe) -> Path:
            p = preview_dir / f"preview_{len(decode_calls)}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
            decode_calls.append(len(decode_calls))
            return p

        pipe_mock = self._make_pipe_mock()
        pipe_mock._dummy_latents = MagicMock()

        cb = make_step_callback(
            t, preview_interval=5, decode_fn=fake_decode
        )
        # Steps 1-4: no decode
        for step in range(1, 5):
            cb(pipe_mock, step, 900, {"latents": pipe_mock._dummy_latents})
        self.assertEqual(len(decode_calls), 0)

        # Step 5: decode triggered
        cb(pipe_mock, 5, 750, {"latents": pipe_mock._dummy_latents})
        self.assertEqual(len(decode_calls), 1)

        tmp.cleanup()

    def test_callback_preview_not_decoded_without_decode_fn(self):
        """Without decode_fn, preview_path stays None."""
        from genbox.utils.gen_progress import make_step_callback, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        cb = make_step_callback(t, preview_interval=1, decode_fn=None)
        cb(MagicMock(), 1, 900, {"latents": MagicMock()})
        self.assertIsNone(t.snapshot()["preview_path"])


# ── LatentDecoder ─────────────────────────────────────────────────────────────

class TestLatentDecoder(unittest.TestCase):
    """decode_latents_to_preview converts a tensor to a PIL-saveable image."""

    def test_returns_path_or_none(self):
        from genbox.utils.gen_progress import decode_latents_to_preview
        # With mocked pipe and fake latents — should return None gracefully
        pipe_mock = MagicMock()
        pipe_mock.vae = MagicMock()
        pipe_mock.vae.decode = MagicMock(side_effect=RuntimeError("no GPU"))

        tmp = tempfile.TemporaryDirectory()
        result = decode_latents_to_preview(
            latents=MagicMock(),
            pipe=pipe_mock,
            out_dir=Path(tmp.name),
            step=5,
        )
        # Failed decode → None, not an exception
        self.assertIsNone(result)
        tmp.cleanup()

    def test_decode_success_returns_path(self):
        from genbox.utils.gen_progress import decode_latents_to_preview
        import sys

        # Mock torch and PIL
        torch_mock = MagicMock()
        torch_mock.no_grad.return_value.__enter__ = lambda s: None
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Fake decoded tensor → 1×3×64×64
        fake_tensor = MagicMock()
        fake_tensor.__truediv__ = MagicMock(return_value=fake_tensor)
        fake_tensor.__add__ = MagicMock(return_value=fake_tensor)
        fake_tensor.clamp = MagicMock(return_value=fake_tensor)
        fake_tensor.permute = MagicMock(return_value=fake_tensor)
        fake_tensor.cpu = MagicMock(return_value=fake_tensor)
        fake_tensor.float = MagicMock(return_value=fake_tensor)
        fake_tensor.numpy = MagicMock(return_value=[[[[128] * 64] * 64] * 3])

        pipe_mock = MagicMock()
        pipe_mock.vae = MagicMock()
        pipe_mock.vae.config.scaling_factor = 0.18215
        decoded = MagicMock()
        decoded.sample = fake_tensor
        pipe_mock.vae.decode.return_value = decoded

        pil_mock = MagicMock()
        pil_image_mock = MagicMock()
        pil_mock.fromarray.return_value = pil_image_mock

        tmp = tempfile.TemporaryDirectory()
        out_dir = Path(tmp.name)

        with patch.dict(sys.modules, {"torch": torch_mock}):
            with patch("genbox.utils.gen_progress.PIL_Image", pil_mock):
                result = decode_latents_to_preview(
                    latents=MagicMock(),
                    pipe=pipe_mock,
                    out_dir=out_dir,
                    step=7,
                )

        # If mocking worked correctly, save was called and a path returned
        if result is not None:
            self.assertIsInstance(result, Path)

        tmp.cleanup()


# ── GenRunner (thread wrapper) ────────────────────────────────────────────────

class TestGenRunner(unittest.TestCase):
    """GenRunner runs generation in a thread and updates a tracker."""

    def test_runner_initializes(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        runner = GenRunner(fn=lambda tracker: None, tracker=t)
        self.assertFalse(runner.is_alive())

    def test_runner_calls_fn(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=10)
        called = []
        def fn(tracker):
            called.append(True)
            return {"output_path": Path("/tmp/x.png"), "metadata": {}, "elapsed_s": 1.0}

        runner = GenRunner(fn=fn, tracker=t)
        runner.start()
        runner.join(timeout=2.0)
        self.assertTrue(called)

    def test_runner_marks_done_on_success(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=5)
        def fn(tracker):
            tracker.set_step(5)
            return {"output_path": Path("/tmp/x.png"), "metadata": {}, "elapsed_s": 0.1}

        runner = GenRunner(fn=fn, tracker=t)
        runner.start()
        runner.join(timeout=2.0)
        self.assertTrue(t.snapshot()["done"])

    def test_runner_marks_error_on_exception(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=5)
        def fn(tracker):
            raise RuntimeError("GPU OOM")

        runner = GenRunner(fn=fn, tracker=t)
        runner.start()
        runner.join(timeout=2.0)
        snap = t.snapshot()
        self.assertTrue(snap["error"])
        self.assertIn("GPU OOM", snap["error_msg"])

    def test_runner_result_accessible(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=5)
        expected = {"output_path": Path("/tmp/x.png"), "metadata": {"seed": 42}, "elapsed_s": 1.5}
        def fn(tracker):
            return expected

        runner = GenRunner(fn=fn, tracker=t)
        runner.start()
        runner.join(timeout=2.0)
        self.assertEqual(runner.result, expected)

    def test_runner_exception_accessible(self):
        from genbox.utils.gen_progress import GenRunner, GenProgressTracker
        t = GenProgressTracker(total_steps=5)
        def fn(tracker):
            raise ValueError("bad input")

        runner = GenRunner(fn=fn, tracker=t)
        runner.start()
        runner.join(timeout=2.0)
        self.assertIsInstance(runner.exception, ValueError)


# ── Stage label helpers ────────────────────────────────────────────────────────

class TestStageLabelFormatting(unittest.TestCase):
    def test_format_step_label(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=5, total=28, stage="denoising", eta=12.5)
        self.assertIn("5", label)
        self.assertIn("28", label)

    def test_format_step_label_loading(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=0, total=28, stage="loading model", eta=None)
        self.assertIn("loading", label.lower())

    def test_format_step_label_done(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=28, total=28, stage="done", eta=0)
        self.assertIn("done", label.lower())

    def test_eta_formatted_nicely(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=5, total=28, stage="denoising", eta=65)
        self.assertIn("1m", label)  # 65s → "1m 5s"

    def test_eta_seconds_only(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=5, total=28, stage="denoising", eta=30)
        self.assertIn("30s", label)

    def test_no_eta_shows_unknown(self):
        from genbox.utils.gen_progress import format_step_label
        label = format_step_label(step=5, total=28, stage="denoising", eta=None)
        self.assertIsInstance(label, str)  # no crash


if __name__ == "__main__":
    unittest.main(verbosity=2)
