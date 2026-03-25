"""
tests/test_ui.py
Tests for genbox/ui.py helper functions — no Streamlit runtime required.
All st.* calls are mocked at import time.
Run: python -m unittest tests.test_ui -v
"""

import json
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Mock streamlit before importing ui ────────────────────────────────────────

_st_mock = MagicMock()
_st_mock.set_page_config = MagicMock()
_st_mock.markdown = MagicMock()
_st_mock.session_state = {}

sys.modules["streamlit"] = _st_mock

# now safe to import ui helpers
import genbox.genbox_ui.ui as ui_mod


# ══════════════════════════════════════════════════════════════════════════════
# _ts()
# ══════════════════════════════════════════════════════════════════════════════

class TestTs(unittest.TestCase):
    def test_returns_string(self):
        result = ui_mod._ts()
        self.assertIsInstance(result, str)

    def test_format_hh_mm_ss(self):
        result = ui_mod._ts()
        parts = result.split(":")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit(), f"non-digit in timestamp part: {p!r}")


# ══════════════════════════════════════════════════════════════════════════════
# _logline()
# ══════════════════════════════════════════════════════════════════════════════

class TestLogline(unittest.TestCase):
    def test_contains_message(self):
        line = ui_mod._logline("hello world")
        self.assertIn("hello world", line)

    def test_contains_timestamp(self):
        line = ui_mod._logline("test")
        self.assertIn("ts", line)

    def test_ok_class_applied(self):
        line = ui_mod._logline("done", "ok")
        self.assertIn('class="ok"', line)

    def test_err_class_applied(self):
        line = ui_mod._logline("failed", "err")
        self.assertIn('class="err"', line)

    def test_accent_class_applied(self):
        line = ui_mod._logline("info", "accent")
        self.assertIn('class="accent"', line)

    def test_no_class_for_empty_kind(self):
        line = ui_mod._logline("neutral")
        self.assertNotIn('class="ok"', line)
        self.assertNotIn('class="err"', line)

    def test_ends_with_br(self):
        line = ui_mod._logline("test")
        self.assertTrue(line.strip().endswith("<br>"))

    def test_unknown_kind_no_class(self):
        line = ui_mod._logline("msg", "unknown_kind")
        # should not inject a class for unknown kinds
        self.assertNotIn('class="unknown_kind"', line)


# ══════════════════════════════════════════════════════════════════════════════
# _vram_color()
# ══════════════════════════════════════════════════════════════════════════════

class TestVramColor(unittest.TestCase):
    def _entry(self, vram_min):
        e = MagicMock()
        e.vram_min_gb = vram_min
        e.fits_vram = lambda v: v >= vram_min
        return e

    def test_no_cfg_returns_ok(self):
        with patch("genbox.genbox_ui.ui.cfg", None):
            result = ui_mod._vram_color(self._entry(24))
        self.assertEqual(result, "vram-ok")

    def test_fits_returns_ok(self):
        mock_cfg = MagicMock()
        mock_cfg.vram_gb = 12
        with patch("genbox.genbox_ui.ui.cfg", mock_cfg):
            result = ui_mod._vram_color(self._entry(10))
        self.assertEqual(result, "vram-ok")

    def test_slight_over_returns_warn(self):
        mock_cfg = MagicMock()
        mock_cfg.vram_gb = 12
        with patch("genbox.genbox_ui.ui.cfg", mock_cfg):
            result = ui_mod._vram_color(self._entry(13))
        self.assertEqual(result, "vram-warn")

    def test_way_over_returns_err(self):
        mock_cfg = MagicMock()
        mock_cfg.vram_gb = 8
        with patch("genbox.genbox_ui.ui.cfg", mock_cfg):
            result = ui_mod._vram_color(self._entry(24))
        self.assertEqual(result, "vram-err")

    def test_exact_fit_is_ok(self):
        mock_cfg = MagicMock()
        mock_cfg.vram_gb = 12
        with patch("genbox.genbox_ui.ui.cfg", mock_cfg):
            result = ui_mod._vram_color(self._entry(12))
        self.assertEqual(result, "vram-ok")


# ══════════════════════════════════════════════════════════════════════════════
# _stars()
# ══════════════════════════════════════════════════════════════════════════════

class TestStars(unittest.TestCase):
    def test_zero(self):
        s = ui_mod._stars(0)
        self.assertEqual(s.count("★"), 0)
        self.assertEqual(s.count("☆"), 5)

    def test_five(self):
        s = ui_mod._stars(5)
        self.assertEqual(s.count("★"), 5)
        self.assertEqual(s.count("☆"), 0)

    def test_three(self):
        s = ui_mod._stars(3)
        self.assertEqual(s.count("★"), 3)
        self.assertEqual(s.count("☆"), 2)

    def test_always_five_total(self):
        for n in range(6):
            s = ui_mod._stars(n)
            self.assertEqual(s.count("★") + s.count("☆"), 5)


# ══════════════════════════════════════════════════════════════════════════════
# _run_pipeline_code()
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipelineCode(unittest.TestCase):
    def test_simple_print(self):
        stdout, stderr = ui_mod._run_pipeline_code('print("genbox ok")')
        self.assertIn("genbox ok", stdout)
        self.assertEqual(stderr.strip(), "")

    def test_syntax_error_in_stderr(self):
        stdout, stderr = ui_mod._run_pipeline_code("def broken(: pass")
        self.assertNotEqual(stderr, "")

    def test_empty_code_no_crash(self):
        stdout, stderr = ui_mod._run_pipeline_code("")
        self.assertIsInstance(stdout, str)
        self.assertIsInstance(stderr, str)

    def test_exception_in_stderr(self):
        stdout, stderr = ui_mod._run_pipeline_code("raise ValueError('intentional')")
        self.assertIn("intentional", stderr)

    def test_math_output(self):
        stdout, stderr = ui_mod._run_pipeline_code("print(2 + 2)")
        self.assertIn("4", stdout)


# ══════════════════════════════════════════════════════════════════════════════
# Template coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestTemplates(unittest.TestCase):
    def test_all_templates_have_pipeline_call(self):
        for name, code in ui_mod._TEMPLATES.items():
            with self.subTest(template=name):
                self.assertIn("pipeline.", code)

    def test_all_templates_are_strings(self):
        for name, code in ui_mod._TEMPLATES.items():
            with self.subTest(template=name):
                self.assertIsInstance(code, str)
                self.assertGreater(len(code), 10)

    def test_template_keys_are_unique(self):
        keys = list(ui_mod._TEMPLATES.keys())
        self.assertEqual(len(keys), len(set(keys)))

    def test_batch_template_has_loop(self):
        code = ui_mod._TEMPLATES.get("Batch variations", "")
        self.assertIn("for", code)

    def test_t2i_template_has_model(self):
        code = ui_mod._TEMPLATES.get("Text → Image (FLUX.2 Klein)", "")
        self.assertIn("flux2_klein", code)

    def test_video_template_has_frames(self):
        code = ui_mod._TEMPLATES.get("Text → Video (WAN 1.3B)", "")
        self.assertIn("frames", code)


# ══════════════════════════════════════════════════════════════════════════════
# CSS / theme
# ══════════════════════════════════════════════════════════════════════════════

class TestTheme(unittest.TestCase):
    def test_css_contains_background_color(self):
        self.assertIn("#0d0d0d", ui_mod.THEME_CSS)

    def test_css_contains_accent_color(self):
        self.assertIn("#4a9eff", ui_mod.THEME_CSS)

    def test_css_contains_info_accent(self):
        self.assertIn("#2dd4a0", ui_mod.THEME_CSS)

    def test_css_contains_font(self):
        self.assertIn("JetBrains Mono", ui_mod.THEME_CSS)

    def test_css_no_gradients(self):
        # design spec: no gradients
        self.assertNotIn("linear-gradient", ui_mod.THEME_CSS)
        self.assertNotIn("radial-gradient", ui_mod.THEME_CSS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
