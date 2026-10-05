import math
import os
import unittest
from unittest.mock import patch

from engine.draft_bypass import DraftBypassPolicy, enabled


class BypassTest(unittest.TestCase):
    def test_off_and_invalid_setting(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
            os.environ['DSV41_DRAFT_BYPASS'] = '1'
            self.assertTrue(enabled())
            os.environ['DSV41_DRAFT_BYPASS'] = 'yes'
            with self.assertRaises(ValueError):
                enabled()

    def test_capture_samples_do_not_price_mode(self):
        p = DraftBypassPolicy()
        for seconds in (None, 0, -1, math.nan, math.inf):
            p.observe(False, 1, seconds)
        self.assertFalse(p.decide())
        self.assertEqual(len(p.spec_samples), 0)

    def test_probe_then_choose_measured_faster_mode(self):
        p = DraftBypassPolicy()
        for _ in range(16):
            p.observe(False, 1, .1)
        self.assertTrue(p.decide())
        p.observe(True, 1, .06)
        self.assertTrue(p.decide())
        p.observe(True, 1, .2)
        self.assertFalse(p.decide())

    def test_many_accepted_tokens_keep_drafting(self):
        p = DraftBypassPolicy()
        for _ in range(32):
            p.observe(False, 5, .1)
        self.assertFalse(p.decide())
        p.observe(True, 1, .05)
        self.assertFalse(p.decide())

    def test_refresh_and_reset(self):
        p = DraftBypassPolicy()
        for _ in range(16):
            p.observe(False, 1, .1)
        for _ in range(16):
            p.observe(True, 1, .05)
        self.assertFalse(p.decide())
        p.pinned = True
        self.assertTrue(p.decide())
        p.reset_request()
        self.assertEqual(p.steps, {'draft': 0, 'bypass': 0})
        self.assertEqual(len(p.bypass_samples), 0)


if __name__ == '__main__':
    unittest.main()
