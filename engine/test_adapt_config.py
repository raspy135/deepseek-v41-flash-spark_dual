"""engine/adapt_config.py: the two adaptation knobs and the legacy settings they replace.

    python3 -m unittest engine.test_adapt_config
"""
from __future__ import annotations

import math
import unittest

from engine.adapt_config import resolve

# The 2026-09 serving profile, spelled the old way (.env.example before the knobs).
LEGACY_PROFILE = {
    "DSV41_PRUNE_MISS": "1", "DSV41_PRUNE_UNIT": "request", "DSV41_PRUNE_PRIOR": "8",
    "DSV41_PRUNE_DB": "results/prune_demand_req.npz", "DSV41_PRUNE_SWAP": "1",
    "DSV41_PRUNE_SWAP_MAX": "512", "DSV41_PRUNE_SWAP_MIN_GROWTH": "0",
    "DSV41_PRUNE_SWAP_MIN_GAIN": "0.005", "DSV41_PRUNE_SWAP_PREFILL": "1",
    "DSV41_PRUNE_SWAP_PREFILL_MIN": "32", "DSV41_PRUNE_SWAP_PREFILL_MIN_MISS": "0.02",
    "DSV41_PRUNE_HALFLIFE": "20",
}
SAME = ("record", "request_unit", "db_path", "prior", "halflife", "swap", "swap_prefill",
        "swap_prefill_min", "swap_prefill_min_miss", "swap_max", "swap_min_gain", "min_growth_req")


class AdaptConfigTest(unittest.TestCase):
    def test_no_knobs_keeps_legacy_defaults(self):
        c = resolve({})
        self.assertEqual(c.source, "legacy")
        self.assertEqual((c.record, c.request_unit, c.swap, c.swap_prefill), (False, False, False, False))
        self.assertEqual((c.prior, c.halflife, c.swap_max, c.swap_min_gain), (2e7, 2e7, 64, 0.05))
        self.assertEqual((c.swap_prefill_min, c.swap_prefill_min_miss), (1024, 0.10))
        self.assertEqual(c.db_path, "results/prune_demand.npz")

    def test_legacy_variables_still_read(self):
        c = resolve(LEGACY_PROFILE)
        self.assertEqual(c.source, "legacy")
        self.assertEqual((c.halflife, c.prior, c.swap_max), (20.0, 8.0, 512))

    def test_medium_is_the_serving_profile(self):
        old, new = resolve(LEGACY_PROFILE), resolve({"DSV41_ADAPT_SENSITIVITY": "medium"})
        for k in SAME:
            self.assertAlmostEqual(getattr(new, k), getattr(old, k), msg=k) \
                if isinstance(getattr(old, k), float) else self.assertEqual(getattr(new, k), getattr(old, k), k)
        self.assertEqual(new.boot_fields()["prune_swap_prefill"], old.boot_fields()["prune_swap_prefill"])

    def test_levels_and_numbers(self):
        for level, half in (("low", 40), ("medium", 20), ("high", 10), ("max", 5)):
            c = resolve({"DSV41_ADAPT_SENSITIVITY": level})
            self.assertAlmostEqual(c.halflife, half)
            self.assertAlmostEqual(c.sensitivity, 1 - 0.5 ** (1 / half))
        c = resolve({"DSV41_ADAPT_SENSITIVITY": "0.1"})
        self.assertAlmostEqual(c.halflife, math.log(0.5) / math.log(0.9))
        self.assertIsNone(c.level)
        # the prefill miss gate scales inversely with sensitivity: 2% at medium, 1% at high
        self.assertAlmostEqual(resolve({"DSV41_ADAPT_SENSITIVITY": "high"}).swap_prefill_min_miss, 0.02 * 0.0341 / 0.0670, places=3)

    def test_off_freezes_but_records(self):
        for v in ("off", "0"):
            c = resolve({"DSV41_ADAPT_SENSITIVITY": v})
            self.assertFalse(c.swap or c.swap_prefill)
            self.assertTrue(c.record and c.use_db)

    def test_prior_alone_implies_medium(self):
        c = resolve({"DSV41_ADAPT_PRIOR": "4"})
        self.assertEqual((c.source, c.prior, c.level), ("knobs", 4.0, "medium"))

    def test_replaced_settings_are_reported_not_read(self):
        c = resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_PRUNE_HALFLIFE": "20",
                     "DSV41_PRUNE_SWAP_MAX": "64", "DSV41_PRUNE_DB": "x.npz"})
        self.assertAlmostEqual(c.halflife, 10)
        self.assertEqual(c.swap_max, 512)
        self.assertEqual(c.ignored, ("DSV41_PRUNE_HALFLIFE", "DSV41_PRUNE_SWAP_MAX"))
        self.assertEqual(c.db_path, "x.npz")          # the DB path is not a tuning knob; still honored
        self.assertIn("ignoring DSV41_PRUNE_HALFLIFE", c.describe())

    def test_scripts_can_still_freeze(self):
        # the benchmarks set these in-process; with knobs in .env they must still freeze
        c = resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_PRUNE_SWAP": "0"})
        self.assertFalse(c.swap or c.swap_prefill)
        self.assertNotIn("DSV41_PRUNE_SWAP", c.ignored)
        c = resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_PRUNE_SWAP_PREFILL": "0"})
        self.assertTrue(c.swap)
        self.assertFalse(c.swap_prefill)
        c = resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_PRUNE_SWAP": "1"})
        self.assertTrue(c.swap)
        self.assertIn("DSV41_PRUNE_SWAP", c.ignored)   # a leftover "on" is simply redundant

    def test_bad_values(self):
        for env in ({"DSV41_ADAPT_SENSITIVITY": "fast"}, {"DSV41_ADAPT_SENSITIVITY": "0.5"},
                    {"DSV41_ADAPT_SENSITIVITY": "-1"}, {"DSV41_ADAPT_PRIOR": "-2"},
                    {"DSV41_ADAPT_PRIOR": "many"}):
            with self.assertRaises(ValueError, msg=env):
                resolve(env)


if __name__ == "__main__":
    unittest.main()
