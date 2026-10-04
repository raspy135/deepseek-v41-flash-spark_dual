"""Rolling window, threshold, cooldown and bounded-memory tests (no GPU)."""
import unittest
from engine.urgent_adapt import UrgentAdaptWindow
from engine.adapt_config import resolve


class UrgentTest(unittest.TestCase):
    def test_strict_threshold_and_full_window(self):
        w = UrgentAdaptWindow(1, (1000, 10000))
        self.assertIsNone(w.observe(30, (1029, 10290)))
        self.assertFalse(w.observe(31, (1030, 10300))['triggered'])
        r = w.observe(32, (1032, 10310))
        self.assertTrue(r['triggered'])
        self.assertEqual(r['window_tokens'], 31)

    def test_trailing_window_forgets_old_bad_period(self):
        w = UrgentAdaptWindow(0, (0, 0))
        for n in range(1, 31):
            r = w.observe(n, (n*2, n*10))
        self.assertTrue(r['triggered'])
        for n in range(31, 61):
            r = w.observe(n, (60, n*10))
        self.assertEqual(r['miss_rate'], 0)
        self.assertFalse(r['triggered'])
        self.assertLessEqual(len(w.samples), 31)

    def test_cooldown_noop_attempt_and_bursts(self):
        w = UrgentAdaptWindow(1, (0, 0))
        r = None
        for n in range(7, 38, 6):
            r = w.observe(n, ((n-1)*2, (n-1)*10))
        self.assertTrue(r['triggered'])
        self.assertEqual(r['window_tokens'], 30)
        w.attempted(37, (72, 360))
        for n in range(43, 187, 6):
            r = w.observe(n, ((n-1)*2, (n-1)*10))
            if r is not None: self.assertFalse(r['triggered'])
        self.assertTrue(w.observe(187, (372, 1860))['triggered'])

    def test_missing_reset_and_warmup(self):
        w = UrgentAdaptWindow(1, None)
        self.assertIsNone(w.observe(10, (10, 100)))
        self.assertIsNone(w.observe(20, None))
        self.assertIsNone(w.observe(30, (20, 200)))
        w.attempted(30, (20, 200))
        w.restart(40, (10000, 20000))
        self.assertEqual(w.last_attempt, 30)
        self.assertIsNone(w.observe(50, (10000, 20100)))
        self.assertEqual(w.observe(70, (10000, 20300))['miss_rate'], 0)

    def test_configuration_and_guard(self):
        env={'DSV41_ADAPT_SENSITIVITY':'high'}
        c=resolve(env)
        self.assertTrue(c.urgent)
        self.assertEqual((c.urgent_window,c.urgent_miss,c.urgent_max,c.urgent_cooldown),(30,.1,64,150))
        self.assertTrue(c.boot_fields()['adapt_urgent_v1'])
        for override in ({'DSV41_ADAPT_URGENT':'0'}, {'DSV41_ADAPT_DECODE_TOKENS':'0'},
                         {'DSV41_PRUNE_SWAP':'0'}, {'DSV41_ADAPT_SENSITIVITY':'off'}):
            self.assertFalse(resolve(env|override).urgent)
        self.assertFalse(resolve({}).urgent)
        with self.assertRaises(ValueError): resolve(env|{'DSV41_ADAPT_URGENT':'yes'})


if __name__=='__main__': unittest.main()
