"""CPU-only checks that diagnostic collection never synchronizes between chunks."""
import unittest

from engine.prefill_timing import PrefillTiming


class TestPrefillTiming(unittest.TestCase):
    def test_deferred_report(self):
        calls = []
        stamps = iter((10, 20, 25, 40))

        class Event:
            def record(self):
                self.stamp = next(stamps)
                calls.append('record')

            def synchronize(self):
                calls.append('sync')

            def elapsed_time(self, other):
                return other.stamp - self.stamp

        host = iter((0, 1, 2, 3, 5))
        timer = PrefillTiming(1, Event, lambda: next(host))
        with timer.chunk(0, 2048):
            pass
        with timer.chunk(2048, 2108):
            pass
        self.assertEqual(calls, ['record'] * 4)
        report = timer.report()
        self.assertEqual(calls, ['record'] * 4 + ['sync'])
        self.assertEqual(report['rank'], 1)
        first, second = report['chunks']
        self.assertEqual(first['tokens'], 2048)
        self.assertIsNone(first['gpu_gap_ms'])
        self.assertEqual(second['tokens'], 60)
        self.assertEqual(second['host_call_ms'], 2000)
        self.assertEqual(second['host_gap_ms'], 1000)
        self.assertEqual(second['gpu_envelope_ms'], 15)
        self.assertEqual(second['gpu_gap_ms'], 5)

    def test_empty(self):
        timer = PrefillTiming(0, lambda: self.fail('no events for empty prefill'))
        self.assertEqual(timer.report(), {'rank': 0, 'chunks': []})


if __name__ == '__main__':
    unittest.main()
