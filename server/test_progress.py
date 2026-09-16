import unittest
from unittest.mock import Mock

from server.progress import DecodeProgress


class DecodeProgressTests(unittest.TestCase):
    def test_interval_thinking_transition_and_throughput(self):
        logger = Mock()
        clock = Mock(side_effect=[0., 20., 25., 30., 40.])
        p = DecodeProgress(logger, 100, 200, True, 99, clock=clock)
        p.update([1, 2])  # first output latency includes prefill
        p.update([3, 4])
        self.assertEqual(logger.info.call_count, 2)
        p.update([5, 99, 6])
        self.assertEqual(p.reasoning, 5)
        self.assertFalse(p.thinking)
        args = logger.info.call_args.args
        self.assertEqual(args[2:6], ('answer', 7, 200, 5))
        self.assertEqual(args[-1], .5)  # five tokens since first output, ten seconds
        p.update([7])
        self.assertEqual(logger.info.call_args.args[-1], .1)
        self.assertEqual(p.reasoning, 5)

    def test_empty_burst_and_non_thinking(self):
        logger = Mock()
        p = DecodeProgress(logger, 10, 20, False, 99, clock=Mock(side_effect=[0., 1.]))
        p.update([])
        self.assertIsNone(p.first)
        p.update([1, 99])
        self.assertEqual(p.reasoning, 0)
        self.assertEqual(p.total, 2)

    def test_requests_have_independent_counters_and_ids(self):
        a = DecodeProgress(Mock(), 10, 20, True, 99, clock=lambda: 0.)
        b = DecodeProgress(Mock(), 10, 20, False, 99, clock=lambda: 0.)
        a.update([1])
        self.assertEqual(b.total, 0)
        self.assertNotEqual(a.request_id, b.request_id)


if __name__ == '__main__':
    unittest.main()
