"""CPU-only checks that diagnostic collection never synchronizes between chunks."""
import unittest

from engine.prefill_timing import PrefillTiming, PrefillMoeTiming


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

    def test_moe_subphases_defer_sync_and_keep_calls(self):
        operations = []
        stamps = iter((0, 2, 5, 10, 14))

        class Event:
            def record(self):
                self.stamp = next(stamps)
                operations.append('record')

            def synchronize(self):
                operations.append('sync')

            def elapsed_time(self, other):
                return other.stamp - self.stamp

        host = iter((0, .001, .003, .010, .012))
        timer = PrefillMoeTiming(1, Event, lambda: next(host))
        timer.start(0, 2048)
        timer.mark('grouping')
        timer.mark('up_gemm')
        timer.start(1, 128)
        timer.mark('grouping')
        self.assertEqual(operations, ['record'] * 5)
        report = timer.report()
        self.assertEqual(operations, ['record'] * 5 + ['sync'])
        self.assertEqual(report['totals']['grouping'],
                         {'host_ms': 3.0, 'gpu_envelope_ms': 6.0})
        self.assertEqual(report['totals']['up_gemm'],
                         {'host_ms': 2.0, 'gpu_envelope_ms': 3.0})
        self.assertEqual([r['tokens'] for r in report['calls']], [2048, 128])

    def test_empty_moe(self):
        timer = PrefillMoeTiming(0, lambda: self.fail('no events for empty prefill'))
        self.assertEqual(timer.report(), {'rank': 0, 'totals': {}, 'calls': []})

    def test_engine_dispatch_forwards_timing_only_when_enabled(self):
        # Exercise the actual nested dispatcher without constructing model weights.
        import ast
        import inspect
        import textwrap
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        from engine.v41_engine import V41Engine
        init = ast.parse(textwrap.dedent(inspect.getsource(V41Engine.__init__)))
        dispatch = next(n for n in init.body[0].body
                        if isinstance(n, ast.FunctionDef) and n.name == 'moe_fn')
        target, callback = Mock(), Mock()
        ns = dict(torch=torch, self=SimpleNamespace(kernel='triton-fp4'),
                  cb3_cls=None, fp4_moe_fn=target)
        exec(compile(ast.Module(body=[dispatch], type_ignores=[]), '<dispatcher-test>', 'exec'), ns)
        ns['moe_fn'](None, None, None, None, 10, stage_mark=callback)
        self.assertIs(target.call_args.kwargs['stage_mark'], callback)
        target.reset_mock()
        ns['moe_fn'](None, None, None, None, 10)
        self.assertNotIn('stage_mark', target.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
