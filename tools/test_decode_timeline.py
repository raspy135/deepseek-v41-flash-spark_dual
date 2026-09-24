import unittest
import json
import os
import tempfile
from concurrent.futures import Future
from types import SimpleNamespace
from tools.bench_decode_timeline_analysis import analyze, intersection, merged
from tools.bench_decode_timeline_hooks import DecodeTimeline


class TimelineTests(unittest.TestCase):
    def test_unions(self):
        self.assertEqual(merged([(0, 4), (2, 6), (9, 10)]), [(0, 6), (9, 10)])
        self.assertEqual(intersection([(0, 6)], [(2, 4), (5, 8)]), [(2, 4), (5, 6)])

    def test_overlap_is_not_double_counted(self):
        def event(name, cat, start, duration):
            return dict(name=name, cat=cat, ts=start, dur=duration)
        report = analyze({'traceEvents': [event('decode/window', 'decode_host', 0, 10000),
            event('_moe_up_kernel', 'kernel', 1000, 4000),
            event('nccl_AllGather', 'kernel', 3000, 4000),
            event('copy', 'gpu_memcpy', 8000, 1000),
            event('engram/1/read_raw', 'decode_host', 0, 2000)]})
        self.assertEqual(report['gpu_busy_union_ms'], 7)
        self.assertEqual(report['gpu_idle_ms'], 3)
        self.assertEqual(report['host_spans']['engram/1/read_raw']['overlaps_gpu_idle_ms'], 1)

    def test_missing_gpu_events_rejected(self):
        with self.assertRaises(ValueError):
            analyze({'traceEvents': []})

    def test_trim_excludes_first_and_last_iterations(self):
        events = [dict(name='decode/control', cat='decode_host', ts=t, dur=1)
                  for t in (0, 1000, 2000, 3000)]
        events.append(dict(name='kernel', cat='kernel', ts=0, dur=4000))
        report = analyze({'traceEvents': events}, trim_edges=True)
        self.assertEqual(report['window_ms'], 2)
        self.assertEqual(report['gpu_busy_union_ms'], 2)
        self.assertEqual(report['gpu_idle_ms'], 0)

    def test_hooks_restore_after_error(self):
        class Fake:
            def step(self): return 7
            def draft(self): return 8
            def capture(self): return 9
        fast = Fake()
        hook = DecodeTimeline(SimpleNamespace(fast=fast, tables={}))
        with self.assertRaises(RuntimeError):
            with hook:
                self.assertEqual(fast.step(), 7)
                self.assertEqual(hook.events, [])
                hook.active = True
                self.assertEqual(fast.step(), 7)
                raise RuntimeError('test')
        self.assertNotIn('step', vars(fast))
        self.assertEqual(hook.events[0]['name'], 'decode/step')

    def test_future_result_preserved(self):
        hook = DecodeTimeline(None)
        future = Future()
        hook.watch_future(future)
        future.set_result(17)
        hook.active = True
        self.assertEqual(future.result(timeout=0), 17)
        self.assertEqual(hook.events[0]['name'], 'engram/future_wait')

    def test_private_export_and_clock_conversion(self):
        class Profiler:
            def export_chrome_trace(self, path):
                os.unlink(path)  # Kineto can replace a pre-reserved output file.
                with open(path, 'w') as f:
                    json.dump(dict(baseTimeNanoseconds=1000000, traceEvents=[]), f)
        hook = DecodeTimeline(None)
        hook.events = [dict(name='test', ts_ns=1002000, dur=1)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'trace.json')
            hook.export(Profiler(), path, {'rank': 0})
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            with open(path) as f:
                self.assertEqual(json.load(f)['traceEvents'][0]['ts'], 2)
            with self.assertRaises(FileExistsError):
                hook.export(Profiler(), path, {})


if __name__ == '__main__':
    unittest.main()
