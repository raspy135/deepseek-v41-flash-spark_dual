"""Scheduler lifecycle tests, without loading CUDA or model weights."""

import threading
import unittest
from types import SimpleNamespace

from engine.decode_events import VerifyStep
from server.concurrency import Scheduler, RequestStream


class FakeRuntime:
    def __init__(self):
        self.events, self.generators, self.stats, self.actions = {}, {}, {}, []
        self.verified = set()

    def execute(self, action):
        self.actions.append(action)
        op, lane = action['op'], action.get('lane')
        if op == 'start':
            self.generators[lane] = iter([VerifyStep(None, 0, None), [action['prompt_ids'][0] + 1]])
            event = action['prompt_ids']
        elif op == 'verify':
            self.verified.update(action['lanes'])
            return None
        elif op == 'close':
            event = None
        else:
            if isinstance(self.events[lane], VerifyStep):
                assert lane in self.verified
                self.verified.remove(lane)
            event = next(self.generators[lane], None)
        self.events[lane] = event
        if event is None:
            del self.generators[lane]
            self.stats[lane] = {'lane': lane}
        return event


def make_scheduler():
    runtime = FakeRuntime()
    state = SimpleNamespace(lock=threading.Lock(), ep_fault=None,
                            ep=SimpleNamespace(broadcast_request=lambda x: None,
                                               gather_objects=lambda x: [x, x]),
                            maintain_experts=lambda: None)
    # Hold the GPU lock so tests can enqueue a pair deterministically.
    state.lock.acquire()
    return Scheduler(state, runtime), state, runtime


class TestScheduler(unittest.TestCase):
    def test_pair_and_lane_reuse(self):
        scheduler, state, runtime = make_scheduler()
        a = scheduler.submit({'prompt_ids': [10], 'kwargs': {}})
        b = scheduler.submit({'prompt_ids': [20], 'kwargs': {}})
        state.lock.release()
        self.assertTrue(a.finished.wait(2))
        self.assertTrue(b.finished.wait(2))
        self.assertEqual(list(a), [[10], [11]])
        self.assertEqual(list(b), [[20], [21]])
        self.assertIn({'op': 'verify', 'lanes': [0, 1]}, runtime.actions)
        c = scheduler.submit({'prompt_ids': [30], 'kwargs': {}})
        self.assertTrue(c.finished.wait(2))
        self.assertEqual(list(c), [[30], [31]])
        self.assertIsNone(state.ep_fault)

    def test_cancel_before_admission(self):
        scheduler, state, runtime = make_scheduler()
        job = scheduler.submit({'prompt_ids': [10], 'kwargs': {}})
        job.cancelled.set()
        state.lock.release()
        self.assertTrue(job.finished.wait(2))
        self.assertEqual(list(job), [])
        self.assertEqual(runtime.actions, [])

    def test_failure_unblocks_consumers(self):
        scheduler, state, runtime = make_scheduler()
        runtime.execute = lambda _: (_ for _ in ()).throw(ValueError('test failure'))
        jobs = [scheduler.submit({'prompt_ids': [10], 'kwargs': {}}) for _ in range(3)]
        state.lock.release()
        for job in jobs:
            self.assertTrue(job.finished.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                next(job)
            job.close()
        self.assertIn('test failure', state.ep_fault)

    def test_cancel_active_lane_keeps_peer_and_queue_running(self):
        scheduler, state, runtime = make_scheduler()
        jobs = [scheduler.submit({'prompt_ids': [10 * i], 'kwargs': {}}) for i in (1, 2, 3)]
        execute = runtime.execute

        def cancelling(action):
            if action['op'] == 'verify' and action['lanes'] == [0, 1]:
                jobs[0].cancelled.set()
            return execute(action)

        runtime.execute = cancelling
        state.lock.release()
        for job in jobs:
            self.assertTrue(job.finished.wait(2))
        self.assertEqual(list(jobs[1]), [[20], [21]])
        self.assertEqual(list(jobs[2]), [[30], [31]])
        self.assertIn({'op': 'close', 'lane': 0}, runtime.actions)
        self.assertIsNone(state.ep_fault)

    def test_stream_eof_is_stable(self):
        job = RequestStream({})
        job.finish({'done': True})
        self.assertEqual(list(job), [])
        self.assertIsNone(next(job, None))
        job.close()

    def test_shutdown_cancels_queued_work(self):
        scheduler, state, runtime = make_scheduler()
        jobs = [scheduler.submit({'prompt_ids': [i], 'kwargs': {}}) for i in range(3)]
        scheduler.stop()
        state.lock.release()
        scheduler.thread.join(2)
        self.assertFalse(scheduler.thread.is_alive())
        self.assertTrue(all(job.finished.is_set() for job in jobs))
        self.assertEqual(runtime.actions, [])
        with self.assertRaisesRegex(RuntimeError, 'shutting down'):
            scheduler.submit({})


if __name__ == '__main__':
    unittest.main()
