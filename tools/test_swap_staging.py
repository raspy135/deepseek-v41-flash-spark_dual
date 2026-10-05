"""CPU tests for bounded expert reads; no model, CUDA or service is needed."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.swap_staging import StaleSwapPlan, SwapStager


PART_BYTES = 32
EXPERT_BYTES = 6 * PART_BYTES


def payload(layer, expert):
    return tuple(bytes([(layer * 7 + expert * 3 + i) % 256]) * PART_BYTES for i in range(6))


def swaps(count=6):
    return [(0, i, 10 + i, 1.0 / (i + 1)) for i in range(count)]


class SwapStagingTest(unittest.TestCase):
    def stager(self, reader=payload, capacity=2, workers=2):
        stager = SwapStager(reader, bytes_per_expert=EXPERT_BYTES,
                            max_bytes=EXPERT_BYTES * capacity, workers=workers)
        self.addCleanup(stager.close)
        return stager

    def test_exact_file_bytes_and_original_order_at_install_boundary(self):
        # A real CPU file read, not a reader returning references to a mutable pool.
        with tempfile.TemporaryDirectory() as directory:
            for _, _, incoming, _ in swaps():
                Path(directory, str(incoming)).write_bytes(b"".join(payload(0, incoming)))
            reads = []
            def read(layer, expert):
                reads.append((layer, expert))
                data = Path(directory, str(expert)).read_bytes()
                return tuple(data[i * PART_BYTES:(i + 1) * PART_BYTES] for i in range(6))
            stager = self.stager(read)
            plan = stager.start(swaps(), generation=4)
            self.assertEqual(plan.wait(swaps(), 4), 2)
            arena = {(0, i): payload(0, i) for i in range(6)}
            before = dict(arena)
            self.assertEqual(arena, before)  # staging has no reference to the arena
            self.assertCountEqual(reads, [(0, 10), (0, 11)])
            installed = []
            for layer, outgoing, incoming, _ in swaps():
                with plan.take((layer, incoming), swaps(), 4) as record:
                    self.assertEqual(record.views, payload(layer, incoming))
                    arena[layer, incoming] = record.views
                    del arena[layer, outgoing]
                    installed.append(incoming)
            self.assertEqual(installed, list(range(10, 16)))
            self.assertEqual(stager.reserved_bytes, 0)
            self.assertLessEqual(stager.peak_reserved_bytes, 2 * EXPERT_BYTES)

    def test_stages_only_owned_keys_and_keeps_full_plan_identity(self):
        stager = self.stager()
        plan = stager.start(swaps(), 0, world=2, rank=1)
        self.assertEqual(plan.keys, ((0, 11), (0, 13), (0, 15)))
        plan.wait(swaps(), 0)
        self.assertIsNone(plan.take((0, 10), swaps(), 0))
        for key in plan.keys:
            plan.take(key, swaps(), 0).release()
        self.assertEqual(stager.reserved_bytes, 0)

    def test_failure_preflight_does_not_claim_or_install_other_records(self):
        def read(layer, expert):
            if expert == 11:
                raise OSError("simulated short checkpoint read")
            return payload(layer, expert)
        stager = self.stager(read)
        plan = stager.start(swaps(), 0)
        with self.assertRaisesRegex(RuntimeError, "staging read failed") as raised:
            plan.wait(swaps(), 0)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        plan.cancel()
        self.assertEqual(stager.reserved_bytes, 0)

    def test_later_pipeline_failure_is_exposed_before_its_installation(self):
        def read(layer, expert):
            if expert == 12:
                raise OSError("late disk failure")
            return payload(layer, expert)
        stager = self.stager(read, capacity=1)
        plan = stager.start(swaps(), 0)
        plan.wait(swaps(), 0)
        for key in ((0, 10), (0, 11)):
            plan.take(key, swaps(), 0).release()
        with self.assertRaisesRegex(OSError, "late disk failure"):
            plan.take((0, 12), swaps(), 0)
        plan.cancel()
        self.assertEqual(stager.reserved_bytes, 0)

    def test_changed_generation_or_plan_drops_ready_records(self):
        for changed_generation, changed_swaps in ((1, swaps()), (0, list(reversed(swaps())))):
            stager = self.stager()
            plan = stager.start(swaps(), 0)
            plan.wait(swaps(), 0)
            with self.assertRaises(StaleSwapPlan):
                plan.take((0, 10), changed_swaps, changed_generation)
            self.assertEqual(stager.reserved_bytes, 0)

    def test_cancellation_during_read_and_new_plan_share_budget(self):
        reading = threading.Event()
        proceed = threading.Event()
        def read(layer, expert):
            if expert == 10:
                reading.set()
                self.assertTrue(proceed.wait(2))
            return payload(layer, expert)
        stager = self.stager(read, capacity=1)
        old = stager.start(swaps(1), 0)
        self.assertTrue(reading.wait(2))
        old.cancel()
        new_swaps = [(1, 0, 20, 1.0)]
        new = stager.start(new_swaps, 1)
        self.assertEqual(stager.reserved_bytes, EXPERT_BYTES)
        proceed.set()
        new.wait(new_swaps, 1)
        new.take((1, 20), new_swaps, 1).release()
        self.assertEqual(stager.reserved_bytes, 0)
        self.assertEqual(stager.peak_reserved_bytes, EXPERT_BYTES)

    def test_claimed_record_survives_plan_cancel_until_installer_releases_it(self):
        stager = self.stager(capacity=1)
        plan = stager.start(swaps(1), 0)
        plan.wait(swaps(1), 0)
        record = plan.take((0, 10), swaps(1), 0)
        plan.cancel()
        self.assertEqual(record.views, payload(0, 10))
        self.assertEqual(stager.reserved_bytes, EXPERT_BYTES)
        record.release()
        record.release()  # double release does not return a permit twice
        self.assertEqual(stager.reserved_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "released"):
            _ = record.views

    def test_consume_once_and_consume_prefix_before_suffix(self):
        stager = self.stager(capacity=1)
        plan = stager.start(swaps(2), 0)
        plan.wait(swaps(2), 0)
        with self.assertRaisesRegex(RuntimeError, "plan order"):
            plan.take((0, 11), swaps(2), 0)
        record = plan.take((0, 10), swaps(2), 0)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            plan.take((0, 10), swaps(2), 0)
        record.release()
        plan.take((0, 11), swaps(2), 0).release()

    def test_invalid_plan_or_payload_fails_before_install(self):
        stager = self.stager()
        with self.assertRaisesRegex(ValueError, "ownership"):
            stager.start([(0, 0, 11, 1.0)], 0, world=2, rank=0)
        with self.assertRaisesRegex(ValueError, "twice"):
            stager.start([(0, 0, 10, 1.0), (0, 1, 10, 0.5)], 0)
        bad = self.stager(lambda layer, expert: (b"short",) * 6)
        plan = bad.start(swaps(1), 0)
        with self.assertRaisesRegex(RuntimeError, "staging read failed"):
            plan.wait(swaps(1), 0)
        self.assertEqual(bad.reserved_bytes, 0)

    def test_empty_peer_and_closed_stager(self):
        stager = self.stager()
        plan = stager.start(swaps(1), 0, world=2, rank=1)
        self.assertEqual(plan.wait(swaps(1), 0), 0)
        self.assertEqual(plan.keys, ())
        stager.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            stager.start(swaps(), 0)

if __name__ == "__main__":
    unittest.main()
