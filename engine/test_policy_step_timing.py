"""CPU regression for graph replacement contaminating speculative step costs.

Execute the actual engine timing guard and loop snapshot without importing the
checkpoint/CUDA runtime: python3 -m unittest engine.test_policy_step_timing
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from engine.spec_depth import DepthPolicy


class PolicyStepTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name('v41_engine.py')
        tree = ast.parse(source.read_text())
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'V41Engine')
        guard = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_policy_step_s')
        decode = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_loop')
        loop = next(n for n in ast.walk(decode) if isinstance(n, ast.While)
                    and ast.unparse(n.test).startswith('self.ep.control('))
        start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 't_iter' for t in n.targets))
        snapshot = copy.deepcopy(loop.body[start + 1])
        assert isinstance(snapshot, ast.Assign) and len(snapshot.targets) == 1
        # Exercise the caller's actual snapshot too: a fixed guard paired with
        # the old live-count snapshot would silently discard every warm sample.
        wrapper = ast.parse('def snapshot(self, pol=True, bypass=None):\n    pass\n').body[0]
        wrapper.body = [snapshot, ast.Return(value=ast.Name(id=snapshot.targets[0].id, ctx=ast.Load()))]
        clock = SimpleNamespace(perf_counter=lambda: 20.)
        namespace = {'time': clock}
        module = ast.Module(body=[copy.deepcopy(guard), wrapper], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        cls.guard = staticmethod(namespace['_policy_step_s'])
        cls.snapshot = staticmethod(namespace['snapshot'])

    def setUp(self):
        self.fast = SimpleNamespace(graph_captures=7,
                                    graphs={(0, 4096, 4): object()}, draft_graphs=object())
        self.engine = SimpleNamespace(fast=self.fast)

    def test_cap_one_replacement_does_not_train_step_cost(self):
        policy = DepthPolicy((3, 5), start=3)
        stamp = self.snapshot(self.engine)
        policy.observe(3, 1, 2, self.guard(self.engine, 19.9, stamp))
        warm_cost = policy.step_s[3]
        self.assertAlmostEqual(warm_cost, .1)

        stamp = self.snapshot(self.engine)
        # DSV41_GRAPHS_MAX=1 rotates the pool on another parity: one main
        # graph and a drafter still exist afterward, despite a new capture.
        self.fast.graphs = {(1, 4096, 4): object()}
        self.fast.draft_graphs = object()
        self.fast.graph_captures += 1
        self.assertEqual(len(self.fast.graphs), 1)
        sample = self.guard(self.engine, 10., stamp)
        self.assertIsNone(sample)
        policy.observe(3, 1, 2, sample)
        self.assertEqual(policy.step_s[3], warm_cost)
        self.assertEqual(len(policy._samples[3]), 1)

    def test_warm_replay_and_bypass_keep_elapsed_time(self):
        for pol, bypass in ((True, None), (None, True)):
            with self.subTest(pol=pol, bypass=bypass):
                stamp = self.snapshot(self.engine, pol, bypass)
                self.assertEqual(stamp, self.fast.graph_captures)
                self.assertAlmostEqual(self.guard(self.engine, 19.875, stamp), .125)

    def test_first_capture_is_excluded(self):
        self.fast.graph_captures = 0
        self.fast.graphs = {}
        self.fast.draft_graphs = None
        stamp = self.snapshot(self.engine)
        self.fast.graph_captures = 1
        self.fast.graphs = {(0, 4096, 4): object()}
        self.fast.draft_graphs = object()
        self.assertIsNone(self.guard(self.engine, 10., stamp))

    def test_missing_counter_omits_unknown_timing(self):
        del self.fast.graph_captures
        stamp = self.snapshot(self.engine)
        self.assertIsNone(stamp)
        self.assertIsNone(self.guard(self.engine, 19.9, stamp))


if __name__ == '__main__':
    unittest.main()
