"""CPU execution of actual decode-mode and commit branches, without checkpoint imports."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import torch

from engine.lookup_draft import ExactDraftCache
from engine.spec_sampling import verify_sampled


class DecodeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name('v41_engine.py')
        tree = ast.parse(source.read_text())
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'V41Engine')
        decode = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_loop')
        mode = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_step_value')
        probability = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_probs')
        loop = next(n for n in ast.walk(decode) if isinstance(n, ast.While)
                    and ast.unparse(n.test).startswith('self.ep.control('))
        spec = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.spec')
        fast = next(n for n in spec.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.fast is not None')
        before_hash = next(i for i, n in enumerate(fast.body) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == '_evh' for t in n.targets))
        stage = ast.parse('''
def proposal_stage(self, temperature, conf_pol):
    pol = conf_pol
    tok, pos, ph = 8, 10, None
    for _ in range(1):
        pass
    return drafts, q, locals().get('block'), depth
''').body[0]
        # Execute the actual mode dispatch and second control, stopping just before
        # hash/Engram work. No copied spellings of the implementation under test.
        first = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 't_iter' for t in n.targets))
        stage_nodes = copy.deepcopy(loop.body[first:loop.body.index(spec)])
        spec_prefix = copy.deepcopy(spec)
        spec_prefix.body = copy.deepcopy(spec.body[:spec.body.index(fast)])
        fast_prefix = copy.deepcopy(fast)
        fast_prefix.body = copy.deepcopy(fast.body[:before_hash])
        fast_prefix.orelse = []
        spec_prefix.body.append(fast_prefix)
        spec_prefix.orelse = []
        stage_nodes.append(spec_prefix)
        stage.body[-2].body = stage_nodes

        tail_start = next(i for i, n in enumerate(spec.body) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == '_ev_sample' for t in n.targets))
        tail = ast.parse('''
def sampled_tail(self, logits, q, drafts, stop_ids, max_tokens, m,
                 copied=False, root_only=False, lookup=None, bypass=None, pol=None):
    temperature, top_p = .6, 1.
    pos, n_out, tok, steps = 10, 0, 0, 0
    ph = pen = grammar = conf_hist = None
    depth = drafts.numel()
    n_graphs, t_iter = 0, time.perf_counter()
    accepted_hist, out, out_st = [], [], {}
    for _ in range(1):
        pass
''').body[0]
        tail.body[-1].body = copy.deepcopy(spec.body[tail_start:])
        ns = {'torch': torch, 'time': time, 'LEAN_STEP': True}
        module = ast.Module(body=[copy.deepcopy(mode), copy.deepcopy(probability), stage, tail], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
        cls.mode = staticmethod(ns['_decode_step_value'])
        cls.stage = staticmethod(ns['proposal_stage'])
        cls.tail = staticmethod(ns['sampled_tail'])

    def test_mode_selection_is_preproposal_and_rank_zero_owned(self):
        calls = []
        policy = SimpleNamespace(decide=lambda: calls.append('policy') or 3)
        lookup = SimpleNamespace(propose=lambda depth: calls.append(('lookup', depth)) or [1, 2, 3])
        bypass = SimpleNamespace(decide=lambda: calls.append('bypass') or True)
        engine = SimpleNamespace(ep=SimpleNamespace(rank=0), _tv=6)
        self.assertEqual(self.mode(engine, policy, lookup, bypass), -3)
        self.assertEqual(engine._lookup_step_tokens, [1, 2, 3])
        self.assertEqual(calls, ['policy', ('lookup', 3)])
        calls.clear()
        lookup.propose = lambda depth: calls.append(('lookup', depth))
        self.assertEqual(self.mode(engine, policy, lookup, bypass), 0)
        self.assertEqual(calls, ['policy', ('lookup', 3), 'bypass'])
        self.assertIsNone(engine._lookup_step_tokens)
        bypass.decide = lambda: False
        self.assertEqual(self.mode(engine, None, None, bypass), 5)
        for rank, keep_going in ((1, True), (0, False), (1, False)):
            engine.ep.rank = rank
            calls.clear()
            self.assertEqual(self.mode(engine, policy, lookup, bypass, keep_going), 0)
            self.assertEqual(calls, [])

    def test_mode_dispatch_and_second_control_at_all_temperatures(self):
        for mode in (-3, 0, 5):
            for rank in (0, 1):
                for temperature in (0., .1, .6, 1., 2.):
                    calls = []
                    chosen = abs(mode) if mode < 0 else 0 if mode == 0 else 3
                    ep = SimpleNamespace(rank=rank, control_value=mode)
                    def control(keep, value):
                        calls.append(('control', keep, value))
                        ep.control_value = chosen
                        return True
                    ep.control = control
                    ep.broadcast_obj = lambda payload: calls.append(('broadcast', payload)) or [1, 2, 3]
                    def draft(*args):
                        calls.append(('draft', args))
                        return torch.arange(5), torch.full((5, 16), 1 / 16)
                    def choose(values):
                        calls.append(('confidence', values))
                        return 3
                    policy = SimpleNamespace(pop_switch=lambda: None, choose=choose, choose_sampled=choose)
                    buffers = {width: (torch.zeros(width, dtype=torch.int64), None, None) for width in (2, 4, 6)}
                    engine = SimpleNamespace(ep=ep, spec=True, device='cpu', args=SimpleNamespace(vocab_size=16),
                        fast=SimpleNamespace(graphs=[], draft_graphs=None, draft=draft, d_conf=torch.zeros(5)),
                        _lookup_step_tokens=[1, 2, 3] if rank == 0 else None, _vbufs=buffers)
                    with self.subTest(mode=mode, rank=rank, temperature=temperature):
                        drafts, q, block, depth = self.stage(engine, temperature, policy)
                        self.assertEqual(depth, chosen)
                        self.assertEqual(sum(c[0] == 'control' for c in calls), 1)
                        self.assertEqual(sum(c[0] == 'broadcast' for c in calls), int(mode < 0))
                        self.assertEqual(sum(c[0] == 'draft' for c in calls), int(mode > 0))
                        self.assertEqual(sum(c[0] == 'confidence' for c in calls), int(mode > 0 and rank == 0))
                        if mode == 0:
                            self.assertEqual(drafts.numel(), 0)
                            self.assertEqual(q.shape, (0, 16))
                            self.assertEqual(block.tolist(), [8, 8])
                        elif mode < 0:
                            self.assertEqual(block.tolist(), [8, 1, 2, 3])
                            self.assertEqual(q.gather(1, drafts[:, None]).tolist(), [[1.], [1.], [1.]])
                            self.assertTrue(torch.equal(q.sum(1), torch.ones(3)))
                        else:
                            self.assertEqual(block.tolist(), [8, 0, 1, 2])

    def test_second_control_cancellation_avoids_verify_block(self):
        for mode in (-3, 0, 5):
            ep = SimpleNamespace(rank=0, control_value=mode, control=lambda *args: False,
                                 broadcast_obj=lambda payload: payload)
            policy = SimpleNamespace(pop_switch=lambda: None, choose_sampled=lambda _: 3)
            engine = SimpleNamespace(ep=ep, spec=True, device='cpu', args=SimpleNamespace(vocab_size=16),
                fast=SimpleNamespace(graphs=[], draft_graphs=None, d_conf=torch.zeros(5),
                    draft=lambda *args: (torch.arange(5), torch.full((5, 16), 1 / 16))),
                _lookup_step_tokens=[1, 2, 3], _vbufs={width: (None, None, None) for width in (2, 4, 6)})
            self.assertIsNone(self.stage(engine, .6, policy)[2])

    def test_copy_commit_rejection_stops_budget_and_history(self):
        for batched in (False, True):
            for rejected, stops, budget, expected in ((False, set(), 10, [1, 2]),
                    (True, set(), 10, [2]), (False, {1}, 10, [1]), (False, set(), 1, [1])):
                cache = ExactDraftCache([2, 3, 1, 2, 3], 2)
                self.assertEqual(cache.propose(1), [1])
                drafts = torch.tensor([1])
                q = torch.tensor([[0., 1., 0., 0.]])
                logits = torch.full((2, 4), -torch.inf)
                logits[0, 2 if rejected else 1] = 0.
                logits[1, 2] = 0.
                rollbacks, observations = [], []
                engine = SimpleNamespace(ep=SimpleNamespace(rank=0), batched_verify=batched,
                    _verify_sampled=verify_sampled, device='cpu', fast=SimpleNamespace(_ev_begin=lambda _: None),
                    _policy_step_s=lambda *args: .02)
                model = SimpleNamespace(c=SimpleNamespace(rollback=rollbacks.append))
                policy = SimpleNamespace(observe=lambda *args: observations.append(args))
                bypass = SimpleNamespace(observe=lambda *args: observations.append(args))
                with self.subTest(batched=batched, rejected=rejected, stops=stops, budget=budget):
                    output = list(self.tail(engine, logits, q, drafts, stops, budget, model,
                                            copied=True, lookup=cache, bypass=bypass, pol=policy))
                    self.assertEqual([t for burst in output for t in burst], expected)
                    self.assertEqual(rollbacks, [11 if rejected else 12])
                    self.assertEqual(cache.history, [2, 3, 1, 2, 3] + expected)
                    self.assertEqual(cache.stats['accepted_tokens'], int(not rejected))
                    self.assertEqual(observations, [])  # copy timings must not train DSpark/bypass costs

    def test_root_only_samples_row_zero_discards_dummy_and_prices_bypass(self):
        logits = torch.full((2, 4), -torch.inf)
        logits[0, 2] = logits[1, 3] = 0.
        rollbacks, policy_observations, bypass_observations = [], [], []
        engine = SimpleNamespace(ep=SimpleNamespace(rank=0), batched_verify=True,
            _verify_sampled=lambda *args: self.fail('root-only must avoid empty batched verifier'),
            fast=SimpleNamespace(_ev_begin=lambda _: None), device='cpu', _policy_step_s=lambda *args: .02)
        model = SimpleNamespace(c=SimpleNamespace(rollback=rollbacks.append))
        policy = SimpleNamespace(observe=lambda *args: policy_observations.append(args))
        bypass = SimpleNamespace(observe=lambda *args: bypass_observations.append(args))
        output = list(self.tail(engine, logits, torch.empty((0, 4)), torch.empty(0, dtype=torch.int64),
                                set(), 10, model, root_only=True, bypass=bypass, pol=policy))
        self.assertEqual(output, [[2]])
        self.assertEqual(rollbacks, [11])
        self.assertEqual(policy_observations, [])
        self.assertEqual(bypass_observations, [(True, 1, .02)])


if __name__ == '__main__':
    unittest.main()
