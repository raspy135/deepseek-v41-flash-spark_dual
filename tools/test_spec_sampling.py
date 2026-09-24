"""Decision, distribution and GPU-capture tests for sampled speculative verification."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from engine.spec_sampling import sample_probs_batch, verify_probabilities, verify_sampled


def sequential(probs, q, drafts, uniforms, stop_ids=()):
    """Independent host oracle with fixed uniforms and the original rejection rule."""
    accepted = 0
    for i, token in enumerate(drafts.tolist()):
        ratio = min(1.0, float(probs[i, token]) / max(float(q[i, token]), 1e-20))
        if float(uniforms[i]) >= ratio:
            weights = (probs[i] - q[i]).clamp_min(0)
            if not float(weights.sum()) > 0:
                weights = probs[i]
            break
        accepted += 1
        if token in stop_ids:
            return [accepted, -1, *drafts.tolist()]
    else:
        weights = probs[-1]
    mass = weights.double().sum()
    threshold = float(uniforms[-1]) * float(mass)
    running = 0.0
    for token, weight in enumerate(weights.tolist()):
        running += weight
        if running > threshold:
            return [accepted, token, *drafts.tolist()]
    raise AssertionError("invalid categorical distribution")


class SamplingTests(unittest.TestCase):
    devices = ("cpu",)

    def check_case(self, p, q, drafts, u, stops=()):
        expected = sequential(p, q, drafts, u, stops)
        for device in self.devices:
            actual = verify_probabilities(p.to(device), q.to(device), drafts.to(device),
                                          u.to(device), stops).cpu().tolist()
            self.assertEqual(actual, expected)

    def test_every_rejection_position_and_stops(self):
        for b in (1, 3, 5, 15):
            p = torch.tensor([.25, .5, .25, 0.]).repeat(b + 1, 1)
            q = torch.tensor([.5, .25, .25, 0.]).repeat(b, 1)
            drafts = torch.zeros(b, dtype=torch.int64)
            for reject in range(b + 1):
                u = torch.full((b + 1,), .125)
                if reject < b:
                    u[reject] = .75
                self.check_case(p, q, drafts, u)
                self.check_case(p, q, drafts, u, {0})
                self.check_case(p, q, drafts, u, {1, 3})

    def test_stop_after_rejection_is_ignored(self):
        p = torch.tensor([[.1, .9, 0.]] * 4)
        q = torch.tensor([[.9, .1, 0.]] * 3)
        self.check_case(p, q, torch.tensor([0, 1, 0]), torch.tensor([.8, .1, .1, .2]), {1})

    def test_categorical_zero_mass_boundaries(self):
        p = torch.tensor([[0., .25, .75, 0.], [0., .25, .75, 0.]])
        q = p[:1].clone()
        for u in (0., .249, .25, .99999994):
            self.check_case(p, q, torch.tensor([1]), torch.tensor([.1, u]))
        # Synthetic rejection at zero target and proposal mass exercises zero-residual fallback.
        self.check_case(p, q, torch.tensor([0]), torch.tensor([.1, .75]))

    def test_probability_rows_match_sequential_rule(self):
        logits = torch.randn(6, 301, generator=torch.Generator().manual_seed(27))
        logits[0, 0] = -torch.inf  # constrained-token masking
        for temperature in (.1, .6, 1., 2.):
            for top_p in (.01, .5, .95, 1.):
                ref = []
                for row in logits:
                    p = torch.softmax(row / temperature, -1)
                    if top_p < 1:
                        values, ids = p.sort(descending=True)
                        values = torch.where(values.cumsum(0) - values < top_p, values, 0.)
                        p = torch.zeros_like(p).scatter_(0, ids, values)
                        p /= p.sum()
                    ref.append(p)
                for device in self.devices:
                    got = sample_probs_batch(logits.to(device), temperature, top_p).cpu()
                    torch.testing.assert_close(got, torch.stack(ref), rtol=2e-6, atol=2e-7)

    def test_random_decisions(self):
        g = torch.Generator().manual_seed(73)
        for _ in range(80):
            b, v = 5, 23
            p = torch.softmax(torch.randn(b + 1, v, generator=g), -1)
            q = torch.softmax(torch.randn(b, v, generator=g), -1)
            drafts = torch.multinomial(q, 1, generator=g).squeeze(-1)
            u = torch.rand(b + 1, generator=g)
            self.check_case(p, q, drafts, u, {1, 2})

    def test_nucleus_ties_and_masked_tail(self):
        # Sorting ties can choose a different support at the nucleus boundary. Check both
        # tiny rows and the serving vocabulary on the same device/algorithm as the baseline.
        for device in self.devices:
            for vocab in (301, 129280):
                logits = (torch.arange(vocab, device=device) % 5).float().repeat(4, 1)
                logits[:, -13:] = -torch.inf
                for top_p in (.5, .95):
                    got = sample_probs_batch(logits, .6, top_p)
                    rows = []
                    for row in logits:
                        p = torch.softmax(row / .6, -1)
                        values, ids = p.sort(descending=True)
                        values = torch.where(values.cumsum(0) - values < top_p, values, 0.)
                        p = torch.zeros_like(p).scatter_(0, ids, values)
                        rows.append(p / p.sum())
                    expected = torch.stack(rows)
                    self.assertTrue(torch.equal(got > 0, expected > 0))
                    torch.testing.assert_close(got, expected, rtol=2e-6, atol=2e-7)

    def test_rejection_sampling_distribution(self):
        # Deterministic quadrature over proposals, acceptance, and residual categorical draws:
        # target [1/4,3/4], proposal [3/4,1/4]. Marginal emitted first token must be target.
        p = torch.tensor([[.25, .75], [.25, .75]])
        q = torch.tensor([[.75, .25]])
        counts = torch.zeros(2)
        for proposed in (0, 1):
            for ai in range(12):
                for ci in range(8):
                    u = torch.tensor([(ai + .5) / 12, (ci + .5) / 8])
                    result = verify_probabilities(p, q, torch.tensor([proposed]), u).tolist()
                    emitted = proposed if result[0] else result[1]
                    counts[emitted] += q[0, proposed] / 96
        torch.testing.assert_close(counts, p[0], rtol=1e-5, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaSamplingTests(SamplingTests):
    devices = ("cuda",)

    def test_graph_replay_changes_decision(self):
        p = torch.tensor([[.25, .75], [.25, .75]], device="cuda")
        q = torch.tensor([[.75, .25]], device="cuda")
        drafts = torch.tensor([0], device="cuda")
        u = torch.tensor([.1, .1], device="cuda")
        verify_probabilities(p, q, drafts, u)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = verify_probabilities(p, q, drafts, u)
        for acceptance in (.1, .9):
            u[0] = acceptance
            graph.replay()
            self.assertEqual(output.cpu().tolist(),
                             sequential(p.cpu(), q.cpu(), drafts.cpu(), u.cpu()))

    def test_seed_reproducibility_and_fixed_rng_consumption(self):
        logits = torch.tensor([[0., 1., 2.]] * 4, device="cuda")
        q = torch.softmax(logits[:3], -1)
        drafts = torch.tensor([0, 1, 2], device="cuda")
        def run(stops):
            g = torch.Generator(device="cuda").manual_seed(42)
            result = verify_sampled(logits, q, drafts, 1., 1., stops, generator=g)
            return result, g.get_state()
        first, first_state = run(())
        second, second_state = run(())
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(first_state, second_state))
        # Accepted stop at the first proposal still consumes exactly B+1 uniforms.
        stopped, stopped_state = run({0})
        self.assertEqual(stopped, (1, [0], None))
        self.assertTrue(torch.equal(first_state, stopped_state))


class EngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real sampled branch and common commit/rollback tail without loading
        # checkpoint weights. This catches accidental bypasses of stops and output limits.
        source = Path(__file__).resolve().parents[1] / "engine/v41_engine.py"
        tree = ast.parse(source.read_text())
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "V41Engine")
        decode = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_loop")
        spec = next(n for n in ast.walk(decode) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "self.spec")
        start = next(i for i, n in enumerate(spec.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_ev_sample" for t in n.targets))
        wrapper = ast.parse('''
def sampled_tail(self, logits, q, drafts, temperature, top_p, stop_ids, max_tokens, m):
    pos, n_out, tok, steps = 10, 0, 0, 0
    ph = pen = grammar = None
    accepted_hist, out, out_st = [], [], {}
    for _ in range(1):
        pass
''')
        wrapper.body[0].body[-1].body = spec.body[start:]
        probability_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name == "sample_probs")
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[probability_fn, wrapper.body[0]],
                                                         type_ignores=[])), str(source), "exec"), namespace)
        cls.run_tail = staticmethod(namespace["sampled_tail"])

    def test_stop_and_token_budget_commit(self):
        for enabled in (False, True):
            for budget, stops, expected in ((2, set(), [1, 2]), (10, {2}, [1, 2]),
                                            (1, {2}, [1]), (10, {1}, [1])):
                logits = torch.zeros(4, 5)
                q = torch.full((3, 5), .2)
                drafts = torch.tensor([1, 2, 3])
                calls, rollback = [], []
                def verify(*args):
                    calls.append(True)
                    return verify_sampled(*args)
                eng = SimpleNamespace(batched_verify=enabled, _verify_sampled=verify, device="cpu",
                                      fast=SimpleNamespace(_ev_begin=lambda _: None))
                model = SimpleNamespace(c=SimpleNamespace(rollback=rollback.append))
                got = list(self.run_tail(eng, logits, q, drafts, .6, 1., stops, budget, model))
                self.assertEqual([t for burst in got for t in burst], expected)
                self.assertEqual(len(calls), int(enabled))
                self.assertEqual(rollback, [12 if 1 in stops else 13 if 2 in stops else 14])


if __name__ == "__main__":
    unittest.main()
