"""Exercise the production sampled verifier without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


def verifier_code():
    tree = ast.parse(Path(__file__).with_name('v41_engine.py').read_text())
    # Extract the real inline acceptance loop, not a duplicate implementation.
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and n.value.id == 'drafts' for n in ast.walk(node)):
            continue
        for parent in ast.walk(tree):
            body = getattr(parent, 'body', None)
            if isinstance(body, list) and node in body:
                i = body.index(node)
                return compile(ast.Module(body=body[i:i + 2], type_ignores=[]),
                               'sampled_verifier', 'exec')
    raise AssertionError('sampled verifier not found')


class TestDraftWidth(unittest.TestCase):
    def run_case(self, k, reject=None, stop=None, temperature=0.7):
        drafts = torch.arange(k)
        q = torch.eye(k + 2)[:k]
        logits = torch.eye(k + 2)[:k + 1].clone()
        if reject is not None:
            logits[reject].zero_()
            logits[reject, k + 1] = 1
        scope = dict(torch=torch, self=SimpleNamespace(device='cpu'),
                     drafts=drafts, q=q, logits=logits, a=0, new=[], bonus=None,
                     temperature=temperature, top_p=1.0,
                     stop_ids=set() if stop is None else {stop},
                     sample_probs=lambda row, temperature, top_p: row)
        exec(verifier_code(), scope)
        return scope['a'], scope['new'], scope['bonus']

    def test_all_accepted_and_bonus(self):
        for k in (1, 3, 5, 7, 15):
            for temperature in (0.0, 0.7):
                with self.subTest(k=k, temperature=temperature):
                    self.assertEqual(self.run_case(k, temperature=temperature),
                                     (k, list(range(k)), k))

    def test_rejection_at_each_position(self):
        for k in (1, 3, 5, 7, 15):
            for i in range(k):
                with self.subTest(k=k, reject=i):
                    self.assertEqual(self.run_case(k, reject=i),
                                     (i, list(range(i)), k + 1))

    def test_accepted_stop_has_no_bonus(self):
        for k in (1, 3, 5, 7, 15):
            self.assertEqual(self.run_case(k, stop=k - 1),
                             (k, list(range(k)), None))


if __name__ == '__main__':
    unittest.main()
