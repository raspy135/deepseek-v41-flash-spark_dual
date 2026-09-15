"""CPU-only regression tests for sampling defaults; no checkpoint or GPU needed."""
import os
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.v41_engine import Penalties


def test_default_preserves_nested_json():
    # Actual tokenizer IDs for the canonical depth-8 answer. After four repeats,
    # the old default banned the correct `n` at position 13, despite valid JSON.
    ids = [24313, 80, 3362] + [28612, 80, 3362] * 7 + [223, 2170, 91428, 91428]
    with patch.dict(os.environ, {}, clear=True):
        pen = Penalties()
    assert not pen.active
    scores = torch.arange(100_000, dtype=torch.float32)
    expected = scores.clone()
    for j in range(len(ids)):
        pen.apply(scores, ids[:j])
    assert torch.equal(scores, expected)


def test_explicit_cycle_break_still_available():
    pen = Penalties(enabled=True, no_repeat_ngram=0)
    history = [24313, 80, 3362] + [28612, 80, 3362] * 3 + [28612]
    assert pen._cycle_token(history) == 80
    scores = torch.zeros(100)
    pen.apply(scores, history)
    assert torch.isneginf(scores[80])
    assert pen.hits == 1


def test_explicit_penalties_still_apply():
    pen = Penalties(presence=0.5, frequency=0.25, enabled=False, no_repeat_ngram=0)
    pen.observe([1, 1, 2])
    scores = torch.zeros(4)
    pen.apply(scores, [1, 1, 2])
    assert torch.equal(scores, torch.tensor([0., -1., -0.75, 0.]))


if __name__ == "__main__":
    for test in (test_default_preserves_nested_json, test_explicit_cycle_break_still_available,
                 test_explicit_penalties_still_apply):
        test()
        print(test.__name__, "PASS")
