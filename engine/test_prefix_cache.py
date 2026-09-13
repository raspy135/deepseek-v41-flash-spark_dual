"""CPU checks for the small-state half of exact prompt-prefix reuse."""

from types import SimpleNamespace

import torch

from engine.v41_engine import V41Engine


def _fixture():
    e = object.__new__(V41Engine)
    e.swa_replay = True
    e._prefix_cache = None
    e.args = SimpleNamespace(window_size=4, candidate_source_layer=1)
    slots = 4096
    c = SimpleNamespace(
        win=[torch.arange(slots * 2).view(slots, 2) + 10000 * L for L in range(3)],
        pending={2: (torch.tensor([7.0]), torch.tensor([8.0]))},
        len=6,
    )
    rep = {
        "h": torch.arange(8).view(4, 2),
        "pre_mix": torch.arange(4).view(4, 1),
        "topk": torch.arange(12).view(4, 3),
        "cand": torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.bool),
    }
    m = SimpleNamespace(
        _positions=torch.arange(slots),
        _rep_tail=lambda: (rep["h"], rep["pre_mix"], rep["topk"], rep["cand"], 2),
    )
    e.caches, e.model = c, m
    return e


def test_prefix_snapshot_restores_overwritten_state():
    e = _fixture()
    ids = torch.tensor([10, 11, 12, 13, 14, 15])
    e._save_prefix(ids, len(ids))
    saved = {L: value.clone() for L, value in e._prefix_cache["win"].items()}

    for ring in e.caches.win:
        ring.zero_()
    e.caches.pending[2] = None
    e.caches.len = 0

    assert e._restore_prefix(ids.tolist() + [16, 17]) == len(ids)
    for L, value in saved.items():
        assert torch.equal(e.caches.win[L][e._prefix_cache["slots"]], value)
    assert e.caches.pending[2] is not None
    assert e.caches.len == len(ids)
    assert torch.equal(e.model._rep["h"][0], torch.arange(8).view(4, 2))


def test_prefix_mismatch_invalidates_snapshot():
    e = _fixture()
    e._save_prefix(torch.tensor([1, 2, 3]), 3)
    assert e._restore_prefix([1, 9, 3, 4]) == 0
    assert e._prefix_cache is None
