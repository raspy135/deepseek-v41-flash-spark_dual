"""CPU check: the streaming prefill indexer matches the monolithic one.

`Model._indexer_stream` (DSV41_INDEX_STREAM=1) reduces the score buffer one query tile at a time
instead of materializing [chunk, context]. The two paths must select the same compressed positions
for every query -- that set is what the sparse attention then reads, so a divergence here changes
the model's output. This test drives both on synthetic q/wts/k with no weights and no GPU, over
three layer roles: before the candidate source, the candidate source itself, and a later layer
that consumes a candidate mask.

Ties are the one thing a per-tile topk can break differently from a per-chunk one, so the inputs
are seeded and the assertion compares the selected *sets*; a failure here is the signal that
test_prefix_invariance would also fail.
"""

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.model import Model, SCORE_DTYPE


class _Fake:
    """The five things _indexer_stream touches, without loading a checkpoint."""

    def __init__(self, n_pad, index_topk, candidate_source_layer):
        self.args = types.SimpleNamespace(candidate_source_layer=candidate_source_layer,
                                          candidate_topk_blocks=2, candidate_block_size=4,
                                          index_topk=index_topk)
        self.dev = "cpu"
        self._positions = torch.arange(n_pad, dtype=torch.long)
        self._select_candidates = Model._select_candidates
        self._pad_topk = lambda idx: Model._pad_topk(self, idx)


def monolithic(f, q, wts, k, n_pad, k_, L, compress_lens, sh, B, NB, T):
    """The old path, copied: full [T, n_pad] score, then mask/select/topk once."""
    a = f.args
    cpos = f._positions[:n_pad]
    score = torch.empty(T, n_pad, dtype=SCORE_DTYPE, device=f.dev)
    for i in range(0, T, B):
        j = min(i + B, T)
        qt, wt = q[i:j], wts[i:j]
        if j - i < B:
            pad = B - (j - i)
            qt = torch.cat([qt, qt.new_zeros(pad, *qt.shape[1:])])
            wt = torch.cat([wt, wt.new_zeros(pad, wt.size(1))])
        for jb in range(0, n_pad, NB):
            sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])
            sc = sc.float().relu_() * wt[:, :, None]
            score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i].to(score.dtype)
    score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
    is_cand_src = L == a.candidate_source_layer
    if is_cand_src:
        sh.candidates = f._select_candidates(score, compress_lens, a.candidate_topk_blocks,
                                             a.candidate_block_size)
    elif 0 <= a.candidate_source_layer < L and sh.candidates is not None:
        score = score.masked_fill(~sh.candidates, float("-inf"))
    idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
    idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
    return f._pad_topk(idx)


class IndexStreamTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.NB = 8
        self.B = 8
        self.n_pad = 32
        self.index_topk = 4
        self.T = 20          # last query tile is partial (20 = 2*8 + 4)
        self.n_heads = 3
        self.head_dim = 6

    def _inputs(self):
        T = self.T
        q = torch.randn(T, self.n_heads, self.head_dim).to(torch.bfloat16)
        wts = torch.randn(T, self.n_heads).float().abs() + 0.1
        k = torch.randn(self.n_pad, self.head_dim).to(torch.bfloat16)
        compress_lens = torch.arange(T, dtype=torch.long) + 3  # all within n_pad
        return q, wts, k, compress_lens

    def _run_pair(self, L, sh_a, sh_b, compress_lens, q, wts, k):
        k_ = min(self.index_topk, self.n_pad)
        got = Model._indexer_stream(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad, k_, L,
                                    compress_lens, sh_a, self.B, self.NB, self.T)
        want = monolithic(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad, k_, L,
                          compress_lens, sh_b, self.B, self.NB, self.T)
        return got, want

    @staticmethod
    def _selected(idx):
        return [sorted(int(v) for v in row if int(v) >= 0) for row in idx]

    def test_matches_before_candidate_source(self):
        q, wts, k, cl = self._inputs()
        got, want = self._run_pair(2, types.SimpleNamespace(), types.SimpleNamespace(), cl, q, wts, k)
        self.assertEqual(self._selected(got), self._selected(want))

    def test_matches_at_candidate_source_and_carries_mask(self):
        q, wts, k, cl = self._inputs()
        sh_a, sh_b = types.SimpleNamespace(), types.SimpleNamespace()
        got, want = self._run_pair(20, sh_a, sh_b, cl, q, wts, k)
        self.assertEqual(self._selected(got), self._selected(want))
        # the candidate mask the tiled path built must equal the monolithic one, or the decoder
        # indexers that consume it would search a different pool
        self.assertTrue(torch.equal(sh_a.candidates, sh_b.candidates))

    def test_matches_later_layer_consuming_mask(self):
        q, wts, k, cl = self._inputs()
        src_a, src_b = types.SimpleNamespace(), types.SimpleNamespace()
        self._run_pair(20, src_a, src_b, cl, q, wts, k)   # fill both masks identically

        def _later(sh):
            # mirror the real flow: the mask is produced at layer 20 and read at layer 24
            sh.candidates = src_a.candidates.clone()
            return sh

        got = Model._indexer_stream(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad,
                                    self.index_topk, 24, cl, _later(types.SimpleNamespace()),
                                    self.B, self.NB, self.T)
        want = monolithic(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad,
                          self.index_topk, 24, cl, _later(types.SimpleNamespace()),
                          self.B, self.NB, self.T)
        self.assertEqual(self._selected(got), self._selected(want))

    def test_small_n_c_uses_shorter_topk(self):
        q, wts, k, cl = self._inputs()
        cl = torch.full((self.T,), 3, dtype=torch.long)
        k_ = 3
        got = Model._indexer_stream(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad, k_, 2,
                                    cl, types.SimpleNamespace(), self.B, self.NB, self.T)
        want = monolithic(_Fake(self.n_pad, self.index_topk, 20), q, wts, k, self.n_pad, k_, 2,
                          cl, types.SimpleNamespace(), self.B, self.NB, self.T)
        self.assertEqual(self._selected(got), self._selected(want))


if __name__ == "__main__":
    unittest.main()
