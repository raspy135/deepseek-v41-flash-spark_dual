"""Immediate current-layer routing and transient-only decode resolution, on CPU."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from engine import model as M
from engine.fastdecode import FastDecoder
from engine.layer_stream import LayerAblation, graph_bounds, parse_layers, validate_capacity


class LayerStreamTest(unittest.TestCase):
    def test_bounds_keep_resident_runs_and_split_selected_layers(self):
        self.assertEqual(graph_bounds(40, [1, 14], []), [0, 1, 14, 40])
        self.assertEqual(graph_bounds(40, [1, 14], [0, 38, 39]), [0, 1, 14, 38, 39, 40])
        self.assertEqual(parse_layers('39,0,39', 40), frozenset([0, 39]))
        for raw in ('-1', '40', 'a', '1,'):
            with self.assertRaises(ValueError):
                parse_layers(raw, 40)

    def test_prefill_capacity_uses_all_cold_experts(self):
        masks = {0: torch.arange(384) < 235}
        validate_capacity([0], masks, 149, 384)
        with self.assertRaises(ValueError):
            validate_capacity([0], masks, 148, 384)

    def test_decode_uses_transients_for_selected_layers(self):
        events = []
        def resolve(layer, ids, prefill):
            events.append((layer, prefill))
            return ids.to(torch.int32) + 7
        decoder = NS(stream_layers={3}, m=NS(store=NS(resolve=resolve)),
                     route_idx=torch.tensor([[1, 2]]), slots=torch.zeros(1, 2, dtype=torch.int32))
        FastDecoder._resolve(decoder, 3)
        self.assertEqual(decoder.slots.tolist(), [[8, 9]])
        FastDecoder._resolve(decoder, 2)
        self.assertEqual(events, [(3, True), (2, False)])

    def test_pruned_wanted_expert_loaded_before_current_moe(self):
        events = []
        y = torch.tensor([[1., 0.]], dtype=torch.bfloat16)
        w = NS(gate_w=torch.tensor([[0., 0.], [1., 0.], [2., 0.], [10., 0.]]),
               gate_bias=torch.zeros(4), sh_w1=None, sh_w2=None, sh_w3=None)
        def resolve(layer, ids, prefill):
            events.append(('load', ids.tolist(), prefill))
            return ids.to(torch.int32)
        def kernel(x, slots, weights, arena, limit, **kwargs):
            events.append(('compute', slots.tolist()))
            self.assertNotIn('routing_ids', kwargs)
            return (slots.float() * weights).sum(1, keepdim=True).expand_as(x)
        store = NS(null_slot=9, resolve=resolve, ep=NS(tensor_parallel=True))
        mask = torch.tensor([True, True, False, False])
        model = NS(args=NS(n_activated_experts=2, route_scale=1., swiglu_limit=10.),
            stream_layers={0}, stream_keep={0: torch.ones(4, dtype=torch.bool)}, prune_mask={0: mask},
            slot_lut=torch.tensor([[0, 1, 9, 9]], dtype=torch.int32),
            prefill_routes={0: (torch.zeros(4, dtype=torch.int32), torch.tensor([9]))},
            _tap=lambda *args: None, moe_fn=kernel, stats={'ep_s': 0., 'ep_calls': 0, 'moe_s': 0.})
        for prefill in (True, False):
            events.clear()
            with patch.object(M, 'PRUNE_MISS', False), patch.object(M.R, 'expert_ffn', return_value=torch.zeros_like(y)):
                out = M.Model.moe(model, y, w, 0, prefill, store, None, 4)
            self.assertEqual(events[0], ('load', [[3, 2]], True))
            self.assertEqual(events[1], ('compute', [[3, 2]]))
            self.assertTrue((out > 2).all())
            self.assertEqual(model.prune_mask[0].tolist(), [True, True, False, False])

    def test_dynamic_mask_control_is_authoritative_and_preserves_addresses(self):
        class Pair:
            packet = None
            def broadcast_obj(self, packet):
                if self.rank == 0:
                    self.packet = packet
                return self.packet
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'control.json'
            ep = Pair()
            models = [NS(stream_keep={l: torch.ones(4, dtype=torch.bool) for l in range(2)},
                         prune_mask={l: torch.tensor([True, True, False, False]) for l in range(2)})
                      for _ in range(2)]
            addresses = [[m.stream_keep[l].data_ptr() for l in range(2)] for m in models]
            controls = [LayerAblation(str(file), 2), LayerAblation('/peer/has/no/file', 2)]
            for ids in ([0], [1], []):
                file.write_text(json.dumps({'pruned_layers': ids}))
                for rank in (0, 1):
                    ep.rank = rank
                    controls[rank].sync(models[rank], ep)
                for layer in range(2):
                    torch.testing.assert_close(models[0].stream_keep[layer], models[1].stream_keep[layer])
                    self.assertEqual(models[0].stream_keep[layer].tolist(),
                        [True, True, False, False] if layer in ids else [True] * 4)
                self.assertEqual(addresses, [[m.stream_keep[l].data_ptr() for l in range(2)] for m in models])
            # A malformed file still broadcasts an error and both ranks reject it.
            file.write_text(json.dumps({'pruned_layers': [True]}))
            for rank in (0, 1):
                ep.rank = rank
                with self.assertRaises(RuntimeError):
                    controls[rank].sync(models[rank], ep)


if __name__ == '__main__':
    unittest.main()
