import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from engine import collective_rails as rails


class Routing(unittest.TestCase):
    def test_phase_restored_on_exception_and_nested_decode(self):
        pg = object()
        with patch.object(rails, '_prefill_group', pg):
            self.assertIsNone(rails.group())
            with self.assertRaises(ValueError):
                with rails.phase(True):
                    self.assertIs(rails.group(), pg)
                    with rails.phase(False):
                        self.assertIsNone(rails.group())
                    self.assertIs(rails.group(), pg)
                    raise ValueError('failed prefill')
            self.assertIsNone(rails.group())

    def test_explicit_phase_not_token_count(self):
        class Model:
            @rails.forward_phase
            def forward(self, ids, S, prefill):
                return rails.group()

            @rails.prefill_phase
            def decoder_replay(self):
                return rails.group()
        pg = object()
        with patch.object(rails, '_prefill_group', pg):
            m = Model()
            self.assertIs(m.forward([1], 0, prefill=True), pg)
            self.assertIsNone(m.forward(list(range(100)), 0, False))
            self.assertIs(m.decoder_replay(), pg)
            self.assertIsNone(rails.group())

    def test_single_rail_graph_retains_gpu_ids_and_both_algorithms(self):
        root = ET.Element('graphs')
        for ident in ('0', '1'):
            graph = ET.SubElement(root, 'graph', id=ident, nchannels='4', speedinter='12')
            for net in ('0', '1', '0', '1'):
                ch = ET.SubElement(graph, 'channel')
                ET.SubElement(ch, 'net', dev=net)
                ET.SubElement(ch, 'gpu', dev='abc')
                ET.SubElement(ch, 'net', dev=net)
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td)/'both.xml', Path(td)/'one.xml'
            ET.ElementTree(root).write(src)
            self.assertEqual(rails.single_rail_graph(src, dst), '0')
            result = ET.parse(dst).getroot()
            self.assertEqual({n.get('dev') for n in result.iter('net')}, {'0'})
            self.assertEqual({n.get('dev') for n in result.iter('gpu')}, {'abc'})
            self.assertEqual([g.get('nchannels') for g in result], ['2', '2'])
            # A missing rail is a topology/config error, never an automatic fallback.
            with self.assertRaises(RuntimeError):
                rails.single_rail_graph(dst, src)

    def test_rank_disagreement_fails_before_optional_group_creation(self):
        def disagree(peers, cfg):
            peers[:] = [cfg, (not cfg[0], cfg[1])]
        with patch.object(rails.dist, 'get_world_size', return_value=2), \
             patch.object(rails.dist, 'all_gather_object', side_effect=disagree), \
             patch.object(rails.dist, 'new_group') as create:
            with self.assertRaisesRegex(RuntimeError, 'differs across ranks'):
                rails.initialize(True, None, None)
            create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
