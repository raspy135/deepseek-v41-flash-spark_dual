import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from engine.test_prefix_cache import _fixture
from engine.model import Model


@patch.dict(os.environ, {'DSV41_PREFIX_RESPONSE': '1', 'DSV41_PREFIX_CACHE': '1'})
class ResponsePrefixTests(unittest.TestCase):
    def make(self):
        e = _fixture()
        e.lock, e.device = threading.Lock(), 'cpu'
        e.replica_slots, e.max_context, e._images = 0, 100, None
        e.prefix_disk = SimpleNamespace(strict=False, save=Mock())
        e.ep = SimpleNamespace(gather_objects=lambda x: [x, x])
        e.model.hash_state = Mock(return_value=torch.zeros(1, 20, 2))
        e.model.forward = Mock()
        e._engram_readahead = Mock(return_value=None)
        e._save_prefix([1, 2, 3, 4, 5, 6], 6)
        return e

    @patch('torch.cuda.synchronize')
    def test_only_response_replayed_and_input_fallback_retained(self, sync):
        e = self.make()
        prompt, response = [1, 2, 3, 4, 5, 6], [7, 8, 9]
        with patch('engine.v41_engine.MAX_CHUNK', 2), patch('engine.v41_engine.PREFIX_SNAPSHOTS', 1):
            report = e.cache_response(prompt, response)
        calls = e.model.forward.call_args_list
        self.assertEqual([c.args[1] for c in calls], [6, 8])
        self.assertEqual([c.args[0].tolist() for c in calls], [[7, 8], [9]])
        self.assertTrue(all(c.kwargs['encoder_only'] and not c.kwargs['need_logits'] for c in calls))
        self.assertEqual(e._prefix_cache['ids'], tuple(prompt + response))
        self.assertEqual(e._prefix_snapshots[6]['ids'], tuple(prompt))
        self.assertEqual(report['cached_tokens'], 9)
        self.assertFalse(e.model._prefix_replay_only)
        e.prefix_disk.save.assert_called_once_with(prompt + response)
        self.assertEqual(e._restore_prefix(prompt + [99]), 6)

    def test_peer_disagreement_skips_all_model_work(self):
        e = self.make()
        e.ep.gather_objects = lambda x: [True, False]
        self.assertEqual(e.cache_response([1, 2, 3, 4, 5, 6], [7])['status'], 'skipped')
        e.model.forward.assert_not_called()

    @patch('torch.cuda.synchronize')
    def test_default_enabled_and_explicit_opt_out(self, sync):
        e = self.make()
        with patch.dict(os.environ):
            os.environ.pop('DSV41_PREFIX_RESPONSE', None)
            self.assertEqual(e.cache_response([1, 2, 3, 4, 5, 6], [7])['status'], 'saved')
        e = self.make()
        with patch.dict(os.environ, {'DSV41_PREFIX_RESPONSE': '0'}):
            self.assertEqual(e.cache_response([1, 2, 3, 4, 5, 6], [7])['status'], 'skipped')
        e.model.forward.assert_not_called()

    def test_strict_route_change_skips(self):
        e = self.make()
        e.prefix_disk.strict, e._prefix_route = True, 'new-route'
        self.assertEqual(e.cache_response([1, 2, 3, 4, 5, 6], [7])['status'], 'skipped')

    @patch('torch.cuda.synchronize')
    def test_replay_flag_cleared_on_failure(self, sync):
        e = self.make()
        e.model.forward.side_effect = RuntimeError('test')
        with self.assertRaises(RuntimeError):
            e.cache_response([1, 2, 3, 4, 5, 6], [7])
        self.assertFalse(e.model._prefix_replay_only)

    def test_replay_does_not_record_demand(self):
        m = SimpleNamespace(_prefix_replay_only=True)
        Model._record_prune_miss(m, None, None, None, 0, 6)


if __name__ == '__main__':
    unittest.main()
