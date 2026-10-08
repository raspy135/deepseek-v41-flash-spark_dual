"""Decode probe transport tests: mocked engine/peer, no CUDA or model service."""
import json
import io
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from server.app import APIError, Handler, State, run_worker


def state(*, paired=True):
    st = object.__new__(State)
    st.lock = threading.Lock()
    st.scheduler, st.ep_fault, st.ep_active = None, None, paired
    st.ep = Mock()
    st.engine = NS(decode_probe_command=Mock(return_value={
        'version': 1, 'action': 'arm', 'ok': True, 'ranks': []}),
        decode_probe_status=Mock(return_value={'armed': True, 'cases': 0}))
    return st


class DecodeProbeStateTests(unittest.TestCase):
    def assert_error(self, status, call):
        with self.assertRaises(APIError) as caught:
            call()
        self.assertEqual(caught.exception.status, status)

    def test_canonical_command_validated_before_broadcast_under_lock(self):
        st = state()
        events = []
        def broadcast(command):
            self.assertTrue(st.lock.locked())
            events.append(('peer', command))
        def command(body):
            self.assertTrue(st.lock.locked())
            events.append(('local', body))
            return {'ok': True}
        st.ep.broadcast_request.side_effect = broadcast
        st.engine.decode_probe_command.side_effect = command
        self.assertEqual(st.decode_probe({'action': 'arm'}), {'ok': True})
        expected = dict(action='arm', names=['*dense*', '*head*', '*markov*'],
                        max_cases=256, snapshot_budget_mb=64, warmup=3,
                        repeats=6, calls=16, flush_mb=64)
        self.assertEqual(events, [('peer', {'cmd': 'decode_probe', 'body': expected}),
                                  ('local', expected)])
        self.assertFalse(st.lock.locked())

    def test_invalid_request_never_reaches_peer(self):
        st = state()
        bad = [dict(action='generate'), dict(unknown=True), dict(names=[]),
               dict(names='*'), dict(names=['x' * 129]), dict(names=['*'] * 65),
               dict(max_cases=0), dict(max_cases=1025), dict(snapshot_budget_mb=0),
               dict(snapshot_budget_mb=257), dict(warmup=-1), dict(warmup=17),
               dict(repeats=0), dict(repeats=17), dict(calls=65), dict(flush_mb=257)]
        for body in bad:
            with self.subTest(body=body):
                self.assert_error(400, lambda: st.decode_probe(body))
        st.ep.broadcast_request.assert_not_called()
        st.engine.decode_probe_command.assert_not_called()
        self.assertFalse(st.lock.locked())

    def test_run_keeps_only_supplied_overrides_after_validation(self):
        st = state()
        st.decode_probe({'action': 'run', 'names': ['*moe*'], 'repeats': 4})
        expected = {'action': 'run', 'names': ['*moe*'], 'repeats': 4}
        st.engine.decode_probe_command.assert_called_once_with(expected)
        st.ep.broadcast_request.assert_called_once_with({'cmd': 'decode_probe', 'body': expected})

    def test_busy_is_nonblocking_and_does_not_broadcast(self):
        st = state()
        st.lock.acquire()
        try:
            self.assert_error(409, lambda: st.decode_probe({'action': 'run'}))
        finally:
            st.lock.release()
        st.ep.broadcast_request.assert_not_called()
        st.engine.decode_probe_command.assert_not_called()

    def test_scheduler_fault_and_unsupported_fail_before_broadcast(self):
        st = state()
        st.scheduler = object()
        self.assert_error(501, lambda: st.decode_probe({'action': 'stop'}))
        st.scheduler = None
        st.ep_fault = 'peer unavailable'
        self.assert_error(503, lambda: st.decode_probe({'action': 'stop'}))
        self.assert_error(503, st.decode_probe_status)
        st.ep_fault = None
        st.engine = NS()
        self.assert_error(501, lambda: st.decode_probe({'action': 'arm'}))
        self.assert_error(501, st.decode_probe_status)
        st.ep.broadcast_request.assert_not_called()

    def test_rank_diagnostic_failure_is_a_report_and_keeps_pair_available(self):
        st = state()
        report = {'version': 1, 'action': 'run', 'ok': False,
                  'ranks': [{'rank': 1, 'ok': False, 'error': 'replay rejected'}]}
        st.engine.decode_probe_command.return_value = report
        self.assertEqual(st.decode_probe({'action': 'run'}), report)
        self.assertIsNone(st.ep_fault)
        self.assertFalse(st.lock.locked())

    def test_unexpected_dispatch_failure_faults_pair_and_releases_lock(self):
        for fail_peer in (False, True):
            with self.subTest(fail_peer=fail_peer):
                st = state()
                failing = st.ep.broadcast_request if fail_peer else st.engine.decode_probe_command
                failing.side_effect = RuntimeError('unexpected dispatch error')
                with self.assertLogs('dsv41.server', level='ERROR'):
                    self.assert_error(503, lambda: st.decode_probe({'action': 'stop'}))
                self.assertIn('unexpected dispatch error', st.ep_fault)
                self.assertFalse(st.lock.locked())
                if fail_peer:
                    st.engine.decode_probe_command.assert_not_called()

    def test_status_is_cached_and_available_while_generation_is_busy(self):
        st = state()
        st.lock.acquire()
        try:
            self.assertEqual(st.decode_probe_status(), {'armed': True, 'cases': 0})
        finally:
            st.lock.release()
        st.ep.broadcast_request.assert_not_called()
        st.engine.decode_probe_command.assert_not_called()

    def test_single_rank_command_needs_no_broadcast(self):
        st = state(paired=False)
        st.decode_probe({'action': 'stop'})
        st.ep.broadcast_request.assert_not_called()
        st.engine.decode_probe_command.assert_called_once()


class DecodeProbeHTTPTests(unittest.TestCase):
    def setUp(self):
        self.state = state()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.state = self.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1/decode-probe'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body):
        return urllib.request.urlopen(urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}), timeout=2)

    def test_get_and_post_route_cache_headers(self):
        self.state.lock.acquire()
        try:
            with urllib.request.urlopen(self.url, timeout=2) as response:
                self.assertEqual(json.load(response), {'armed': True, 'cases': 0})
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
        finally:
            self.state.lock.release()
        with self.post({'action': 'arm'}) as response:
            self.assertTrue(json.load(response)['ok'])
            self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_dashboard_html_does_not_touch_cuda_status_or_generation_lock(self):
        html = b'<!doctype html><title>Decode probe</title>'
        self.state.lock.acquire()
        try:
            with patch('server.app.open', create=True, return_value=io.BytesIO(html)) as page:
                with urllib.request.urlopen(self.url.replace('/v1/decode-probe', '/decode-probe'),
                                            timeout=2) as response:
                    self.assertEqual(response.headers.get_content_type(), 'text/html')
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    self.assertEqual(response.read(), html)
                self.assertTrue(page.call_args.args[0].endswith('/server/decode_probe.html'))
                self.assertEqual(page.call_args.args[1], 'rb')
        finally:
            self.state.lock.release()
        self.state.engine.decode_probe_status.assert_not_called()
        self.state.engine.decode_probe_command.assert_not_called()
        self.state.ep.broadcast_request.assert_not_called()

    def test_invalid_and_busy_http_errors_do_not_broadcast(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'names': []})
        self.assertEqual(caught.exception.code, 400)
        self.state.lock.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post({'action': 'run'})
            self.assertEqual(caught.exception.code, 409)
        finally:
            self.state.lock.release()
        self.state.ep.broadcast_request.assert_not_called()


class DecodeProbeWorkerTests(unittest.TestCase):
    @patch.dict(os.environ, {'DSV41_MAX_CONCURRENCY': '1'})
    def test_worker_dispatches_diagnostic_without_generic_generate(self):
        ep = Mock(rank=1)
        body = {'action': 'run'}
        ep.broadcast_request.side_effect = [
            {'cmd': 'decode_probe', 'body': body}, {'cmd': 'ping'}, {'cmd': 'shutdown'}]
        engine = NS(ep=ep, decode_probe_command=Mock(return_value={'ok': False}),
                    generate=Mock(), close=Mock())
        with patch('server.app.os._exit') as die:
            run_worker(engine)
        engine.decode_probe_command.assert_called_once_with(body)
        engine.generate.assert_not_called()
        die.assert_not_called()
        engine.close.assert_called_once()
        ep.destroy.assert_called_once()

    @patch.dict(os.environ, {'DSV41_MAX_CONCURRENCY': '1'})
    def test_unexpected_worker_error_exits_instead_of_serving_next_request(self):
        ep = Mock(rank=1)
        ep.broadcast_request.side_effect = [{'cmd': 'decode_probe', 'body': {'action': 'run'}}]
        engine = NS(ep=ep, decode_probe_command=Mock(side_effect=RuntimeError('broken guard')),
                    generate=Mock(), close=Mock())
        with patch('server.app.os._exit', side_effect=SystemExit(1)) as die:
            with self.assertLogs('dsv41.server', level='ERROR'), self.assertRaises(SystemExit):
                run_worker(engine)
        die.assert_called_once_with(1)
        engine.generate.assert_not_called()
        engine.close.assert_called_once()
        ep.destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
