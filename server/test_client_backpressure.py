"""A stalled HTTP consumer must release the paired-engine lock, without CUDA."""
import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from server.app import ClientWriteError, Handler


class ClientBackpressureTests(unittest.TestCase):
    def stalled_response(self, path, *, final=False, stream=True):
        lock = threading.Lock()
        entered, closed, finished = (threading.Event() for _ in range(3))

        class ShortTimeoutHandler(Handler):
            timeout = .1

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                finally:
                    finished.set()

        def generate(*args, result, **kwargs):
            entered.set()
            try:
                result.gen_ids = [1]
                result.router = NS(content='answer', reasoning='')
                result.stats = {'large_stats': 's' * (8 * 1024 * 1024)} if final else {}
                yield 'content', 'ok' if final else 'x' * (8 * 1024 * 1024)
            finally:
                closed.set()

        state = NS(lock=lock, request_context=lambda: lock, generate=generate,
                   cache_response=Mock(), ep_fault=None, model_name='deepseek', bos_id=None,
                   tok=NS(encode=lambda _: [1]), enc=NS(), engine=NS(),
                   args=NS(default_thinking=False, default_effort=75))
        server = ThreadingHTTPServer(('127.0.0.1', 0), ShortTimeoutHandler)
        server.daemon_threads = True
        server.state = state
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = json.dumps({'prompt': [1], 'messages': [], 'stream': stream}).encode()
        started = time.monotonic()
        try:
            with patch('server.app.build_chat_prompt', return_value=('', [1], None, None)):
                with socket.socket() as client:
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                    client.settimeout(2)
                    client.connect(server.server_address)
                    client.sendall((f'POST {path} HTTP/1.1\r\nHost: localhost\r\n'
                                    f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                                    'Connection: close\r\n\r\n').encode() + body)
                    # Keep the socket open but consume no answer bytes: a dead or
                    # suspended harness leaves the server blocked in exactly this way.
                    self.assertTrue(entered.wait(1), 'mock generation did not start')
                    self.assertTrue(finished.wait(2), 'HTTP write held engine lock indefinitely')
            self.assertTrue(closed.is_set(), 'active generator was not explicitly closed')
            self.assertFalse(lock.locked())
            self.assertIsNone(state.ep_fault, 'client failure must not fault the engine')
            state.cache_response.assert_not_called()
            self.assertLess(time.monotonic() - started, 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_stalled_chat_stream_closes_active_generator_and_unlocks(self):
        self.stalled_response('/v1/chat/completions')

    def test_stalled_completion_stream_closes_active_generator_and_unlocks(self):
        self.stalled_response('/v1/completions')

    def test_final_stats_write_is_bounded_after_generation_finishes(self):
        self.stalled_response('/v1/chat/completions', final=True)

    def test_nonstream_response_write_is_bounded(self):
        self.stalled_response('/v1/chat/completions', final=True, stream=False)

    def test_socket_write_error_is_marked_client_failure(self):
        handler = object.__new__(Handler)
        handler.wfile = Mock()
        handler.wfile.write.side_effect = OSError(113, 'No route to host')
        with self.assertRaises(ClientWriteError):
            handler._write_client(b'answer')
        self.assertTrue(handler.close_connection)

    def test_faulted_health_returns_503_with_details(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.state = NS(ep_fault='heartbeat failed', model_name='deepseek',
                          args=NS(engine='v41'), lock=threading.Lock(),
                          engine=NS(max_context=524288), started=int(time.time()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/health', timeout=2)
            self.assertEqual(caught.exception.code, 503)
            body = json.load(caught.exception)
            self.assertEqual(body['status'], 'degraded')
            self.assertEqual(body['ep_fault'], 'heartbeat failed')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
