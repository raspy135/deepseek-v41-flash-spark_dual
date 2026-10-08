"""Keepalive failure safety: mocked transport, no model or process group."""
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from server.app import APIError, State, _ep_heartbeat


def state():
    st = object.__new__(State)
    st.lock = threading.Lock()
    st.ep_fault = None
    st.ep = NS(broadcast_request=Mock())
    st.maintain_experts = Mock(return_value=False)
    st.scheduler = None
    st.engine = NS(decode_probe_status=Mock(return_value={}))
    return st


def one_tick():
    return NS(wait=Mock(side_effect=[False, True]), is_set=Mock(return_value=False))


class HeartbeatTests(unittest.TestCase):
    def test_idle_ping_holds_request_lock_and_maintenance_replaces_ping(self):
        for maintenance in (False, True):
            with self.subTest(maintenance=maintenance):
                st = state()
                st.maintain_experts.side_effect = lambda: self.assert_locked(st, maintenance)
                st.ep.broadcast_request.side_effect = lambda cmd: self.assert_locked(st, None)
                _ep_heartbeat(st, 30, one_tick())
                self.assertIsNone(st.ep_fault)
                self.assertFalse(st.lock.locked())
                if maintenance:
                    st.ep.broadcast_request.assert_not_called()
                else:
                    st.ep.broadcast_request.assert_called_once_with({'cmd': 'ping'})

    def assert_locked(self, st, result):
        self.assertTrue(st.lock.locked())
        return result

    def test_transport_failure_latches_before_unlock_and_blocks_later_commands(self):
        st = state()
        released_faults = []
        lock = st.lock

        class ObservedLock:
            def __enter__(self):
                lock.acquire()

            def __exit__(self, *exc):
                released_faults.append(st.ep_fault)
                lock.release()

        st.lock = ObservedLock()

        def fail(command):
            self.assertTrue(lock.locked())
            raise RuntimeError('connection closed by peer')

        st.ep.broadcast_request.side_effect = fail
        with self.assertLogs('dsv41.server', level='WARNING'):
            _ep_heartbeat(st, 30, one_tick())
        self.assertEqual(st.ep_fault, 'heartbeat failed: RuntimeError: connection closed by peer')
        self.assertEqual(released_faults, [st.ep_fault])
        self.assertFalse(lock.locked())
        _ep_heartbeat(st, 30, one_tick())
        st.ep.broadcast_request.assert_called_once_with({'cmd': 'ping'})
        with self.assertRaises(APIError) as caught:
            st.decode_probe_status()
        self.assertEqual(caught.exception.status, 503)
        st.engine.decode_probe_status.assert_not_called()

    def test_tick_waiting_on_request_rechecks_fault_before_broadcast(self):
        st = state()
        lock = st.lock

        class RequestFinishedLock:
            def __enter__(self):
                # The previous request faults while the heartbeat is queued for the lock.
                st.ep_fault = 'request failed after peer dispatch'
                lock.acquire()

            def __exit__(self, *exc):
                lock.release()

        st.lock = RequestFinishedLock()
        _ep_heartbeat(st, 30, one_tick())
        st.maintain_experts.assert_not_called()
        st.ep.broadcast_request.assert_not_called()
        self.assertEqual(st.ep_fault, 'request failed after peer dispatch')
        self.assertFalse(lock.locked())

    def test_escaping_maintenance_failure_is_bounded_and_stops_the_tick(self):
        st = state()
        st.maintain_experts.side_effect = RuntimeError('maintenance failed ' * 80)
        stop = one_tick()
        with self.assertLogs('dsv41.server', level='WARNING'):
            _ep_heartbeat(st, 30, stop)
        self.assertTrue(st.ep_fault.startswith('heartbeat failed: RuntimeError: maintenance failed'))
        self.assertEqual(len(st.ep_fault), 200)
        self.assertFalse(st.lock.locked())
        stop.wait.assert_called_once_with(30)
        st.ep.broadcast_request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
