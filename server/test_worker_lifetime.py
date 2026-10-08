"""Worker queue failure exits PID 1 without entering potentially stuck teardown."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkerLifetimeTests(unittest.TestCase):
    def run_worker_process(self, broadcast, cleanup):
        # No CUDA import/process group: run the actual server worker against a tiny queue.
        # A subprocess is necessary to prove os._exit skips finally; mocking it would unwind
        # Python and execute exactly the teardown that the production fix must avoid.
        code = '''
import sys, types, time
torch = types.ModuleType('torch')
torch.distributed = types.ModuleType('torch.distributed')
sys.modules['torch'] = torch
sys.modules['torch.distributed'] = torch.distributed
from server.app import run_worker
class Peer:
    rank = 1
    def broadcast_request(self, payload):
        BROADCAST
    def destroy(self):
        print('DESTROY_ENTERED', flush=True)
class Engine:
    ep = Peer()
    def close(self):
        CLEANUP
run_worker(Engine())
print('WORKER_RETURNED', flush=True)
'''.replace('        BROADCAST', textwrap.indent(textwrap.dedent(broadcast).strip(), '        '))
        code = code.replace('        CLEANUP', textwrap.indent(textwrap.dedent(cleanup).strip(), '        '))
        env = dict(os.environ, DSV41_MAX_CONCURRENCY='1')
        return subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=5)

    def test_broadcast_failure_exits_nonzero_before_blocking_cleanup(self):
        result = self.run_worker_process('''
            print('BROADCAST_FAILED', end='')
            raise RuntimeError('simulated queue timeout')
        ''', '''
            print('CLEANUP_ENTERED', flush=True)
            time.sleep(30)
        ''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('BROADCAST_FAILED', result.stdout)  # buffered output was flushed
        self.assertNotIn('CLEANUP_ENTERED', result.stdout)
        self.assertNotIn('DESTROY_ENTERED', result.stdout)
        self.assertNotIn('WORKER_RETURNED', result.stdout)
        self.assertIn('communicator is unusable', result.stderr)

    def test_explicit_shutdown_and_closed_queue_still_clean_up_normally(self):
        for response in ("{'cmd': 'shutdown'}", 'None'):
            with self.subTest(response=response):
                result = self.run_worker_process('return ' + response,
                                                 "print('CLEANUP_ENTERED', flush=True)")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 ['CLEANUP_ENTERED', 'DESTROY_ENTERED', 'WORKER_RETURNED'])


if __name__ == '__main__':
    unittest.main()
