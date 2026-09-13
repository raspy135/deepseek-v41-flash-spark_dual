"""Test clean SIGTERM handling with the mock server as Docker PID 1; no GPU needed."""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='deepseek-v41-flash-spark:local')
    parser.add_argument('--model-dir', default=os.environ.get('MODEL_DIR', os.path.expanduser('~/models/DeepSeek-V4.1-Flash')))
    args = parser.parse_args()
    name = 'prefill-sigterm-check-' + uuid.uuid4().hex[:8]
    subprocess.run(['docker', 'run', '-d', '--name', name, '--entrypoint', 'python3',
                    '-p', '127.0.0.1::8000', '-v', args.model_dir+':/models/test:ro', args.image,
                    '/app/server/app.py', '--engine', 'mock', '--model-dir', '/models/test',
                    '--host', '0.0.0.0', '--port', '8000'], check=True, stdout=subprocess.DEVNULL)
    try:
        port = subprocess.check_output(['docker', 'port', name, '8000/tcp'], text=True).strip().split(':')[-1]
        for _ in range(60):
            try:
                urllib.request.urlopen('http://127.0.0.1:'+port+'/health', timeout=1).close()
                break
            except Exception:
                time.sleep(.2)
        else:
            raise AssertionError('mock server never became healthy')
        start = time.monotonic()
        subprocess.run(['docker', 'stop', '-t', '5', name], check=True, stdout=subprocess.DEVNULL)
        elapsed = time.monotonic()-start
        state = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]['State']
        assert state['ExitCode'] == 0, state
        assert elapsed < 3, elapsed
        print(json.dumps(dict(pid1_sigterm_exit_code=state['ExitCode'], shutdown_s=elapsed)))
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
