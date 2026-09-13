"""Exercise stop orchestration with fake Docker/SSH; never touch live containers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'scripts').mkdir()
        shutil.copyfile(Path(__file__).with_name('dual-down.sh'), root / 'scripts/dual-down.sh')
        (root / '.env').write_text('PEER=test-peer\nSTOP_TIMEOUT=30\nMIN_FREE_GIB=90\n')
        bindir = root / 'bin'
        bindir.mkdir()
        docker = bindir / 'docker'
        docker.write_text('''#!/usr/bin/env python3
import json,os,sys,time
with open(os.environ['STOP_TEST_LOG'],'a') as f:
    f.write(json.dumps([time.monotonic(),sys.argv[1:]])+'\\n')
if sys.argv[1]=='stop': time.sleep(.3)
if sys.argv[1]=='rm' and os.environ.get('STOP_TEST_FAIL'): sys.exit(9)
''')
        ssh = bindir / 'ssh'
        ssh.write_text('#!/usr/bin/env python3\nimport os,sys\nos.execlp("bash","bash","-c",sys.argv[-1])\n')
        docker.chmod(0o755)
        ssh.chmod(0o755)
        log = root / 'calls.jsonl'
        env = dict(os.environ, PATH=str(bindir)+':'+os.environ['PATH'],
                   STOP_TEST_LOG=str(log), STOP_TIMEOUT='1', MIN_FREE_GIB='0')
        for args, expected, fail in (([], '1', False), (['--force'], '0', False), ([], '1', True)):
            log.write_text('')
            trial = dict(env)
            if fail:
                trial['STOP_TEST_FAIL'] = '1'
            start = time.monotonic()
            result = subprocess.run(['bash', str(root/'scripts/dual-down.sh'), *args],
                                    env=trial, capture_output=True, text=True, timeout=5)
            elapsed = time.monotonic()-start
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            stops = [call for call in calls if call[1][0]=='stop']
            assert len(stops)==2, calls
            assert all(call[1][2]==expected for call in stops), calls
            assert abs(stops[0][0]-stops[1][0]) < .25, calls
            assert (result.returncode != 0) == fail, result
            print(json.dumps(dict(force=bool(args), failure_propagated=fail, elapsed_s=round(elapsed,3))))


if __name__ == '__main__':
    main()
