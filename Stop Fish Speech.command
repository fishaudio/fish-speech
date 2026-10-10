#!/bin/zsh
set -eu
cd "${0:A:h}"
./venv/bin/python - <<'PY'
from pathlib import Path
import os,signal,subprocess
p=Path('.local-voice/server.pid')
if not p.exists():
    print('No Fish Speech server was started by this setup.')
else:
    pid=int(p.read_text().strip())
    command=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if command.returncode==0 and 'local_voice.py' in command.stdout:
        try:
            os.kill(pid,signal.SIGINT)
            print('Stopped Fish Speech. Use Start Fish Speech.command to restart it.')
        except ProcessLookupError:
            print('Fish Speech is already stopped.')
    else:
        print('Fish Speech is already stopped.')
        p.unlink(missing_ok=True)
PY
