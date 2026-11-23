"""Fresh CPU v2 baseline followed by corrected-timing controls and visualization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/tetris/v2_baseline_seed0'


def status(phase, **extra):
    payload = dict(phase=phase, pid=os.getpid(), updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temp = RUN / 'run_status.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(RUN / 'run_status.json')


if __name__ == '__main__':
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'config.yaml').exists():
        raise FileExistsError('Baseline already exists; refusing duplicate launch')
    sources = ['train.py','octax/env.py','octax/agents/ppo.py','experiment_utils.py',
               'evaluate_checkpoint.py','visualize_run.py','conf/config.yaml',
               'run_v2_baseline.py','requirements-macos.lock.txt']
    manifest = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        dest = RUN / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (RUN / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, JAX_PLATFORMS='cpu', MPLCONFIGDIR=str(ROOT / '../../work/mpl'))
    commands = [
        ('training', [sys.executable, '-u', 'train.py', 'total_timesteps=65536',
                      'run_dir=runs/tetris/v2_baseline_seed0', 'resume=null', 'encoder_checkpoint=null']),
        ('heldout_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py',
                                str(RUN / 'checkpoint.msgpack'), '--episodes', '128',
                                '--seed-start', '50000', '--out', str(RUN / 'heldout')]),
        ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(RUN)]),
    ]
    try:
        for phase, command in commands:
            status(phase, command=command)
            print(f'PHASE {phase}', flush=True)
            subprocess.run(command, env=env, check=True)
        status('complete', total_timesteps=65536)
        print('V2 BASELINE COMPLETE', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
