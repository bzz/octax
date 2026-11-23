"""Run the fixed-budget longer-credit ablation and matched-seed evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/tetris/v2_credit_horizon_seed0'


def status(phase, **extra):
    payload = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temp = RUN / 'run_status.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(RUN / 'run_status.json')


if __name__ == '__main__':
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'config.yaml').exists():
        raise FileExistsError('Ablation already exists; refusing to overwrite it')
    sources = ['train.py', 'octax/env.py', 'octax/agents/ppo.py', 'octax/checkpoints.py',
               'octax/wrappers.py', 'experiment_utils.py', 'evaluate_checkpoint.py',
               'visualize_run.py', 'conf/config.yaml', 'conf/credit_horizon.yaml',
               'run_credit_ablation.py', 'requirements-macos.lock.txt',
               'roms/Tetris [Fran Dachille, 1991].ch8']
    manifest = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
        dest = RUN / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (RUN / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, JAX_PLATFORMS='cpu', MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR', str(ROOT / '.matplotlib')))
    commands = [
        ('training', [sys.executable, '-u', 'train.py', '--config-name', 'credit_horizon',
                      'total_timesteps=65536', 'seed=0', 'resume=null', 'encoder_checkpoint=null']),
        ('matched_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py',
                                str(RUN.relative_to(ROOT) / 'checkpoint.msgpack'), '--episodes', '128',
                                '--seed-start', '50000', '--out', str(RUN.relative_to(ROOT) / 'heldout')]),
        ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(RUN.relative_to(ROOT))]),
    ]
    try:
        for phase, command in commands:
            status(phase, command=command)
            print(f'PHASE {phase}', flush=True)
            subprocess.run(command, env=env, check=True)
        status('complete', total_timesteps=65536)
        print('CREDIT ABLATION COMPLETE', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
