"""Run the authorized single-seed continuation, then evaluate and render it."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/tetris/continued_seed0'


def status(phase, **extra):
    payload = dict(phase=phase, pid=os.getpid(), updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temp = RUN / 'run_status.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(RUN / 'run_status.json')


if __name__ == '__main__':
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS='cpu', MPLCONFIGDIR=str(ROOT / '../../work/mpl'))
    commands = [
        ('training', [sys.executable, '-u', 'train.py', 'total_timesteps=499712',
                      'resume=runs/tetris/ppo_seed0_smoke/checkpoint.msgpack',
                      'run_dir=runs/tetris/continued_seed0', 'experiment_version=1',
                      'env_kwargs.timer_mode=legacy', 'advantage_std_floor=0.0',
                      'grad_clip_mode=elementwise', 'zero_init_value=false']),
        ('heldout_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py',
                                str(RUN / 'checkpoint.msgpack'), '--episodes', '64',
                                '--out', str(RUN / 'heldout')]),
        ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(RUN)]),
    ]
    try:
        for phase, command in commands:
            status(phase, command=command)
            print(f'PHASE {phase}', flush=True)
            subprocess.run(command, env=env, check=True)
        status('complete', total_timesteps=499712)
        print('PILOT COMPLETE', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
