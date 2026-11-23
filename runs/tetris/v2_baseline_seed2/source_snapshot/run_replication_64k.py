"""Sequential, fixed-budget baseline/longer-credit replication for seeds 1 and 2."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SERIES = ROOT / 'runs/tetris/v2_replication_64k'
RUNS = [(seed, condition, profile) for seed in (1, 2)
        for condition, profile in [('baseline', 'config'), ('credit_horizon', 'credit_horizon')]]
SOURCES = ['train.py', 'octax/env.py', 'octax/agents/ppo.py', 'octax/checkpoints.py',
           'octax/wrappers.py', 'experiment_utils.py', 'evaluate_checkpoint.py',
           'visualize_run.py', 'compare_random_budget.py', 'conf/config.yaml',
           'conf/credit_horizon.yaml', 'run_replication_64k.py', 'requirements-macos.lock.txt',
           'roms/Tetris [Fran Dachille, 1991].ch8']


def write_status(directory, phase, **extra):
    temp = directory / 'run_status.tmp'
    temp.write_text(json.dumps(dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2)+'\n')
    temp.replace(directory / 'run_status.json')


if __name__ == '__main__':
    os.chdir(ROOT)
    SERIES.mkdir(parents=True, exist_ok=True)
    for seed, condition, _ in RUNS:
        if (ROOT / f'runs/tetris/v2_{condition}_seed{seed}/config.yaml').exists():
            raise FileExistsError(f'{condition} seed {seed} already exists; refusing duplicate launch')
    sources = {name: (ROOT / name).read_bytes() for name in SOURCES}
    env = dict(os.environ, JAX_PLATFORMS='cpu', MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR', str(ROOT / '.matplotlib')))
    completed = []
    try:
        for seed, condition, profile in RUNS:
            relative = Path(f'runs/tetris/v2_{condition}_seed{seed}')
            run = ROOT / relative
            run.mkdir(parents=True, exist_ok=True)
            manifest = {}
            for name, data in sources.items():
                # All four runs use the launch-time source, not subsequent edits.
                if (ROOT / name).read_bytes() != data:
                    raise RuntimeError(f'Training-series source changed after launch: {name}')
                destination = run / 'source_snapshot' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                manifest[name] = hashlib.sha256(data).hexdigest()
            (run / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            commands = [
                ('training', [sys.executable, '-u', 'train.py', '--config-name', profile,
                              f'seed={seed}', 'num_seeds=1', 'total_timesteps=65536',
                              f'run_dir={relative}', 'resume=null', 'encoder_checkpoint=null']),
                ('matched_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py',
                                        str(relative/'checkpoint.msgpack'), '--episodes', '128',
                                        '--seed-start', '50000', '--out', str(relative/'heldout')]),
                ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(relative), '--seed', '30016']),
            ]
            if condition == 'baseline':
                commands.append(('random_budget', [sys.executable, '-u', 'compare_random_budget.py', '--run', str(relative)]))
            with (run/'job.log').open('w') as log:
                for phase, command in commands:
                    write_status(SERIES, phase, run=str(relative), pid=os.getpid(), completed=completed)
                    write_status(run, phase)
                    print(f'{relative}: {phase}', flush=True)
                    print(f'PHASE {phase}', file=log, flush=True)
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_status(run, 'complete', total_timesteps=65536)
            completed.append(str(relative))
            print(f'{relative}: complete', flush=True)
        write_status(SERIES, 'complete', completed=completed, total_new_training_steps=262144)
    except Exception as exc:
        write_status(SERIES, 'failed', completed=completed, error=str(exc))
        raise
