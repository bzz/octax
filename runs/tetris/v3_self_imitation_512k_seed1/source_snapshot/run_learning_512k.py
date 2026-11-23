"""Continue matched seed-1 PPO and score-replay runs from 64K to 512K."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SERIES = ROOT/'runs/tetris/learning_512k_seed1'
TARGET = 524288
SOURCES = ['run_learning_512k.py', 'train.py', 'train_self_imitation.py',
           'self_imitation.py', 'octax/agents/ppo.py', 'octax/env.py',
           'octax/checkpoints.py', 'octax/wrappers.py', 'experiment_utils.py',
           'conf/config.yaml', 'conf/credit_horizon.yaml', 'conf/self_imitation.yaml',
           'evaluate_checkpoint.py', 'visualize_run.py', 'requirements-macos.lock.txt',
           'LEARNING_512K.md', 'roms/Tetris [Fran Dachille, 1991].ch8']
ARMS = [('credit', 'v2_credit_horizon_seed1', 'v2_credit_horizon_512k_seed1', 'train.py', 'credit_horizon'),
        ('replay', 'v3_self_imitation_seed1', 'v3_self_imitation_512k_seed1', 'train_self_imitation.py', 'self_imitation')]


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.chdir(ROOT)
    SERIES.mkdir(parents=True, exist_ok=True)
    if (SERIES/'run_status.json').exists():
        raise FileExistsError('Continuation series exists; refusing duplicate launch.')
    manifest = {name: digest(ROOT/name) for name in SOURCES}
    for _, parent, output, _, _ in ARMS:
        assert load(ROOT/'runs/tetris'/parent/'run_status.json')['phase'] == 'complete'
        assert load(ROOT/'runs/tetris'/parent/'summary.json')['step'] == 65536
        for name, expected in load(ROOT/'runs/tetris'/parent/'source_manifest.json').items():
            assert digest(ROOT/name) == expected, ('Parent source changed', name)
        if (ROOT/'runs/tetris'/output/'config.yaml').exists():
            raise FileExistsError(output)
    completed = []
    current = None
    env = dict(os.environ, JAX_PLATFORMS='cpu')

    def status(phase, **extra):
        data = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    pid=os.getpid(), completed=completed.copy(), **extra)
        write(SERIES/'run_status.json', data)
        if current is not None:
            write(current/'run_status.json', data)

    try:
        for arm, parent_name, output, trainer, config in ARMS:
            parent = ROOT/'runs/tetris'/parent_name
            current = ROOT/'runs/tetris'/output; current.mkdir(parents=True, exist_ok=True)
            relative = current.relative_to(ROOT)
            assert all(digest(ROOT/name) == expected for name, expected in manifest.items())
            for name in SOURCES:
                path = current/'source_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT/name).read_bytes())
            write(current/'source_manifest.json', manifest)
            parent_files = ['checkpoint.msgpack'] + (['replay.npz', 'checkpoint_pair.json'] if arm == 'replay' else [])
            provenance = dict(parent_run=str(parent.relative_to(ROOT)), starting_step=65536,
                              target_step=TARGET, added_training_steps=TARGET-65536,
                              parent_sha256={name: digest(parent/name) for name in parent_files})
            write(current/'continuation.json', provenance)
            commands = [
                ('training', [sys.executable, '-u', trainer, '--config-name', config,
                              'seed=1', 'num_seeds=1', f'total_timesteps={TARGET}', 'eval_freq=65536',
                              f'run_dir={relative}', f'resume={parent.relative_to(ROOT)}/checkpoint.msgpack',
                              'encoder_checkpoint=null']),
                ('matched_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py', str(relative/'checkpoint.msgpack'),
                                        '--episodes', '128', '--seed-start', '50000', '--out', str(relative/'heldout')]),
                ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(relative), '--seed', '30016']),
            ]
            for phase, command in commands:
                assert all(digest(ROOT/name) == expected for name, expected in manifest.items())
                assert all(digest(parent/name) == expected for name, expected in provenance['parent_sha256'].items())
                status(phase, arm=arm, run=str(relative), command=command)
                print('PHASE', arm, phase, flush=True)
                with (current/'job.log').open('a') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if phase == 'training':
                    history = load(current/'evaluation.json')
                    assert history[0]['step'] == 65536
                    assert history[0]['episodes'] == load(parent/'evaluation.json')[-1]['episodes']
                    assert load(current/'summary.json')['step'] == TARGET
            completed.append(str(relative))
            status('arm_complete', arm=arm, run=str(relative), total_timesteps=TARGET)
            write(current/'run_status.json', {**load(current/'run_status.json'), 'phase': 'complete'})
        current = None
        status('complete', total_new_training_steps=2*(TARGET-65536))
        print('MATCHED 512K SERIES COMPLETE', flush=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
