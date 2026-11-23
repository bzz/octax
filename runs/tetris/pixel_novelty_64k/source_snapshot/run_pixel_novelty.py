"""Run the frozen two-seed 64K on-policy discovery pilot after smoke gates."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from flax import serialization
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
SERIES = ROOT/'runs/tetris/pixel_novelty_64k'


def load(path):
    return json.loads(path.read_text())


def smoke_gates():
    results = {}
    configs = {}
    for label, suffix, coefficient in [('control', '_control', 0.), ('novelty', '', .01)]:
        run = ROOT/f'runs/tetris/pixel_novelty_smoke{suffix}'
        assert load(run/'run_status.json')['phase'] == 'complete'
        cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml')); configs[label] = cfg
        assert cfg['seed'] == 101 and cfg['total_timesteps'] == 8192
        assert cfg['novelty']['coefficient'] == coefficient and cfg['normalization_warmup_decisions'] == 0
        summary = load(run/'summary.json')
        assert summary['step'] == 8192 and summary['optimizer_updates'] == 256 and summary['target_unchanged']
        with (run/'metrics.csv').open() as stream: rows = list(csv.DictReader(stream))
        assert [int(r['step']) for r in rows] == [2048, 4096, 6144, 8192]
        assert all(np.isfinite(float(v)) for r in rows for v in r.values())
        assert all(float(r['clipped_grad_norm']) <= .500001 for r in rows)
        full = serialization.msgpack_restore((run/'checkpoint_full.msgpack').read_bytes())
        assert full['format_version'] == 2 and full['method'] == 'online_pixel_novelty'
        assert full['state']['ppo']['agent_ts']['step'] == 256
        assert full['state']['intrinsic_value']['step'] == (256 if coefficient else 0)
        assert full['state']['novelty']['predictor']['step'] == (256 if coefficient else 0)
        assert not full['state']['pending_update']
        evaluations = load(run/'evaluation.json')
        assert [e['step'] for e in evaluations] == [0, 8192]
        assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
        if coefficient:
            assert all(float(r['normalized_novelty_mean']) > 0 for r in rows)
        results[label] = dict(summary=summary, initial=evaluations[0],
            initial_parameter_sha256=load(run/'preflight.json')['initial_parameter_sha256'])
    assert results['control']['initial']['episodes'] == results['novelty']['initial']['episodes']
    assert results['control']['initial_parameter_sha256'] == results['novelty']['initial_parameter_sha256']
    differences = {k for k in configs['control'] if configs['control'][k] != configs['novelty'][k]}
    assert differences == {'run_dir', 'novelty'}
    assert {k for k in configs['control']['novelty'] if configs['control']['novelty'][k] != configs['novelty']['novelty'][k]} == {'coefficient'}
    return results


def main():
    os.chdir(ROOT)
    smoke = smoke_gates()
    assert not SERIES.exists(), 'Series already exists; inspect its actual process/checkpoint before any restart'
    for seed in [1, 2]:
        assert not (ROOT/f'runs/tetris/pixel_novelty_seed{seed}').exists()
    SERIES.mkdir(parents=True)
    (SERIES/'smoke_validation.json').write_text(json.dumps(smoke, indent=2)+'\n')
    frozen = {name: (ROOT/name).read_bytes() for name in
        ['run_pixel_novelty.py', 'train_pixel_novelty.py', 'on_policy_novelty.py', 'pixel_novelty.py',
         'PIXEL_EXPLORATION_PROPOSAL.md', 'evaluate_checkpoint.py', 'visualize_run.py',
         'summarize_pixel_novelty.py', 'summarize_pixel_novelty_smoke.py', 'round_card.py']}
    for name, data in frozen.items():
        dest = SERIES/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    (SERIES/'source_manifest.json').write_text(json.dumps({n: hashlib.sha256(d).hexdigest()
        for n, d in frozen.items()}, indent=2)+'\n')
    completed = []

    def status(phase, **extra):
        temp = SERIES/'status.tmp'
        temp.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), completed=completed,
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2)+'\n')
        temp.replace(SERIES/'run_status.json')

    env = dict(os.environ, JAX_PLATFORMS='cpu')
    try:
        for seed in [1, 2]:
            run = Path(f'runs/tetris/pixel_novelty_seed{seed}')
            commands = [
                ('training', [sys.executable, '-u', 'train_pixel_novelty.py', '--seed', str(seed), '--run', str(run)]),
                ('matched_evaluation', [sys.executable, '-u', 'evaluate_checkpoint.py', str(run/'checkpoint.msgpack'),
                    '--episodes', '128', '--seed-start', '50000', '--out', str(run/'heldout')]),
                ('visualization', [sys.executable, '-u', 'visualize_run.py', '--run', str(run), '--seed', '30016'])]
            with (SERIES/f'seed{seed}.log').open('w') as log:
                for phase, command in commands:
                    assert all((ROOT/n).read_bytes() == d for n, d in frozen.items()), 'Frozen source changed'
                    status(phase, seed=seed, run=str(run))
                    print('PHASE', seed, phase, flush=True)
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if phase == 'training':
                        assert load(ROOT/run/'run_status.json')['phase'] == 'complete'
                        assert load(ROOT/run/'summary.json')['step'] == 65536
            completed.append(str(run))
        status('complete', total_new_training_steps=131072, separate_smoke_training_steps=16384)
        print('PILOT COMPLETE', flush=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
