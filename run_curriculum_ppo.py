"""Fixed two-seed online PPO pilot; each trainer reads only offline weight tensors."""
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
SERIES = ROOT/'runs/tetris/human_bc_100epoch_ppo_64k'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def admission():
    assert read(ROOT/'runs/tetris/human_bc_duration_replication_seed2/run_status.json')['phase'] == 'complete'
    result = {}
    for seed in [1, 2]:
        parent = ROOT/f'runs/tetris/human_bc_100epoch_seed{seed}'
        summary = read(parent/'pretrain/summary.json')
        audit = read(parent/'analysis.json')
        assert summary['seed'] == seed and summary['epochs'] == 100
        assert summary['training_fit_gate'] and summary['full_cloning_checkpoint_roundtrip_exact'] and summary['weights_roundtrip_exact']
        assert audit['source_snapshot_audits'] and audit['original_controls_exact']
        assert audit['matched_20_epoch_audit']['full_cloning_state_exact']
        assert read(parent/'pretrain/heldout/comparison.json')['monitoring_arrays_exact']
        digest = hashlib.sha256((parent/'pretrain/policy_weights.msgpack').read_bytes()).hexdigest()
        assert digest == summary['policy_weights_sha256']
        assert not (ROOT/f'runs/tetris/human_bc_100epoch_ppo_seed{seed}').exists()
        result[seed] = dict(weights_sha256=digest, offline_candidate_screen=audit['candidate_screen'],
            original_scoring_games=audit['scoring_games'], offline_updates=summary['optimizer_updates'],
            example_presentations=summary['examples_seen'])
    assert result[1]['offline_candidate_screen'], 'Seed-one exploratory trigger must exist'
    return result


def main():
    os.chdir(ROOT)
    parents = admission()
    SERIES.mkdir(parents=True, exist_ok=False)
    write(SERIES/'admission.json', parents)
    names = ['run_curriculum_ppo.py', 'train_pretrained_ppo.py', 'train.py', 'policy_initialization.py',
        'experiment_utils.py', 'evaluate_checkpoint.py', 'visualize_run.py', 'CURRICULUM_PPO_EXPERIMENT.md',
        'requirements-macos.lock.txt', 'roms/Tetris [Fran Dachille, 1991].ch8']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'octax').rglob('*.py'))]
    sources = {name: (ROOT/name).read_bytes() for name in names}
    for name, data in sources.items():
        target = SERIES/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    write(SERIES/'source_manifest.json', {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
    completed = []
    def status(phase, **kwargs):
        temp = SERIES/'status.tmp'
        write(temp, dict(phase=phase, pid=os.getpid(), completed=completed,
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs))
        temp.replace(SERIES/'run_status.json')
    try:
        for seed in [1, 2]:
            parent = Path(f'runs/tetris/human_bc_100epoch_seed{seed}/pretrain')
            run = Path(f'runs/tetris/human_bc_100epoch_ppo_seed{seed}')
            commands = [
                ('training', ['train_pretrained_ppo.py', '--seed', str(seed), '--entropy-coef', '0',
                    '--weights', str(parent/'policy_weights.msgpack'), '--weights-sha256', parents[seed]['weights_sha256'], '--run', str(run)]),
                ('evaluation', ['evaluate_checkpoint.py', str(run/'checkpoint.msgpack'), '--episodes', '128',
                    '--seed-start', '50000', '--out', str(run/'heldout')]),
                ('visualization', ['visualize_run.py', '--run', str(run), '--seed', '30016'])]
            with (SERIES/f'seed{seed}.log').open('w') as log:
                for phase, args in commands:
                    assert all((ROOT/name).read_bytes() == data for name, data in sources.items()), 'Frozen source changed'
                    status(phase, seed=seed, run=str(run))
                    subprocess.run([sys.executable, '-u', *args], check=True, stdout=log, stderr=subprocess.STDOUT)
                    if phase == 'training':
                        assert read(run/'run_status.json')['phase'] == 'complete'
                        cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
                        control = OmegaConf.to_container(OmegaConf.load(ROOT/f'runs/tetris/v2_entropy_zero_seed{seed}/config.yaml'))
                        differences = {key: [control.get(key), cfg.get(key)] for key in cfg.keys() | control.keys() if control.get(key) != cfg.get(key)}
                        assert set(differences) <= {'run_dir', 'initialization', 'initialization_weights_sha256', 'purpose'}, differences
                        evaluations = read(run/'evaluation.json')
                        assert [e['step'] for e in evaluations] == [0, 65536]
                        assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
                        assert evaluations[0]['episodes'] == read(parent/'pretrained_evaluation.json')['episodes']
                        full = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
                        assert int(full['train_state']['global_step']) == 65536
                        assert int(full['train_state']['agent_ts']['step']) == 2048
                        rows = list(csv.DictReader((run/'metrics.csv').read_text().splitlines()))
                        assert [int(row['step']) for row in rows] == list(range(2048, 65537, 2048))
                        assert all(np.isfinite(float(value)) for row in rows for value in row.values())
                        assert all(float(row['clipped_grad_norm']) <= .500001 for row in rows)
                        write(run/'handoff_audit.json', dict(initial_episode_arrays_exact=True,
                            scratch_control_config_differences=differences, source_frozen=True,
                            training_decisions=65536, optimizer_updates=2048,
                            parent_weights_sha256=parents[seed]['weights_sha256']))
                    elif phase == 'evaluation':
                        evaluation = read(run/'heldout/comparison.json')
                        before = read(parent/'heldout/comparison.json')
                        for mode in ['random', 'noop']:
                            assert evaluation[mode]['episodes'] == before[mode]['episodes']
            completed.append(str(run))
        status('complete', total_fresh_ppo_decisions=131072, total_ppo_updates=4096,
            offline_work_counted_separately=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
