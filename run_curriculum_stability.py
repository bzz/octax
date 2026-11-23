"""Predeclared smaller-step PPO pilot; independent weights-only training stages."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from flax import serialization
import jax
import numpy as np
from omegaconf import OmegaConf

from policy_initialization import read_policy_weights

ROOT = Path(__file__).resolve().parent
SERIES = ROOT/'runs/tetris/human_bc_stability_ppo_64k'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def audit(run, parent, *, smoke=False):
    budget = 2048 if smoke else 65536
    assert read(run/'run_status.json')['phase'] == 'complete'
    cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
    seed = cfg['seed']
    original = OmegaConf.to_container(OmegaConf.load(ROOT/f'runs/tetris/human_bc_100epoch_ppo_seed{seed}/config.yaml'))
    differences = {key: [original.get(key), cfg.get(key)] for key in cfg.keys() | original.keys() if original.get(key) != cfg.get(key)}
    allowed = {'run_dir', 'learning_rate', 'target_kl'}
    if smoke:
        allowed |= {'purpose', 'total_timesteps', 'eval_freq', 'eval_episodes'}
    assert set(differences) <= allowed, differences
    assert cfg['learning_rate'] == .000005 and cfg['target_kl'] == .015
    assert cfg['ent_coef'] == 0 and cfg['resume'] is None and cfg['encoder_checkpoint'] is None
    for name, digest in read(run/'source_manifest.json').items():
        assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    evaluations = read(run/'evaluation.json')
    assert [e['step'] for e in evaluations] == [0, budget]
    assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
    initial = read(parent/'pretrained_evaluation.json')['episodes']
    if smoke:
        initial = {name: values[:2] for name, values in initial.items()}
    assert evaluations[0]['episodes'] == initial
    full = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
    assert int(full['train_state']['global_step']) == budget
    rows = list(csv.DictReader((run/'metrics.csv').read_text().splitlines()))
    assert [int(row['step']) for row in rows] == list(range(2048, budget+1, 2048))
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())
    assert all(float(row['clipped_grad_norm']) <= .500001 for row in rows)
    counts = {key: sum(int(float(row[key])) for row in rows) for key in
        ['optimizer_updates', 'gradient_evaluations', 'minibatches_checked', 'skipped_minibatches', 'kl_stop_triggered']}
    assert counts['optimizer_updates'] == counts['gradient_evaluations'] == int(full['train_state']['agent_ts']['step'])
    assert counts['optimizer_updates'] == read(run/'summary.json')['optimizer_updates']
    assert counts['optimizer_updates']+counts['skipped_minibatches'] == len(rows)*64
    for row in rows:
        updates, checked, stopped = [int(float(row[k])) for k in ['optimizer_updates', 'minibatches_checked', 'kl_stop_triggered']]
        assert checked == updates+stopped
        if not stopped:
            assert updates == 64
    digest = read(parent/'summary.json')['policy_weights_sha256']
    weights, _ = read_policy_weights(run/'initialization_weights.msgpack', digest)
    params = full['train_state']['agent_ts']['params']['params']
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))
    rewards = sum(float(row['reward_sum']) for row in rows)
    if smoke:
        assert rewards == 0 and counts['optimizer_updates'] == 64
        assert all(float(row['grad_norm']) == 0 for row in rows)
        actual = {'params': {key: params[key] for key in ['features', 'actor']}}
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(weights)):
            np.testing.assert_array_equal(a, b)
        assert evaluations[-1]['episodes'] == initial
    prefix_check = None
    if not smoke and seed == 1:
        old_rows = list(csv.DictReader((ROOT/'runs/tetris/human_bc_100epoch_ppo_seed1/metrics.csv').read_text().splitlines()))
        keys = ['reward_sum', 'positive_reward_steps', 'done_count']+[f'action_{i}_fraction' for i in range(5)]
        for new, old in zip(rows[:14], old_rows[:14]):
            assert all(float(new[k]) == float(old[k]) for k in keys)
        assert all(float(r['grad_norm']) == 0 for r in rows[:13])
        assert float(rows[13]['reward_sum']) == 1 and int(rows[13]['step']) == 28672
        probe_kl = .004025235306471586
        np.testing.assert_allclose(float(rows[13]['final_full_batch_kl']), probe_kl, rtol=1e-4, atol=1e-6)
        assert int(float(rows[13]['optimizer_updates'])) == 64
        prefix_check = dict(recorded_prefix_summaries_exact=True, first_reward_step=28672,
            first_reward_full_batch_kl=float(rows[13]['final_full_batch_kl']), diagnostic_kl=probe_kl)
    result = dict(initial_episode_arrays_exact=True, original_rate_config_differences=differences,
        source_frozen=True, training_decisions=budget, counts=counts,
        parent_weights_sha256=digest, smoke=smoke, seed1_first_reward_check=prefix_check)
    write(run/'handoff_audit.json', result)
    return result


def main():
    os.chdir(ROOT)
    parents = {}
    for seed in [1, 2]:
        parent = ROOT/f'runs/tetris/human_bc_100epoch_seed{seed}/pretrain'
        summary = read(parent/'summary.json')
        assert summary['epochs'] == 100 and summary['seed'] == seed
        assert summary['training_fit_gate'] and summary['full_cloning_checkpoint_roundtrip_exact'] and summary['weights_roundtrip_exact']
        assert read(parent/'heldout/comparison.json')['monitoring_arrays_exact']
        digest = hashlib.sha256((parent/'policy_weights.msgpack').read_bytes()).hexdigest()
        assert digest == summary['policy_weights_sha256']
        parents[seed] = dict(weights_sha256=digest)
        assert not (ROOT/f'runs/tetris/human_bc_stability_ppo_seed{seed}').exists()
    assert not (ROOT/'runs/tetris/human_bc_stability_smoke_seed1').exists()
    SERIES.mkdir(parents=True, exist_ok=False)
    write(SERIES/'admission.json', parents)
    names = ['run_curriculum_stability.py', 'train_pretrained_ppo.py', 'train.py', 'policy_initialization.py',
        'experiment_utils.py', 'evaluate_checkpoint.py', 'visualize_run.py', 'CURRICULUM_STABILITY_EXPERIMENT.md',
        'tests/test_ppo_kl_stop.py', 'requirements-macos.lock.txt', 'roms/Tetris [Fran Dachille, 1991].ch8']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'octax').rglob('*.py'))]
    sources = {name: (ROOT/name).read_bytes() for name in names}
    for name, data in sources.items():
        target = SERIES/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    write(SERIES/'source_manifest.json', {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
    completed, audits = [], {}

    def status(phase, **kwargs):
        temp = SERIES/'status.tmp'
        write(temp, dict(phase=phase, pid=os.getpid(), completed=completed,
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs))
        temp.replace(SERIES/'run_status.json')

    try:
        for label, seed, smoke in [('smoke', 1, True), ('seed1', 1, False), ('seed2', 2, False)]:
            parent = Path(f'runs/tetris/human_bc_100epoch_seed{seed}/pretrain')
            run = Path('runs/tetris/human_bc_stability_smoke_seed1' if smoke else f'runs/tetris/human_bc_stability_ppo_seed{seed}')
            args = ['train_pretrained_ppo.py', '--seed', str(seed), '--entropy-coef', '0',
                '--learning-rate', '.000005', '--target-kl', '.015', '--weights', str(parent/'policy_weights.msgpack'),
                '--weights-sha256', parents[seed]['weights_sha256'], '--run', str(run)]
            if smoke:
                args += ['--smoke']
            commands = [('training', args)]
            if not smoke:
                commands += [('evaluation', ['evaluate_checkpoint.py', str(run/'checkpoint.msgpack'), '--episodes', '128',
                    '--seed-start', '50000', '--out', str(run/'heldout')]),
                    ('visualization', ['visualize_run.py', '--run', str(run), '--seed', '30016'])]
            with (SERIES/f'{label}.log').open('w') as log:
                for phase, args in commands:
                    assert all((ROOT/name).read_bytes() == data for name, data in sources.items()), 'Frozen source changed'
                    status(phase, seed=seed, smoke=smoke, run=str(run))
                    subprocess.run([sys.executable, '-u', *args], check=True, stdout=log, stderr=subprocess.STDOUT)
                    if phase == 'training':
                        audits[label] = audit(run, parent, smoke=smoke)
                    elif phase == 'evaluation':
                        current, before = read(run/'heldout/comparison.json'), read(parent/'heldout/comparison.json')
                        for mode in ['random', 'noop']:
                            assert current[mode]['episodes'] == before[mode]['episodes']
            completed.append(str(run))
        write(SERIES/'audits.json', audits)
        status('complete', total_fresh_ppo_decisions=131072, smoke_decisions=2048,
            total_ppo_updates=sum(audits[label]['counts']['optimizer_updates'] for label in ['seed1', 'seed2']),
            offline_work_counted_separately=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
