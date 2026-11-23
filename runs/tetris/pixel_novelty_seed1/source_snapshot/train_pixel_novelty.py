"""Fresh-run harness for the separately labeled, on-policy pixel-RND experiment.

Smoke: --smoke --run runs/tetris/pixel_novelty_smoke
Pilot: --seed 1 --run runs/tetris/pixel_novelty_seed1
Full checkpoints are restored/verified during the run; checkpoint.msgpack is an
inference export for the existing original-score evaluator, never a resume file.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from flax import serialization
import jax
import numpy as np
from omegaconf import OmegaConf

from experiment_utils import environment_from_config, evaluate, make_evaluator, system_info, write_json
from octax.agents import PPOOctax
from octax.wrappers import OctaxGymnaxWrapper
from on_policy_novelty import OnlinePixelPPO, export_policy, restore_online_checkpoint, save_online_checkpoint
from pixel_novelty import PixelNovelty

ROOT = Path(__file__).resolve().parent
OPTIONS = ['num_envs', 'num_steps', 'num_minibatches', 'num_epochs', 'learning_rate',
    'total_timesteps', 'eval_freq', 'gamma', 'gae_lambda', 'clip_eps', 'vf_coef',
    'ent_coef', 'max_grad_norm', 'advantage_std_floor', 'grad_clip_mode', 'zero_init_value']
SOURCES = ['train_pixel_novelty.py', 'on_policy_novelty.py', 'pixel_novelty.py',
    'experiment_utils.py', 'evaluate_initial_policy.py', 'evaluate_checkpoint.py',
    'visualize_run.py', 'round_card.py', 'PIXEL_EXPLORATION_PROPOSAL.md',
    'requirements-macos.lock.txt', 'roms/Tetris [Fran Dachille, 1991].ch8']
SOURCES += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'octax').rglob('*.py'))]


def canonical_hash(tree):
    return hashlib.sha256(serialization.to_bytes(jax.tree.map(np.asarray, tree))).hexdigest()


def exact(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def run(out, seed, smoke=False, coefficient=.01):
    os.chdir(ROOT)
    out = Path(out).resolve()
    # A new directory is mandatory, including for failed/partial previous runs.
    relative = str(out.relative_to(ROOT))
    out.mkdir(parents=True, exist_ok=False)
    control = ROOT/f'runs/tetris/v2_credit_horizon_seed{seed if not smoke else 1}'
    reference_cfg = OmegaConf.to_container(OmegaConf.load(control/'config.yaml'))
    cfg = dict(reference_cfg)
    cfg.update(seed=seed, run_dir=relative, total_timesteps=8192 if smoke else 65536,
        eval_freq=8192 if smoke else 65536, eval_episodes=2 if smoke else 8,
        method='online_pixel_novelty', purpose='smoke_only' if smoke else 'discovery_pilot',
        novelty=dict(coefficient=coefficient, learning_rate=.0001, gamma=.99,
            variance_floor=.0001, features=128, intrinsic_value_coefficient=.5),
        normalization_warmup_decisions=0)
    assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None
    batch_size = cfg['num_envs'] * cfg['num_steps']
    assert cfg['total_timesteps'] % batch_size == 0
    assert batch_size // cfg['num_minibatches'] == 128
    OmegaConf.save(OmegaConf.create(cfg), out/'config.yaml')

    def status(phase, **extra):
        temp = out/'status.tmp'
        write_json(temp, dict(phase=phase, pid=os.getpid(),
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))
        temp.replace(out/'run_status.json')

    try:
        status('preflight', step=0)
        source_bytes = {name: (ROOT/name).read_bytes() for name in SOURCES}
        control_sources = json.loads((control/'source_manifest.json').read_text())
        for name in ['octax/agents/ppo.py', 'octax/env.py', 'octax/wrappers.py']:
            assert hashlib.sha256(source_bytes[name]).hexdigest() == control_sources[name]
        for name, data in source_bytes.items():
            target = out/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        write_json(out/'source_manifest.json', {name: hashlib.sha256(data).hexdigest()
                                               for name, data in source_bytes.items()})
        write_json(out/'system.json', system_info())
        raw, _ = environment_from_config(cfg)
        env = OctaxGymnaxWrapper(raw)
        agent = PPOOctax.create_agent({}, env, env.default_params)
        base = PPOOctax(env=env, env_params=env.default_params, agent=agent,
            eval_callback=None, **{key: cfg[key] for key in OPTIONS})
        novelty_cfg = cfg['novelty']
        online = OnlinePixelPPO(base, raw, coefficient=coefficient,
            intrinsic_value_coefficient=novelty_cfg['intrinsic_value_coefficient'],
            novelty=PixelNovelty(**{k: novelty_cfg[k] for k in
                ['learning_rate', 'gamma', 'variance_floor', 'features']}))
        initial = online.init(jax.random.PRNGKey(seed))
        state = initial
        original_params = jax.device_get(initial.ppo.agent_ts.params)
        initial_hash = hashlib.sha256(serialization.msgpack_serialize(
            jax.device_get(serialization.to_state_dict(original_params)))).hexdigest()
        target_hash = canonical_hash(initial.novelty.target_params)
        if not smoke:
            reference = json.loads((ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json').read_text())
            assert initial_hash == reference['initial_parameter_sha256']
        preflight = dict(initial_parameter_sha256=initial_hash, target_sha256=target_hash,
            matched_control=None if smoke else str(control.relative_to(ROOT)),
            training_decisions=cfg['total_timesteps'], normalization_warmup_decisions=0,
            expected_optimizer_updates=cfg['total_timesteps']//batch_size*cfg['num_epochs']*cfg['num_minibatches'],
            control_config_sha256=hashlib.sha256((control/'config.yaml').read_bytes()).hexdigest(),
            unchanged_environment_and_base_ppo_sources=True,
            config_differences={k: [reference_cfg.get(k), cfg.get(k)] for k in cfg.keys()|reference_cfg.keys()
                                if cfg.get(k) != reference_cfg.get(k)})
        write_json(out/'preflight.json', preflight)
        evaluator = make_evaluator(raw, agent, cfg['eval_episodes'], cfg['eval_seed_start'])
        evaluations = []

        def checkpoint():
            assert all((ROOT/name).read_bytes() == data for name, data in source_bytes.items()), 'Frozen source changed'
            assert canonical_hash(state.novelty.target_params) == target_hash
            save_online_checkpoint(out/'checkpoint_full.msgpack', state, cfg)

        def assess():
            checkpoint()
            restored = restore_online_checkpoint(out/'checkpoint_full.msgpack', initial, cfg)
            exact(state, restored)
            export_policy(out/'checkpoint.msgpack', restored, cfg)
            result = evaluate(evaluator, restored.ppo.agent_ts.params, 'policy')
            result.update(step=int(state.ppo.global_step), checkpoint_roundtrip_exact=True)
            if not smoke and result['step'] == 0:
                expected = json.loads((control/'evaluation.json').read_text())[0]
                assert result['episodes'] == expected['episodes']
            evaluations.append(result); write_json(out/'evaluation.json', evaluations)
            print('EVAL', result['step'], result['summary'], flush=True)

        status('initial_evaluation', step=0); assess()
        metric_keys = ['loss', 'actor_loss', 'value_loss', 'intrinsic_value_loss', 'entropy',
            'approx_kl', 'clip_fraction', 'extrinsic_advantage_std', 'intrinsic_advantage_std',
            'combined_advantage_std', 'grad_norm', 'clipped_grad_norm', 'predictor_loss',
            'raw_novelty_mean', 'normalized_novelty_mean', 'original_score_sum']
        fields = ['step', *metric_keys, 'weighted_intrinsic_advantage_std', 'reward_sum',
            'positive_reward_steps', 'done_count', 'collect_seconds', 'update_seconds', 'steps_per_second']
        fields += [f'action_{i}_fraction' for i in range(raw.num_actions)]
        elapsed = []; reward_events = completed = 0
        with (out/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader()
            while int(state.ppo.global_step) < cfg['total_timesteps']:
                status('training', step=int(state.ppo.global_step))
                start = time.perf_counter()
                collected, rollout = online.collect(state); jax.block_until_ready((collected, rollout))
                collect_time = time.perf_counter() - start
                start = time.perf_counter()
                state, metrics = online.update(collected, rollout); jax.block_until_ready((state, metrics))
                update_time = time.perf_counter() - start
                assert all(np.isfinite(x).all() for x in jax.tree.leaves(jax.device_get(state)))
                values = {k: float(metrics.get(k, 0.)) for k in metric_keys}
                if coefficient == 0:
                    values['extrinsic_advantage_std'] = float(metrics['advantage_std'])
                    values['combined_advantage_std'] = float(metrics['advantage_std'])
                values['original_score_sum'] = float(rollout.trajectory.reward.sum())
                assert all(np.isfinite(list(values.values())))
                counts = np.bincount(np.asarray(rollout.trajectory.action).ravel(), minlength=raw.num_actions)
                row = dict(values, step=int(state.ppo.global_step),
                    weighted_intrinsic_advantage_std=coefficient*values['intrinsic_advantage_std'],
                    reward_sum=float(rollout.trajectory.reward.sum()),
                    positive_reward_steps=int((rollout.trajectory.reward > 0).sum()),
                    done_count=int(rollout.trajectory.done.sum()), collect_seconds=collect_time,
                    update_seconds=update_time, steps_per_second=batch_size/(collect_time+update_time))
                row.update({f'action_{i}_fraction': float(c/counts.sum()) for i, c in enumerate(counts)})
                writer.writerow(row); stream.flush()
                elapsed.append((collect_time, update_time))
                reward_events += row['positive_reward_steps']; completed += row['done_count']
                checkpoint()
                print('TRAIN', row, flush=True)
                # No previous-rollout transitions survive to the next collect.
                del collected, rollout, metrics
        status('final_evaluation', step=int(state.ppo.global_step)); assess()
        expected_updates = preflight['expected_optimizer_updates']
        assert int(state.ppo.agent_ts.step) == expected_updates
        assert int(state.intrinsic_value.step) == (expected_updates if coefficient else 0)
        assert int(state.novelty.predictor.step) == (expected_updates if coefficient else 0)
        parameter_change = sum(float(np.square(a-b).sum()) for a, b in zip(
            jax.tree.leaves(original_params), jax.tree.leaves(jax.device_get(state.ppo.agent_ts.params))))**.5
        write_json(out/'summary.json', dict(step=int(state.ppo.global_step), seed=seed,
            parameter_change_l2=parameter_change, training_reward_events=reward_events,
            training_completed_games=completed, optimizer_updates=expected_updates,
            training_steps_per_second=cfg['total_timesteps']/np.sum(elapsed),
            steady_training_steps_per_second=(len(elapsed)-1)*batch_size/np.sum(elapsed[1:]),
            collect_seconds=float(np.sum(elapsed, axis=0)[0]), update_seconds=float(np.sum(elapsed, axis=0)[1]),
            evaluation_decisions=sum(sum(e['episodes']['lengths']) for e in evaluations),
            normalization_warmup_decisions=0, target_unchanged=True,
            final_evaluation=evaluations[-1]['summary'], purpose=cfg['purpose']))
        status('complete', step=int(state.ppo.global_step)); print('COMPLETE', relative, flush=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, choices=[1, 2], default=1)
    parser.add_argument('--smoke', action='store_true', help='Separate 8192-step, seed-101 diagnostic')
    parser.add_argument('--coefficient', type=float, choices=[0., .01], default=.01)
    args = parser.parse_args()
    run(args.run, 101 if args.smoke else args.seed, args.smoke, args.coefficient)
