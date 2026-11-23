"""Reproduce the first rewarded PPO batch and inspect isolated update choices.

This diagnostic does not modify the running pilot or export candidate policies.
Its environment transitions repeat a recorded prefix; they are not new evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from flax import serialization, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from experiment_utils import environment_from_config, write_json
from octax.agents.ppo import PPOOctax, gradient_clipper
from octax.checkpoints import restore_checkpoint
from octax.wrappers import OctaxGymnaxWrapper

ROOT = Path(__file__).resolve().parent


@struct.dataclass
class TraceBatch:
    trajectories: object
    advantages: object
    targets: object
    old_log_probs: object


def all_log_probs(module, obs):
    return jax.nn.log_softmax(module.actor._action_dist(module.encode(obs)).logits, axis=-1)


def traced_updater(algo, kl_limit=None):
    def update(ts, trajectories):
        last = algo.agent.apply(ts.agent_ts.params, ts.last_obs, method='call_critic')
        advantages, targets = algo.calculate_gae(trajectories, jnp.where(ts.last_done, 0, last))
        shape = trajectories.obs.shape
        old_lp = algo.agent.apply(ts.agent_ts.params, trajectories.obs.reshape(-1, *shape[2:]), method=all_log_probs)
        old_lp = old_lp.reshape(*shape[:2], -1)
        batch = TraceBatch(trajectories, advantages, targets, old_lp)
        def epoch(carry, unused):
            ts, stopped = carry
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatches = algo.shuffle_and_split(batch, minibatch_rng)
            def minibatch(carry, mb):
                ts, stopped = carry
                lp = algo.agent.apply(ts.agent_ts.params, mb.trajectories.obs, method=all_log_probs)
                exact_kl = (jnp.exp(mb.old_log_probs)*(mb.old_log_probs-lp)).sum(axis=-1).mean()
                proposed, metrics = algo.update_with_metrics(ts, mb)
                if kl_limit is not None:
                    stopped = stopped | (exact_kl > kl_limit)
                ts = jax.lax.cond(stopped, lambda: ts, lambda: proposed)
                return (ts, stopped), dict(metrics, exact_categorical_kl_pre=exact_kl,
                    accepted=(~stopped).astype(jnp.int32))
            return jax.lax.scan(minibatch, (ts, stopped), minibatches)
        (state, stopped), metrics = jax.lax.scan(epoch, (ts, jnp.bool_(False)), None, algo.num_epochs)
        return state, metrics
    return jax.jit(update)


def max_difference(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    differences = []
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if np.asarray(x).dtype.kind in 'biu':
            np.testing.assert_array_equal(x, y)
        else:
            differences.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
    return max(differences, default=0.)


def policy_metrics(agent, before, after, obs):
    initial = agent.apply(before, obs, method=all_log_probs)
    final = agent.apply(after, obs, method=all_log_probs)
    p, q = jnp.exp(initial), jnp.exp(final)
    result = dict(exact_full_batch_kl=float((p*(initial-final)).sum(axis=-1).mean()),
        final_entropy=float(-(q*final).sum(axis=-1).mean()),
        final_mean_action_probabilities=np.asarray(q.mean(axis=0)).tolist(),
        final_argmax_action_fractions=np.bincount(np.asarray(final.argmax(axis=-1)), minlength=5).astype(float).tolist())
    result['final_argmax_action_fractions'] = [v/len(obs) for v in result['final_argmax_action_fractions']]
    diffs = [np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
        for name in ['features', 'actor'] for a, b in zip(jax.tree.leaves(after['params'][name]), jax.tree.leaves(before['params'][name]))]
    result['encoder_actor_change_l2'] = float(np.sqrt(sum(np.sum(d*d) for d in diffs)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=Path('runs/tetris/human_bc_100epoch_ppo_seed1'))
    parser.add_argument('--out', type=Path, default=Path('runs/tetris/curriculum_first_reward_diagnostic'))
    parser.add_argument('--fast-prefix', action='store_true', help='Validate then analytically advance the known zero-gradient prefix')
    parser.add_argument('--stability-followup', action='store_true', help='Compare original with smaller-rate plus KL-stop profiles')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.run/'config.yaml'))
    recorded = list(csv.DictReader((args.run/'metrics.csv').read_text().splitlines()))
    event_index = next(i for i, r in enumerate(recorded) if int(r['positive_reward_steps']))
    assert event_index == 13 and int(recorded[event_index]['step']) == 28672
    args.out.mkdir(parents=True, exist_ok=False)
    def status(phase, **extra):
        write_json(args.out/'run_status.json', dict(phase=phase, **extra))
    status('reproducing_prefix')
    source_names = ['diagnose_curriculum_first_reward.py', 'octax/agents/ppo.py', 'octax/checkpoints.py',
        'octax/wrappers.py', 'experiment_utils.py', 'requirements-macos.lock.txt']
    manifest = {}
    for name in source_names:
        data = (ROOT/name).read_bytes(); target = args.out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(args.out/'source_manifest.json', manifest)
    for name, digest in json.loads((args.run/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    data = args.initial_checkpoint.read_bytes()
    (args.out/'initial_checkpoint.msgpack').write_bytes(data)
    raw, _ = environment_from_config(cfg); env = OctaxGymnaxWrapper(raw)
    agent = PPOOctax.create_agent({}, env, env.default_params)
    keys = ['num_envs', 'num_steps', 'num_minibatches', 'num_epochs', 'learning_rate', 'total_timesteps',
        'eval_freq', 'gamma', 'gae_lambda', 'clip_eps', 'vf_coef', 'ent_coef', 'max_grad_norm', 'advantage_std_floor',
        'grad_clip_mode', 'zero_init_value']
    algo = PPOOctax(env=env, env_params=env.default_params, agent=agent,
        eval_callback=lambda *args: (), **{k: cfg[k] for k in keys})
    state, _ = restore_checkpoint(args.initial_checkpoint, algo.init_state(jax.random.PRNGKey(cfg['seed'])))
    assert int(state.global_step) == int(state.agent_ts.step) == 0
    initial_params = state.agent_ts.params
    collector, updater = jax.jit(algo.collect_trajectories), jax.jit(algo.update_trajectories)
    @jax.jit
    def advance_zero_gradient(ts):
        rng = ts.rng
        for _ in range(cfg['num_epochs']):
            rng, _ = jax.random.split(rng)
        count = cfg['num_epochs']*cfg['num_minibatches']
        # This fixed optimizer has exactly one integer leaf: Adam's clock.
        opt_state = jax.tree.map(lambda x: x+count if np.issubdtype(x.dtype, np.integer) else x, ts.agent_ts.opt_state)
        return ts.replace(rng=rng, agent_ts=ts.agent_ts.replace(step=ts.agent_ts.step+count, opt_state=opt_state))
    audits = []
    fast_proof = None
    try:
        for i in range(event_index+1):
            state, trajectory = collector(state)
            jax.block_until_ready((state, trajectory))
            row = recorded[i]
            assert int(state.global_step) == int(row['step'])
            assert float(trajectory.reward.sum()) == float(row['reward_sum'])
            assert int((trajectory.reward > 0).sum()) == int(row['positive_reward_steps'])
            assert int(trajectory.done.sum()) == int(row['done_count'])
            for action in range(5):
                assert float((trajectory.action == action).mean()) == float(row[f'action_{action}_fraction'])
            if i == event_index:
                break
            if args.fast_prefix:
                assert cfg['ent_coef'] == 0 and float(trajectory.reward.sum()) == 0
                assert np.all(np.asarray(trajectory.reward) == 0) and np.all(np.asarray(trajectory.value) == 0)
                assert np.all(np.asarray(agent.apply(state.agent_ts.params, state.last_obs, method='call_critic')) == 0)
                assert np.isfinite(np.asarray(trajectory.log_prob)).all()
                leaves = jax.tree.leaves(state.agent_ts.opt_state)
                clocks = [x for x in leaves if np.issubdtype(x.dtype, np.integer)]
                assert len(clocks) == 1 and clocks[0].shape == ()
                assert all(np.all(np.asarray(x) == 0) for x in leaves if np.issubdtype(x.dtype, np.floating))
                advanced = advance_zero_gradient(state)
                if fast_proof is None:
                    ordinary, proof_metrics = updater(state, trajectory)
                    jax.block_until_ready((ordinary, proof_metrics))
                    error = max_difference(ordinary, advanced)
                    assert error == 0
                    fast_proof = dict(full_state_max_error=error, actual_proof_optimizer_updates=64,
                        all_optimizer_moments_zero=True, sole_integer_optimizer_leaf_is_clock=True)
                    metric_keys = tuple(proof_metrics)
                flat = trajectory.obs.reshape(-1, *trajectory.obs.shape[2:])
                lp = agent.apply(state.agent_ts.params, flat, method=all_log_probs)
                metrics = {key: jnp.float32(0) for key in metric_keys}
                metrics['entropy'] = -(jnp.exp(lp)*lp).sum(axis=-1).mean()
                state = advanced
            else:
                state, metrics = updater(state, trajectory)
            jax.block_until_ready((state, metrics))
            for key, value in metrics.items():
                np.testing.assert_allclose(value, float(row[key]), atol=1e-7, rtol=1e-5)
            assert max_difference(state.agent_ts.params, initial_params) == 0.
            audits.append(dict(step=int(state.global_step), metrics_match=True, parameters_unchanged=True,
                method='validated_zero_gradient_advance' if args.fast_prefix else 'ordinary_updater'))
            print('PREFIX EXACT', int(state.global_step), flush=True)
            status('reproducing_prefix', step=int(state.global_step))
        base = state
        oracle, oracle_metrics = updater(base, trajectory)
        jax.block_until_ready((oracle, oracle_metrics))
        for key, value in oracle_metrics.items():
            np.testing.assert_allclose(value, float(recorded[event_index][key]), atol=1e-7, rtol=1e-5)
        obs = trajectory.obs.reshape(-1, *trajectory.obs.shape[2:])
        # Validate the all-action probability route against the public method.
        lp = agent.apply(base.agent_ts.params, obs, method=all_log_probs)
        chosen = trajectory.action.reshape(-1)
        public = agent.apply(base.agent_ts.params, obs, None, chosen)[1]
        np.testing.assert_allclose(np.asarray(lp)[np.arange(len(obs)), np.asarray(chosen)], public, atol=1e-6, rtol=1e-6)
        branches = [
            ('original', algo, base, None),
            ('learning_rate_10x_smaller', algo, base.replace(agent_ts=base.agent_ts.replace(tx=optax.chain(
                gradient_clipper(cfg['grad_clip_mode'], cfg['max_grad_norm']), optax.adam(cfg['learning_rate']/10)))), None),
            ('fresh_adam_moments_and_clock', algo, base.replace(agent_ts=base.agent_ts.replace(opt_state=base.agent_ts.tx.init(base.agent_ts.params))), None),
            ('actor_loss_only', algo.replace(vf_coef=0.), base, None),
            ('exact_kl_stop_0015', algo, base, .015)]
        if args.stability_followup:
            def smaller_rate(divisor):
                return base.replace(agent_ts=base.agent_ts.replace(tx=optax.chain(
                    gradient_clipper(cfg['grad_clip_mode'], cfg['max_grad_norm']), optax.adam(cfg['learning_rate']/divisor))))
            branches = [('original', algo, base, None),
                ('learning_rate_10x_smaller_and_kl_stop', algo, smaller_rate(10), .015),
                ('learning_rate_100x_smaller_and_kl_stop', algo, smaller_rate(100), .015)]
        results = {}
        for name, branch_algo, branch_state, limit in branches:
            status('counterfactual_update', branch=name)
            final, metrics = traced_updater(branch_algo, limit)(branch_state, trajectory)
            jax.block_until_ready((final, metrics))
            arrays = {k: np.asarray(v).reshape(-1) for k, v in metrics.items()}
            with (args.out/f'{name}_minibatches.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, ['minibatch', *arrays]); writer.writeheader()
                for i in range(64):
                    writer.writerow(dict(minibatch=i+1, **{k: float(v[i]) for k, v in arrays.items()}))
            result = dict(policy_metrics(agent, base.agent_ts.params, final.agent_ts.params, obs),
                accepted_optimizer_steps=int(metrics['accepted'].sum()), gradient_calculations=64,
                mean_approx_kl_pre=float(metrics['approx_kl'].mean()),
                max_exact_minibatch_kl_pre=float(metrics['exact_categorical_kl_pre'].max()))
            if name == 'original':
                result['traced_vs_original_full_state_max_error'] = max_difference(final, oracle)
                assert result['traced_vs_original_full_state_max_error'] < 1e-5
                for key, value in oracle_metrics.items():
                    np.testing.assert_allclose(metrics[key].mean(), value, atol=1e-6, rtol=1e-5)
            results[name] = result
            write_json(args.out/'branches.json', results)
            print('BRANCH', name, result, flush=True)
        summary = dict(recorded_run=str(args.run), reproduced_reward_step=28672,
            initial_checkpoint_sha256=hashlib.sha256(data).hexdigest(), prefix_audits=audits,
            original_rewarded_update_metrics_match=True, all_action_probabilities_match_public_api=True,
            optimizer_steps_before_reward=int(base.agent_ts.step),
            fast_prefix=args.fast_prefix, fast_prefix_proof=fast_proof,
            diagnostic_repeated_environment_decisions=28672, newly_independent_experience_decisions=0,
            prefix_optimizer_state_advances=832,
            prefix_gradient_calculations=64 if args.fast_prefix else 832, oracle_optimizer_updates=64,
            counterfactual_gradient_calculations=64*len(branches), policies_exported=False, branches=results,
            limitations=['Single recorded first-reward batch, not an online learning experiment.',
                'No policy branch is retained for training; current pilot remains unchanged.',
                'KL-stop checks current minibatch before accepting a proposed step; it is not a hard full-rollout bound.',
                'Instrumented stopping still computes all proposals to retain diagnostics; accepted update counts are separate.',
                'Lower KL/retained preferences alone do not establish better gameplay.'])
        write_json(args.out/'analysis.json', summary)
        status('complete', repeated_environment_decisions=28672, policies_exported=False)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
