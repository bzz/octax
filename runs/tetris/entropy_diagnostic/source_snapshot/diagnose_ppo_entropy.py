"""Inspect PPO gradient components on fresh rollouts; save no updated policy."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from rejax.algos.ppo import AdvantageMinibatch

from experiment_utils import environment_from_config, restore_checkpoint
from octax.agents.ppo import PPOOctax, normalize_advantages
from octax.wrappers import OctaxGymnaxWrapper


def diagnose(run):
    run = Path(run)
    cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
    raw, _ = environment_from_config(cfg)
    env = OctaxGymnaxWrapper(raw)
    agent = PPOOctax.create_agent({}, env, env.default_params)
    names = ['num_envs','num_steps','num_minibatches','num_epochs','learning_rate',
             'total_timesteps','eval_freq','gamma','gae_lambda','clip_eps','vf_coef',
             'ent_coef','max_grad_norm','advantage_std_floor','grad_clip_mode','zero_init_value']
    algo = PPOOctax(env=env, env_params=env.default_params, agent=agent,
                    eval_callback=None, **{k:cfg[k] for k in names})
    state, _ = restore_checkpoint(run/'checkpoint.msgpack', algo.init_state(jax.random.PRNGKey(cfg['seed'])))
    start_step = int(state.global_step)
    state, trajectory = jax.jit(algo.collect_trajectories)(state)
    last_value = agent.apply(state.agent_ts.params, state.last_obs, method='call_critic')
    advantages, targets = algo.calculate_gae(trajectory, jnp.where(state.last_done, 0, last_value))
    _, shuffle_key = jax.random.split(state.rng)
    batches = algo.shuffle_and_split(AdvantageMinibatch(trajectory, advantages, targets), shuffle_key)

    def components(params, batch):
        _, lp, entropy, value = agent.apply(params, batch.trajectories.obs, None, batch.trajectories.action)
        ratio = jnp.exp(lp-batch.trajectories.log_prob)
        adv = normalize_advantages(batch.advantages, cfg['advantage_std_floor'])
        policy = -jnp.minimum(ratio*adv, jnp.clip(ratio, 1-cfg['clip_eps'], 1+cfg['clip_eps'])*adv).mean()
        clipped_value = batch.trajectories.value + jnp.clip(value-batch.trajectories.value, -cfg['clip_eps'], cfg['clip_eps'])
        critic = .5*cfg['vf_coef']*jnp.maximum((value-batch.targets)**2, (clipped_value-batch.targets)**2).mean()
        return jnp.stack([policy, -cfg['ent_coef']*entropy.mean(), critic])

    def gradients(batch):
        jacobian = jax.jacrev(components)(state.agent_ts.params, batch)
        return [jax.tree.map(lambda x:x[i], jacobian) for i in range(3)]

    def measure(_, batch):
        policy, entropy, critic = gradients(batch)
        pg, eg, vg = [optax.global_norm(g) for g in [policy, entropy, critic]]
        dot = sum(jnp.sum(a*b) for a,b in zip(jax.tree.leaves(policy), jax.tree.leaves(entropy)))
        return None, dict(policy_gradient_norm=pg, entropy_gradient_norm=eg,
                          value_gradient_norm=vg, entropy_to_policy_norm=eg/jnp.maximum(pg, 1e-12),
                          policy_entropy_cosine=dot/jnp.maximum(pg*eg, 1e-12))
    _, raw_metrics = jax.jit(lambda:jax.lax.scan(measure, None, batches))()
    metrics = jax.device_get(raw_metrics)
    assert all(np.isfinite(v).all() for v in metrics.values())
    first = jax.tree.map(lambda x:x[0], batches)
    policy, entropy, critic = jax.jit(gradients)(first)
    reference_state, reference_metrics = jax.jit(algo.update_with_metrics)(state, first)
    np.testing.assert_allclose(float(components(state.agent_ts.params, first).sum()),
                               float(reference_metrics['loss']), rtol=1e-5, atol=1e-7)
    counterfactual = {}
    validation = {}
    for label, use_entropy in [('configured_entropy', True), ('zero_entropy', False)]:
        total = jax.tree.map(lambda p,e,v:p+v+(e if use_entropy else 0), policy, entropy, critic)
        updated = state.agent_ts.apply_gradients(grads=total)
        if use_entropy:
            differences = []
            for a,b in zip(jax.tree.leaves(updated), jax.tree.leaves(reference_state.agent_ts)):
                a,b = np.asarray(a),np.asarray(b)
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
                differences.append(float(np.max(np.abs(a-b))))
            validation = dict(component_sum_matches_ppo_loss=True,
                              reconstructed_update_matches_ppo=True,
                              maximum_state_element_difference=max(differences))
        _, _, h, _ = agent.apply(updated.params, first.trajectories.obs, None, first.trajectories.action)
        delta = jax.tree.map(lambda a,b:a-b, updated.params, state.agent_ts.params)
        counterfactual[label] = dict(mean_entropy_after_one_minibatch=float(h.mean()),
                                    parameter_delta_l2=float(optax.global_norm(delta)))
    _, _, h, _ = agent.apply(state.agent_ts.params, first.trajectories.obs, None, first.trajectories.action)
    return dict(run=str(run), checkpoint_sha256=hashlib.sha256((run/'checkpoint.msgpack').read_bytes()).hexdigest(),
        checkpoint_step=start_step, fresh_diagnostic_decisions=int(trajectory.reward.size),
        diagnostic_reward_sum=float(trajectory.reward.sum()), diagnostic_positive_events=int((trajectory.reward>0).sum()),
        entropy_coefficient=cfg['ent_coef'], mean_entropy_before=float(h.mean()),
        minibatch_count=len(metrics['policy_gradient_norm']),
        minibatch_gradients={k:v.tolist() for k,v in metrics.items()},
        gradient_summary={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),min=float(np.min(v)),max=float(np.max(v))) for k,v in metrics.items()},
        single_minibatch_counterfactual=counterfactual,
        validation=validation,
        limitations=['Fresh diagnostic rollout from a saved policy; no updated weights are saved or used for training.',
                     'All minibatch gradients are measured at the same parameter state, not along the four-epoch PPO update path.',
                     'Raw gradient norms do not account for Adam scaling; the separate one-minibatch counterfactual includes the saved optimizer state.',
                     'This diagnoses local update pressure, not long-run causal effects or scoring performance.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        result = diagnose(run); results.append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['run','diagnostic_positive_events','gradient_summary','single_minibatch_counterfactual']},indent=2),flush=True)
