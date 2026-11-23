"""Integration diagnostics only: tiny rollouts, no learning-performance claim."""
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from octax.agents import PPOOctax
from octax.checkpoints import restore_checkpoint
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from on_policy_novelty import (OnlinePixelPPO, export_policy,
    restore_online_checkpoint, save_online_checkpoint)


def equal(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def changed(a, b):
    return any(not np.array_equal(x, y) for x, y in
               zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.fixture(scope='module')
def models():
    raw, _ = create_environment('tetris', timer_mode='frame')
    env = OctaxGymnaxWrapper(raw)
    agent = PPOOctax.create_agent({}, env, env.default_params)
    base = PPOOctax(env=env, env_params=env.default_params, agent=agent,
        eval_callback=lambda *args: (), num_envs=2, num_steps=8,
        num_epochs=2, num_minibatches=2, learning_rate=.0005,
        total_timesteps=32, eval_freq=32, gamma=.995, gae_lambda=.98,
        clip_eps=.2, vf_coef=.5, ent_coef=.01, max_grad_norm=.5,
        advantage_std_floor=.01, grad_clip_mode='global_norm', zero_init_value=True)
    return base, OnlinePixelPPO(base, raw, coefficient=0), OnlinePixelPPO(base, raw)


def test_disabled_path_exactly_matches_initialization_collection_and_updates(models):
    base, inactive, _ = models
    key = jax.random.PRNGKey(1)
    original = base.init_state(key)
    state = initial = inactive.init(key)
    equal(original, state.ppo)
    collect, update = jax.jit(base.collect_trajectories), jax.jit(base.update_trajectories)
    for i in range(2):
        original, trajectory = collect(original)
        state, rollout = inactive.collect(state)
        equal(original, state.ppo)
        equal(trajectory, rollout.trajectory)
        if i == 1:
            # Validation-only score/terminal injection exercises nonzero GAE and
            # both policy/value updates; this is not recorded as gameplay.
            trajectory = trajectory.replace(reward=trajectory.reward.at[3, 0].set(1),
                done=trajectory.done.at[3, 0].set(True))
            rollout = rollout.replace(trajectory=trajectory)
        original, metrics = update(original, trajectory)
        state, actual = inactive.update(state, rollout)
        equal(original, state.ppo)
        equal(metrics, actual)
    equal(initial.novelty, state.novelty)
    equal(initial.intrinsic_value, state.intrinsic_value)
    assert int(state.rollouts) == 2 and not bool(state.pending_update)
    assert changed(initial.ppo.agent_ts.params, state.ppo.agent_ts.params)


def test_two_gae_streams_match_independent_terminal_aware_reference(models):
    _, _, online = models
    state, rollout = online.collect(online.init(jax.random.PRNGKey(3)))
    rewards = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 9
    values = rewards / 3
    dones = jnp.zeros((8, 2), bool).at[2, 0].set(True).at[-1, 1].set(True)
    trajectory = rollout.trajectory.replace(reward=rewards, value=values, done=dones)
    rollout = rollout.replace(trajectory=trajectory, intrinsic_values=values * 2)
    # Nonzero bootstrap outputs make erroneous rollout-end masking visible.
    params = state.ppo.agent_ts.params
    params = jax.tree.map(lambda x: x, params)
    params['params']['critic']['Dense_1']['bias'] = jnp.ones_like(params['params']['critic']['Dense_1']['bias']) * 3
    head = jax.tree.map(lambda x: x, state.intrinsic_value.params)
    head['params']['Dense_1']['bias'] = jnp.ones_like(head['params']['Dense_1']['bias']) * 5
    state = state.replace(ppo=state.ppo.replace(last_done=dones[-1],
        agent_ts=state.ppo.agent_ts.replace(params=params)),
        intrinsic_value=state.intrinsic_value.replace(params=head))
    result = online.targets(state, rollout, rewards * 4)

    def reference(reward, value, gamma, bootstrap):
        advantage = np.zeros(2); next_value = np.array(bootstrap); rows = []
        for t in reversed(range(8)):
            alive = 1 - np.asarray(dones[t])
            delta = reward[t] + gamma * next_value * alive - value[t]
            advantage = delta + gamma * .98 * alive * advantage
            rows.append(advantage.copy()); next_value = value[t]
        adv = np.array(rows[::-1])
        return adv, adv + value

    expected = (*reference(np.array(rewards), np.array(values), .995, [3, 0]),
                *reference(np.array(rewards * 4), np.array(values * 2), .99, [5, 0]))
    for a, b in zip(result, expected):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)


def test_enabled_update_is_finite_changes_policy_and_preserves_target_and_environment(models):
    _, _, online = models
    initial = online.init(jax.random.PRNGKey(4))
    collected, rollout = online.collect(initial)
    updated, metrics = online.update(collected, rollout)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(updated))
    assert float(metrics['clipped_grad_norm']) <= online.base.max_grad_norm + 1e-6
    assert float(metrics['normalized_novelty_mean']) > 0
    assert changed(initial.ppo.agent_ts.params, updated.ppo.agent_ts.params)
    assert changed(initial.intrinsic_value.params, updated.intrinsic_value.params)
    assert changed(initial.novelty.predictor.params, updated.novelty.predictor.params)
    equal(initial.novelty.target_params, updated.novelty.target_params)
    equal(collected.ppo.env_state, updated.ppo.env_state)
    equal(collected.ppo.last_obs, updated.ppo.last_obs)
    assert int(updated.ppo.global_step) == 16
    assert int(updated.ppo.agent_ts.step) == int(updated.intrinsic_value.step) == 4
    assert int(updated.novelty.predictor.step) == 4
    assert int(updated.novelty.rollouts) == int(updated.rollouts) == 1


def test_lifecycle_rejects_pending_duplicate_and_stale_batches(models, tmp_path):
    _, _, online = models
    state = online.init(jax.random.PRNGKey(5))
    collected, rollout = online.collect(state)
    with pytest.raises(ValueError, match='Consume'): online.collect(collected)
    with pytest.raises(ValueError, match='unconsumed'):
        save_online_checkpoint(tmp_path/'pending.msgpack', collected, {})
    with pytest.raises(ValueError, match='No fresh'): online.update(state, rollout)
    for wrong in [rollout.replace(generation=rollout.generation+1),
                  rollout.replace(end_step=rollout.end_step+1),
                  rollout.replace(policy_step=rollout.policy_step+1),
                  rollout.replace(end_rng=jax.random.PRNGKey(900))]:
        with pytest.raises(ValueError, match='Stale'): online.update(collected, wrong)
    updated, _ = online.update(collected, rollout)
    with pytest.raises(ValueError, match='No fresh'): online.update(updated, rollout)
    next_state, _ = online.collect(updated)
    with pytest.raises(ValueError, match='Stale'): online.update(next_state, rollout)


def test_full_restore_reproduces_next_collection_and_update_export_cannot_resume(models, tmp_path):
    _, _, online = models
    state, rollout = online.collect(online.init(jax.random.PRNGKey(6)))
    state, _ = online.update(state, rollout)
    path = tmp_path/'full.msgpack'; config = {'coefficient': .01, 'seed': 6}
    save_online_checkpoint(path, state, config)
    template = online.init(jax.random.PRNGKey(99))
    restored = restore_online_checkpoint(path, template, config)
    equal(state, restored)
    a, ra = online.collect(state); b, rb = online.collect(restored)
    equal(a, b); equal(ra, rb)
    ua, ma = online.update(a, ra); ub, mb = online.update(b, rb)
    equal(ua, ub); equal(ma, mb)
    with pytest.raises(ValueError, match='configuration'):
        restore_online_checkpoint(path, template, dict(config, coefficient=.02))
    export = tmp_path/'policy.msgpack'
    export_policy(export, state, config)
    payload = serialization.msgpack_restore(export.read_bytes())
    equal(payload['train_state']['agent_ts']['params'], state.ppo.agent_ts.params)
    assert payload['purpose'] == 'inference_only'
    with pytest.raises(ValueError, match='Not a full'):
        restore_online_checkpoint(export, template, config)
    with pytest.raises(ValueError): restore_checkpoint(export, state.ppo)
