"""Cloning correctness and the offline-to-on-policy stage boundary."""
import hashlib
import json

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from octax.agents import PPOOctax
from octax.checkpoints import save_checkpoint, restore_checkpoint
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from pixel_pretraining import PixelCloner, actor_log_probs, load_human_pairs
from policy_initialization import (transfer_to_fresh_ppo, save_policy_weights,
    read_policy_weights, initialize_ppo_from_file)


def equal(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope='module')
def setup():
    raw, _ = create_environment('tetris', timer_mode='frame')
    env = OctaxGymnaxWrapper(raw)
    agent = PPOOctax.create_agent({}, env, env.default_params)
    base = PPOOctax(env=env, env_params=env.default_params, agent=agent, eval_callback=None,
        num_envs=2, num_steps=4, num_epochs=1, num_minibatches=1,
        total_timesteps=8, eval_freq=8, grad_clip_mode='global_norm', zero_init_value=True,
        advantage_std_floor=.01, learning_rate=.0005)
    fresh = base.init_state(jax.random.PRNGKey(1))
    cloner = PixelCloner(agent, batch_size=4)
    state = cloner.init(fresh.agent_ts.params, jax.random.PRNGKey(500))
    observations = jnp.zeros((5, 4, 32, 64), bool).at[1::2].set(True)
    actions = jnp.array([0, 1, 0, 1, 0])
    return base, fresh, cloner, state, observations, actions


def test_loader_selects_pre_action_pixels_and_rejects_wrong_source_hash_timing(tmp_path):
    path = tmp_path/'prefix.npz'
    observations = np.zeros((4, 4, 32, 64), bool)
    observations[1, :, 2, 3] = True; observations[-1] = True
    metadata = dict(format_version=1, source='human', environment='tetris',
        observation_shape=[4, 32, 64], env_kwargs=dict(timer_mode='frame', frame_skip=4,
            fps=60, instruction_frequency=700, max_num_steps_per_episodes=4500),
        steps=3, complete=False, return_sum=1., episode_id='prefix', reset_seed=200000)
    def write(meta):
        np.savez_compressed(path, observations=observations, actions=np.array([0, 2, 4]),
            metadata=json.dumps(meta), state__secret=np.full(3, 999), rewards=np.array([0, 1, 0]))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = write(metadata)
    data = load_human_pairs(path, digest)
    np.testing.assert_array_equal(data.observations, observations[:-1])
    np.testing.assert_array_equal(data.actions, [0, 2, 4])
    assert set(vars(data)) == {'observations', 'actions', 'provenance'}
    assert not data.provenance['complete'] and data.provenance['decisions'] == 3
    with pytest.raises(ValueError, match='hash'): load_human_pairs(path, 'wrong')
    digest = write(dict(metadata, source='automated_test'))
    with pytest.raises(ValueError, match='human recording'): load_human_pairs(path, digest)
    digest = write(dict(metadata, env_kwargs=dict(metadata['env_kwargs'], frame_skip=8)))
    with pytest.raises(ValueError, match='timing'): load_human_pairs(path, digest)


def test_cloning_reduces_action_loss_without_critic_or_persistent_examples(setup):
    base, fresh, cloner, state, observations, actions = setup
    initial = state
    before = float(cloner.fit_metrics(state, observations, actions)['cross_entropy'])
    for _ in range(12):
        state, metrics = cloner.epoch(state, observations, actions)
        assert float(metrics['examples']) == 5 # Last batch is partial; padding has zero weight.
        assert np.isfinite(float(metrics['loss']))
    after = float(cloner.fit_metrics(state, observations, actions)['cross_entropy'])
    assert after < before*.5, (before, after)
    assert int(state.policy.step) == 24 and int(state.examples_seen) == 60 and int(state.epochs) == 12
    assert 'critic' not in state.policy.params['params']
    assert set(serialization.to_state_dict(state)) == {'policy', 'rng', 'epochs', 'examples_seen'}
    feature_keys = [k for k in state.policy.params['params'] if k != 'actor']
    assert any(not np.array_equal(x, y) for k in feature_keys for x, y in zip(
        jax.tree.leaves(initial.policy.params['params'][k]), jax.tree.leaves(state.policy.params['params'][k])))


def test_full_cloning_checkpoint_reproduces_next_epoch(setup, tmp_path):
    _, fresh, cloner, state, observations, actions = setup
    state, _ = cloner.epoch(state, observations, actions)
    path = tmp_path/'offline.msgpack'
    save_checkpoint(path, state, {'stage': 'offline_cloning'})
    restored, cfg = restore_checkpoint(path, cloner.init(fresh.agent_ts.params, jax.random.PRNGKey(999)))
    assert cfg['stage'] == 'offline_cloning'
    equal(state, restored)
    a, ma = cloner.epoch(state, observations, actions)
    b, mb = cloner.epoch(restored, observations, actions)
    equal(a, b); equal(ma, mb)


def test_disk_handoff_preserves_source_actor_and_fresh_ppo_state_and_collects_new_actions(setup, tmp_path):
    base, fresh, cloner, state, observations, actions = setup
    state, _ = cloner.epoch(state, observations, actions)
    path = tmp_path/'weights.msgpack'
    digest = save_policy_weights(path, state.policy.params, {'source':'test'})
    transferred, provenance = initialize_ppo_from_file(fresh, path, digest)
    assert provenance['components'] == 'encoder_actor'
    equal(transferred.agent_ts.params['params']['critic'], fresh.agent_ts.params['params']['critic'])
    equal(transferred.agent_ts.opt_state, fresh.agent_ts.opt_state)
    # The only changed field is the intended policy weight tree.
    equal(transferred.replace(agent_ts=transferred.agent_ts.replace(params=fresh.agent_ts.params)), fresh)
    expected_lp = base.agent.apply(state.policy.params, observations, actions, method=actor_log_probs)
    actual_lp = base.agent.apply(transferred.agent_ts.params, observations, None, actions)[1:3]
    equal(expected_lp, actual_lp)
    _, next_rng = jax.random.split(fresh.rng)
    _, action_key = jax.random.split(next_rng)
    expected_actions = base.agent.apply(state.policy.params, fresh.last_obs, action_key, method='call_actor')[0]
    collected, trajectory = jax.jit(base.collect_trajectories)(transferred)
    np.testing.assert_array_equal(trajectory.action[0], expected_actions)
    assert int(collected.global_step) == 8 and int(collected.agent_ts.step) == 0
    # Ordinary PPO consumes only the new rollout; no offline dataset argument.
    updated, _ = jax.jit(base.update_trajectories)(collected, trajectory)
    assert int(updated.agent_ts.step) == 1


def test_weights_schema_rejects_full_training_states_critic_and_wrong_checksum(setup, tmp_path):
    _, fresh, _, state, _, _ = setup
    path = tmp_path/'weights.msgpack'
    digest = save_policy_weights(path, state.policy.params, {'source':'test'})
    with pytest.raises(ValueError, match='checksum'): read_policy_weights(path, 'wrong')
    payload = serialization.msgpack_restore(path.read_bytes())
    assert set(payload) == {'format_version','kind','components','parameters','metadata'}
    assert 'critic' not in payload['parameters']['params']
    with pytest.raises(ValueError, match='critic'):
        save_policy_weights(path, fresh.agent_ts.params, {})
    save_checkpoint(path, fresh, {})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='weights-only'): read_policy_weights(path, digest)


def test_encoder_only_transfer_keeps_both_fresh_heads_and_rejects_used_state(setup):
    base, fresh, cloner, state, observations, actions = setup
    state, _ = cloner.epoch(state, observations, actions)
    transferred = transfer_to_fresh_ppo(fresh, state.policy.params, include_actor=False)
    for key in ['actor', 'critic']:
        equal(transferred.agent_ts.params['params'][key], fresh.agent_ts.params['params'][key])
    with pytest.raises(ValueError, match='fresh PPO state'):
        transfer_to_fresh_ppo(fresh.replace(global_step=jnp.int32(1)), state.policy.params, include_actor=True)
    used = fresh.agent_ts.apply_gradients(grads=jax.tree.map(jnp.ones_like, fresh.agent_ts.params)).replace(step=0)
    with pytest.raises(ValueError, match='optimizer moments'):
        transfer_to_fresh_ppo(fresh.replace(agent_ts=used), state.policy.params, include_actor=True)
    bad = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), state.policy.params)
    with pytest.raises(ValueError, match='nonfinite'):
        transfer_to_fresh_ppo(fresh, bad, include_actor=True)
