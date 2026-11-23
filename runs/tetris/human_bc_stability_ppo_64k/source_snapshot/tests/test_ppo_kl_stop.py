"""Regression against archived PPO plus exercised stopping and continuation."""
import importlib.util
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from octax.agents.ppo import Agent, PPOOctax
from octax.checkpoints import restore_checkpoint, save_checkpoint
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from train import run


def equal(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope='module')
def models():
    # Independent pre-change implementation retained with the completed pilot.
    source = Path(__file__).resolve().parents[1]/'runs/tetris/human_bc_100epoch_ppo_seed1/source_snapshot/octax/agents/ppo.py'
    spec = importlib.util.spec_from_file_location('ppo_before_kl_stop', source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    raw, _ = create_environment('tetris', timer_mode='frame')
    env = OctaxGymnaxWrapper(raw)
    network = dict(action_dim=5, features_list=(4,), kernel_sizes=((3, 3),),
        strides_list=((2, 2),), hidden_layer_sizes=(8,))
    kwargs = dict(env=env, env_params=env.default_params, eval_callback=None,
        num_envs=2, num_steps=4, num_epochs=4, num_minibatches=1,
        learning_rate=.01, total_timesteps=16, eval_freq=16,
        gamma=.995, gae_lambda=.98, clip_eps=.2, vf_coef=.5, ent_coef=0.,
        max_grad_norm=.5, advantage_std_floor=.01,
        grad_clip_mode='global_norm', zero_init_value=True)
    return PPOOctax(agent=Agent(**network), **kwargs), reference.PPOOctax(agent=reference.Agent(**network), **kwargs)


def collected(model):
    return jax.jit(model.collect_trajectories)(model.init_state(jax.random.PRNGKey(7)))


def rewarded(model, state, trajectory):
    # Synthetic regression input only, not recorded as gameplay or shaping.
    obs = jax.random.bernoulli(jax.random.PRNGKey(23), shape=trajectory.obs.shape)
    rewards = jnp.arange(1, 9, dtype=jnp.float32).reshape(4, 2)/8
    actions = (rewards > .5).astype(jnp.int32)
    _, lp, _, values = model.agent.apply(state.agent_ts.params, obs.reshape(8, 4, 32, 64), None, actions.reshape(8))
    return trajectory.replace(obs=obs, action=actions, log_prob=lp.reshape(4, 2),
        value=values.reshape(4, 2), reward=rewards, done=jnp.ones((4, 2), bool))


def test_disabled_path_exactly_matches_archived_initialization_collection_and_rewarded_update(models):
    base, reference = models
    a = base.init_state(jax.random.PRNGKey(7)); b = reference.init_state(jax.random.PRNGKey(7))
    equal(a, b)
    a, ta = jax.jit(base.collect_trajectories)(a)
    b, tb = jax.jit(reference.collect_trajectories)(b)
    equal(a, b); equal(ta, tb)
    ta = rewarded(base, a, ta); tb = rewarded(reference, b, tb)
    equal(ta, tb)
    ua, ma = jax.jit(base.update_trajectories)(a, ta)
    ub, mb = jax.jit(reference.update_trajectories)(b, tb)
    equal(ua, ub); equal(ma, mb)


def test_nonbinding_guard_preserves_real_updates_and_accounts_for_gradients(models):
    base, _ = models
    state, trajectory = collected(base); trajectory = rewarded(base, state, trajectory)
    expected, metrics = jax.jit(base.update_trajectories)(state, trajectory)
    actual, guarded = jax.jit(base.replace(target_kl=1e6).update_trajectories)(state, trajectory)
    equal(actual, expected)
    for key, value in metrics.items():
        np.testing.assert_allclose(guarded[key], value, atol=1e-7, rtol=1e-6)
    assert int(guarded['optimizer_updates']) == int(guarded['gradient_evaluations']) == 4
    assert int(guarded['minibatches_checked']) == 4 and not bool(guarded['kl_stop_triggered'])


def test_stop_preserves_exact_first_update_including_critic_and_optimizer(models):
    base, _ = models
    guard = base.replace(target_kl=1e-12)
    state, trajectory = collected(base); trajectory = rewarded(base, state, trajectory)
    updated, metrics = jax.jit(guard.update_trajectories)(state, trajectory)
    # Use the ordinary whole compiled update as oracle: splitting GAE and
    # backprop into separately compiled calls changes floating-point fusion.
    expected, _ = jax.jit(base.replace(num_epochs=1).update_trajectories)(state, trajectory)
    rng = expected.rng
    for _ in range(3):
        rng, _ = jax.random.split(rng)
    expected = expected.replace(rng=rng)
    equal(updated, expected)  # All encoder, actor, critic, optimizer and environment fields.
    assert int(metrics['optimizer_updates']) == int(metrics['gradient_evaluations']) == 1
    assert int(metrics['minibatches_checked']) == 2 and int(metrics['skipped_minibatches']) == 3
    assert bool(metrics['kl_stop_triggered'])
    assert float(metrics['max_checked_exact_kl']) > guard.target_kl
    assert float(metrics['final_full_batch_kl']) > guard.target_kl  # Early stopping is not rollback.


def test_reward_free_zero_gradient_case_retains_ordinary_optimizer_clock(models):
    base, _ = models
    state, trajectory = collected(base)
    assert float(trajectory.reward.sum()) == 0
    original, _ = jax.jit(base.update_trajectories)(state, trajectory)
    actual, metrics = jax.jit(base.replace(target_kl=.015).update_trajectories)(state, trajectory)
    equal(original, actual); equal(state.agent_ts.params, actual.agent_ts.params)
    assert int(actual.agent_ts.step) == 4
    assert int(metrics['optimizer_updates']) == 4 and not bool(metrics['kl_stop_triggered'])
    assert float(metrics['grad_norm']) == 0


def test_stop_skips_remaining_minibatches_as_well_as_later_epochs(models):
    base, _ = models
    guard = base.replace(num_minibatches=2, target_kl=1e-6)
    state, trajectory = collected(base); trajectory = rewarded(base, state, trajectory)
    updated, metrics = jax.jit(guard.update_trajectories)(state, trajectory)
    # With this deliberately large regression-test rate, the second minibatch
    # triggers. Four epochs must end at exactly the first epoch's full state,
    # except for its prescribed remaining RNG splits.
    expected, first = jax.jit(guard.replace(num_epochs=1).update_trajectories)(state, trajectory)
    rng = expected.rng
    for _ in range(3):
        rng, _ = jax.random.split(rng)
    equal(updated, expected.replace(rng=rng))
    assert int(first['optimizer_updates']) == int(metrics['optimizer_updates']) == 1
    assert int(metrics['gradient_evaluations']) == 1
    assert int(metrics['minibatches_checked']) == 2
    assert int(metrics['skipped_minibatches']) == 7


def test_checkpoint_continuation_and_guard_resets_each_rollout(models, tmp_path):
    base, _ = models; guard = base.replace(target_kl=1e-12)
    state, trajectory = collected(base); trajectory = rewarded(base, state, trajectory)
    state, _ = jax.jit(guard.update_trajectories)(state, trajectory)
    path = tmp_path/'state.msgpack'; cfg = {'target_kl': 1e-12}
    save_checkpoint(path, state, cfg)
    restored, actual_cfg = restore_checkpoint(path, guard.init_state(jax.random.PRNGKey(999)))
    equal(state, restored); assert actual_cfg == cfg
    a, ta = jax.jit(guard.collect_trajectories)(state)
    b, tb = jax.jit(guard.collect_trajectories)(restored)
    equal(a, b); equal(ta, tb)
    ua, ma = jax.jit(guard.update_trajectories)(a, ta)
    ub, mb = jax.jit(guard.update_trajectories)(b, tb)
    equal(ua, ub); equal(ma, mb)
    assert int(ma['optimizer_updates']) >= 1
    assert int(ua.agent_ts.step)-int(a.agent_ts.step) == int(ma['optimizer_updates'])


@pytest.mark.parametrize('threshold', [0., -1., float('nan'), float('inf')])
def test_invalid_threshold_rejected(models, threshold):
    base, _ = models; state, trajectory = collected(base)
    with pytest.raises(ValueError, match='positive and finite'):
        base.replace(target_kl=threshold).update_trajectories(state, trajectory)


def test_resume_rejects_changed_kl_semantics_before_output_creation(tmp_path):
    checkpoint = tmp_path/'old.msgpack'
    checkpoint.write_bytes(serialization.msgpack_serialize({'config': {}}))
    out = tmp_path/'new'
    with pytest.raises(ValueError, match='Resume changes target_kl'):
        run({'agent': 'PPO', 'resume': str(checkpoint), 'target_kl': .015, 'run_dir': str(out)})
    assert not out.exists()
