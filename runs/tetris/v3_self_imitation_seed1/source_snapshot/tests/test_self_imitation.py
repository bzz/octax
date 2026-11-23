import jax
import jax.numpy as jnp
import numpy as np
import pytest
from self_imitation import EpisodeReplay, self_imitation_loss


def test_completed_returns_do_not_cross_environment_or_reset_boundaries():
    replay = EpisodeReplay((1,2,3), 2, capacity=20, gamma=.5)
    obs = np.zeros((3,2,1,2,3), np.uint8)
    obs[1,0,0,0,0] = 1
    actions = np.array([[0,1],[2,3],[4,0]])
    rewards = np.array([[0.,2.],[1.,0.],[8.,4.]])
    dones = np.array([[False,True],[True,False],[False,True]])
    completed = replay.add_rollout(obs, actions, rewards, dones)
    assert [x['return_sum'] for x in completed] == [2.,1.,4.]
    # Stored in completion order: env1 single-step; env0 first episode;
    # env1 second episode. Env0's new +8 is pending and must not leak back.
    np.testing.assert_array_equal(replay.returns[:5], [2.,.5,1.,2.,4.])
    assert [len(x) for x in replay.pending] == [1,0]
    assert replay.completed_episodes == replay.positive_episodes == 3
    decoded=np.unpackbits(replay.observations[2],count=6).reshape(1,2,3)
    np.testing.assert_array_equal(decoded,obs[1,0])


def test_policy_imitation_gradient_has_correct_sign_and_no_value_shortcut():
    def objective(logits, value):
        logp=jax.nn.log_softmax(logits)[jnp.array([1])]
        return self_imitation_loss(logp,jnp.array([value]),jnp.array([1.]),jnp.ones(1),value_weight=0.)[0]
    grad_logits, grad_value=jax.grad(objective,argnums=(0,1))(jnp.zeros(3),0.)
    assert grad_logits[1] < 0 and grad_logits[0] > 0 and grad_logits[2] > 0
    assert grad_value == 0  # Actor loss cannot be reduced by inflating the critic.
    grad_logits,grad_value=jax.grad(objective,argnums=(0,1))(jnp.zeros(3),2.)
    np.testing.assert_array_equal(grad_logits,np.zeros(3))
    assert grad_value == 0
    loss=lambda value:self_imitation_loss(jnp.zeros(1),jnp.array([value]),jnp.ones(1),jnp.ones(1),value_weight=1.)[0]
    assert jax.grad(loss)(0.) < 0  # Value moves toward the better observed return.


def test_replay_checkpoint_preserves_pending_episodes_and_next_draw(tmp_path):
    replay=EpisodeReplay((1,2,3),2,capacity=7,gamma=.9,seed=42)
    replay.add_rollout(np.ones((2,2,1,2,3)),np.zeros((2,2),int),np.array([[0.,0.],[1.,0.]]),np.array([[False,False],[True,False]]))
    sample=replay.sample(4);replay.update_priorities(sample['indices'],np.arange(4,dtype=float))
    path=tmp_path/'replay.npz';replay.save(path,1024)
    restored=EpisodeReplay.load(path,expected_step=1024)
    a,b=replay.sample(16),restored.sample(16)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert [len(x) for x in restored.pending]==[0,2]
    completed=restored.add_rollout(np.zeros((1,2,1,2,3)),np.zeros((1,2),int),np.array([[0.,2.]]),np.array([[False,True]]))
    assert completed[0]['length']==3 and completed[0]['return_sum']==2
    with pytest.raises(ValueError,match='step mismatch'):EpisodeReplay.load(path,expected_step=512)


def test_no_replay_optimizer_work_before_first_scoring_episode():
    replay=EpisodeReplay((1,2,3),1,capacity=3)
    replay.add_rollout(np.zeros((4,1,1,2,3)),np.zeros((4,1),int),np.zeros((4,1)),np.ones((4,1),bool))
    assert replay.size==3 and not replay.ready
    with pytest.raises(ValueError,match='scoring episode'):replay.sample(4)
    with pytest.raises(ValueError,match='binary'):replay.add_rollout(np.full((1,1,1,2,3),.5),np.zeros((1,1)),np.zeros((1,1)),np.ones((1,1),bool))


def test_inactive_replay_does_not_move_real_network_or_adam_momentum():
    import optax
    from flax.training.train_state import TrainState
    from octax.agents.ppo import Agent
    from self_imitation import make_self_imitation_updater
    agent=Agent(action_dim=5)
    obs=jnp.zeros((2,4,32,64))
    params=agent.init(jax.random.PRNGKey(19),obs,jax.random.PRNGKey(20))
    state=TrainState.create(apply_fn=(),params=params,tx=optax.adam(.0005))
    state=state.apply_gradients(grads=jax.tree.map(jnp.ones_like,params))
    update=make_self_imitation_updater(agent)
    updated,metrics=update(state,obs,jnp.array([0,1]),jnp.full(2,-100.),jnp.ones(2))
    assert not bool(metrics['applied'])
    for before,after in zip(jax.tree.leaves(state),jax.tree.leaves(updated)):
        np.testing.assert_array_equal(before,after)


def test_positive_replay_improves_selected_action_and_zero_value_head():
    import optax
    from flax.training.train_state import TrainState
    from octax.agents.ppo import Agent
    from self_imitation import make_self_imitation_updater
    agent=Agent(action_dim=5)
    obs=jnp.ones((2,4,32,64))
    params=agent.init(jax.random.PRNGKey(31),obs,jax.random.PRNGKey(32))
    params['params']['critic']['Dense_1']=jax.tree.map(jnp.zeros_like,params['params']['critic']['Dense_1'])
    state=TrainState.create(apply_fn=(),params=params,tx=optax.chain(optax.clip_by_global_norm(.5),optax.adam(.0001)))
    actions=jnp.array([2,2])
    before=agent.apply(state.params,obs,None,actions)
    updated,metrics=make_self_imitation_updater(agent)(state,obs,actions,jnp.ones(2),jnp.ones(2))
    after=agent.apply(updated.params,obs,None,actions)
    assert bool(metrics['applied']) and float(metrics['active_fraction'])==1
    assert np.all(np.asarray(after[1]) > np.asarray(before[1]))
    assert np.all(np.asarray(after[3]) > 0)
