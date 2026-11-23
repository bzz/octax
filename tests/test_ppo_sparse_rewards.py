import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from octax.agents.ppo import PPOOctax, normalize_advantages, gradient_clipper
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from train import run


def test_normalization_floor_does_not_amplify_tiny_value_noise():
 tiny=jnp.array([-1e-5,1e-5])
 fixed=normalize_advantages(tiny,0.01)
 assert float(fixed.std()) < .002
 np.testing.assert_array_equal(normalize_advantages(jnp.zeros(4),.01),jnp.zeros(4))
 # Real score-scale differences still get the ordinary standardization.
 big=jnp.array([0.,0.,0.,1.])
 np.testing.assert_allclose(normalize_advantages(big,.01),normalize_advantages(big,0.0))


def test_global_clip_really_limits_full_gradient_norm():
 grads={'w':jnp.ones((100,))}
 tx=gradient_clipper('global_norm',.5)
 clipped,_=tx.update(grads,tx.init(grads))
 assert float(optax.global_norm(clipped)) <= .500001


def test_fresh_zero_value_output_on_pixels():
 raw,_=create_environment('tetris')
 env=OctaxGymnaxWrapper(raw)
 agent=PPOOctax.create_agent({},env,env.default_params)
 algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,eval_callback=None,
  num_envs=2,zero_init_value=True,grad_clip_mode='global_norm')
 state=algo.init_state(jax.random.PRNGKey(2))
 obs=jax.random.bernoulli(jax.random.PRNGKey(3),shape=(2,4,32,64))
 value=agent.apply(state.agent_ts.params,obs,method='call_critic')
 np.testing.assert_array_equal(value,jnp.zeros(2))


def test_resume_rejects_changed_semantics_before_creating_output(tmp_path):
 checkpoint=tmp_path/'old.msgpack'
 checkpoint.write_bytes(serialization.msgpack_serialize({'config':{'env':'tetris'}}))
 destination=tmp_path/'new'
 with pytest.raises(ValueError,match='Resume changes experiment_version'):
  run({'agent':'PPO','resume':str(checkpoint),'experiment_version':2,'run_dir':str(destination)})
 assert not destination.exists()


def test_zero_value_head_can_learn_when_a_score_arrives():
 from rejax.algos.ppo import AdvantageMinibatch,Trajectory
 raw,_=create_environment('tetris')
 env=OctaxGymnaxWrapper(raw)
 agent=PPOOctax.create_agent({},env,env.default_params)
 algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,eval_callback=None,
  num_envs=2,zero_init_value=True,grad_clip_mode='global_norm',max_grad_norm=.5,
  advantage_std_floor=.01)
 state=algo.init_state(jax.random.PRNGKey(2))
 obs=jax.random.bernoulli(jax.random.PRNGKey(3),shape=(2,4,32,64))
 action,logp,_,value=agent.apply(state.agent_ts.params,obs,jax.random.PRNGKey(4))
 # An isolated score-scale training example, not a reward-shaping rule.
 target=jnp.array([1.,0.])
 traj=Trajectory(obs,action,logp,target,value,jnp.ones(2,dtype=bool))
 batch=AdvantageMinibatch(traj,target,target)
 updated,metrics=jax.jit(algo.update_with_metrics)(state,batch)
 learned=agent.apply(updated.agent_ts.params,obs,method='call_critic')
 assert float(learned[0])>0
 assert float(metrics['clipped_grad_norm'])<=.500001
 assert float(metrics['normalized_advantage_std'])>.99
 assert np.isfinite(float(metrics['loss']))
