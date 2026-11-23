"""Inspect PPO gradient sources on a fixed reward-free rollout from old checkpoint."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from octax.agents.ppo import PPOOctax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from experiment_utils import restore_checkpoint,write_json

raw,_=create_environment('tetris',timer_mode='legacy')
env=OctaxGymnaxWrapper(raw)
agent=PPOOctax.create_agent({},env,env.default_params)
algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,eval_callback=lambda *a:(),
 num_envs=16,num_steps=32,num_minibatches=4,num_epochs=4,learning_rate=5e-4,max_grad_norm=.5)
ts=algo.init_state(jax.random.PRNGKey(0))
ts,cfg=restore_checkpoint('runs/tetris/continued_seed0/checkpoint.msgpack',ts)
ts,traj=jax.jit(algo.collect_trajectories)(ts)
last=agent.apply(ts.agent_ts.params,ts.last_obs,method='call_critic')
adv,target=algo.calculate_gae(traj,jnp.where(ts.last_done,0,last))
obs=traj.obs.reshape((-1,*traj.obs.shape[2:]));actions=traj.action.reshape(-1)
old_log=traj.log_prob.reshape(-1); advantages=adv.reshape(-1)
results=dict(reward_sum=float(traj.reward.sum()),raw_advantage_std=float(adv.std()),
 raw_advantage_mean=float(adv.mean()),value_mean=float(traj.value.mean()),
 value_std=float(traj.value.std()),settings=cfg,gradients=[])
for floor in [1e-8,1e-3,1e-2]:
 def policy_loss(params):
  _,logprob,entropy,_=agent.apply(params,obs,None,actions)
  normalized=(advantages-advantages.mean())/jnp.maximum(advantages.std(),floor)
  return -(jnp.exp(logprob-old_log)*normalized).mean()
 def entropy_loss(params):
  return -.01*agent.apply(params,obs,None,actions)[2].mean()
 pg=jax.jit(jax.grad(policy_loss))(ts.agent_ts.params)
 eg=jax.jit(jax.grad(entropy_loss))(ts.agent_ts.params)
 results['gradients'].append(dict(std_floor=floor,
  policy_grad_norm=float(optax.global_norm(pg)),entropy_grad_norm=float(optax.global_norm(eg)),
  normalized_advantage_std=float((advantages/jnp.maximum(advantages.std(),floor)).std())))
write_json(Path('runs/tetris/reward_diagnosis/ppo_noise.json'),results)
print({k:v for k,v in results.items() if k!='settings'},flush=True)
