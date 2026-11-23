"""Uniform-random control with the v2 learner's reset scheme and step budget."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from octax.agents.ppo import PPOOctax
from octax.wrappers import OctaxGymnaxWrapper
from experiment_utils import environment_from_config, write_json

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=Path('runs/tetris/v2_baseline_seed0'))
args=parser.parse_args()
run=args.run
cfg=OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
raw,_=environment_from_config(cfg);env=OctaxGymnaxWrapper(raw)
agent=PPOOctax.create_agent({},env,env.default_params)
options={k:cfg[k] for k in ['num_envs','num_steps','num_epochs','num_minibatches','learning_rate',
 'total_timesteps','eval_freq','gamma','gae_lambda','clip_eps','vf_coef','ent_coef',
 'max_grad_norm','advantage_std_floor','grad_clip_mode','zero_init_value']}
algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,eval_callback=None,**options)
# Match initial environment states and base RNG. Uniform keypress draws do not
# necessarily match categorical samples or policy-dependent episode/reset times.
ts=algo.init_state(jax.random.PRNGKey(cfg['seed']))
@jax.jit
def rollout(state,rng):
 def step(carry,index):
  state,rng=carry
  rng,new_rng=jax.random.split(rng)
  keys,ak=jax.random.split(new_rng)
  keys=jax.random.split(keys,cfg['num_envs'])
  actions=jax.random.randint(ak,(cfg['num_envs'],),0,raw.num_actions)
  obs,state,reward,done,_=algo.vmap_step(keys,state,actions,env.default_params)
  return (state,rng),(reward,done,actions)
 return jax.lax.scan(step,(state,rng),jnp.arange(cfg['total_timesteps']//cfg['num_envs']))[1]
start=time.perf_counter()
rewards,dones,actions=jax.device_get(rollout(ts.env_state,ts.rng))
counts=np.bincount(actions.ravel(),minlength=raw.num_actions)
indices=np.argwhere(rewards>0)
result=dict(policy='uniform_random',timer_mode='frame',seed=cfg['seed'],
 total_steps=int(rewards.size),positive_reward_events=int((rewards>0).sum()),
 reward_sum=float(rewards.sum()),completed_episodes=int(dones.sum()),
 positive_events=[{'vector_step':int(i+1),'environment_steps_consumed':int((i+1)*cfg['num_envs']),
                   'env_index':int(j),'reward':float(rewards[i,j])} for i,j in indices],
 action_distribution=(counts/counts.sum()).tolist(),elapsed_seconds=time.perf_counter()-start,
 note='Matched budget and initial reset states; no claim of identical action RNG or episode boundaries.')
write_json(run/'random_training_budget.json',result)
print(json.dumps(result,indent=2),flush=True)
