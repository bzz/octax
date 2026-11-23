"""Matched-seed, no-learning timing/action probes. State fields are diagnostics only."""
import argparse
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from octax.environments import create_environment
from experiment_utils import write_json

p=argparse.ArgumentParser()
p.add_argument('--episodes',type=int,default=64)
p.add_argument('--out',default='runs/tetris/reward_diagnosis')
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
results=[]
for timing,repeat,policy in [('legacy',1,'random'),('step_saturating',1,'random'),('frame',1,'random'),('frame',4,'random'),('legacy',1,'noop'),('frame',1,'noop')]:
 env,_=create_environment('tetris',timer_mode=timing)
 def single(seed):
  key,rk=jax.random.split(jax.random.PRNGKey(seed));s,o,_=env.reset(rk)
  # steps, score, high-delay count, unchanged pixels, reward events, reward sum
  initial=(s,key,jnp.int32(0),jnp.int32(4),jnp.zeros(4,jnp.float32),jnp.bool_(False),jnp.bool_(False))
  def body(c):
   s,key,n,action,stats,_,_=c
   key,ak=jax.random.split(key)
   if policy=='random':
    action=jnp.where(n%repeat==0,jax.random.randint(ak,(),0,env.num_actions),action)
   ns,_,reward,term,trunc,_=env.step(s,action)
   stats=stats+jnp.array([ns.delay_timer>16,jnp.all(ns.display==s.display),reward>0,reward],jnp.float32)
   return ns,key,n+1,action,stats,term,trunc
  s,key,n,action,stats,term,trunc=jax.lax.while_loop(lambda c:~(c[-1]|c[-2]),body,initial)
  return dict(length=n,score=s.current_score,stats=stats,terminated=term,truncated=trunc)
 start=time.perf_counter()
 raw=jax.device_get(jax.jit(jax.vmap(single))(jnp.arange(40000,40000+a.episodes)))
 stats=raw['stats'].sum(axis=0);steps=int(raw['length'].sum())
 result=dict(timer_mode=timing,action_repeat=repeat,policy=policy,episodes=a.episodes,
  seed_start=40000,steps=steps,mean_length=float(raw['length'].mean()),mean_score=float(raw['score'].mean()),
  scoring_episodes=int((raw['score']>0).sum()),positive_reward_events=int(stats[2]),
  high_delay_fraction=float(stats[0]/steps),unchanged_pixels_fraction=float(stats[1]/steps),
  terminated=int(raw['terminated'].sum()),truncated=int(raw['truncated'].sum()),
  elapsed_seconds=time.perf_counter()-start,
  scores=raw['score'].tolist(),lengths=raw['length'].tolist())
 results.append(result);write_json(out/'timing_action_ablation.json',results)
 print({k:v for k,v in result.items() if k not in ['scores','lengths']},flush=True)
