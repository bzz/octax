"""Held-out evaluation of a checkpoint against random and no-op controls."""
import argparse
from pathlib import Path
import jax
from flax import serialization
from octax.agents.ppo import Agent
from octax.environments import create_environment
from experiment_utils import evaluate, make_evaluator, save_gif, write_json, environment_from_config

p=argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('--episodes',type=int,default=64)
p.add_argument('--seed-start',type=int,default=30000)
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
payload=serialization.msgpack_restore(Path(a.checkpoint).read_bytes())
env,_=environment_from_config(payload['config'])
agent=Agent(action_dim=env.num_actions)
params=payload['train_state']['agent_ts']['params']
evaluator=make_evaluator(env,agent,a.episodes,a.seed_start)
result={'checkpoint':str(Path(a.checkpoint).resolve()), 'step':int(payload['train_state']['global_step']),
        'episodes_per_policy':a.episodes,'seed_start':a.seed_start}
for mode in ['random','noop','policy']:
 result[mode]=evaluate(evaluator,params,mode)
 write_json(out/'comparison.json',result)
 print(mode,result[mode]['summary'],flush=True)
save_gif(env,out/'eval.gif',agent,params,'policy',seed=a.seed_start+a.episodes,max_steps=4500)
