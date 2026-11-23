"""Reproduce a positive random-policy reward and capture its surrounding frames."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw
from octax.environments import create_environment
from experiment_utils import write_json

env,_ = create_environment('tetris', timer_mode='legacy')
key, rk = jax.random.split(jax.random.PRNGKey(10002))
s,o,_ = env.reset(rk)
@jax.jit
def rollout(s,key):
 def step(c,_):
  s,key = c
  key,ak=jax.random.split(key)
  a=jax.random.randint(ak,(),0,env.num_actions)
  s,o,r,t,tr,i=env.step(s,a)
  return (s,key),(s.display,r,s.V[10],s.current_score,t)
 return jax.lax.scan(step,(s,key),None,length=3198)[1]
displays,rewards,register,scores,term=jax.device_get(rollout(s,key))
events=np.flatnonzero(rewards>0)
assert events.size and np.allclose(rewards,np.diff(np.r_[0.,scores]))
assert np.array_equal(scores,register.astype(float))
canvas=Image.new('RGB',(512*3,285),'#171721'); d=ImageDraw.Draw(canvas)
event=int(events[0])
for col,step in enumerate([event-5,event,event+5]):
 im=Image.fromarray(displays[step].T.astype('uint8')*255).resize((512,256),Image.Resampling.NEAREST)
 canvas.paste(im,(col*512,29));d.text((col*512+5,7),f'Step {step+1}, score {scores[step]}, reward {rewards[step]}',fill='white')
out=Path('runs/tetris/sanity')
canvas.save(out/'positive_reward.png')
write_json(out/'reward_validation.json',dict(seed=10002,positive_reward_steps=(events+1).tolist(),
    rewards=rewards[events].tolist(),score_equals_V10=True,reward_equals_score_difference=True,
    final_score=float(scores[-1]),terminated=bool(term[-1])))
p=out/'baselines.json';data=json.loads(p.read_text());c=data['checks']
c['noop_first_120_steps_identical_across_seeds']=c.pop('noop_trajectories_identical_across_seeds',True)
c['noop_full_episode_lengths_vary']=len(set(data['noop']['episodes']['lengths']))>1
write_json(p,data)
print('Validated positive reward',events+1,flush=True)
