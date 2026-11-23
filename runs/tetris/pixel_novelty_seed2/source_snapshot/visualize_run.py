"""Matplotlib/Seaborn gameplay and diagnostics, following MinAtar's renderer.

Reference: https://github.com/kenjyoung/MinAtar/blob/master/minatar/environment.py
Uses the same plotting primitives: imshow, discrete colormap, cubehelix palette,
nearest-pixel rendering. This is still Octax Tetris, with one observed channel.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import serialization
from octax.agents.ppo import Agent
from octax.environments import create_environment
from experiment_utils import write_json, environment_from_config

ACTIONS=['Rotate','Left','Right','Down','No-op']


def capture_episode(env, agent, params, mode, seed):
    """Record exact emulator frames; terminal transition is retained."""
    @jax.jit
    def run(params):
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        state, obs, _ = env.reset(reset_key)
        def step(carry, _):
            state,obs,key,done = carry
            key,ak = jax.random.split(key)
            if mode == 'random':
                action=jax.random.randint(ak,(),0,env.num_actions)
            elif mode == 'noop':
                action=jnp.int32(env.num_actions-1)
            else:
                action=agent.apply(params,obs.transpose(0,2,1)[None],ak)[0][0]
            ns,no,r,term,trunc,_=env.step(state,action)
            # Freeze at terminal state rather than recording a restarted game.
            ns=jax.tree.map(lambda x,y:jnp.where(done,x,y),state,ns)
            no=jnp.where(done,obs,no)
            done=done | term | trunc
            return (ns,no,key,done),(ns.display.T,ns.current_score,action,ns.time,done)
        return jax.lax.scan(step,(state,obs,key,jnp.bool_(False)),None,
                            length=env.max_num_steps_per_episodes)[1]
    frames,scores,actions,steps,done=jax.device_get(run(params))
    terminal=np.flatnonzero(done)
    length=int(terminal[0]+1) if terminal.size else len(frames)
    selection=np.unique(np.r_[np.arange(0,length,8),length-1])
    return dict(frames=frames[selection],scores=scores[selection],actions=actions[selection],
                steps=steps[selection],length=length,final_score=float(scores[length-1]),seed=seed)


def gameplay(traces, out):
    # MinAtar uses one cubehelix color per channel, with black for empty cells.
    # Octax is monochrome, so there is exactly one foreground channel.
    cmap=ListedColormap([(0.,0.,0.),*sns.color_palette('cubehelix',1)])
    norm=BoundaryNorm([0,1,2],2)
    names=list(traces)
    fig, axes=plt.subplots(1,len(names),figsize=(6.4*len(names),3.9),squeeze=False)
    artists=[]
    for ax,name in zip(axes[0],names):
        artists.append(ax.imshow(traces[name]['frames'][0].astype(float)+.5,cmap=cmap,norm=norm,interpolation='none'))
        ax.set_xticks([0,16,32,48,63]); ax.set_yticks([0,8,16,24,31])
        ax.set_xlabel('CHIP-8 pixels (64 × 32)')
        ax.grid(False)
    fig.suptitle('Tetris · CPU PPO · MinAtar plotting approach · 4× playback',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91))
    rendered=[]
    length=max(len(v['frames']) for v in traces.values())
    for index in range(length):
        for ax,artist,name in zip(axes[0],artists,names):
            t=traces[name]; i=min(index,len(t['frames'])-1)
            artist.set_data(t['frames'][i].astype(float)+.5)
            finished=' · ended' if i==len(t['frames'])-1 else ''
            ax.set_title(f"{name} | score {t['scores'][i]:.0f} | step {t['steps'][i]}\n{ACTIONS[int(t['actions'][i])]}{finished}",fontsize=11)
        fig.canvas.draw()
        rendered.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
    # Eight actions per frame, each action = four 60 Hz emulator frames, at 4×.
    rendered[0].save(out/'gameplay.gif',save_all=True,append_images=rendered[1:],duration=130,loop=0)
    rendered[len(rendered)//2].save(out/'gameplay.png')
    plt.close(fig)


def plot_results(run, baselines, out):
    m=pd.read_csv(run/'metrics.csv')
    e=json.loads((run/'evaluation.json').read_text())
    b=json.loads(baselines.read_text())
    sns.set_theme(style='whitegrid',context='notebook')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    x=m.step/1000
    ax=axes[0,0]
    ax.plot([v['step']/1000 for v in e],[v['summary']['mean_return'] for v in e],marker='o',label='PPO (8 episodes)')
    ax.axhline(b['random']['summary']['mean_return'],color='tab:orange',ls='--',label='Random (8 episodes)')
    ax.axhline(b['noop']['summary']['mean_return'],color='gray',ls=':',label='No-op (8 episodes)')
    ax.set(title='Evaluation game score — 8 episodes per checkpoint',ylabel='Mean episode return');ax.legend(fontsize=9)
    ax=axes[0,1];ax.step(x,m.positive_reward_steps.cumsum(),where='post')
    ax.set(title='Positive reward events in training',ylabel='Cumulative events',ylim=(-0.05,max(1,float(m.positive_reward_steps.sum())*1.1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax=axes[1,0]
    ax.plot(x,m.loss,label='Total loss');ax.plot(x,m.value_loss,label='Weighted value loss')
    ax.set(title='PPO losses',ylabel='Loss');ax.legend(fontsize=9)
    ax=axes[1,1];ax.plot(x,m.entropy,label='Policy entropy');ax.axhline(np.log(5),ls=':',color='gray',label='Uniform policy')
    ax.set(title='Policy diversity',ylabel='Entropy (nats)',ylim=(0,np.log(5)*1.05));ax.legend(fontsize=9)
    ax=axes[2,0]
    for i,label in enumerate(ACTIONS):
        ax.plot(x,m[f'action_{i}_fraction'].rolling(8,min_periods=1).mean(),label=label)
    ax.set(title='Action distribution (8-update moving average)',ylabel='Fraction',ylim=(0,1));ax.legend(ncol=3,fontsize=8)
    ax=axes[2,1]
    ax.plot(x,m.steps_per_second,label='Training steps/s')
    ax.set(title='CPU throughput (collection + PPO update)',ylabel='Environment steps/s')
    for ax in axes.ravel():
        ax.set_xlabel('Environment steps (thousands)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Tetris PPO · {int(m.step.iloc[-1]):,} steps · one seed · pixel observations',fontsize=16)
    fig.savefig(out/'learning_curves.png',dpi=150)
    fig.savefig(out/'learning_curves.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='runs/tetris/ppo_seed0_smoke')
    p.add_argument('--baselines',default=None)
    p.add_argument('--seed',type=int,default=30016)
    a=p.parse_args();run=Path(a.run);out=run/'visualizations';out.mkdir(exist_ok=True)
    payload=serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
    mode=payload['config'].get('env_kwargs',{'timer_mode':'legacy'})['timer_mode']
    default_baseline='runs/tetris/sanity' if mode=='legacy' else 'runs/tetris/sanity_frame_v2'
    baseline=Path(a.baselines) if a.baselines else Path(default_baseline)/'baselines.json'
    baseline_mode=json.loads(baseline.read_text()).get('timer_mode','legacy')
    if baseline_mode != mode:
        raise ValueError('Baseline and checkpoint use different timer modes')
    plot_results(run,baseline,out)
    env,_=environment_from_config(payload['config']);agent=Agent(action_dim=env.num_actions)
    params=payload['train_state']['agent_ts']['params']
    traces={}
    for label,mode in [('Random','random'),(f"PPO {int(payload['train_state']['global_step']):,} steps",'policy')]:
        traces[label]=capture_episode(env,agent,params,mode,a.seed)
        np.savez_compressed(out/f'{mode}_trajectory.npz',**traces[label])
        print(label,traces[label]['length'],traces[label]['final_score'],flush=True)
    write_json(out/'gameplay.json',{k:{f:v[f] for f in ['length','final_score','seed']} for k,v in traces.items()})
    gameplay(traces,out)
