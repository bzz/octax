"""Experimental PPO plus completed-episode score replay, with pixel-only inputs."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from octax.agents import PPOOctax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from self_imitation import EpisodeReplay, make_self_imitation_updater
from experiment_utils import (evaluate, make_evaluator, restore_checkpoint, save_checkpoint,
                              save_gif, system_info, write_json, environment_from_config)


def verify_checkpoint_pair(checkpoint):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint.parent/'checkpoint_pair.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((checkpoint.parent/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Model/replay checkpoint pair is inconsistent')
    return manifest['step']


def run(cfg):
    if cfg['agent'] != 'PPO':
        raise ValueError('This experimental runner supports PPO only.')
    if cfg['resume']:
        verify_checkpoint_pair(cfg['resume'])
        previous = serialization.msgpack_restore(Path(cfg['resume']).read_bytes())['config']
        if cfg['self_imitation'] != previous.get('self_imitation'):
            raise ValueError('Resume changes the self-imitation configuration')
        defaults = dict(experiment_version=1, env_kwargs={'timer_mode': 'legacy'},
                        advantage_std_floor=0.0, grad_clip_mode='elementwise', zero_init_value=False)
        for key, default in defaults.items():
            if cfg.get(key, default) != previous.get(key, default):
                raise ValueError(f'Resume changes {key}; start a fresh run for changed environment/PPO semantics.')
    out = Path(cfg['run_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.yaml').exists():
        raise FileExistsError(f'{out} already contains a run; use a new run_dir, including for resume.')
    if cfg['total_timesteps'] % (cfg['num_envs'] * cfg['num_steps']):
        raise ValueError('total_timesteps must be divisible by num_envs * num_steps (no silent overshoot).')
    OmegaConf.save(OmegaConf.create(cfg), out/'config.yaml')
    write_json(out/'system.json', system_info())
    raw, _ = environment_from_config(cfg)
    env = OctaxGymnaxWrapper(raw)
    options = {k:cfg[k] for k in ['num_envs', 'num_steps', 'num_minibatches', 'num_epochs',
               'learning_rate', 'total_timesteps', 'eval_freq', 'gamma', 'gae_lambda',
               'clip_eps', 'vf_coef', 'ent_coef', 'max_grad_norm', 'advantage_std_floor',
               'grad_clip_mode', 'zero_init_value']}
    agent = PPOOctax.create_agent({}, env, env.default_params)
    algo = PPOOctax(env=env, env_params=env.default_params, agent=agent,
                    eval_callback=lambda *args: (), **options)
    ts = algo.init_state(jax.random.PRNGKey(cfg['seed']))
    if cfg['resume']:
        ts, previous = restore_checkpoint(cfg['resume'], ts)
        for key in ['env','seed','num_envs','num_steps','num_minibatches','num_epochs',
                    'learning_rate','gamma','gae_lambda','clip_eps','vf_coef','ent_coef','max_grad_norm']:
            if cfg[key] != previous[key]:
                raise ValueError(f'Resume config differs in {key}')
    if cfg['encoder_checkpoint']:
        if cfg['resume']:
            raise ValueError('Choose resume OR encoder initialization')
        payload = serialization.msgpack_restore(Path(cfg['encoder_checkpoint']).read_bytes())
        source = payload['train_state']['agent_ts']['params']['params']
        target = ts.agent_ts.params
        # Heads remain at the target seed's fresh initialization; optimizer moments are fresh.
        feature_keys = [k for k in target['params'] if k not in ('actor', 'critic')]
        for key in feature_keys:
            if jax.tree.structure(source[key]) != jax.tree.structure(target['params'][key]):
                raise ValueError('Incompatible encoder structure')
            for x,y in zip(jax.tree.leaves(source[key]),jax.tree.leaves(target['params'][key])):
                if x.shape != y.shape:
                    raise ValueError('Incompatible encoder shape')
            target['params'][key] = jax.tree.map(jnp.asarray,source[key])
        ts = ts.replace(agent_ts=ts.agent_ts.replace(params=target))
    sil = cfg['self_imitation']
    replay = EpisodeReplay(env.observation_space(env.default_params).shape, cfg['num_envs'],
        capacity=sil['capacity'], gamma=cfg['gamma'], seed=cfg['seed'] + 1000000,
        alpha=sil['priority_alpha'], beta=sil['priority_beta'])
    if cfg['resume']:
        replay = EpisodeReplay.load(Path(cfg['resume']).parent/'replay.npz', expected_step=int(ts.global_step))
    sil_update = make_self_imitation_updater(agent, sil['loss_weight'], sil['value_weight'])
    total_sil_updates = total_sil_seconds = 0
    first_sil_step = None
    initial_params = jax.device_get(ts.agent_ts.params)
    collector = jax.jit(algo.collect_trajectories)
    updater = jax.jit(algo.update_trajectories)
    evaluator = make_evaluator(raw, agent, cfg['eval_episodes'], cfg['eval_seed_start'])
    evaluations = []

    def checkpoint_and_eval(state):
        step = int(state.global_step)
        save_checkpoint(out/'checkpoint.msgpack', state, cfg)
        replay.save(out/'replay.npz', step)
        pair = dict(step=step, sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                                      for name in ['checkpoint.msgpack','replay.npz']})
        temp = out/'checkpoint_pair.tmp'
        temp.write_text(json.dumps(pair,indent=2)+'\n');temp.replace(out/'checkpoint_pair.json')
        assert verify_checkpoint_pair(out/'checkpoint.msgpack') == step
        restored_replay = EpisodeReplay.load(out/'replay.npz', expected_step=step)
        assert restored_replay.rng.bit_generator.state == replay.rng.bit_generator.state
        assert (restored_replay.size, restored_replay.position, restored_replay.completed_episodes, restored_replay.positive_episodes) == (replay.size, replay.position, replay.completed_episodes, replay.positive_episodes)
        for field in ['observations','actions','returns','priorities']:
            np.testing.assert_array_equal(getattr(restored_replay,field)[:replay.size],getattr(replay,field)[:replay.size])
        for a,b in zip(restored_replay.pending,replay.pending):
            assert len(a)==len(b)
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x[0],y[0]);assert x[1:]==y[1:]
        restored, _ = restore_checkpoint(out/'checkpoint.msgpack', state)
        key = jax.random.PRNGKey(123456)
        expected = agent.apply(state.agent_ts.params, state.last_obs, key)
        actual = agent.apply(restored.agent_ts.params, state.last_obs, key)
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
            np.testing.assert_array_equal(a,b)
        for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
            np.testing.assert_array_equal(a,b)
        result = evaluate(evaluator, restored.agent_ts.params, 'policy')
        result['step'] = step
        result['checkpoint_roundtrip_exact'] = True
        result['replay_roundtrip_exact'] = True
        evaluations.append(result)
        write_json(out/'evaluation.json', evaluations)
        save_gif(raw, out/f'eval_{step:09d}.gif', agent, restored.agent_ts.params, 'policy',
                 seed=cfg['eval_seed_start'] + cfg['eval_episodes'])
        (out/'eval.gif').write_bytes((out/f'eval_{step:09d}.gif').read_bytes())
        print('EVAL',step,result['summary'],flush=True)
        return result

    checkpoint_and_eval(ts)
    fields = ['step','loss','actor_loss','value_loss','entropy','grad_norm','reward_sum',
              'positive_reward_steps','done_count','collect_seconds','update_seconds','steps_per_second']
    fields += ['clipped_grad_norm', 'approx_kl', 'clip_fraction', 'advantage_std', 'normalized_advantage_std']
    fields += [f'action_{i}_fraction' for i in range(raw.num_actions)]
    fields += ['sil_loss','sil_actor_loss','sil_value_loss','sil_active_fraction','sil_updates_applied',
               'sil_seconds','replay_size','replay_positive_episodes','completed_scoring_episodes']
    total_collect = total_update = 0.0
    steady_collect = steady_update = 0.0
    iterations = 0
    start_step = int(ts.global_step)
    start = time.perf_counter()
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        while int(ts.global_step) < cfg['total_timesteps']:
            t = time.perf_counter()
            ts, traj = collector(ts)
            jax.block_until_ready((ts,traj))
            collect_seconds = time.perf_counter()-t
            t = time.perf_counter()
            ts, stats = updater(ts,traj)
            jax.block_until_ready((ts,stats))
            sil_start = time.perf_counter()
            rollout_start_step = int(ts.global_step)-algo.iteration_size
            def record_success(episode, returns, metadata):
                if metadata['return_sum'] <= 0:
                    return
                destination = out/'successful_episodes'
                destination.mkdir(exist_ok=True)
                np.savez_compressed(destination/f'episode_{replay.positive_episodes:06d}.npz',
                    observations_packed=np.stack([row[0] for row in episode]),
                    observation_shape=np.array(replay.observation_shape),
                    actions=np.array([row[1] for row in episode],np.int32),
                    rewards=np.array([row[2] for row in episode],np.float32), returns=returns,
                    environment_steps_at_end=rollout_start_step+metadata['rollout_offset']*cfg['num_envs'])
            completed = replay.add_rollout(jax.device_get(traj.obs), jax.device_get(traj.action),
                jax.device_get(traj.reward), jax.device_get(traj.done), on_episode=record_success)
            assert len(completed) == int(traj.done.sum())
            with (out/'completed_episodes.jsonl').open('a') as episodes_file:
                for episode in completed:
                    episodes_file.write(json.dumps({**episode, 'environment_steps_at_end':rollout_start_step+episode['rollout_offset']*cfg['num_envs']})+'\n')
            sil_metrics = []
            if replay.ready and sil['enabled']:
                if first_sil_step is None:
                    first_sil_step = int(ts.global_step)
                for _ in range(sil['updates_per_rollout']):
                    batch = replay.sample(sil['batch_size'])
                    agent_state, metrics = sil_update(ts.agent_ts, batch['observations'],
                        batch['actions'], batch['returns'], batch['weights'])
                    jax.block_until_ready((agent_state, metrics))
                    metrics = jax.device_get(metrics)
                    replay.update_priorities(batch['indices'], metrics['priorities'])
                    ts = ts.replace(agent_ts=agent_state)
                    sil_metrics.append(metrics)
            sil_seconds = time.perf_counter()-sil_start
            applied = sum(int(m['applied']) for m in sil_metrics)
            total_sil_updates += applied;total_sil_seconds += sil_seconds
            update_seconds = time.perf_counter()-t
            row = {k: float(v) for k,v in stats.items()}
            for name in ['loss','actor_loss','value_loss','active_fraction']:
                row['sil_'+name] = float(np.mean([m[name] for m in sil_metrics])) if sil_metrics else 0.
            row.update(sil_updates_applied=applied, sil_seconds=sil_seconds,
                       replay_size=replay.size, replay_positive_episodes=replay.positive_episodes,
                       completed_scoring_episodes=sum(e['return_sum'] > 0 for e in completed))
            if not all(np.isfinite(list(row.values()))):
                raise FloatingPointError(f'Nonfinite PPO diagnostics: {row}')
            if not all(np.all(np.isfinite(x)) for x in jax.tree.leaves(jax.device_get(ts.agent_ts.params))):
                raise FloatingPointError('Nonfinite parameters')
            counts = np.bincount(np.asarray(traj.action).ravel(),minlength=raw.num_actions)
            row.update(step=int(ts.global_step),reward_sum=float(traj.reward.sum()),
                       positive_reward_steps=int((traj.reward>0).sum()),done_count=int(traj.done.sum()),
                       collect_seconds=collect_seconds,update_seconds=update_seconds,
                       steps_per_second=algo.iteration_size/(collect_seconds+update_seconds))
            row.update({f'action_{i}_fraction':float(v/counts.sum()) for i,v in enumerate(counts)})
            writer.writerow(row); f.flush()
            total_collect += collect_seconds; total_update += update_seconds
            if iterations:
                steady_collect += collect_seconds; steady_update += update_seconds
            iterations += 1
            if iterations == 1 or iterations % 8 == 0:
                print('TRAIN',row,flush=True)
            previous_step = int(ts.global_step)-algo.iteration_size
            if int(ts.global_step)//cfg['eval_freq'] > previous_step//cfg['eval_freq'] or int(ts.global_step) == cfg['total_timesteps']:
                checkpoint_and_eval(ts)
    delta = sum(float(np.square(a-b).sum()) for a,b in zip(jax.tree.leaves(initial_params),jax.tree.leaves(jax.device_get(ts.agent_ts.params))))**0.5
    write_json(out/'summary.json', dict(step=int(ts.global_step),seed=cfg['seed'],
        wall_training_and_periodic_eval_seconds=time.perf_counter()-start,
        collect_seconds=total_collect,update_seconds=total_update,
        training_steps_per_second=(int(ts.global_step)-start_step)/(total_collect+total_update),
        steady_training_steps_per_second=(iterations-1)*algo.iteration_size/max(steady_collect+steady_update,1e-9),
        parameter_change_l2=delta, final_evaluation=evaluations[-1]['summary'],
        sil_updates_applied=total_sil_updates, sil_seconds=total_sil_seconds,
        first_sil_step=first_sil_step, replay_size=replay.size, replay_positive_episodes=replay.positive_episodes))
    print('COMPLETE',out,flush=True)


@hydra.main(version_base=None, config_path='conf', config_name='self_imitation')
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    count = config.pop('num_seeds')
    for offset in range(count):
        c = dict(config)
        c['seed'] = config['seed']+offset
        if count > 1:
            c['run_dir'] = str(Path(config['run_dir'])/f"seed{c['seed']}")
        run(c)


if __name__ == '__main__':
    main()
