"""Stage A: controls, resets, rewards and completed baseline episodes."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from octax.environments import create_environment
from experiment_utils import evaluate, make_evaluator, save_gif, write_json, system_info

p = argparse.ArgumentParser()
p.add_argument('--env', default='tetris')
p.add_argument('--episodes', type=int, default=8)
p.add_argument('--out', default='runs/tetris/sanity_frame_v2')
p.add_argument('--timer-mode', default='frame', choices=['legacy', 'step_saturating', 'frame'])
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
env, meta = create_environment(a.env, timer_mode=a.timer_mode)
write_json(out/'system.json', system_info())
print('Environment ready', meta['title'], flush=True)
eval_fn = make_evaluator(env, episodes=a.episodes)
results = {'timer_mode': a.timer_mode}
for mode in ['random', 'noop']:
    results[mode] = evaluate(eval_fn, None, mode)
    write_json(out/'baselines.json', results)
    print(mode, results[mode]['summary'], flush=True)
    save_gif(env, out/f'{mode}.gif', mode=mode)

@jax.jit
def probe(seed, action):
    state, obs, _ = env.reset(jax.random.PRNGKey(seed))
    def step(s, _):
        ns, o, r, term, trunc, _ = env.step(s, action)
        return ns, (ns.display, r, ns.current_score, term, ns.V)
    state, trace = jax.lax.scan(step, state, None, length=120)
    return obs, trace

probes = [jax.device_get(probe(0, jnp.int32(i))) for i in range(env.num_actions)]
np.savez_compressed(out/'control_probes.npz', **{f'action_{i}':x[1][0] for i,x in enumerate(probes)})
reset_equal = np.array_equal(probe(0, 4)[0], probe(1, 4)[0])
trajectory_equal = np.array_equal(probe(0, 4)[1][0], probe(1, 4)[1][0])
results['checks'] = dict(raw_observation_shape=list(probes[0][0].shape),
    action_keys=np.asarray(env.action_set).tolist(), reset_pixels_identical=reset_equal,
    noop_first_120_steps_identical_across_seeds=trajectory_equal,
    noop_full_episode_lengths_vary=len(set(results['noop']['episodes']['lengths'])) > 1,
    action_pixels_differ_from_noop=[bool(np.any(x[1][0] != probes[-1][1][0])) for x in probes[:-1]],
    reward_matches_score_difference=all(np.allclose(x[1][1],np.diff(np.r_[float(env.cached_reset_state.current_score),x[1][2]])) for x in probes))
write_json(out/'baselines.json',results)
print('Checks',results['checks'],flush=True)
