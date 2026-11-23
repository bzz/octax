"""Evaluation, provenance and checkpoint helpers for pixel-only Octax PPO."""
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def environment_from_config(config):
    """Old checkpoints retain their original timer semantics when evaluated."""
    from octax.environments import create_environment
    return create_environment(config['env'], **config.get('env_kwargs', {'timer_mode': 'legacy'}))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def system_info():
    return dict(platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), backend=jax.default_backend(),
                devices=[str(x) for x in jax.devices()],
                versions={p: importlib.metadata.version(p) for p in
                          ['jax', 'jaxlib', 'flax', 'optax', 'rejax', 'gymnax', 'numpy']},
                upstream_commit='3aa53b516152e97f2ed91eae6e33b6ee9a97596b',
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_dirty=bool(subprocess.check_output(
                    ['git', 'status', '--porcelain', '--untracked-files=normal'], text=True).strip()))


def make_evaluator(env, agent=None, episodes=8, seed_start=10000):
    """Evaluate raw env to retain final scores and termination flags (no auto-reset)."""
    def single(params, seed, mode):
        key = jax.random.PRNGKey(seed)
        key, reset_key = jax.random.split(key)
        state, obs, _ = env.reset(reset_key)
        initial = (state, obs, key, jnp.float32(0), jnp.int32(0),
                   jnp.zeros(env.num_actions, jnp.int32), jnp.bool_(False), jnp.bool_(False))
        def body(carry):
            state, obs, key, ret, length, counts, _, _ = carry
            key, ak = jax.random.split(key)
            if mode == 'random':
                action = jax.random.randint(ak, (), 0, env.num_actions)
            elif mode == 'noop':
                action = jnp.int32(env.num_actions - 1)
            else:
                action = agent.apply(params, obs.transpose(0, 2, 1)[None], ak)[0][0]
            state, obs, reward, term, trunc, _ = env.step(state, action)
            return state, obs, key, ret + reward, length + 1, counts.at[action].add(1), term, trunc
        result = jax.lax.while_loop(lambda c: ~(c[-2] | c[-1]), body, initial)
        state, _, _, ret, length, counts, term, trunc = result
        return dict(returns=ret, lengths=length, action_counts=counts, terminated=term,
                    truncated=trunc, final_score=state.current_score)
    seeds = jnp.arange(seed_start, seed_start + episodes)
    return jax.jit(lambda params, mode: jax.vmap(single, in_axes=(None, 0, None))(params, seeds, mode), static_argnums=1)


def evaluate(evaluator, params, mode):
    start = time.perf_counter()
    raw = jax.device_get(evaluator(params, mode))
    elapsed = time.perf_counter() - start
    counts = np.sum(raw['action_counts'], axis=0)
    summary = dict(mean_return=float(np.mean(raw['returns'])),
                   scoring_fraction=float(np.mean(raw['returns'] > 0)),
                   mean_length=float(np.mean(raw['lengths'])),
                   terminated=int(np.sum(raw['terminated'])), truncated=int(np.sum(raw['truncated'])),
                   action_distribution=(counts / max(counts.sum(), 1)).tolist(), seconds=elapsed)
    return dict(summary=summary, episodes={k: v.tolist() for k, v in raw.items()})


def save_gif(env, path, agent=None, params=None, mode='random', seed=20000, max_steps=600):
    key = jax.random.PRNGKey(seed)
    state, obs, _ = env.reset(key)
    if agent is not None:
        act = jax.jit(lambda p, o, k: agent.apply(p, o.transpose(0, 2, 1)[None], k)[0][0])
    frames = []
    for i in range(max_steps):
        key, ak = jax.random.split(key)
        if mode == 'random':
            action = jax.random.randint(ak, (), 0, env.num_actions)
        elif mode == 'noop':
            action = env.num_actions - 1
        else:
            action = act(params, obs, ak)
        state, obs, _, term, trunc, _ = env.step(state, action)
        # Each environment action spans four 60Hz frames; sample every two actions.
        if i % 2 == 0:
            frame = Image.fromarray(np.asarray(state.display.T, dtype=np.uint8) * 255)
            frames.append(frame.resize((512, 256), Image.Resampling.NEAREST))
        if bool(term | trunc):
            break
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=130, loop=0)


# Re-export for existing experiment entry points.
from octax.checkpoints import save_checkpoint, restore_checkpoint
