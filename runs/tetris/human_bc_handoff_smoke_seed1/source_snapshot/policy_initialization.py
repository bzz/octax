"""Weights-only disk boundary; no dataset loader or pretraining optimizer imports."""
import hashlib
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np


def save_policy_weights(path, params, metadata, *, include_actor=True):
    """Write initialization weights, never a resumable training state."""
    if set(params) != {'params'} or 'critic' in params['params']:
        raise ValueError('Export encoder/actor weights only; critic and training state are forbidden')
    if ('actor' in params['params']) != include_actor:
        raise ValueError('Actor presence must match the declared component selection')
    if set(params['params']) != ({'features', 'actor'} if include_actor else {'features'}):
        raise ValueError('Only the declared CNN/actor parameter collections may cross the boundary')
    payload = dict(format_version=1, kind='policy_initialization',
        components='encoder_actor' if include_actor else 'encoder', parameters=serialization.to_state_dict(params), metadata=metadata)
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_bytes(serialization.msgpack_serialize(jax.device_get(payload))); temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_policy_weights(path, expected_sha256):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Initialization weights checksum mismatch')
    payload = serialization.msgpack_restore(data)
    if set(payload) != {'format_version', 'kind', 'components', 'parameters', 'metadata'}:
        raise ValueError('Not a weights-only initialization artifact')
    if payload['format_version'] != 1 or payload['kind'] != 'policy_initialization' or payload['components'] not in ('encoder', 'encoder_actor'):
        raise ValueError('Unsupported initialization artifact')
    params = payload['parameters']
    if set(params) != {'params'} or 'critic' in params['params']:
        raise ValueError('Initialization artifact contains forbidden parameter collections')
    if ('actor' in params['params']) != (payload['components'] == 'encoder_actor'):
        raise ValueError('Declared components do not match weights')
    expected = {'features', 'actor'} if payload['components'] == 'encoder_actor' else {'features'}
    if set(params['params']) != expected:
        raise ValueError('Unexpected parameter collection in weights-only artifact')
    return params, dict(sha256=digest, components=payload['components'], metadata=payload['metadata'])


def initialize_ppo_from_file(fresh, path, expected_sha256):
    weights, provenance = read_policy_weights(path, expected_sha256)
    initialized = transfer_to_fresh_ppo(fresh, weights, include_actor=provenance['components'] == 'encoder_actor')
    return initialized, provenance

def transfer_to_fresh_ppo(fresh, pretrained_params, *, include_actor):
    """Transfer selected weights; preserve the fresh target critic/RNG/env/Adam.

    The source is a weight tree, not a checkpoint containing rollout/replay or
    optimizer state. Refuse an already-running PPO state at this stage boundary.
    """
    if int(fresh.global_step) != 0 or int(fresh.agent_ts.step) != 0:
        raise ValueError('Pretraining transfer requires fresh PPO state')
    expected_opt = fresh.agent_ts.tx.init(fresh.agent_ts.params)
    if jax.tree.structure(expected_opt) != jax.tree.structure(fresh.agent_ts.opt_state) or any(
            not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(expected_opt), jax.tree.leaves(fresh.agent_ts.opt_state))):
        raise ValueError('Pretraining transfer requires fresh PPO optimizer moments')
    target = fresh.agent_ts.params['params']
    source = pretrained_params['params']
    keys = [k for k in target if k != 'critic' and (include_actor or k != 'actor')]
    for key in keys:
        if key not in source or jax.tree.structure(target[key]) != jax.tree.structure(source[key]):
            raise ValueError(f'Incompatible pretrained parameter structure: {key}')
        for a, b in zip(jax.tree.leaves(target[key]), jax.tree.leaves(source[key])):
            if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(b).all():
                raise ValueError(f'Incompatible/nonfinite pretrained weights: {key}')
    merged = dict(target)
    for key in keys:
        merged[key] = jax.tree.map(jnp.asarray, source[key])
    params = dict(fresh.agent_ts.params, params=merged)
    return fresh.replace(agent_ts=fresh.agent_ts.replace(params=params))
