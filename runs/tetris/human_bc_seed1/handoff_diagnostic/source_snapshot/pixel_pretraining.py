"""Offline pixel/action cloning and an explicit boundary into fresh on-policy PPO.

This module is not imported by either existing live trainer. Persistent cloning
state contains parameters/optimizer/RNG/counters, never observations or actions.
"""
from dataclasses import dataclass
from functools import partial
import hashlib
import io
import json
from pathlib import Path

from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class PixelActions:
    observations: np.ndarray
    actions: np.ndarray
    provenance: dict


def _check_pairs(observations, actions):
    if observations.shape != (len(actions), 4, 32, 64) or len(actions) == 0:
        raise ValueError('Nonempty aligned [time,4,32,64] pixel/action pairs required')
    if not np.all((observations == 0) | (observations == 1)):
        raise ValueError('Binary pixels required')
    if actions.ndim != 1 or actions.dtype.kind not in 'iu' or not np.all((actions >= 0) & (actions < 5)):
        raise ValueError('Integer actions in [0,5) required')


def load_human_pairs(path, expected_sha256):
    """Load one immutable, separately replay-validated human recording/prefix.

    Hash pinning protects the selected version if an autosave path changes.
    Full emulator snapshots and score-derived metadata never enter inputs.
    This structural loader does not replace validate_demonstration.
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Dataset hash differs from the validated version')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('source') != 'human' or metadata.get('format_version') != 1:
            raise ValueError('A version-1 human recording is required; exclude automated tests')
        if metadata.get('environment') != 'tetris' or metadata.get('observation_shape') != [4, 32, 64]:
            raise ValueError('Unexpected environment/observation format')
        if metadata.get('env_kwargs') != dict(timer_mode='frame', frame_skip=4, fps=60,
                instruction_frequency=700, max_num_steps_per_episodes=4500):
            raise ValueError('Recording timing differs from the corrected PPO environment')
        actions = archive['actions'].copy()
        stacks = archive['observations']
        if stacks.shape != (len(actions)+1, 4, 32, 64):
            raise ValueError('Human recordings must include the final successor observation')
        observations = stacks[:-1].copy()
        _check_pairs(observations, actions)
        if metadata['steps'] != len(actions):
            raise ValueError('Recording step count mismatch')
        return PixelActions(observations.astype(np.bool_), actions.astype(np.int32),
            dict(sha256=digest, source='human', decisions=len(actions), complete=bool(metadata['complete']),
                 return_sum=float(metadata['return_sum']), episode_id=metadata['episode_id'],
                 reset_seed=metadata['reset_seed'], full_replay_validation_required=True))


def actor_log_probs(module, observations, actions):
    # Access only the encoder and actor, so the cloning parameter tree has no
    # critic and cannot accidentally fit score targets or alter value weights.
    return module.actor.log_prob_entropy(module.encode(observations), actions)


@struct.dataclass
class CloningState:
    policy: TrainState
    rng: jax.Array
    epochs: jax.Array
    examples_seen: jax.Array


class PixelCloner:
    def __init__(self, agent, learning_rate=.0005, max_grad_norm=.5, batch_size=128):
        if not np.isfinite(learning_rate) or learning_rate <= 0 or max_grad_norm <= 0 or not np.isfinite(max_grad_norm):
            raise ValueError('Positive finite optimizer settings required')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive integer batch size required')
        self.agent, self.learning_rate, self.max_grad_norm, self.batch_size = agent, learning_rate, max_grad_norm, batch_size

    def init(self, full_policy_params, key):
        params = {'params': {k: v for k, v in full_policy_params['params'].items() if k != 'critic'}}
        if 'actor' not in params['params']:
            raise ValueError('Actor parameters missing')
        policy = TrainState.create(apply_fn=self.agent.apply, params=params,
            tx=optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(self.learning_rate)))
        return CloningState(policy, key, jnp.int32(0), jnp.int32(0))

    def epoch(self, state, observations, actions):
        _check_pairs(np.asarray(observations), np.asarray(actions))
        return self._epoch(state, jnp.asarray(observations), jnp.asarray(actions))

    @partial(jax.jit, static_argnums=(0,))
    def _epoch(self, state, observations, actions):
        n = actions.shape[0]
        rng, key = jax.random.split(state.rng)
        order = jax.random.permutation(key, n)
        padded = ((n+self.batch_size-1)//self.batch_size)*self.batch_size
        indices = jnp.pad(order, (0, padded-n)).reshape(-1, self.batch_size)
        masks = (jnp.arange(padded) < n).reshape(-1, self.batch_size)

        def step(policy, batch):
            index, mask = batch
            weights = mask.astype(jnp.float32)
            count = weights.sum()
            def objective(params):
                lp, entropy = self.agent.apply(params, observations[index], actions[index], method=actor_log_probs)
                return -(weights*lp).sum()/count, (-(weights*lp).sum(), (weights*entropy).sum())
            (loss, (nll, entropy)), grads = jax.value_and_grad(objective, has_aux=True)(policy.params)
            return policy.apply_gradients(grads=grads), dict(nll_sum=nll, entropy_sum=entropy,
                examples=count, grad_norm=optax.global_norm(grads))
        policy, metrics = jax.lax.scan(step, state.policy, (indices, masks))
        return state.replace(policy=policy, rng=rng, epochs=state.epochs+1,
            examples_seen=state.examples_seen+n), dict(loss=metrics['nll_sum'].sum()/n,
                entropy=metrics['entropy_sum'].sum()/n, examples=metrics['examples'].sum(),
                max_raw_grad_norm=metrics['grad_norm'].max())

    @partial(jax.jit, static_argnums=(0,))
    def fit_metrics(self, state, observations, actions):
        lp, entropy = self.agent.apply(state.policy.params, observations, actions, method=actor_log_probs)
        return dict(cross_entropy=-lp.mean(), entropy=entropy.mean())
