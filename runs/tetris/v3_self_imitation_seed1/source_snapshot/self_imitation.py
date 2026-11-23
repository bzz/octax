"""Experimental pixel-only self-imitation replay; not enabled by the v2 runner.

Uses completed on-policy episodes and original score returns. The policy receives
only its original observations. Based on Oh et al., ICML 2018, equations 1–3:
https://proceedings.mlr.press/v80/oh18b/oh18b.pdf
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def self_imitation_loss(log_probs, values, returns, weights, value_weight=.01):
    """Reinforce only past returns above the current value estimate.

    Stop the policy term's gradient through its advantage: otherwise it could
    reduce the classification loss by inflating the value prediction.
    """
    advantage = jax.nn.relu(returns - values)
    actor = -jnp.mean(weights * log_probs * jax.lax.stop_gradient(advantage))
    critic = .5 * jnp.mean(weights * jnp.square(advantage))
    return actor + value_weight * critic, dict(actor_loss=actor, value_loss=critic,
        active_fraction=jnp.mean(advantage > 0), mean_advantage=jnp.mean(advantage),
        priorities=advantage)


def make_self_imitation_updater(agent, loss_weight=.1, value_weight=.01):
    """Reuse the existing actor/critic and optimizer; no extra policy inputs."""
    @jax.jit
    def update(train_state, observations, actions, returns, weights):
        def objective(params):
            _, log_probs, _, values = agent.apply(params, observations, None, actions)
            loss, metrics = self_imitation_loss(log_probs, values, returns, weights, value_weight)
            return loss_weight * loss, metrics
        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(train_state.params)
        # An inactive batch should not advance Adam moments or parameters.
        active = metrics['active_fraction'] > 0
        updated = jax.lax.cond(active, lambda ts: ts.apply_gradients(grads=grads), lambda ts: ts, train_state)
        metrics = {**metrics, 'loss': loss, 'applied': active}
        return updated, metrics
    return update


class EpisodeReplay:
    """Bounded bit-packed binary-observation replay, with exact RNG persistence.

    Store complete episodes before computing discounted returns. Pending episodes
    are separate per environment, so neither terminal resets nor parallel workers
    can leak rewards into another trajectory. All completed episodes enter replay;
    prioritized sampling emphasizes transitions with positive return advantage.
    """
    def __init__(self, observation_shape, num_envs, capacity=50000, gamma=.995,
                 seed=0, alpha=.6, beta=.1, epsilon=1e-6):
        if capacity < 1 or num_envs < 1 or not 0 <= gamma <= 1:
            raise ValueError('Invalid replay capacity, environment count or discount')
        self.observation_shape = tuple(observation_shape)
        self.bits = int(np.prod(self.observation_shape))
        self.num_envs, self.capacity = num_envs, capacity
        self.gamma, self.alpha, self.beta, self.epsilon = gamma, alpha, beta, epsilon
        self.observations = np.zeros((capacity, (self.bits+7)//8), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.full(capacity, epsilon, dtype=np.float32)
        self.pending = [[] for _ in range(num_envs)]
        self.size = self.position = self.completed_episodes = self.positive_episodes = 0
        self.rng = np.random.default_rng(seed)

    def add_rollout(self, observations, actions, rewards, dones, on_episode=None):
        observations, actions, rewards, dones = map(np.asarray, (observations, actions, rewards, dones))
        if observations.shape[:2] != actions.shape or actions.shape != rewards.shape or rewards.shape != dones.shape:
            raise ValueError('Expected matching [time, environment] rollout axes')
        if actions.shape[1] != self.num_envs or observations.shape[2:] != self.observation_shape:
            raise ValueError('Observation/environment shape differs from replay')
        if not np.all((observations == 0) | (observations == 1)):
            raise ValueError('Lossless bit packing requires binary pixel observations')
        if not np.all(np.isfinite(rewards)):
            raise ValueError('Nonfinite reward')
        packed = np.packbits(observations.reshape(*actions.shape, self.bits).astype(np.uint8), axis=-1)
        completed = []
        for t in range(actions.shape[0]):
            for env in range(self.num_envs):
                self.pending[env].append((packed[t, env].copy(), int(actions[t, env]), float(rewards[t, env])))
                if dones[t, env]:
                    episode = self.pending[env]
                    discounted = np.empty(len(episode), dtype=np.float32)
                    value = 0.
                    for i in range(len(episode)-1, -1, -1):
                        value = episode[i][2] + self.gamma * value
                        discounted[i] = value
                    total = sum(row[2] for row in episode)
                    self._insert(episode, discounted)
                    self.completed_episodes += 1
                    self.positive_episodes += int(total > 0)
                    completed.append(dict(environment=env, length=len(episode), return_sum=total,
                                          positive_events=sum(row[2] > 0 for row in episode), rollout_offset=t+1))
                    if on_episode is not None:
                        on_episode(episode, discounted, completed[-1])
                    self.pending[env] = []
        return completed

    def _insert(self, episode, discounted):
        # Retain the most recent transitions if an episode exceeds capacity.
        for (observation, action, _), target in zip(episode, discounted):
            i = self.position
            self.observations[i] = observation
            self.actions[i] = action
            self.returns[i] = target
            self.priorities[i] = max(float(target), self.epsilon)
            self.position = (i+1) % self.capacity
            self.size = min(self.size+1, self.capacity)

    @property
    def ready(self):
        # Preserve the v2 trajectory/optimizer exactly until a real scoring episode
        # has completed. This is an explicit experimental activation rule.
        return self.size > 0 and self.positive_episodes > 0

    def sample(self, batch_size):
        if not self.ready:
            raise ValueError('Replay activates only after a completed scoring episode')
        probabilities = np.power(np.maximum(self.priorities[:self.size], self.epsilon), self.alpha)
        probabilities = probabilities / probabilities.sum()
        indices = self.rng.choice(self.size, size=batch_size, replace=True, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        observations = np.unpackbits(self.observations[indices], axis=-1, count=self.bits)
        return dict(indices=indices, observations=observations.reshape(batch_size, *self.observation_shape).astype(np.float32),
                    actions=self.actions[indices].copy(), returns=self.returns[indices].copy(), weights=weights)

    def update_priorities(self, indices, advantages):
        indices = np.asarray(indices); advantages = np.asarray(advantages)
        if indices.shape != advantages.shape or not np.all(np.isfinite(advantages)):
            raise ValueError('Invalid replay priorities')
        # Duplicate samples share one priority; keep their maximum rather than
        # depending on assignment order if a caller supplies distinct estimates.
        for i in np.unique(indices):
            self.priorities[i] = max(float(np.max(advantages[indices == i])), self.epsilon)

    def save(self, path, step):
        path = Path(path)
        metadata = dict(format_version=1, step=int(step), observation_shape=self.observation_shape,
            num_envs=self.num_envs, capacity=self.capacity, gamma=self.gamma, alpha=self.alpha,
            beta=self.beta, epsilon=self.epsilon, size=self.size, position=self.position,
            completed_episodes=self.completed_episodes, positive_episodes=self.positive_episodes,
            rng_state=self.rng.bit_generator.state)
        pending = [row for episode in self.pending for row in episode]
        arrays = dict(observations=self.observations[:self.size], actions=self.actions[:self.size],
            returns=self.returns[:self.size], priorities=self.priorities[:self.size],
            pending_lengths=np.array([len(x) for x in self.pending], dtype=np.int32),
            pending_observations=np.stack([x[0] for x in pending]) if pending else np.empty((0,(self.bits+7)//8),np.uint8),
            pending_actions=np.array([x[1] for x in pending], dtype=np.int32),
            pending_rewards=np.array([x[2] for x in pending], dtype=np.float32),
            metadata=np.array(json.dumps(metadata)))
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)

    @classmethod
    def load(cls, path, expected_step=None):
        with np.load(path, allow_pickle=False) as arrays:
            m = json.loads(str(arrays['metadata']))
            if m['format_version'] != 1 or (expected_step is not None and m['step'] != expected_step):
                raise ValueError('Replay/model checkpoint version or step mismatch')
            replay = cls(m['observation_shape'], m['num_envs'], m['capacity'], m['gamma'],
                         alpha=m['alpha'], beta=m['beta'], epsilon=m['epsilon'])
            for field in ['size','position','completed_episodes','positive_episodes']:
                setattr(replay, field, m[field])
            replay.rng.bit_generator.state = m['rng_state']
            for field in ['observations','actions','returns','priorities']:
                getattr(replay, field)[:replay.size] = arrays[field]
            index = 0
            for env, length in enumerate(arrays['pending_lengths']):
                replay.pending[env] = [(arrays['pending_observations'][i].copy(), int(arrays['pending_actions'][i]), float(arrays['pending_rewards'][i])) for i in range(index,index+length)]
                index += length
        return replay
