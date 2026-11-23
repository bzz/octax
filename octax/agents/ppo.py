from collections.abc import Callable
import math
from typing import Tuple

import chex
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState
import jax.numpy as jnp
from rejax import Algorithm
from rejax.algos.algorithm import register_init
from rejax.algos.mixins import OnPolicyMixin

from flax import linen as nn
from rejax.algos.ppo import AdvantageMinibatch, Trajectory
from rejax.networks import DiscretePolicy, VNetwork


def normalize_advantages(advantages, std_floor=0.0):
    """Avoid turning tiny, reward-free critic errors into unit-scale signals."""
    scale = jnp.maximum(advantages.std() + 1e-8, std_floor)
    return (advantages - advantages.mean()) / scale


def gradient_clipper(mode, limit):
    if mode == "global_norm":
        return optax.clip_by_global_norm(limit)
    if mode == "elementwise":
        return optax.clip(limit)
    raise ValueError(f"Unknown gradient clipping mode: {mode}")

class Agent(nn.Module):
    action_dim: int
    features_list: Tuple[int] = (32, 64, 64)
    kernel_sizes: Tuple[Tuple[int, int] | int] = ((8, 4), (4, 4), (3, 3))
    strides_list: Tuple[Tuple[int, int] | int] = ((4, 2), (2, 2), (1, 1))
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    hidden_layer_sizes: Tuple[int] = (256,)

    def setup(self) -> None:
        layers = []

        for features, kernel_size, strides in zip(self.features_list, self.kernel_sizes, self.strides_list):
            layers.append(nn.Conv(features=features, kernel_size=kernel_size, strides=strides))
            layers.append(self.activation)
        layers.append(lambda x: x.reshape(x.shape[0], -1))

        self.features = nn.Sequential(layers)

        self.actor = DiscretePolicy(
            action_dim=self.action_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
        )

        self.critic = VNetwork(hidden_layer_sizes=self.hidden_layer_sizes,activation=self.activation)

    def __call__(self, obs, rng, action=None):
        features = self.encode(obs)

        value = self.critic(features)

        #  return action, log_prob, entropy, value
        if action is None:
            return *self.actor(features, rng), value
        else:
            return action, *self.actor.log_prob_entropy(features, action), value

    def call_critic(self, obs):
        features = self.encode(obs)
        return self.critic(features)

    def call_actor(self, obs, rng):
        features = self.encode(obs)
        return self.actor(features, rng)

    def encode(self, obs):
        # Gymnax gives NCHW; Flax convolutions require NHWC.
        return self.features(jnp.transpose(obs, (0, 2, 3, 1)).astype(jnp.float32))


class PPOOctax(OnPolicyMixin, Algorithm):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    advantage_std_floor: chex.Scalar = struct.field(pytree_node=True, default=0.0)
    grad_clip_mode: str = struct.field(pytree_node=False, default="elementwise")
    zero_init_value: bool = struct.field(pytree_node=False, default=False)
    target_kl: float | None = struct.field(pytree_node=False, default=None)

    def make_act(self, ts):
        def act(obs, rng):
            obs = jnp.expand_dims(obs, 0)
            action, _, _, _ = self.agent.apply(ts.agent_ts.params, obs, rng)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        if "activation" in agent_kwargs:
            agent_kwargs["activation"] = getattr(nn, agent_kwargs["activation"])
        return Agent(action_dim=env.action_space(env_params).n, **agent_kwargs)

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.zeros([1, *self.env.observation_space(self.env_params).shape])
        agent_params = self.agent.init(rng, obs_ph, rng)
        if self.zero_init_value:
            # Same architecture/parameter names; only the fresh value output changes.
            output_name = f"Dense_{len(self.agent.hidden_layer_sizes)}"
            output = agent_params['params']['critic'][output_name]
            agent_params['params']['critic'][output_name] = jax.tree.map(jnp.zeros_like, output)

        tx = optax.chain(
            gradient_clipper(self.grad_clip_mode, self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        return {"agent_ts": TrainState.create(apply_fn=(), params=agent_params, tx=tx)}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)
        ts, _ = self.update_trajectories(ts, trajectories)
        return ts

    def update_trajectories(self, ts, trajectories):

        if self.target_kl is not None:
            if not math.isfinite(self.target_kl) or self.target_kl <= 0:
                raise ValueError('target_kl must be positive and finite, or None')
            from octax.agents.ppo_kl_stop import update_with_kl_stop
            return update_with_kl_stop(self, ts, trajectories)

        last_val = self.agent.apply(
            ts.agent_ts.params, ts.last_obs, method="call_critic"
        )

        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, metrics = jax.lax.scan(
                self.update_with_metrics,
                ts,
                minibatches,
            )
            return ts, metrics

        ts, metrics = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts, jax.tree.map(jnp.mean, metrics)

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)


            action, log_prob, _, value = self.agent.apply(ts.agent_ts.params, ts.last_obs, rng_action)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, action, log_prob, reward, value, done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                    transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                    + self.gamma * next_value * (1 - transition.done)
                    - transition.value
            )
            advantage = (
                    delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update(self, ts, batch):
        ts, _ = self.update_with_metrics(ts, batch)
        return ts

    def update_with_metrics(self, ts, batch):
        def loss_fn(params):
            _, log_prob, entropy, value = self.agent.apply(params, batch.trajectories.obs, None, batch.trajectories.action)

            # Actor loss
            entropy = entropy.mean()
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = normalize_advantages(batch.advantages, self.advantage_std_floor)
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            actor_loss = pi_loss - self.ent_coef * entropy

            # Value loss
            value_pred_clipped = batch.trajectories.value + (
                    value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            value_loss =  self.vf_coef * value_loss

            log_ratio = log_prob - batch.trajectories.log_prob
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = (jnp.abs(ratio - 1) > self.clip_eps).mean()
            return actor_loss + value_loss, (actor_loss, value_loss, entropy, approx_kl, clip_fraction)

        (loss, (actor_loss, value_loss, entropy, approx_kl, clip_fraction)), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.agent_ts.params)
        clipper = gradient_clipper(self.grad_clip_mode, self.max_grad_norm)
        clipped, _ = clipper.update(grads, clipper.init(grads))
        metrics = dict(loss=loss, actor_loss=actor_loss, value_loss=value_loss,
                       entropy=entropy, grad_norm=optax.global_norm(grads),
                       clipped_grad_norm=optax.global_norm(clipped), approx_kl=approx_kl,
                       clip_fraction=clip_fraction, advantage_std=batch.advantages.std(),
                       normalized_advantage_std=normalize_advantages(batch.advantages, self.advantage_std_floor).std())
        return ts.replace(agent_ts=ts.agent_ts.apply_gradients(grads=grads)), metrics
