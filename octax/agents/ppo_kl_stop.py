"""Optional current-rollout KL stopping for the shared actor/critic optimizer.

The check precedes an update and cannot undo an earlier overshoot. It is a
minibatch stopping rule, not a hard bound on full-rollout policy divergence.
No guard state or observations persist beyond this update call.
"""
from flax import struct
import jax
import jax.numpy as jnp


LOSS_METRICS = ('loss', 'actor_loss', 'value_loss', 'entropy', 'grad_norm',
    'clipped_grad_norm', 'approx_kl', 'clip_fraction', 'advantage_std',
    'normalized_advantage_std')
GUARD_METRICS = ('optimizer_updates', 'gradient_evaluations', 'minibatches_checked',
    'skipped_minibatches', 'kl_stop_triggered', 'max_checked_exact_kl',
    'final_full_batch_kl')


@struct.dataclass
class GuardBatch:
    trajectories: object
    advantages: object
    targets: object
    old_log_probs: object


def all_log_probs(module, observations):
    return jax.nn.log_softmax(module.actor._action_dist(module.encode(observations)).logits, axis=-1)


def update_with_kl_stop(algo, ts, trajectories):
    last = algo.agent.apply(ts.agent_ts.params, ts.last_obs, method='call_critic')
    advantages, targets = algo.calculate_gae(trajectories, jnp.where(ts.last_done, 0, last))
    shape = trajectories.obs.shape
    observations = trajectories.obs.reshape(-1, *shape[2:])
    old_flat = algo.agent.apply(ts.agent_ts.params, observations, method=all_log_probs)
    batch = GuardBatch(trajectories, advantages, targets, old_flat.reshape(*shape[:2], -1))
    zero = jnp.zeros((), dtype=old_flat.dtype)

    def empty_metrics():
        return dict({key: zero for key in LOSS_METRICS}, accepted=jnp.int32(0),
            checked=jnp.int32(0), exact_kl=zero)

    def epoch(carry, unused):
        ts, stopped = carry
        # Preserve the ordinary epoch RNG schedule even after stopping.
        rng, minibatch_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        minibatches = algo.shuffle_and_split(batch, minibatch_rng)

        def step(carry, minibatch):
            ts, stopped = carry

            def inactive(_):
                return (ts, stopped), empty_metrics()

            def active(_):
                lp = algo.agent.apply(ts.agent_ts.params, minibatch.trajectories.obs, method=all_log_probs)
                kl = (jnp.exp(minibatch.old_log_probs)*(minibatch.old_log_probs-lp)).sum(axis=-1).mean()

                def reject(_):
                    # No gradient calculation or optimizer/moment update here.
                    return (ts, jnp.bool_(True)), dict(empty_metrics(), checked=jnp.int32(1), exact_kl=kl)

                def accept(_):
                    updated, metrics = algo.update_with_metrics(ts, minibatch)
                    return (updated, jnp.bool_(False)), dict(metrics, accepted=jnp.int32(1), checked=jnp.int32(1), exact_kl=kl)

                return jax.lax.cond(kl > algo.target_kl, reject, accept, operand=None)

            return jax.lax.cond(stopped, inactive, active, operand=None)

        return jax.lax.scan(step, (ts, stopped), minibatches)

    (updated, stopped), metrics = jax.lax.scan(epoch, (ts, jnp.bool_(False)), None, algo.num_epochs)
    accepted = metrics['accepted'].sum()
    denominator = jnp.maximum(accepted, 1)
    result = {key: metrics[key].sum()/denominator for key in LOSS_METRICS}
    final_flat = algo.agent.apply(updated.agent_ts.params, observations, method=all_log_probs)
    result.update(optimizer_updates=accepted, gradient_evaluations=accepted,
        minibatches_checked=metrics['checked'].sum(),
        skipped_minibatches=algo.num_epochs*algo.num_minibatches-accepted,
        kl_stop_triggered=stopped.astype(jnp.int32),
        max_checked_exact_kl=metrics['exact_kl'].max(),
        final_full_batch_kl=(jnp.exp(old_flat)*(old_flat-final_flat)).sum(axis=-1).mean())
    return updated, result
