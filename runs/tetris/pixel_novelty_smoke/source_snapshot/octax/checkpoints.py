"""Atomic, versioned checkpoints for model, optimizer, RNG and environment state."""
from pathlib import Path
import jax
from flax import serialization


def save_checkpoint(path, ts, config):
    # Full state includes weights, optimizer moments, RNG, envs and exact global step.
    payload = dict(format_version=1, config=config, train_state=serialization.to_state_dict(ts))
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(serialization.msgpack_serialize(jax.device_get(payload)))
    temp.replace(path)


def restore_checkpoint(path, template):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    return serialization.from_state_dict(template, payload['train_state']), payload['config']
