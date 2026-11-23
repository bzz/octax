from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from octax.checkpoints import restore_checkpoint, save_checkpoint


def test_checkpoint_preserves_optimizer_rng_and_exact_next_update(tmp_path):
    class State(struct.PyTreeNode):
        agent_ts: TrainState
        rng: jax.Array
        global_step: int
    model=TrainState.create(apply_fn=(),params={'w':jnp.array([1.,2.])},tx=optax.adam(0.01))
    model=model.apply_gradients(grads={'w':jnp.array([0.5,-0.1])})
    state=State(model,jax.random.PRNGKey(3),512)
    path=tmp_path/'checkpoint.msgpack'
    save_checkpoint(path,state,{'env':'tetris'})
    restored,config=restore_checkpoint(path,state)
    assert config['env']=='tetris'
    assert int(restored.global_step)==512
    np.testing.assert_array_equal(state.rng,restored.rng)
    gradients={'w':jnp.array([0.3,0.8])}
    a=state.agent_ts.apply_gradients(grads=gradients)
    b=restored.agent_ts.apply_gradients(grads=gradients)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)
