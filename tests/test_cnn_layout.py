import jax
import jax.numpy as jnp
import numpy as np
from octax.agents.ppo import Agent


def test_cnn_has_three_spatial_convolutions_and_four_input_channels():
    agent = Agent(action_dim=5)
    obs = jnp.zeros((2, 4, 32, 64))
    key = jax.random.PRNGKey(0)
    params = agent.init(key, obs, key)
    kernels = [x for x in jax.tree.leaves(params) if x.ndim == 4]
    assert len(kernels) == 3
    assert sorted(x.shape for x in kernels) == sorted([(8,4,4,32),(4,4,32,64),(3,3,64,64)])
    output = agent.apply(params, obs, key)
    assert all(x.shape == (2,) for x in output)
    np.testing.assert_array_equal(output[-1],agent.apply(params,obs,method='call_critic'))
    np.testing.assert_array_equal(output[0],agent.apply(params,obs,key,method='call_actor')[0])
