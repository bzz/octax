import jax
import jax.numpy as jnp
import numpy as np
from octax.agents.ppo import Agent, PPOOctax
from rejax.algos.ppo import Trajectory


def test_gae_stops_at_terminal_and_bootstraps_unfinished_episode():
    algo = PPOOctax(env=None, env_params=None, eval_callback=None, gamma=0.9, gae_lambda=1.0)
    traj = Trajectory(obs=jnp.zeros((2,2,1)),action=jnp.zeros((2,2)),
        log_prob=jnp.zeros((2,2)), reward=jnp.array([[1.,1.],[2.,2.]]),
        value=jnp.zeros((2,2)),done=jnp.array([[True,False],[False,False]]))
    adv, targets = algo.calculate_gae(traj,jnp.array([10.,10.]))
    np.testing.assert_allclose(targets,[[1.,10.9],[11.,11.]],rtol=1e-6)
    np.testing.assert_array_equal(adv,targets)
