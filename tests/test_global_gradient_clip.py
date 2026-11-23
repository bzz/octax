import jax
import jax.numpy as jnp
import optax
from octax.agents.ppo import PPOOctax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper


def test_optimizer_limits_full_gradient_norm(monkeypatch):
    # Isolate the clipping transform from Adam's subsequent rescaling.
    monkeypatch.setattr(optax, 'adam', lambda learning_rate: optax.identity())
    raw,_=create_environment('tetris')
    env=OctaxGymnaxWrapper(raw)
    agent=PPOOctax.create_agent({},env,env.default_params)
    algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,
                  eval_callback=None,max_grad_norm=.5,grad_clip_mode='global_norm')
    state=algo.initialize_network_params(jax.random.PRNGKey(0))['agent_ts']
    grads=jax.tree.map(jnp.ones_like,state.params)
    clipped,_=state.tx.update(grads,state.opt_state,state.params)
    assert float(optax.global_norm(clipped)) <= .500001
