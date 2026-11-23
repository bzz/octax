import jax
import jax.numpy as jnp
import numpy as np
import pytest
from octax.env import OctaxEnv


@pytest.mark.parametrize('disabled,expected', [(False,1),(True,0)])
def test_timer_ticks_once_for_each_of_four_frames(tmp_path,disabled,expected):
    rom=tmp_path/'loop.ch8';rom.write_bytes(bytes([0x12,0x00]))
    env=OctaxEnv(str(rom),disable_delay=disabled)
    state,_,_=env.reset(jax.random.PRNGKey(0))
    state=state.replace(delay_timer=jnp.uint8(5),sound_timer=jnp.uint8(5))
    state,obs,_,_,_,_=env.step(state,env.num_actions-1)
    assert int(state.delay_timer)==expected
    assert int(state.sound_timer)==expected
    assert obs.shape==(4,64,32)
    np.testing.assert_array_equal(obs,jnp.zeros_like(obs))
