import jax
import jax.numpy as jnp
import pytest
from octax.env import tick_timers
from octax.state import create_state


@pytest.mark.parametrize('value,expected', [(0,0),(1,0),(16,15),(255,254)])
def test_timers_saturate_without_unsigned_wrap(value,expected):
    state=create_state().replace(delay_timer=jnp.uint8(value),sound_timer=jnp.uint8(value))
    result=jax.jit(tick_timers)(state)
    assert int(result.delay_timer)==expected
    assert int(result.sound_timer)==expected
    assert result.delay_timer.dtype==jnp.uint8
