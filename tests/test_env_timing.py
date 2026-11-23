import jax
import jax.numpy as jnp
import numpy as np
import pytest
from octax.env import OctaxEnv, tick_timers
from octax.state import create_state
from experiment_utils import environment_from_config


@pytest.mark.parametrize('mode,expected',[('legacy',4),('step_saturating',4)])
def test_timing_runs_one_tick_per_frame(tmp_path,mode,expected):
 rom=tmp_path/'loop.ch8';rom.write_bytes(bytes([0x12,0x00]))
 env=OctaxEnv(str(rom),disable_delay=False,timer_mode=mode)
 s,_,_=env.reset(jax.random.PRNGKey(0))
 s=s.replace(delay_timer=jnp.uint8(5),sound_timer=jnp.uint8(5))
 s,obs,_,_,_,_=env.step(s,env.num_actions-1)
 assert int(s.delay_timer)==expected
 assert int(s.sound_timer)==expected
 assert obs.shape==(4,64,32)
 np.testing.assert_array_equal(obs,jnp.zeros_like(obs))


def test_old_checkpoint_environment_keeps_legacy_timers():
 env,_=environment_from_config({'env':'tetris'})
 assert env.timer_mode=='legacy'
 env,_=environment_from_config({'env':'tetris','env_kwargs':{'timer_mode':'frame'}})
 assert env.timer_mode=='frame'
