import hashlib

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from octax.checkpoints import save_checkpoint,restore_checkpoint
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from pixel_novelty import Moments,NoveltyObservationWrapper,PixelNovelty


def equal_trees(a,b):
    # Compare persisted values, excluding freshly constructed optimizer callback
    # identities. The restored next update is checked independently below.
    a,b=serialization.to_state_dict(a),serialization.to_state_dict(b)
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)


def canonical_hash(tree):
    # JIT reconstructs dicts in sorted-key order; canonicalize both sides before
    # hashing so insertion order cannot masquerade as changed model weights.
    return hashlib.sha256(serialization.to_bytes(jax.tree.map(np.asarray,tree))).hexdigest()


def pixels():
    return jax.random.bernoulli(jax.random.PRNGKey(12),.15,(16,4,4,32,64))


@pytest.fixture(scope='module')
def prepared():
    model=PixelNovelty()
    state=model.init(jax.random.PRNGKey(2),4)
    obs=pixels();dones=jnp.zeros((16,4),bool).at[7,0].set(True).at[-1,1].set(True)
    batch=jax.jit(model.prepare)(state,obs,dones)
    return model,state,obs,dones,batch


def test_running_moments_match_direct_combined_population():
    x=np.arange(60,dtype=np.float32).reshape(12,5)/7
    m=Moments.empty((5,)).update(jnp.array(x[:3])).update(jnp.array(x[3:]))
    np.testing.assert_allclose(m.mean,x.mean(axis=0),rtol=1e-6)
    np.testing.assert_allclose(m.variance,x.var(axis=0),rtol=1e-6)
    assert float(m.count)==12
    np.testing.assert_array_equal(Moments.empty((5,)).variance,np.ones(5))
    with pytest.raises(ValueError,match='empty'):
        m.update(jnp.zeros((0,5)))


def test_bonus_uses_full_newest_pixels_and_freezes_rollout_values(prepared):
    model,state,obs,dones,batch=prepared
    expected=np.asarray(obs[:,:,-1]).reshape(-1,32,64,1).astype(np.float32)
    np.testing.assert_array_equal(batch.pixels,expected) # No /255 or crop.
    altered=obs.at[:,:,:-1].set(~obs[:,:,:-1])
    same=model.prepare(state,altered,dones)
    equal_trees(batch,same) # Older frames do not enter RND.
    before=serialization.to_bytes(batch)
    target_hash=canonical_hash(state.target_params)
    updated,loss=model.finish(state,batch,epochs=12,minibatches=4)
    assert np.isfinite(loss).all()
    assert serialization.to_bytes(batch)==before
    equal_trees(updated.target_params,state.target_params)
    assert canonical_hash(updated.target_params)==target_hash
    after=np.mean(np.square(model.predictor.apply(updated.predictor.params,batch.pixels)-batch.targets))
    before_error=float(np.mean(batch.raw_bonus))
    assert float(after)<before_error*.5,(after,before_error)
    assert int(updated.predictor.step)==48 and int(updated.rollouts)==1
    assert float(state.observation_moments.count)==0 and float(updated.observation_moments.count)==64
    with pytest.raises(ValueError,match='Stale'):
        model.finish(updated,batch)


def test_normalization_filter_respects_episode_boundaries(prepared):
    model,state,obs,dones,batch=prepared
    trace=np.zeros(4);values=[]
    for reward,done in zip(np.asarray(batch.raw_bonus),np.asarray(dones)):
        value=model.gamma*trace+reward
        values.append(value.copy());trace=np.where(done,0,value)
    np.testing.assert_allclose(batch.next_discounted_trace,trace,rtol=1e-6)
    expected_var=np.var(values)
    np.testing.assert_allclose(batch.next_return_moments.variance,expected_var,rtol=1e-6)
    np.testing.assert_allclose(batch.normalized_bonus,np.asarray(batch.raw_bonus)/np.sqrt(max(expected_var,model.variance_floor)),rtol=1e-6)
    assert batch.next_discounted_trace[1]==0


def test_zero_variance_pixels_and_zero_bonus_remain_finite(prepared):
    model,state,obs,dones,_=prepared
    zeros=jnp.zeros_like(obs)
    constant=model.prepare(state,zeros,dones)
    np.testing.assert_array_equal(constant.raw_bonus,np.zeros((16,4)))
    np.testing.assert_array_equal(constant.normalized_bonus,np.zeros((16,4)))
    calibrated=state.replace(observation_moments=constant.next_observation_moments)
    bright=model.prepare(calibrated,jnp.ones_like(obs),dones)
    np.testing.assert_array_equal(bright.pixels,np.full((64,32,64,1),5))
    assert np.isfinite(bright.raw_bonus).all() and np.isfinite(bright.normalized_bonus).all()


def test_full_auxiliary_checkpoint_reproduces_next_bonus_and_update(prepared,tmp_path):
    model,state,obs,dones,batch=prepared
    state,_=model.finish(state,batch,epochs=1,minibatches=4)
    path=tmp_path/'rnd.msgpack'
    save_checkpoint(path,state,{'component':'inactive_pixel_novelty','learning_rate':model.learning_rate})
    restored,cfg=restore_checkpoint(path,model.init(jax.random.PRNGKey(99),4))
    assert cfg['component']=='inactive_pixel_novelty'
    equal_trees(state,restored)
    a=model.prepare(state,obs,dones);b=model.prepare(restored,obs,dones)
    equal_trees(a,b)
    ua,la=model.finish(state,a,epochs=1,minibatches=4)
    ub,lb=model.finish(restored,b,epochs=1,minibatches=4)
    equal_trees(ua,ub);np.testing.assert_array_equal(la,lb)
    # Persistent state contains no observations, batches, targets-per-transition
    # or reward arrays. The frozen target parameters are model weights only.
    assert set(serialization.to_state_dict(ua))=={'predictor','target_params','observation_moments','return_moments','discounted_trace','rng','rollouts'}


def test_terminal_pixels_survive_autoreset_without_changing_environment():
    raw,_=create_environment('tetris',timer_mode='frame')
    original=OctaxGymnaxWrapper(raw);wrapped=NoveltyObservationWrapper(raw)
    key=jax.random.PRNGKey(43);obs,state=original.reset(key)
    stepkey=jax.random.PRNGKey(44)
    normal=original.step(stepkey,state,4)
    extra=wrapped.step(stepkey,state,4)
    equal_trees(normal[:4],extra[:4])
    np.testing.assert_array_equal(extra[4]['transition_observation'],normal[0])

    # Find a natural terminal transition under no-op in the actual game.
    @jax.jit
    def before_terminal(state):
        def step(carry):
            current,previous,done=carry
            following,_,_,term,trunc,_=raw.step(current,4)
            return following,current,term|trunc
        return jax.lax.while_loop(lambda c:~c[2],step,(state,state,jnp.bool_(False)))[1]
    terminal_input=before_terminal(state)
    actual_state,actual_obs,reward,term,trunc,_=raw.step(terminal_input,4)
    assert bool(term|trunc)
    baseline=original.step(stepkey,terminal_input,4)
    result=wrapped.step(stepkey,terminal_input,4)
    equal_trees(baseline[:4],result[:4])
    np.testing.assert_array_equal(result[4]['transition_observation'],actual_obs.transpose(0,2,1))
    assert not np.array_equal(result[0],result[4]['transition_observation'])
    assert float(result[2])==float(reward)


def test_invalid_shapes_and_configuration_are_rejected(prepared):
    model,state,obs,dones,batch=prepared
    with pytest.raises(ValueError):PixelNovelty(variance_floor=0)
    with pytest.raises(ValueError):model.init(jax.random.PRNGKey(1),0)
    with pytest.raises(ValueError):model.prepare(state,obs[:,:,:3],dones)
    with pytest.raises(ValueError):model.prepare(state,obs,dones[:,:3])
    with pytest.raises(ValueError):model.finish(state,batch,minibatches=3)
