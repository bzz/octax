"""Inactive RND component; no import or integration in the PPO trainer.

Only current-rollout pixels are supplied to this component. Persistent state
contains weights, optimizer/RNG state and running moments, never observations.
"""
from dataclasses import dataclass
from functools import partial

from flax import linen as nn, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from octax.wrappers import OctaxGymnaxWrapper


class NoveltyObservationWrapper(OctaxGymnaxWrapper):
    """Expose real transition pixels before Gymnax substitutes reset pixels."""
    def step_env(self, key, state, action, params):
        obs, state, reward, done, info = super().step_env(key,state,action,params)
        return obs,state,reward,done,dict(info,transition_observation=obs)


class NoveltyNetwork(nn.Module):
    predictor: bool = False
    features: int = 128

    @nn.compact
    def __call__(self, pixels):
        # Full newest frame, channels last; policy input is unchanged.
        x=pixels.astype(jnp.float32)
        for channels,kernel,stride in zip((16,32,32),((8,4),(4,4),(3,3)),((4,2),(2,2),(1,1))):
            x=nn.leaky_relu(nn.Conv(channels,kernel,strides=stride)(x))
        x=x.reshape((x.shape[0],-1))
        if self.predictor:
            x=nn.relu(nn.Dense(256)(x))
        return nn.Dense(self.features)(x)


@struct.dataclass
class Moments:
    count: jax.Array
    mean: jax.Array
    m2: jax.Array

    @classmethod
    def empty(cls,shape=()):
        return cls(jnp.float32(0),jnp.zeros(shape),jnp.zeros(shape))

    @property
    def variance(self):
        return jnp.where(self.count>0,jnp.maximum(self.m2/jnp.maximum(self.count,1),0),jnp.ones_like(self.mean))

    def update(self,values):
        if values.shape[0]==0:
            raise ValueError('Cannot update moments with an empty batch')
        count=jnp.float32(values.shape[0])
        mean=jnp.mean(values,axis=0)
        m2=jnp.sum(jnp.square(values-mean),axis=0)
        total=self.count+count
        delta=mean-self.mean
        return Moments(total,self.mean+delta*count/total,
                       self.m2+m2+jnp.square(delta)*self.count*count/total)


@struct.dataclass
class NoveltyState:
    predictor: TrainState
    target_params: object
    observation_moments: Moments
    return_moments: Moments
    discounted_trace: jax.Array
    rng: jax.Array
    rollouts: jax.Array


@struct.dataclass
class NoveltyBatch:
    pixels: jax.Array
    targets: jax.Array
    raw_bonus: jax.Array
    normalized_bonus: jax.Array
    next_observation_moments: Moments
    next_return_moments: Moments
    next_discounted_trace: jax.Array
    generation: jax.Array


@dataclass(frozen=True)
class PixelNovelty:
    learning_rate: float = 0.0001
    gamma: float = 0.99
    variance_floor: float = 0.0001
    features: int = 128

    def __post_init__(self):
        if not 0<=self.gamma<=1 or self.learning_rate<=0 or self.variance_floor<=0:
            raise ValueError('Invalid novelty settings')

    @property
    def target(self):
        return NoveltyNetwork(features=self.features)

    @property
    def predictor(self):
        return NoveltyNetwork(predictor=True,features=self.features)

    def init(self,key,num_envs):
        if num_envs<1:
            raise ValueError('At least one environment is required')
        target_key,predictor_key,rng=jax.random.split(key,3)
        pixels=jnp.zeros((1,32,64,1))
        target_params=self.target.init(target_key,pixels)
        predictor=TrainState.create(apply_fn=self.predictor.apply,
            params=self.predictor.init(predictor_key,pixels),tx=optax.adam(self.learning_rate))
        return NoveltyState(predictor,target_params,Moments.empty((32,64,1)),Moments.empty(),
                            jnp.zeros(num_envs),rng,jnp.int32(0))

    def prepare(self,state,next_observations,dones):
        """Freeze inputs, target features and rewards before predictor/PPO updates.

        next_observations are actual successor stacks, including terminal pixels,
        not Gymnax auto-reset observations. No state is changed by this method.
        """
        if next_observations.ndim!=5 or next_observations.shape[2:]!=(4,32,64):
            raise ValueError('Expected [time, environment, 4, 32, 64] pixel stacks')
        if next_observations.shape[:2]!=dones.shape or dones.shape[1:]!=state.discounted_trace.shape:
            raise ValueError('Transition and environment dimensions do not match')
        if next_observations.shape[0]==0:
            raise ValueError('Empty rollout')
        newest=next_observations[:,:,-1].reshape((-1,32,64,1)).astype(jnp.float32)
        variance=jnp.maximum(state.observation_moments.variance,self.variance_floor)
        pixels=jnp.clip((newest-state.observation_moments.mean)/jnp.sqrt(variance),-5,5)
        targets=jax.lax.stop_gradient(self.target.apply(state.target_params,pixels))
        predicted=self.predictor.apply(state.predictor.params,pixels)
        raw_bonus=jnp.mean(jnp.square(targets-predicted),axis=-1).reshape(dones.shape)

        def step(trace,transition):
            reward,done=transition
            value=self.gamma*trace+reward
            return jnp.where(done,0,value),value
        next_trace,returns=jax.lax.scan(step,state.discounted_trace,(raw_bonus,dones))
        next_returns=state.return_moments.update(returns.reshape(-1))
        normalized=raw_bonus/jnp.sqrt(jnp.maximum(next_returns.variance,self.variance_floor))
        return NoveltyBatch(pixels,targets,jax.lax.stop_gradient(raw_bonus),
            jax.lax.stop_gradient(normalized),state.observation_moments.update(newest),
            next_returns,next_trace,state.rollouts)

    def finish(self,state,batch,*,epochs=4,minibatches=16):
        """Fit this rollout once, then advance statistics and discard the batch.

        Host-side generation validation prevents accidental replay of a consumed
        rollout. The numeric update is compiled separately for the host runner.
        """
        if int(batch.generation)!=int(state.rollouts):
            raise ValueError('Stale novelty batch; this rollout was already consumed')
        if epochs<1 or minibatches<1 or batch.pixels.shape[0]%minibatches:
            raise ValueError('Positive epochs and evenly divisible minibatches required')
        return self._fit(state,batch,epochs=epochs,minibatches=minibatches)

    @partial(jax.jit,static_argnums=(0,),static_argnames=('epochs','minibatches'))
    def _fit(self,state,batch,*,epochs,minibatches):
        def epoch(carry,_):
            predictor,rng=carry
            rng,key=jax.random.split(rng)
            indices=jax.random.permutation(key,batch.pixels.shape[0]).reshape((minibatches,-1))
            def update(predictor,index):
                def loss(params):
                    predicted=self.predictor.apply(params,batch.pixels[index])
                    return jnp.mean(jnp.square(predicted-batch.targets[index]))
                value,gradient=jax.value_and_grad(loss)(predictor.params)
                return predictor.apply_gradients(grads=gradient),value
            predictor,losses=jax.lax.scan(update,predictor,indices)
            return (predictor,rng),losses
        (predictor,rng),losses=jax.lax.scan(epoch,(state.predictor,state.rng),None,length=epochs)
        return state.replace(predictor=predictor,rng=rng,rollouts=state.rollouts+1,
            observation_moments=batch.next_observation_moments,
            return_moments=batch.next_return_moments,discounted_trace=batch.next_discounted_trace),losses
