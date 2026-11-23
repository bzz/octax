"""Opt-in current-rollout PPO/RND integration, not used by train.py.

The original PPO state is composed unchanged with auxiliary state. This avoids
changing Rejax's registered-initializer RNG ordering and preserves controls.
"""
from functools import partial
from pathlib import Path

from flax import serialization, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rejax.algos.ppo import Trajectory
from rejax.networks import VNetwork

from octax.agents.ppo import normalize_advantages
from pixel_novelty import NoveltyObservationWrapper,PixelNovelty


@struct.dataclass
class OnlineNoveltyState:
    ppo: object
    intrinsic_value: TrainState
    novelty: object
    rollouts: jax.Array
    pending_update: jax.Array


@struct.dataclass
class OnlineRollout:
    trajectory: object
    successor_pixels: jax.Array
    intrinsic_values: jax.Array
    generation: jax.Array
    end_step: jax.Array
    policy_step: jax.Array
    end_rng: jax.Array


class OnlinePixelPPO:
    """Host-controlled rollout lifecycle with compiled numeric operations."""
    def __init__(self,base,raw_env,*,coefficient=0.01,novelty=None,intrinsic_value_coefficient=0.5):
        if not np.isfinite(coefficient) or coefficient<0:
            raise ValueError('Nonnegative finite novelty coefficient required')
        if intrinsic_value_coefficient<0 or not np.isfinite(intrinsic_value_coefficient):
            raise ValueError('Invalid intrinsic value coefficient')
        if base.grad_clip_mode!='global_norm':
            raise ValueError('This experiment requires global-norm PPO clipping')
        self.base=base
        self.coefficient=float(coefficient)
        self.intrinsic_value_coefficient=float(intrinsic_value_coefficient)
        self.novelty=novelty or PixelNovelty()
        self.wrapper=NoveltyObservationWrapper(raw_env)
        self.head=VNetwork(hidden_layer_sizes=base.agent.hidden_layer_sizes,activation=base.agent.activation)
        # The joint PPO gradients are clipped once before Adam. EmptyState
        # matches the existing clip transform's state, preserving base moments.
        self.preclipped_tx=optax.chain(optax.identity(),optax.adam(base.learning_rate))
        self.base_update=jax.jit(base.update_trajectories)
        self.prepare_novelty=jax.jit(self.novelty.prepare)

    def init(self,key):
        ppo=self.base.init_state(key)
        # Independent tagged keys; no consumption of the PPO collection RNG.
        features=self.base.agent.apply(ppo.agent_ts.params,ppo.last_obs[:1],method='encode')
        params=self.head.init(jax.random.fold_in(key,41001),features)
        output=f'Dense_{len(self.base.agent.hidden_layer_sizes)}'
        params['params'][output]=jax.tree.map(jnp.zeros_like,params['params'][output])
        head=TrainState.create(apply_fn=self.head.apply,params=params,tx=optax.adam(self.base.learning_rate))
        auxiliary=self.novelty.init(jax.random.fold_in(key,41002),self.base.num_envs)
        return OnlineNoveltyState(ppo,head,auxiliary,jnp.int32(0),jnp.bool_(False))

    def intrinsic_values(self,policy_params,value_params,observations):
        features=self.base.agent.apply(policy_params,observations,method='encode')
        return self.head.apply(value_params,features)

    def collect(self,state):
        if bool(state.pending_update):
            raise ValueError('Consume the current rollout before collecting another')
        return self._collect(state)

    @partial(jax.jit,static_argnums=(0,))
    def _collect(self,state):
        def step(ppo,_):
            rng,new_rng=jax.random.split(ppo.rng)
            ppo=ppo.replace(rng=rng)
            rng_steps,rng_action=jax.random.split(new_rng,2)
            rng_steps=jax.random.split(rng_steps,self.base.num_envs)
            action,log_prob,_,value=self.base.agent.apply(ppo.agent_ts.params,ppo.last_obs,rng_action)
            intrinsic=self.intrinsic_values(ppo.agent_ts.params,state.intrinsic_value.params,ppo.last_obs)
            obs,env_state,reward,done,info=jax.vmap(self.wrapper.step,in_axes=(0,0,0,None))(
                rng_steps,ppo.env_state,action,self.base.env_params)
            trajectory=Trajectory(ppo.last_obs,action,log_prob,reward,value,done)
            ppo=ppo.replace(env_state=env_state,last_obs=obs,last_done=done,
                            global_step=ppo.global_step+self.base.num_envs)
            return ppo,(trajectory,info['transition_observation'],intrinsic)
        ppo,(trajectory,successors,intrinsic)=jax.lax.scan(step,state.ppo,None,length=self.base.num_steps)
        rollout=OnlineRollout(trajectory,successors,intrinsic,state.rollouts,ppo.global_step,
                              ppo.agent_ts.step,ppo.rng)
        return state.replace(ppo=ppo,pending_update=jnp.bool_(True)),rollout

    def targets(self,state,rollout,normalized_bonus):
        last_ext=self.base.agent.apply(state.ppo.agent_ts.params,state.ppo.last_obs,method='call_critic')
        last_int=self.intrinsic_values(state.ppo.agent_ts.params,state.intrinsic_value.params,state.ppo.last_obs)
        last_ext=jnp.where(state.ppo.last_done,0,last_ext)
        last_int=jnp.where(state.ppo.last_done,0,last_int)
        ext_adv,ext_targets=self.base.calculate_gae(rollout.trajectory,last_ext)
        intrinsic=rollout.trajectory.replace(reward=normalized_bonus,value=rollout.intrinsic_values)
        int_adv,int_targets=self.base.replace(gamma=self.novelty.gamma).calculate_gae(intrinsic,last_int)
        return ext_adv,ext_targets,int_adv,int_targets

    def update(self,state,rollout):
        if not bool(state.pending_update):
            raise ValueError('No fresh rollout to update')
        if (int(rollout.generation)!=int(state.rollouts) or
            int(rollout.end_step)!=int(state.ppo.global_step) or
            int(rollout.policy_step)!=int(state.ppo.agent_ts.step) or
            not np.array_equal(rollout.end_rng,state.ppo.rng)):
            raise ValueError('Stale or mismatched on-policy rollout')
        if self.coefficient==0:
            # Exact old update, including optimizer/RNG; auxiliaries stay fixed.
            ppo,metrics=self.base_update(state.ppo,rollout.trajectory)
            return state.replace(ppo=ppo,rollouts=state.rollouts+1,pending_update=jnp.bool_(False)),metrics
        bonus=self.prepare_novelty(state.novelty,rollout.successor_pixels,rollout.trajectory.done)
        updated,metrics=self._update_policy(state,rollout,bonus.normalized_bonus)
        auxiliary,predictor_losses=self.novelty.finish(state.novelty,bonus,
            epochs=self.base.num_epochs,minibatches=self.base.num_minibatches)
        metrics=dict(metrics,predictor_loss=jnp.mean(predictor_losses),
                     raw_novelty_mean=jnp.mean(bonus.raw_bonus),
                     normalized_novelty_mean=jnp.mean(bonus.normalized_bonus),
                     original_score_sum=jnp.sum(rollout.trajectory.reward))
        return updated.replace(novelty=auxiliary,rollouts=state.rollouts+1,pending_update=jnp.bool_(False)),metrics

    @partial(jax.jit,static_argnums=(0,))
    def _update_policy(self,state,rollout,bonus):
        ext_adv,ext_targets,int_adv,int_targets=self.targets(state,rollout,bonus)
        data=(rollout.trajectory,rollout.intrinsic_values,ext_adv,ext_targets,int_adv,int_targets)
        def epoch(state,_):
            rng,shuffle_key=jax.random.split(state.ppo.rng)
            state=state.replace(ppo=state.ppo.replace(rng=rng))
            batches=self.base.shuffle_and_split(data,shuffle_key)
            return jax.lax.scan(self._update_minibatch,state,batches)
        state,metrics=jax.lax.scan(epoch,state,None,length=self.base.num_epochs)
        return state,jax.tree.map(jnp.mean,metrics)

    def _update_minibatch(self,state,batch):
        trajectory,old_int,ext_adv,ext_targets,int_adv,int_targets=batch
        mixed_adv=ext_adv+self.coefficient*int_adv
        def loss(parameters):
            policy_params,value_params=parameters
            _,lp,entropy,value=self.base.agent.apply(policy_params,trajectory.obs,None,trajectory.action)
            intrinsic=self.intrinsic_values(policy_params,value_params,trajectory.obs)
            ratio=jnp.exp(lp-trajectory.log_prob)
            advantage=normalize_advantages(mixed_adv,self.base.advantage_std_floor)
            clipped_ratio=jnp.clip(ratio,1-self.base.clip_eps,1+self.base.clip_eps)
            actor=-jnp.minimum(ratio*advantage,clipped_ratio*advantage).mean()-self.base.ent_coef*entropy.mean()
            def value_loss(current,old,target):
                clipped=old+jnp.clip(current-old,-self.base.clip_eps,self.base.clip_eps)
                return .5*jnp.maximum(jnp.square(current-target),jnp.square(clipped-target)).mean()
            ext_loss=self.base.vf_coef*value_loss(value,trajectory.value,ext_targets)
            int_loss=self.intrinsic_value_coefficient*value_loss(intrinsic,old_int,int_targets)
            total=actor+ext_loss+int_loss
            return total,dict(loss=total,actor_loss=actor,value_loss=ext_loss,
                intrinsic_value_loss=int_loss,entropy=entropy.mean(),
                approx_kl=((ratio-1)-(lp-trajectory.log_prob)).mean(),
                clip_fraction=(jnp.abs(ratio-1)>self.base.clip_eps).mean(),
                extrinsic_advantage_std=ext_adv.std(),intrinsic_advantage_std=int_adv.std(),
                combined_advantage_std=mixed_adv.std())
        params=(state.ppo.agent_ts.params,state.intrinsic_value.params)
        (_,metrics),grads=jax.value_and_grad(loss,has_aux=True)(params)
        clipper=optax.clip_by_global_norm(self.base.max_grad_norm)
        clipped,_=clipper.update(grads,clipper.init(grads))
        policy,head=clipped
        original_tx=state.ppo.agent_ts.tx
        updated=state.ppo.agent_ts.replace(tx=self.preclipped_tx).apply_gradients(grads=policy).replace(tx=original_tx)
        state=state.replace(ppo=state.ppo.replace(agent_ts=updated),
                            intrinsic_value=state.intrinsic_value.apply_gradients(grads=head))
        return state,dict(metrics,grad_norm=optax.global_norm(grads),clipped_grad_norm=optax.global_norm(clipped))


def save_online_checkpoint(path,state,config):
    """Full resume checkpoint; save only at a consumed-rollout boundary."""
    if bool(state.pending_update):
        raise ValueError('Cannot checkpoint an unconsumed rollout')
    payload=dict(format_version=2,method='online_pixel_novelty',config=config,
                 state=serialization.to_state_dict(state))
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_bytes(serialization.msgpack_serialize(jax.device_get(payload)));temp.replace(path)


def restore_online_checkpoint(path,template,expected_config):
    payload=serialization.msgpack_restore(Path(path).read_bytes())
    if payload.get('format_version')!=2 or payload.get('method')!='online_pixel_novelty':
        raise ValueError('Not a full online-novelty checkpoint')
    if payload['config']!=expected_config:
        raise ValueError('Checkpoint configuration mismatch')
    state=serialization.from_state_dict(template,payload['state'])
    if bool(state.pending_update):
        raise ValueError('Checkpoint contains an unconsumed rollout')
    return state


def export_policy(path,state,config):
    """Inference-only export understood by the existing evaluator, not resume."""
    payload=dict(format_version=1,purpose='inference_only',method='online_pixel_novelty',config=config,
        train_state=dict(global_step=int(state.ppo.global_step),agent_ts=dict(params=serialization.to_state_dict(state.ppo.agent_ts.params))))
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_bytes(serialization.msgpack_serialize(jax.device_get(payload)));temp.replace(path)
