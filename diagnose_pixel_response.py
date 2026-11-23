"""Read-only pixel-response audit on a fresh rollout from a frozen policy."""
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from evaluate_initial_policy import initial_state
from experiment_utils import restore_checkpoint
from octax.agents.ppo import PPOOctax
from octax.wrappers import OctaxGymnaxWrapper

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/tetris/v2_long_trace_512k_seed1'
CONTROL = ROOT/'runs/tetris/v2_credit_horizon_512k_seed1'
OUT = ROOT/'runs/tetris/pixel_response_diagnostic'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect_network(agent, obs):
    x = obs.transpose(0,2,3,1).astype(jnp.float32)
    result = {}
    for index, layer in enumerate(agent.features.layers):
        x = layer(x)
        if index in [1,3,5]:
            result[f'conv{(index+1)//2}'] = x
    result['encoder'] = x
    result['actor_hidden'] = agent.actor.features(x)
    result['probabilities'] = agent.actor._action_dist(x).probs
    result['values'] = agent.critic(x)
    return result


def summarize(agent, params, observations):
    infer = jax.jit(lambda x:agent.apply(params,x,method=inspect_network))
    chunks = [jax.device_get(infer(jnp.asarray(x))) for x in np.split(observations,16)]
    arrays = {name:np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    assert all(np.isfinite(a).all() for a in arrays.values())
    # Verify the instrumented path against the actual policy and critic,
    # including every action's probability on the first minibatch.
    batch = jnp.asarray(observations[:128])
    for action in range(5):
        _, lp, entropy, value = agent.apply(params,batch,None,jnp.full((128,),action))
        np.testing.assert_allclose(np.exp(lp),arrays['probabilities'][:128,action],rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(value,arrays['values'][:128],rtol=1e-5,atol=1e-7)
    probabilities = arrays.pop('probabilities')
    values = arrays.pop('values')
    layer_stats = {}
    for name,array in arrays.items():
        axes = tuple(range(array.ndim-1))
        varying = array.std(axis=0)
        layer_stats[name] = dict(shape=list(array.shape[1:]),zero_fraction=float(np.mean(array==0)),
            channels_never_positive=int(np.sum(np.max(array,axis=axes)<=0)),
            channels=int(array.shape[-1]),mean_observation_std=float(varying.mean()),
            max_abs=float(np.abs(array).max()))
    mean_probability = probabilities.mean(axis=0)
    entropy = -(probabilities*np.log(np.maximum(probabilities,1e-30))).sum(axis=1)
    conditional_kl = (probabilities*np.log(np.maximum(probabilities,1e-30)/mean_probability)).sum(axis=1)
    return dict(layers=layer_stats,mean_action_probabilities=mean_probability.tolist(),
        action_probability_std=probabilities.std(axis=0).tolist(),
        action_probability_ranges=np.ptp(probabilities,axis=0).tolist(),
        mean_entropy=float(entropy.mean()),mean_kl_to_uniform=float(np.log(5)-entropy.mean()),
        mean_kl_to_mean_policy=float(conditional_kl.mean()),
        value_mean=float(values.mean()),value_std=float(values.std()),
        instrumented_probabilities_and_values_match_actual_network=True)


def main():
    assert json.loads((RUN/'run_status.json').read_text())['phase']=='complete'
    checkpoints = {name:(path/'checkpoint.msgpack').read_bytes() for name,path in [('long_trace',RUN),('credit_control',CONTROL)]}
    cfg = OmegaConf.to_container(OmegaConf.load(RUN/'config.yaml'))
    raw,agent,initial = initial_state(cfg)
    initial_bytes = serialization.msgpack_serialize(jax.device_get(serialization.to_state_dict(initial.agent_ts.params)))
    expected = json.loads((ROOT/'runs/tetris/v2_replication_64k/initial_seed1.json').read_text())
    assert sha(initial_bytes)==expected['initial_parameter_sha256']
    keys=['num_envs','num_steps','num_epochs','num_minibatches','learning_rate','total_timesteps',
          'eval_freq','gamma','gae_lambda','clip_eps','vf_coef','ent_coef','max_grad_norm',
          'advantage_std_floor','grad_clip_mode','zero_init_value']
    settings={key:cfg[key] for key in keys}
    settings.update(num_steps=128,num_minibatches=16)
    env=OctaxGymnaxWrapper(raw)
    algo=PPOOctax(env=env,env_params=env.default_params,agent=agent,eval_callback=None,**settings)
    state,_=restore_checkpoint(RUN/'checkpoint.msgpack',algo.init_state(jax.random.PRNGKey(cfg['seed'])))
    _,trajectory=jax.jit(algo.collect_trajectories)(state)
    observations=np.asarray(trajectory.obs).reshape(-1,4,32,64)
    assert observations.shape==(2048,4,32,64) and np.isin(observations,[0,1]).all()
    packed=np.packbits(observations.astype(np.uint8),axis=None)
    weights={'untrained':initial.agent_ts.params}
    weights.update({name:serialization.msgpack_restore(data)['train_state']['agent_ts']['params'] for name,data in checkpoints.items()})
    results={name:summarize(agent,params,observations) for name,params in weights.items()}
    for name,path in [('long_trace',RUN),('credit_control',CONTROL)]:
        assert (path/'checkpoint.msgpack').read_bytes()==checkpoints[name]
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'observations.npz',packed=packed,shape=np.array(observations.shape))
    record=dict(collection='First 128 per-environment decisions from the frozen long-trace final checkpoint, without updates.',
        executed_diagnostic_decisions=2048,
        overlap='These repeat the first 2048 decisions of the preceding 16384-decision gradient diagnostic; they are not new independent trajectories.',
        positive_rewards=int(np.sum(np.asarray(trajectory.reward)>0)),
        observation_sha256=sha(packed.tobytes()),unique_pixel_stacks=int(np.unique(observations.reshape(2048,-1),axis=0).shape[0]),
        checkpoint_sha256={name:sha(data) for name,data in checkpoints.items()},
        initial_parameter_sha256=sha(initial_bytes),results=results,
        limitations=['One rollout distribution from one final policy; not a universal test of representation quality.',
                     'Activation and input-response measurements cannot establish learned game skill or explain causality.',
                     'The same observed pixels are supplied to all three networks; controls do not collect separate trajectories.',
                     'No weights are updated, saved or used to start training; no board features or game-state registers enter inference.'])
    (OUT/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
