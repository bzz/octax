"""Evaluate the exact fresh initialization associated with an existing run."""
import argparse
import hashlib
from pathlib import Path
import json
import jax
from flax import serialization
from omegaconf import OmegaConf
from octax.agents.ppo import PPOOctax
from octax.wrappers import OctaxGymnaxWrapper
from experiment_utils import environment_from_config, make_evaluator, evaluate, write_json


def initial_state(config):
    raw, _ = environment_from_config(config)
    env = OctaxGymnaxWrapper(raw)
    agent = PPOOctax.create_agent({}, env, env.default_params)
    keys = ['num_envs', 'num_steps', 'num_epochs', 'num_minibatches', 'learning_rate',
            'total_timesteps', 'eval_freq', 'gamma', 'gae_lambda', 'clip_eps', 'vf_coef',
            'ent_coef', 'max_grad_norm', 'advantage_std_floor', 'grad_clip_mode', 'zero_init_value']
    algo = PPOOctax(env=env, env_params=env.default_params, agent=agent, eval_callback=None,
                    target_kl=config.get('target_kl'), **{key: config[key] for key in keys})
    return raw, agent, algo.init_state(jax.random.PRNGKey(config['seed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed-start', type=int, default=50000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.run/'config.yaml'))
    if config['resume'] or config['encoder_checkpoint']:
        raise ValueError('This comparator requires a fresh, non-transferred training run.')
    raw, agent, state = initial_state(config)
    recorded = json.loads((args.run/'evaluation.json').read_text())[0]
    assert recorded['step'] == 0
    dev = evaluate(make_evaluator(raw, agent, config['eval_episodes'], config['eval_seed_start']), state.agent_ts.params, 'policy')
    assert dev['episodes'] == recorded['episodes'], 'Fresh initialization does not reproduce stored initial evaluation'
    final = evaluate(make_evaluator(raw, agent, args.episodes, args.seed_start), state.agent_ts.params, 'policy')
    params = serialization.msgpack_serialize(jax.device_get(serialization.to_state_dict(state.agent_ts.params)))
    result = dict(training_seed=config['seed'], seed_start=args.seed_start,
                  episodes_per_policy=args.episodes, initial_development_arrays_exact=True,
                  initial_parameter_sha256=hashlib.sha256(params).hexdigest(), policy=final)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps({'training_seed': config['seed'], 'initial_evaluation': final['summary']}, indent=2))
