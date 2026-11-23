"""Evaluate a weights-only actor artifact without a PPO or pretraining state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiment_utils import environment_from_config, evaluate, make_evaluator, write_json
from octax.agents.ppo import Agent
from policy_initialization import read_policy_weights
from visualize_run import capture_episode

ROOT = Path(__file__).resolve().parent


class ActorOnly:
    def __init__(self, actions):
        self.agent = Agent(action_dim=actions)

    def apply(self, params, observations, rng):
        return self.agent.apply(params, observations, rng, method='call_actor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--verify-monitoring', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    params, provenance = read_policy_weights(args.weights, args.weights_sha256)
    if provenance['components'] != 'encoder_actor':
        raise ValueError('A complete actor/encoder initialization is required')
    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    env, _ = environment_from_config(cfg); actor = ActorOnly(env.num_actions)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = ['evaluate_policy_weights.py', 'policy_initialization.py', 'experiment_utils.py',
        'visualize_run.py', 'octax/agents/ppo.py']
    manifest = {}
    for name in sources:
        data = (ROOT/name).read_bytes(); path = args.out/'source_snapshot'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(args.out/'source_manifest.json', manifest)
    monitoring = evaluate(make_evaluator(env, actor, 8, 10000), params, 'policy')
    assert monitoring['episodes'] == json.loads(args.verify_monitoring.read_text())['episodes'], 'Actor-only inference differs'
    result = dict(checkpoint=str(args.weights.resolve().relative_to(ROOT)), step=0,
        kind='pretrained_weights', initialization=provenance, episodes_per_policy=128, seed_start=50000,
        monitoring_arrays_exact=True, additional_monitoring_decisions=sum(monitoring['episodes']['lengths']))
    evaluator = make_evaluator(env, actor, 128, 50000)
    for mode in ['random', 'noop', 'policy']:
        result[mode] = evaluate(evaluator, params, mode)
        write_json(args.out/'comparison.json', result)
        print(mode, result[mode]['summary'], flush=True)
    trace = capture_episode(env, actor, params, 'policy', 30016)
    np.savez_compressed(args.out/'policy_trajectory.npz', **trace)
    write_json(args.out/'illustration.json', {k: trace[k] for k in ['seed', 'length', 'final_score']})


if __name__ == '__main__':
    main()
