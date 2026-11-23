"""Inspect recorded pre-replay reward timing; never feed it to a learner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def diagnose(recorded_run):
    recorded_run = Path(recorded_run)
    cfg = OmegaConf.to_container(OmegaConf.load(recorded_run/'config.yaml'))
    activation = json.loads((recorded_run/'summary.json').read_text())['first_sil_step']
    verification = json.loads((recorded_run/'analysis.json').read_text())
    assert verification['pre_activation_matching_rollouts'] == 16
    result = []
    for path in sorted((recorded_run/'successful_episodes').glob('*.npz')):
        with np.load(path, allow_pickle=False) as episode:
            start = int(episode['environment_steps_at_end'])//cfg['num_envs']-len(episode['rewards'])
            for index in np.flatnonzero(episode['rewards'] > 0):
                vector_step = start+int(index)+1
                if vector_step*cfg['num_envs'] > activation:
                    continue
                result.append(dict(episode=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    reward_vector_step=vector_step, global_decisions_at_reward=vector_step*cfg['num_envs'],
                    episode_transitions_through_reward=int(index)+1,
                    current_rollout_credit_transitions=min((vector_step-1)%cfg['num_steps']+1, int(index)+1),
                    hypothetical_1024_credit_transitions=min((vector_step-1)%1024+1, int(index)+1),
                    current_gae_decay=cfg['gamma']*cfg['gae_lambda']))
    return dict(recorded_run=str(recorded_run), pre_replay_rewards=result,
        limitations=['Only recorded rewards before the first auxiliary replay update are included.',
                     'Episode termination bounds credit even within a longer rollout.',
                     'Long-rollout figures are hypothetical alignment for the recorded trajectory; changing rollout length also changes subsequent RNG draws and trajectories.',
                     'This is diagnostic use of old records, not replay or demonstration data for training.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorded-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.recorded_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
