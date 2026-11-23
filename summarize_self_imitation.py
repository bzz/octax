"""Audit and report the completed 64K pixel-only score-replay mechanism test."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from round_card import write_round_card
from self_imitation import EpisodeReplay
from summarize_credit_ablation import training, compare
from train_self_imitation import verify_checkpoint_pair

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/tetris/v3_self_imitation_seed1'
CONTROL = ROOT / 'runs/tetris/v2_credit_horizon_seed1'


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    assert load(RUN/'run_status.json')['phase'] == 'complete'
    cfg = OmegaConf.to_container(OmegaConf.load(RUN/'config.yaml'))
    control_cfg = OmegaConf.to_container(OmegaConf.load(CONTROL/'config.yaml'))
    differences = {k: [control_cfg.get(k), cfg.get(k)] for k in cfg.keys() | control_cfg.keys()
                   if cfg.get(k) != control_cfg.get(k)}
    assert set(differences) == {'experiment_version', 'run_dir', 'self_imitation'}
    assert cfg['seed'] == 1 and cfg['total_timesteps'] == 65536
    for name, digest in load(RUN/'source_manifest.json').items():
        assert hashlib.sha256((RUN/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    assert verify_checkpoint_pair(RUN/'checkpoint.msgpack') == 65536
    replay = EpisodeReplay.load(RUN/'replay.npz', expected_step=65536)
    evaluation = load(RUN/'heldout/comparison.json')
    control = load(CONTROL/'heldout/comparison.json')
    initial = load(ROOT/'runs/tetris/v2_replication_64k/initial_seed1.json')
    for result in [evaluation, control, initial]:
        assert result['seed_start'] == 50000 and result['episodes_per_policy'] == 128
    for mode in ['random', 'noop']:
        assert evaluation[mode]['episodes'] == control[mode]['episodes']
    dev = load(RUN/'evaluation.json')
    assert dev[0]['episodes'] == load(CONTROL/'evaluation.json')[0]['episodes']
    assert all(e['checkpoint_roundtrip_exact'] and e['replay_roundtrip_exact'] for e in dev)
    metrics, reference = rows(RUN/'metrics.csv'), rows(CONTROL/'metrics.csv')
    assert len(metrics) == len(reference) == 32
    first = next((int(r['step']) for r in metrics if int(r['sil_updates_applied']) > 0), None)
    # PPO diagnostics are computed before auxiliary updates. They must match the
    # original trainer through the first activation rollout, excluding timings.
    keys = set(reference[0]) - {'collect_seconds', 'update_seconds', 'steps_per_second'}
    matched = 0
    for a, b in zip(metrics, reference):
        if first is not None and int(a['step']) > first:
            break
        assert all(a[k] == b[k] for k in keys), (a['step'], 'pre-activation drift')
        matched += 1
    train, prior = training(RUN), training(CONTROL)
    assert sum(int(r['sil_updates_applied']) for r in metrics) == train['sil_updates_applied']
    assert first == train['first_sil_step']
    episodes = [json.loads(line) for line in (RUN/'completed_episodes.jsonl').read_text().splitlines()]
    assert len(episodes) == train['completed_episodes'] == replay.completed_episodes
    assert sum(e['return_sum'] > 0 for e in episodes) == replay.positive_episodes == train['replay_positive_episodes']
    success_files = sorted((RUN/'successful_episodes').glob('*.npz'))
    assert len(success_files) == replay.positive_episodes
    for path in success_files:
        with np.load(path, allow_pickle=False) as episode:
            rewards, targets = episode['rewards'], episode['returns']
            assert rewards.sum() > 0 and len(rewards) == len(episode['actions']) == len(episode['observations_packed'])
            expected = np.empty(len(rewards), np.float32); value = 0.
            for i in range(len(rewards)-1, -1, -1):
                value = float(rewards[i]) + cfg['gamma'] * value; expected[i] = value
            np.testing.assert_array_equal(targets, expected)
    policies = {'replay': evaluation['policy'], 'longer_credit': control['policy'],
                'untrained': initial['policy'], 'random': evaluation['random']}
    counts = {name: dict(scoring_games=sum(x > 0 for x in p['episodes']['returns']), **p['summary'])
              for name, p in policies.items()}
    comparisons = {name: compare(policies['replay'], p) for name, p in policies.items() if name != 'replay'}
    promising = all(v['mean_return_difference'] > 0 for v in comparisons.values())
    conclusion = ('Replay has a higher score in this exploratory sample; freeze this checkpoint and test unused seeds before claiming improvement.'
                  if promising else 'Replay does not beat all scoring controls in this exploratory sample; useful learning remains unestablished.')
    analysis = dict(conclusion=conclusion, candidate_for_fresh_evaluation=promising,
        config_differences=differences, training=train, control_training=prior,
        policies=counts, paired_comparisons=comparisons,
        pre_activation_matching_rollouts=matched, pre_activation_matching_keys=sorted(keys),
        completed_scoring_episodes=replay.positive_episodes, replay_checkpoint_step=65536,
        checks='Sources, initial/repeated controls, paired checkpoint hashes, replay restoration, finite metrics and completed-episode targets passed.')
    (RUN/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    table = '\n'.join(f"| {name.replace('_', ' ')} | {r['scoring_games']}/128 | {r['mean_return']:.5f} | {r['mean_length']:.1f} |" for name, r in counts.items())
    text = f'''# Pixel-only score replay · fresh seed 1 · 64K

{conclusion}

## Matched experiment

The same longer-credit CNN/PPO configuration receives completed-episode self-imitation updates after its first scoring episode completes. Four monochrome frames, keypress actions, corrected frame timing and original score reward are unchanged. No board features, demonstrations or shaping are used. Seed 1 was selected because it previously discovered feedback; this is a targeted mechanism test. Both runs use 65,536 environment decisions and 2,048 PPO optimizer updates. Replay adds {train['sil_updates_applied']} applied auxiliary updates and {train['sil_seconds']:.1f} seconds of replay processing, so compute is not matched. See [method](../../../SELF_IMITATION.md).

## Reward discovery and retention

Replay encountered {train['positive_reward_events']} positive reward events, versus {prior['positive_reward_events']} in the control. It completed {train['completed_episodes']} games, including {replay.positive_episodes} scoring games. First auxiliary update rollout: {first}. Final entropy: {train['final_entropy']:.6f}. Successful completed trajectories are retained as packed pixels/actions/rewards/returns; a reward event and a completed scoring game are different units.

All non-timing PPO diagnostics and action/reward counts matched the control for {matched} rollouts through the first activation rollout. This extends the separately verified 2,048-step complete-state equality check; diagnostic equality alone is not proof of complete-state identity at every later step. Replay/model checkpoint hashes and roundtrips passed, as did independent discounted-return reconstruction of every exported scoring episode.

## Original game scores

| Policy | Scoring games | Mean score | Mean decisions per game |
| --- | ---: | ---: | ---: |
{table}

The 128 evaluation seeds, 50000–50127, have been inspected in earlier rounds. Random is the same shared control, not a new independent sample. Paired comparisons against the original longer-credit policy, its exact untrained network and random are in `analysis.json`; they are exploratory. One selected training seed and these reused games cannot establish generalization. Seeds 60000 onward remain reserved for a frozen candidate's confirmation.

The result card uses the preselected gameplay seed 30016, not a selected scoring game. Extra optimizer activity, memorized experience and changed weights alone do not meet the goal: a saved pixel policy must demonstrate scoring improvement against random.
'''
    (RUN/'RESULT.md').write_text(text)
    write_round_card(RUN, '64K · replaying rare scoring experience', conclusion,
        [('Training reward events', train['positive_reward_events']),
         ('Replay / longer-credit scored', f"{counts['replay']['scoring_games']} / {counts['longer_credit']['scoring_games']} of 128"),
         ('Untrained / random scored', f"{counts['untrained']['scoring_games']} / {counts['random']['scoring_games']} of 128")],
        [('Full result', 'RESULT.md'), ('Method', '../../../SELF_IMITATION.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    heading = '## 2026-11-23 — 64K score-replay mechanism test completed'
    log = ROOT/'IMPLEMENTATION_LOG.md'
    if heading not in log.read_text():
        log.write_text(log.read_text()+f"\n{heading}\n\n{conclusion} Training reward events: {train['positive_reward_events']} (control {prior['positive_reward_events']}); completed scoring episodes retained: {replay.positive_episodes}; auxiliary updates applied: {train['sil_updates_applied']}. Final scored games: replay {counts['replay']['scoring_games']}/128, longer credit {counts['longer_credit']['scoring_games']}/128, untrained {counts['untrained']['scoring_games']}/128, random {counts['random']['scoring_games']}/128. Evaluation remains exploratory on reused seeds. Full checks and evidence: `runs/tetris/v3_self_imitation_seed1/RESULT.md`. The broader learning goal remains dependent on stronger scoring evidence.\n")
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
