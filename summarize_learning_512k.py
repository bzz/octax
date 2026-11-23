"""Audit the full parent+continuation budget and compare 512K scoring results."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf

from round_card import write_round_card
from run_learning_512k import ARMS, TARGET
from self_imitation import EpisodeReplay
from summarize_credit_ablation import compare
from train_self_imitation import verify_checkpoint_pair

ROOT = Path(__file__).resolve().parent
SERIES = ROOT/'runs/tetris/learning_512k_seed1'


def load(path):
    return json.loads(path.read_text())


def read_rows(run):
    with (run/'metrics.csv').open() as stream:
        return list(csv.DictReader(stream))


def combine_training(parent, child):
    before, after = read_rows(parent), read_rows(child)
    assert int(before[-1]['step']) == 65536
    assert int(after[0]['step']) == 67584 and int(after[-1]['step']) == TARGET
    rows = before + after
    assert [int(r['step']) for r in rows] == list(range(2048, TARGET+1, 2048))
    assert all(np.isfinite(float(v)) for r in rows for v in r.values())
    p, c = load(parent/'summary.json'), load(child/'summary.json')
    assert p['step'] == 65536 and c['step'] == TARGET
    assert c['seed'] == p['seed'] == 1
    events = [dict(rollout_end_step=int(r['step']), events=int(r['positive_reward_steps']),
                   reward=float(r['reward_sum'])) for r in rows if int(r['positive_reward_steps'])]
    collect = sum(float(r['collect_seconds']) for r in rows)
    update = sum(float(r['update_seconds']) for r in rows)
    aux = sum(int(r.get('sil_updates_applied', 0)) for r in rows)
    assert aux == p.get('sil_updates_applied', 0) + c.get('sil_updates_applied', 0)
    return dict(total_steps=TARGET, added_steps=TARGET-65536, rollouts=len(rows),
        reward_events=sum(int(r['positive_reward_steps']) for r in rows),
        added_reward_events=sum(int(r['positive_reward_steps']) for r in after),
        total_reward=sum(float(r['reward_sum']) for r in rows),
        completed_episodes=sum(int(r['done_count']) for r in rows),
        event_rollouts=events, final_entropy=float(rows[-1]['entropy']),
        ppo_optimizer_updates=len(rows)*64, auxiliary_updates=aux,
        replay_processing_seconds=sum(float(r.get('sil_seconds', 0)) for r in rows),
        collection_seconds=collect, update_seconds=update,
        training_decisions_per_second=TARGET/(collect+update),
        final_action_distribution=[float(rows[-1][f'action_{i}_fraction']) for i in range(5)])


def main():
    status = load(SERIES/'run_status.json')
    assert status['phase'] == 'complete' and status['total_new_training_steps'] == 917504
    initial = load(ROOT/'runs/tetris/v2_replication_64k/initial_seed1.json')
    assert initial['initial_development_arrays_exact'] and initial['seed_start'] == 50000
    records, controls = {}, None
    for arm, parent_name, output, _, _ in ARMS:
        parent, run = ROOT/'runs/tetris'/parent_name, ROOT/'runs/tetris'/output
        assert load(run/'run_status.json')['phase'] == 'complete'
        cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
        old = OmegaConf.to_container(OmegaConf.load(parent/'config.yaml'))
        assert {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)} == {'run_dir', 'resume', 'total_timesteps'}
        assert cfg['total_timesteps'] == TARGET and cfg['seed'] == 1
        assert cfg['resume'] == str(parent.relative_to(ROOT)/'checkpoint.msgpack')
        provenance = load(run/'continuation.json')
        for name, expected in provenance['parent_sha256'].items():
            assert hashlib.sha256((parent/name).read_bytes()).hexdigest() == expected
        for name, expected in load(run/'source_manifest.json').items():
            assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest() == expected
        dev, old_dev = load(run/'evaluation.json'), load(parent/'evaluation.json')
        assert [e['step'] for e in dev] == list(range(65536, TARGET+1, 65536))
        assert dev[0]['episodes'] == old_dev[-1]['episodes']
        assert all(e['checkpoint_roundtrip_exact'] for e in dev)
        final, prior = load(run/'heldout/comparison.json'), load(parent/'heldout/comparison.json')
        assert final['step'] == TARGET and final['episodes_per_policy'] == 128 and final['seed_start'] == 50000
        if controls is None:
            controls = {mode: final[mode] for mode in ['random', 'noop']}
        for mode in controls:
            assert final[mode]['episodes'] == prior[mode]['episodes'] == controls[mode]['episodes']
        training = combine_training(parent, run)
        if arm == 'replay':
            assert verify_checkpoint_pair(run/'checkpoint.msgpack') == TARGET
            replay = EpisodeReplay.load(run/'replay.npz', expected_step=TARGET)
            assert replay.completed_episodes == training['completed_episodes']
            assert all(e['replay_roundtrip_exact'] for e in dev)
            episodes = [json.loads(line) for directory in [parent, run]
                        for line in (directory/'completed_episodes.jsonl').read_text().splitlines()]
            assert len(episodes) == replay.completed_episodes
            assert sum(e['return_sum'] > 0 for e in episodes) == replay.positive_episodes
            training['completed_scoring_episodes'] = replay.positive_episodes
        policy = final['policy']
        comparisons = {name: compare(policy, other) for name, other in
                       [('parent_64k', prior['policy']), ('untrained', initial['policy']), ('random', controls['random'])]}
        records[arm] = dict(training=training, final=policy, parent_64k=prior['policy'],
            scoring_games=sum(x > 0 for x in policy['episodes']['returns']),
            paired_comparisons=comparisons, development_evaluations=dev)
    candidates = [arm for arm, r in records.items()
                  if all(r['paired_comparisons'][name]['mean_return_difference'] > 0 for name in ['random', 'untrained'])]
    conclusion = ('At least one policy exceeds random and its untrained network on the exploratory bank; freeze it and evaluate unused seeds before claiming learning.'
                  if candidates else 'Neither 512K policy exceeds both random and its untrained network; the learning objective remains unmet.')
    analysis = dict(conclusion=conclusion, records=records, candidates_for_confirmation=candidates,
                    controls=controls, initial_policy=initial,
                    replay_vs_credit=compare(records['replay']['final'], records['credit']['final']),
                    total_new_training_steps=917504, total_steps_per_arm=TARGET,
                    limitations=['One selected training seed; continuations are not independent replications.',
                                 'Previously inspected evaluation seeds and a single shared random control.',
                                 'Replay adds optimization compute; environment decisions are matched.',
                                 'Development games are repeated monitoring data, not fresh confirmation.'])
    (SERIES/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n')
    table = '\n'.join(f"| {arm} | {r['training']['reward_events']} ({r['training']['added_reward_events']} added) | {r['scoring_games']}/128 | {r['final']['summary']['mean_return']:.5f} | {r['training']['auxiliary_updates']} |" for arm, r in records.items())
    random_count = sum(x > 0 for x in controls['random']['episodes']['returns'])
    initial_count = sum(x > 0 for x in initial['policy']['episodes']['returns'])
    text = f'''# Corrected PPO and score replay · 512K total steps

{conclusion}

## Fixed continuation protocol

Both seed-1 arms resume their full 65,536-step states to {TARGET:,} total decisions, adding 458,752 decisions each. These are continuations of the same selected training seed, not independent replications. Four-frame monochrome pixels, CNN, original score reward, keypress actions, corrected timing and longer-credit PPO settings remain fixed. Replay retains its full buffer, pending episodes and sampler RNG. Both arms receive 16,384 cumulative PPO optimizer updates; additional replay work is reported below.

## Cumulative experience and final scores

| Arm | Training reward events | Scoring games | Mean score | Extra replay updates |
| --- | ---: | ---: | ---: | ---: |
{table}

The exact untrained seed-1 network scored {initial_count}/128; the shared random control scored {random_count}/128. The corresponding 64K parents scored 1/128 for PPO and 0/128 for replay. Final evaluation uses the previously inspected seeds 50000–50127; paired comparisons and complete episode arrays are in `analysis.json`. These small counts on reused games cannot establish statistically convincing improvement. A candidate must be frozen and tested on unused seeds before a learning claim, with independent training-seed replication required for a dependable setup.

## Integrity and interpretation

The audit checks every global rollout step across parent and continuation, avoiding gaps or double counting. Reward/optimizer counters are summed across segments; replay's completed-episode counter is cumulative. Source/ROM snapshots and parent hashes, exact resumed initial evaluation, checkpoint roundtrips, finite metrics, repeated controls and the replay/model checkpoint pair all passed. Initial and periodic evaluation decisions are excluded from the training budget.

Final optimizer activity or increased survival alone does not meet the scoring objective. Replay is compared at equal environment experience, not equal compute. The illustrative card uses the preselected replay policy on gameplay seed 30016; it is not chosen for a high score.

See [protocol](../../../LEARNING_512K.md), [implementation log](../../../IMPLEMENTATION_LOG.md), and each arm's raw files and learning curves: [PPO](../v2_credit_horizon_512k_seed1/) and [replay](../v3_self_imitation_512k_seed1/).
'''
    (SERIES/'RESULT.md').write_text(text)
    visual = SERIES/'visualizations'; visual.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'runs/tetris/v3_self_imitation_512k_seed1/visualizations/policy_trajectory.npz', visual/'policy_trajectory.npz')
    write_round_card(SERIES, '512K · PPO versus score replay', conclusion,
        [('PPO / replay scoring games', f"{records['credit']['scoring_games']} / {records['replay']['scoring_games']} of 128"),
         ('Untrained / random scoring games', f'{initial_count} / {random_count} of 128'),
         ('Total decisions per arm', f'{TARGET:,}')],
        [('Full comparison', 'RESULT.md'), ('Protocol', '../../../LEARNING_512K.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    heading = '## 2026-11-23 — Matched 512K continuation completed'
    log = ROOT/'IMPLEMENTATION_LOG.md'
    if heading not in log.read_text():
        log.write_text(log.read_text()+f"\n{heading}\n\n{conclusion} Cumulative training reward events: PPO {records['credit']['training']['reward_events']}, replay {records['replay']['training']['reward_events']}. Final scoring games: PPO {records['credit']['scoring_games']}/128, replay {records['replay']['scoring_games']}/128, untrained {initial_count}/128, random {random_count}/128. All continuation/source/checkpoint/control audits passed. This is one selected seed on reused evaluation games; full evidence and the actual-gameplay card are in `runs/tetris/learning_512k_seed1/RESULT.md`.\n")
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
