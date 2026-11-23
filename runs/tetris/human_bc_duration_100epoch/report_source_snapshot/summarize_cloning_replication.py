"""Combine two completed offline duration experiments without pooling game seeds."""
import hashlib
import json
from pathlib import Path
import shutil

from experiment_utils import write_json
from round_card import write_round_card


def read(path):
    return json.loads(path.read_text())


def main():
    out = Path('runs/tetris/human_bc_duration_100epoch')
    runs = {seed: Path(f'runs/tetris/human_bc_100epoch_seed{seed}') for seed in [1, 2]}
    analyses = {seed: read(run/'analysis.json') for seed, run in runs.items()}
    assert read(Path('runs/tetris/human_bc_duration_replication_seed2/run_status.json'))['phase'] == 'complete'
    for seed, analysis in analyses.items():
        assert analysis['candidate_screen'] == (analysis['scoring_games']['clone_100_epochs'] > analysis['scoring_games']['random']
            and analysis['scoring_games']['clone_100_epochs'] > analysis['scoring_games']['clone_20_epochs'])
        assert analysis['matched_20_epoch_audit']['full_cloning_state_exact']
        assert analysis['original_controls_exact'] and analysis['source_snapshot_audits']
        assert analysis['ppo_decisions'] == 0 and not analysis['reserved_confirmation_bank_used']
    replicated = all(a['candidate_screen'] for a in analyses.values())
    conclusion = ('Both initializations pass the exploratory offline scoring screen; test fresh on-policy PPO next.' if replicated else
        'The exploratory offline scoring advantage does not replicate across both initializations; useful learning remains unestablished.')
    out.mkdir(parents=True, exist_ok=False)
    combined = dict(conclusion=conclusion, replicated_candidate_screen=replicated,
        seeds=analyses, game_bank='Shared exploratory seeds 50000–50127; not 256 independent game seeds.',
        dataset_shared=True, independent_human_trajectories=1, offline_updates_both_seeds=4320,
        example_presentations_both_seeds=537600, ppo_decisions=0,
        budgets_include_each_20_epoch_reference_and_each_fresh_100_epoch_fit=True,
        new_human_decisions_in_this_round=0, raw_human_archive_published=False)
    write_json(out/'analysis.json', combined)
    rows = []
    for label, key in [('20-epoch clone', 'clone_20_epochs'), ('100-epoch clone', 'clone_100_epochs'),
        ('Scratch longer-credit PPO, 64K', 'scratch_ppo_64k'), ('Random (shared)', 'random')]:
        rows.append(f'| {label} | {analyses[1]["scoring_games"][key]}/128 | {analyses[2]["scoring_games"][key]}/128 |')
    table = '\n'.join(rows)
    fit_rows = '\n'.join(f'| {seed} | {a["fit_20_epochs"]["cross_entropy"]:.4f} → {a["fit_100_epochs"]["cross_entropy"]:.4f} | '
        f'{a["fit_20_epochs"]["per_action_recall"][0]:.1%} → {a["fit_100_epochs"]["per_action_recall"][0]:.1%} |'
        for seed, a in analyses.items())
    next_action = ('A separately declared 64K original-score PPO comparison can now test whether on-policy updates add useful performance. Use independent processes and the weights-only disk boundary, compare against each exact pretrained initialization and matched scratch-PPO controls, and account for offline work separately.' if replicated else
        'Do not interpret seed-1 selection as a replicated initialization. Inspect the seed difference and action coverage before a long PPO commitment; further experiments must remain separately declared and preserve the failed replication result.')
    (out/'RESULT.md').write_text(f'''# Two-seed offline cloning duration comparison

**{conclusion}**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
{table}

Seed 2 was triggered by seed 1 passing the declared screen, so this is a
sequential exploratory investigation. Both initializations share one human
prefix and the same 128 evaluation game seeds. Do not pool the columns as 256
independent games or treat the random control as two independent baselines.
The screen is not a significance test. No unused confirmation bank was consumed.

| Training seed | Training CE, 20 → 100 epochs | Training rotation recall, 20 → 100 |
| --- | ---: | ---: |
{fit_rows}

In both seeds, the fresh 100-epoch run exactly reproduces its archived complete
20-epoch state, including optimizer/RNG/counters, before continuing. Inputs,
data order, architecture, optimizer and timing are fixed. All action-fit metrics
use training examples. Final-only weights are evaluated; there is no selection
of the best checkpoint by game score. Original-score controls and source audits
pass. Full evidence: [seed 1](../human_bc_100epoch_seed1/RESULT.md) and
[seed 2](../human_bc_100epoch_seed2/RESULT.md).

Across both seeds, the 20-epoch reference plus fresh 100-epoch run costs 4,320
offline updates and 537,600 example presentations, including seed 1's previously
completed 20-epoch run. There are only 2,240 unique human decisions from one
incomplete, score-11 prefix. Validation replays known actions; it is not new
experience. **No PPO training occurs in this comparison.** The raw archive stays
local; published weights/configurations support evaluation and fresh PPO, while
offline reproduction requires the separately supplied checksum-matched archive.

{next_action}

The card retains the preselected seed-1 policy/game-30016 illustration, score
zero, and does not choose a successful game. [Card](result_card.html) ·
[Machine-readable evidence](analysis.json) ·
[Protocol](../../../CLONING_DURATION_EXPERIMENT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    (out/'visualizations').mkdir()
    shutil.copyfile(runs[1]/'visualizations/policy_trajectory.npz', out/'visualizations/policy_trajectory.npz')
    write_round_card(out, 'Offline duration: two seeds', conclusion,
        [('Seed 1: 100 / 20 epochs', f'{analyses[1]["scoring_games"]["clone_100_epochs"]} / {analyses[1]["scoring_games"]["clone_20_epochs"]} scoring games'),
         ('Seed 2: 100 / 20 epochs', f'{analyses[2]["scoring_games"]["clone_100_epochs"]} / {analyses[2]["scoring_games"]["clone_20_epochs"]} scoring games'),
         ('Shared random / evaluation bank', '3 / 128 games'),
         ('Full states at epoch 20', 'Both exactly matched'),
         ('PPO training decisions', '0')],
        [('Report', 'RESULT.md'), ('Evidence', 'analysis.json'),
         ('Protocol', '../../../CLONING_DURATION_EXPERIMENT.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    manifest = {}
    for name in ['summarize_cloning_replication.py', 'round_card.py']:
        data = Path(name).read_bytes(); target = out/'report_source_snapshot'/name
        target.parent.mkdir(exist_ok=True); target.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(out/'report_source_manifest.json', manifest)
    print(conclusion, {seed: a['scoring_games'] for seed, a in analyses.items()})


if __name__ == '__main__':
    main()
