"""Audit and report the fixed 20-versus-100-epoch offline cloning experiment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnose_cloning_handoff import fit_metrics
from experiment_utils import write_json
from octax.agents.ppo import Agent
from pixel_pretraining import load_human_pairs
from policy_initialization import read_policy_weights
from round_card import write_round_card

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    root = ROOT/'runs/tetris/human_bc_100epoch_seed1'
    run = root/'pretrain'
    previous = ROOT/'runs/tetris/human_bc_seed1'
    assert read(run/'run_status.json')['phase'] == 'complete'
    summary = read(run/'summary.json')
    old_summary = read(previous/'pretrain/summary.json')
    assert summary['epochs'] == 100 and summary['optimizer_updates'] == 1800 and summary['examples_seen'] == 224000
    assert summary['dataset_sha256'] == old_summary['dataset_sha256']
    assert summary['initial_parameter_sha256'] == old_summary['initial_parameter_sha256']
    audit = read(run/'matched_20_epoch_audit.json')
    assert audit['full_cloning_state_exact']
    assert audit['reference_sha256'] == hashlib.sha256((previous/'pretrain/cloning_checkpoint.msgpack').read_bytes()).hexdigest()
    for directory in [run, run/'heldout']:
        for name, digest in read(directory/'source_manifest.json').items():
            assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    weights, provenance = read_policy_weights(run/'policy_weights.msgpack', summary['policy_weights_sha256'])
    pairs = load_human_pairs(run/'dataset.npz', summary['dataset_sha256'])
    fit = fit_metrics(Agent(action_dim=5), weights, pairs.observations, pairs.actions)
    assert np.isclose(fit['cross_entropy'], summary['final_training_cross_entropy'], atol=1e-6)
    old_diag = read(previous/'handoff_diagnostic/diagnostic.json')
    old_fit = old_diag['pretrained']
    evaluations = {
        'clone_100_epochs': read(run/'heldout/comparison.json'),
        'clone_20_epochs': read(previous/'pretrain/heldout/comparison.json'),
        'scratch_ppo_64k': read(ROOT/'runs/tetris/v2_credit_horizon_seed1/heldout/comparison.json'),
        'untrained_seed1': read(ROOT/'runs/tetris/v2_replication_64k/initial_seed1.json')}
    reference = evaluations['clone_100_epochs']
    assert reference['monitoring_arrays_exact']
    for label, evaluation in evaluations.items():
        assert evaluation['seed_start'] == 50000 and evaluation['episodes_per_policy'] == 128
        for mode in ['random', 'noop']:
            if mode in evaluation:
                assert evaluation[mode]['episodes'] == reference[mode]['episodes'], (label, mode)
    scores = {label: sum(x > 0 for x in evaluation['policy']['episodes']['returns'])
        for label, evaluation in evaluations.items()}
    scores['random'] = sum(x > 0 for x in reference['random']['episodes']['returns'])
    scores['noop'] = sum(x > 0 for x in reference['noop']['episodes']['returns'])
    candidate = scores['clone_100_epochs'] > scores['random'] and scores['clone_100_epochs'] > scores['clone_20_epochs']
    conclusion = ('Longer offline fitting passes the exploratory scoring screen; seed replication and fresh on-policy RL remain necessary.' if candidate else
        'Longer offline fitting does not pass the original-score screen; useful gameplay remains unestablished.')
    analysis = dict(conclusion=conclusion, candidate_screen=candidate, scoring_games=scores,
        episodes_per_policy=128, seed_start=50000, offline_summary=summary,
        fit_20_epochs=old_fit, fit_100_epochs=fit, matched_20_epoch_audit=audit,
        initialization=provenance, source_snapshot_audits=True, original_controls_exact=True,
        diagnostic_optimization_updates=0, diagnostic_environment_decisions=0,
        ppo_decisions=0, reserved_confirmation_bank_used=False,
        limitations=['One training seed and one human trajectory prefix.',
            'All action-prediction metrics use training examples, not independent action validation.',
            'The game bank is reused exploratory evidence, not an independent confirmation.',
            'Raw human recording remains local; full offline reproduction requires separately supplied data.'])
    write_json(root/'analysis.json', analysis)
    rows = list(csv.DictReader((run/'metrics.csv').read_text().splitlines()))
    old_rows = list(csv.DictReader((previous/'pretrain/metrics.csv').read_text().splitlines()))
    for a, b in zip(rows[:20], old_rows):
        assert all(a[k] == b[k] for k in b if k != 'seconds')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    axes[0].plot([int(r['epoch']) for r in rows], [float(r['loss_during_epoch']) for r in rows], label='Loss during epoch')
    axes[0].scatter([20, 100], [old_fit['cross_entropy'], fit['cross_entropy']], label='Final training CE', color='black')
    axes[0].axhline(old_diag['empirical_observation_conditional_ce_lower_bound'], color='gray', linestyle=':', label='Empirical input ambiguity bound')
    axes[0].set(xlabel='Offline epoch', ylabel='Cross-entropy (nats)', title='Training fit; no held-out action data')
    axes[0].legend(fontsize=7)
    actions = ['Rotate', 'Left', 'Right', 'Down', 'No-op']; x = np.arange(5)
    axes[1].bar(x-.18, old_fit['per_action_recall'], .36, label='20 epochs')
    axes[1].bar(x+.18, fit['per_action_recall'], .36, label='100 epochs')
    axes[1].set(xticks=x, xticklabels=actions, ylim=(0, 1.05), ylabel='Argmax recall', title='Training-prefix actions')
    axes[1].legend(fontsize=8); fig.savefig(root/'fit_comparison.png', dpi=160); plt.close(fig)
    score_table = '\n'.join(f'| {label} | {score}/128 |' for label, score in scores.items())
    fit_table = '\n'.join(f'| {name} | {old_fit[key]:.4f} | {fit[key]:.4f} |' for name, key in
        [('Cross-entropy', 'cross_entropy'), ('Entropy (nats)', 'entropy'), ('Argmax accuracy', 'argmax_accuracy')])
    recall_table = '\n'.join(f'| {action} | {old_fit["per_action_recall"][i]:.2%} | {fit["per_action_recall"][i]:.2%} |'
        for i, action in enumerate(actions))
    next_step = ('Replicate the fixed offline setup with seed 2 before declaring a learning setup. Any PPO follow-up must be separately declared and preserve the independent weights-only handoff.' if candidate else
        'Do not automatically spend a full PPO budget on this initialization. Diagnose remaining imitation errors and coverage/observation limitations. Better in-sample action fitting alone does not solve reward discovery.')
    (root/'RESULT.md').write_text(f'''# Offline cloning duration: 20 versus 100 epochs

**{conclusion}**

The [predeclared protocol](../../../CLONING_DURATION_EXPERIMENT.md) changes only
offline duration. Both runs use seed 1, the same human prefix, CNN, four pixel
frames, action space, optimizer and shuffled minibatches. The 100-epoch run
starts fresh; its entire state at epoch 20 exactly matches the prior archived
cloning checkpoint, including parameters, Adam moments, RNG and counters.
Initial network hashes, first-20-epoch metrics and source snapshots also match
their declared checks. No PPO updates occur in this experiment.

## Original-score evaluation

| Policy/control | Scoring games |
| --- | ---: |
{score_table}

All policies use the same 128 exploratory game seeds 50000–50127 and corrected
timing. Random and no-op episode arrays match the earlier controls exactly.
The weights-only evaluator exactly reproduces all eight monitoring episodes.
No checkpoint was selected by evaluation score; these are final-epoch weights.
The candidate screen is not a significance test. The fresh confirmation bank
60000–60511 remains unused. There is only one training seed and one human prefix.

## Offline fit

| Diagnostic | 20 epochs | 100 epochs |
| --- | ---: | ---: |
{fit_table}

| True action | 20-epoch recall | 100-epoch recall |
| --- | ---: | ---: |
{recall_table}

All fit and recall values are in-sample. The prefix has 76.56% no-op labels and
only 47 rotation labels. Conflicting actions on identical inputs impose an
empirical training CE lower bound of {old_diag['empirical_observation_conditional_ce_lower_bound']:.5f}
and argmax accuracy upper bound of {old_diag['empirical_observation_argmax_accuracy_upper_bound']:.2%};
these do not predict gameplay performance or prove this CNN can reach them.
The curve shows loss averaged during each epoch; black points are the separately
measured final training losses, which are different quantities.

![Training fit and per-action recall](fit_comparison.png)

## Budget and next action

The new run uses 1,800 offline Adam updates and 224,000 presentations of the
same 2,240 demonstration decisions. It repeats the first 20 epochs for a strict
audit; it is not an 80-epoch continuation. Its recorded training time is
{summary['training_seconds']:.2f} seconds, excluding validation and gameplay
evaluation. It executes 2,241 diagnostic replay transitions for dataset checking.
These are repeated known actions, not new experience. Evaluation episode lengths
are retained separately in the monitoring/comparison files. The offline report
itself collects zero environment decisions and performs zero updates.

{next_step}

The human prefix is incomplete (`complete=false`, original score 11); it is not
held-out data or a completed expert episode. Its raw archive remains local and
excluded from GitHub. Reproducing offline training/fit requires the separately
supplied checksum-matched recording. Published weights, sources and config permit
standalone gameplay evaluation and an independent fresh PPO process.

The actual-policy animation uses preselected seed 30016, without selecting a
successful game. See [the card](result_card.html), [full evidence](analysis.json),
[20-epoch result](../human_bc_seed1/RESULT.md), and
[implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    (root/'visualizations').mkdir(exist_ok=True)
    shutil.copyfile(run/'heldout/policy_trajectory.npz', root/'visualizations/policy_trajectory.npz')
    write_round_card(root, 'Longer offline fit: 100 epochs', conclusion,
        [('100 / 20 epochs: scoring games', f'{scores["clone_100_epochs"]} / {scores["clone_20_epochs"]} out of 128'),
         ('Random scoring games', f'{scores["random"]}/128'),
         ('Training cross-entropy', f'{old_fit["cross_entropy"]:.3f} → {fit["cross_entropy"]:.3f}'),
         ('Training rotation recall', f'{old_fit["per_action_recall"][0]:.1%} → {fit["per_action_recall"][0]:.1%}'),
         ('Matched state at epoch 20', 'Exact'), ('PPO training decisions', '0')],
        [('Report', 'RESULT.md'), ('Evidence', 'analysis.json'),
         ('Protocol', '../../../CLONING_DURATION_EXPERIMENT.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    manifest = {}
    for name in ['summarize_cloning_duration.py', 'diagnose_cloning_handoff.py', 'round_card.py',
        'pixel_pretraining.py', 'policy_initialization.py', 'experiment_utils.py', 'octax/agents/ppo.py']:
        data = (ROOT/name).read_bytes(); target = root/'report_source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(root/'report_source_manifest.json', manifest)
    print(conclusion, scores, fit)


if __name__ == '__main__':
    main()
