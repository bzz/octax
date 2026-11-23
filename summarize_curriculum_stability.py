"""Read completed smaller-step PPO evidence and render its actual-policy card."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from flax import serialization
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import write_json
from policy_initialization import read_policy_weights
from round_card import write_round_card
from summarize_credit_ablation import training, compare

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check_sources(directory):
    for name, digest in read(directory/'source_manifest.json').items():
        assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest() == digest


def main():
    series = ROOT/'runs/tetris/human_bc_stability_ppo_64k'
    status = read(series/'run_status.json')
    assert status['phase'] == 'complete'
    check_sources(series)
    smoke = read(ROOT/'runs/tetris/human_bc_stability_smoke_seed1/handoff_audit.json')
    assert smoke['smoke'] and smoke['training_decisions'] == 2048 and smoke['counts']['optimizer_updates'] == 64
    results = {}
    for seed in [1, 2]:
        run = ROOT/f'runs/tetris/human_bc_stability_ppo_seed{seed}'
        parent = ROOT/f'runs/tetris/human_bc_100epoch_seed{seed}/pretrain'
        assert read(run/'run_status.json')['phase'] == 'complete'
        check_sources(run)
        audit = read(run/'handoff_audit.json')
        assert audit['initial_episode_arrays_exact'] and audit['training_decisions'] == 65536
        evaluations = read(run/'evaluation.json')
        assert evaluations[0]['episodes'] == read(parent/'pretrained_evaluation.json')['episodes']
        assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
        full = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
        assert int(full['train_state']['global_step']) == 65536
        assert int(full['train_state']['agent_ts']['step']) == audit['counts']['optimizer_updates']
        weights, provenance = read_policy_weights(run/'initialization_weights.msgpack', audit['parent_weights_sha256'])
        assert hashlib.sha256((parent/'policy_weights.msgpack').read_bytes()).hexdigest() == audit['parent_weights_sha256']
        after = full['train_state']['agent_ts']['params']['params']
        policy_after = {'params': {name: after[name] for name in ['features', 'actor']}}
        assert jax.tree.structure(weights) == jax.tree.structure(policy_after)
        delta = float(np.sqrt(sum(np.square(np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)).sum()
            for a, b in zip(jax.tree.leaves(policy_after), jax.tree.leaves(weights)))))
        train = training(run)
        assert train['steps'] == 65536 and train['rollouts'] == 32 and train['max_clipped_grad_norm'] <= .500001
        if train['positive_reward_events'] == 0:
            assert delta == 0
        rows = list(csv.DictReader((run/'metrics.csv').read_text().splitlines()))
        assert sum(int(float(r['optimizer_updates'])) for r in rows) == audit['counts']['optimizer_updates']
        kl = dict(max_final_full_batch_kl=max(float(r['final_full_batch_kl']) for r in rows),
            max_checked_minibatch_kl=max(float(r['max_checked_exact_kl']) for r in rows),
            stop_rollout_end_steps=[int(r['step']) for r in rows if float(r['kl_stop_triggered'])],
            first_reward=[{key: float(row[key]) for key in ['step', 'reward_sum', 'approx_kl',
                'final_full_batch_kl', 'optimizer_updates', 'kl_stop_triggered']} for row in rows if float(row['reward_sum'])][:1])
        current = read(run/'heldout/comparison.json')
        controls = {'pretrained': read(parent/'heldout/comparison.json'),
            'original_rate_ppo': read(ROOT/f'runs/tetris/human_bc_100epoch_ppo_seed{seed}/heldout/comparison.json'),
            'scratch_entropy_zero': read(ROOT/f'runs/tetris/v2_entropy_zero_seed{seed}/heldout/comparison.json'),
            'scratch_entropy_001': read(ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}/heldout/comparison.json'),
            'untrained': read(ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json')}
        for other in [current, *controls.values()]:
            assert other['seed_start'] == 50000 and other['episodes_per_policy'] == 128
            for mode in ['random', 'noop']:
                if mode in other:
                    assert other[mode]['episodes'] == current[mode]['episodes']
        policies = dict(stability_ppo=current['policy'], **{name: value['policy'] for name, value in controls.items()},
            random=current['random'], noop=current['noop'])
        scores = {name: sum(x > 0 for x in value['episodes']['returns']) for name, value in policies.items()}
        candidate = scores['stability_ppo'] > scores['pretrained'] and scores['stability_ppo'] > scores['random']
        results[seed] = dict(scores=scores, candidate_screen=candidate, training=train, kl=kl,
            encoder_actor_parameter_change_l2=delta, parent=provenance, handoff_audit=audit,
            checkpoint_sha256=hashlib.sha256((run/'checkpoint.msgpack').read_bytes()).hexdigest(),
            policies={name: value['summary'] for name, value in policies.items()},
            paired_comparisons={name: compare(current['policy'], policies[name])
                for name in ['pretrained', 'random', 'original_rate_ppo']})
    candidate = all(r['candidate_screen'] for r in results.values())
    conclusion = ('Both seeds pass the exploratory scoring screen above their own initializations and random; independent confirmation remains required.' if candidate else
        'Smaller-step PPO does not establish a replicated scoring gain above its starting policies and random play.')
    updates = sum(r['handoff_audit']['counts']['optimizer_updates'] for r in results.values())
    assert updates == status['total_ppo_updates']
    analysis = dict(conclusion=conclusion, candidate_screen=candidate, seeds=results,
        total_fresh_ppo_decisions=131072, total_ppo_optimizer_updates=updates,
        separate_smoke_decisions=2048, separate_smoke_optimizer_updates=64,
        offline_updates_including_reference_fits=4320, offline_example_presentations=537600,
        original_human_decisions=2240, shared_human_prefix=True, shared_game_bank=True,
        reserved_confirmation_bank_used=False, offline_fit_data_replayed_during_ppo=False,
        limitations=['Only two initialization seeds, sharing one incomplete human prefix and 128 exploratory game seeds.',
            'Bundled learning-rate and stopping change; effects cannot be attributed separately.',
            'The guard checks before an update and cannot undo overshoot; full-rollout KL is not hard bounded.',
            'Seed-2 throughput varied substantially during high host load; timings are not an isolated CPU benchmark.',
            'Paired tests are unadjusted exploratory diagnostics after multiple investigations.',
            'Raw human recording stays local; published weights suffice for PPO and gameplay reproduction.'])
    write_json(series/'analysis.json', analysis)
    labels = {'stability_ppo': 'Smaller-step PPO', 'pretrained': 'Pretrained starting policy',
        'original_rate_ppo': 'Previous PPO rate', 'scratch_entropy_zero': 'Scratch PPO, entropy 0',
        'scratch_entropy_001': 'Scratch longer-credit PPO, entropy .01', 'untrained': 'Untrained network',
        'random': 'Uniform random', 'noop': 'No-op'}
    table = '\n'.join(f'| {label} | {results[1]["scores"][name]}/128 | {results[2]["scores"][name]}/128 |'
        for name, label in labels.items())
    details = '\n'.join(f'| {seed} | {r["training"]["positive_reward_events"]} | {r["handoff_audit"]["counts"]["optimizer_updates"]} | '
        f'{r["handoff_audit"]["counts"]["kl_stop_triggered"]} | {r["kl"]["max_final_full_batch_kl"]:.5f} |'
        for seed, r in results.items())
    (series/'RESULT.md').write_text(f'''# Smaller-step on-policy PPO: two seeds at 64K

**{conclusion}**

| Policy | Training seed 1 | Training seed 2 |
| --- | ---: | ---: |
{table}

Each policy was evaluated on the same 128 exploratory game seeds. The columns
are not 256 independent games. Original returns, episode lengths, action
distributions, paired comparisons and all audit results are in
[analysis.json](analysis.json). Offline gains must not be attributed to PPO.

## Fixed change and actual work

Two independent fresh PPO processes read the original offline weights from disk
and construct their own optimizer, critic, RNG and environments. Learning rate
is .000005 (100× below the failed pilot), with categorical-KL stopping at .015.
All other training settings remain fixed, including entropy zero, pixel CNN,
original rewards and stochastic actions. No demonstration replay or auxiliary
loss is used. This bundled candidate does not isolate the two changes' effects.

| Seed | Training reward events | Accepted optimizer updates | Rollouts stopped | Maximum final full-batch KL |
| --- | ---: | ---: | ---: | ---: |
{details}

Each seed completes 32 fresh rollouts and 65,536 decisions. Actual accepted
updates total {updates:,}; gradient evaluations equal accepted updates. No
gradients or shared-optimizer changes occur after stopping within a rollout.
The stop checks the current minibatch before the next update: it cannot undo
overshoot and does not impose a hard full-rollout KL cap.

The separate seed-1 smoke used 2,048 decisions and 64 optimizer updates. It
retained encoder/actor weights and initial episode arrays exactly with zero
reward and gradients. Full pilots reproduce all eight initial monitoring
episodes, pass source/config/checkpoint/count audits, and reproduce random/no-op
controls. Seed 1's known pre-reward rollout summaries and first reward at 28,672
match the previous pilot; its first-reward KL matches the diagnostic probe.

Offline work remains separate: 4,320 updates and 537,600 example presentations,
including reference fits, from only 2,240 human decisions. Diagnostic replays,
evaluation and illustrations are not fresh training experience. Raw human
recordings remain local; published weight artifacts suffice for PPO reproduction.
Seed-2 throughput varied substantially during high host load. Raw timings remain
in the metrics, with a [runtime observation](runtime_observation.json); they are
not an isolated CPU benchmark.

## Interpretation

Training reward events are {[results[s]['training']['positive_reward_events'] for s in [1, 2]]}
across seeds, compared with [1, 1] in the previous pilot. Its final policies
scored [0, 0]/128; the current final policies score
{[results[s]['scores']['stability_ppo'] for s in [1, 2]]}/128. These comparisons
distinguish continued reward discovery from held-out scoring gains. The earlier
seed-1 first-reward diagnostic ended at exact full-batch KL 15.7814; the actual
new update ends at {results[1]['kl']['first_reward'][0]['final_full_batch_kl']:.5f}.
This validates control of that previously destructive update, not successful RL.

The exploratory screen requires both seeds to score above their own pretrained
policy and random; outcome: `{candidate}`. Small KL, retained preferences and
parameter movement do not establish useful learning. Paired test results are
unadjusted diagnostics after multiple investigations. No checkpoint was selected
by evaluation, and confirmation seeds 60000–60511 remain unused.

The card uses the preselected final seed-1/game-30016 trajectory, whether it wins
or loses. [Actual-policy card](result_card.html) ·
[Protocol](../../../CURRICULUM_STABILITY_EXPERIMENT.md) ·
[Previous pilot](../human_bc_100epoch_ppo_64k/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for seed, result in results.items():
        run = ROOT/f'runs/tetris/human_bc_stability_ppo_seed{seed}'
        rows = list(csv.DictReader((run/'metrics.csv').read_text().splitlines()))
        steps = [int(r['step']) for r in rows]
        axes[0].step([0]+steps, [0]+list(np.cumsum([float(r['reward_sum']) for r in rows])), where='post', label=f'Seed {seed}')
        axes[1].plot(steps, [float(r['final_full_batch_kl']) for r in rows], label=f'Seed {seed}')
    axes[0].set(xlabel='Fresh PPO decisions', ylabel='Cumulative original reward')
    axes[1].set(xlabel='Fresh PPO decisions', ylabel='Final full-batch exact KL')
    for ax in axes:
        ax.legend()
    fig.savefig(series/'training_diagnostics.png', dpi=160); plt.close(fig)
    (series/'visualizations').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'runs/tetris/human_bc_stability_ppo_seed1/visualizations/policy_trajectory.npz', series/'visualizations/policy_trajectory.npz')
    write_round_card(series, 'Smaller-step on-policy PPO', conclusion,
        [('Seed 1: PPO / starting policy', f'{results[1]["scores"]["stability_ppo"]} / {results[1]["scores"]["pretrained"]} scoring games'),
         ('Seed 2: PPO / starting policy', f'{results[2]["scores"]["stability_ppo"]} / {results[2]["scores"]["pretrained"]} scoring games'),
         ('Shared random', f'{results[1]["scores"]["random"]}/128 games'), ('Fresh PPO decisions', '2 × 65,536'),
         ('Accepted PPO updates', f'{updates:,}'), ('Replay during PPO', 'None')],
        [('Report', 'RESULT.md'), ('Evidence', 'analysis.json'), ('Protocol', '../../../CURRICULUM_STABILITY_EXPERIMENT.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    manifest = {}
    for name in ['summarize_curriculum_stability.py', 'summarize_credit_ablation.py', 'summarize_v2_baseline.py',
        'round_card.py', 'experiment_utils.py', 'policy_initialization.py']:
        data = (ROOT/name).read_bytes(); target = series/'report_source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(series/'report_source_manifest.json', manifest)
    print(conclusion, {seed: r['scores'] for seed, r in results.items()})


if __name__ == '__main__':
    main()
