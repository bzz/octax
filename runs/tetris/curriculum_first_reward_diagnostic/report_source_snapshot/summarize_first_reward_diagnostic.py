"""Report an isolated first-reward update diagnostic, without gameplay claims."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import write_json


def main():
    root = Path('runs/tetris/curriculum_first_reward_diagnostic')
    analysis = json.loads((root/'analysis.json').read_text())
    assert json.loads((root/'run_status.json').read_text())['phase'] == 'complete'
    assert analysis['original_rewarded_update_metrics_match'] and analysis['all_action_probabilities_match_public_api']
    assert analysis['branches']['original']['traced_vs_original_full_state_max_error'] < 1e-5
    assert not analysis['policies_exported']
    for name, digest in json.loads((root/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    names = dict(original='Original PPO update', learning_rate_10x_smaller='Learning rate ÷10',
        fresh_adam_moments_and_clock='Reset Adam state at first reward', actor_loss_only='Actor loss only',
        exact_kl_stop_0015='Exact minibatch KL stop at .015')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for key, result in analysis['branches'].items():
        trace = list(csv.DictReader((root/f'{key}_minibatches.csv').read_text().splitlines()))
        kl = np.array([float(r['exact_categorical_kl_pre']) for r in trace])
        accepted = np.array([float(r['accepted']) for r in trace])
        assert len(trace) == 64 and int(accepted.sum()) == result['accepted_optimizer_steps']
        assert np.isfinite(kl).all()
        axes[0].plot(np.arange(1, 65), np.maximum(kl, 1e-8), label=names[key])
        axes[1].plot(np.arange(1, 65), np.cumsum(accepted), label=names[key])
        rows.append(f'| {names[key]} | {result["accepted_optimizer_steps"]} | {result["exact_full_batch_kl"]:.5f} | '
            f'{result["final_entropy"]:.5f} | {result["encoder_actor_change_l2"]:.5f} |')
    axes[0].set(yscale='log', xlabel='Minibatch proposal', ylabel='Exact categorical KL before proposal',
        title='Current minibatch; old policy → current policy')
    axes[0].axhline(.015, color='black', linestyle=':', linewidth=1)
    axes[1].set(xlabel='Minibatch proposal', ylabel='Accepted optimizer updates', title='Diagnostic stopping behavior')
    axes[0].legend(fontsize=7); fig.savefig(root/'update_comparison.png', dpi=160); plt.close(fig)
    table = '\n'.join(rows)
    state_error = analysis['branches']['original']['traced_vs_original_full_state_max_error']
    (root/'RESULT.md').write_text(f'''# First-reward PPO update: isolated counterfactuals

This diagnostic reproduces the update associated with seed 1's collapse, then
changes one update setting at a time. **It does not establish a better policy or
perform a new online learning experiment.** No counterfactual policy is exported.

## Reproduction and scope

Start from the actual saved step-zero full PPO checkpoint. Re-execute the
recorded prefix under the same configuration and sources. Thirteen reward-free
rollouts reproduce recorded rewards, terminal/action counts and all update
metrics; their parameters stay exactly unchanged. The first reward occurs in
rollout 14, ending at 28,672 decisions. The ordinary rewarded updater reproduces
the recorded metrics, including the large mean approximate KL. The instrumented
updater agrees with its full resulting state to maximum absolute floating error
{state_error:.3g}, with exact integer/boolean state equality.

The original raw trajectory batch was not archived; there is no bytewise
comparison against an original batch file. This is deterministic reconstruction
from the saved initial state, validated against the recorded per-rollout evidence.
All-action log probabilities are checked against the public policy API.

The online pilot's original first rewarded update had mean approximate KL about
9.78 and was followed by rotation-only behavior. This diagnostic's final full-
batch KL below is a different quantity: exact categorical KL over all five
actions and all 2,048 fixed observations, measured after the counterfactual
updates. The plotted KL is measured on each shuffled minibatch before its
proposal. Do not equate these three measurements.

## Same current-policy batch, isolated update choices

| Update choice | Accepted steps | Final full-batch exact KL | Final full-batch entropy | Encoder/actor L2 change |
| --- | ---: | ---: | ---: | ---: |
{table}

All branches begin with the same parameters, collected batch and shuffle RNG.
The smaller-rate branch keeps Adam state and changes its step scale from .0005
to .00005. The reset branch resets Adam moments/clock only at this diagnostic
point, retaining the model, collection state and training-step counter. The
actor-only branch sets the value-loss coefficient to zero. None changes the
main pilot. The original reward-free prefix advanced 832 optimizer steps before
task feedback; optimizer-clock effects are a hypothesis tested by the reset
comparison, not proof that standard Adam is incorrect.

The stopping branch checks exact categorical KL on the current minibatch before
accepting its proposed update. Once KL exceeds .015, it accepts no further
updates in that rollout. It does not undo an earlier overshoot or impose a hard
bound over the full batch. Instrumentation still calculates every proposal so
all traces can be inspected; accepted updates and computational work are separate.

![KL and accepted updates](update_comparison.png)

## Interpretation and accounting

Use these results to choose the next controlled stability experiment. Retaining
an initialization or reducing KL alone is not learning; a subsequent fresh run
must discover rewards and improve original-score gameplay above its own starting
policy and random. No architecture change, offline replay or auxiliary loss is
introduced here. The current two-seed PPO pilot remains unchanged.

PPO clipping is not a hard limit on policy divergence. OpenAI's implementation
description uses KL-based early stopping, and Stable Baselines3 documents an
optional target-KL limit for this reason. These sources motivate testing a guard;
they do not establish that a particular guard fixes this experiment.
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
[Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

The diagnostic re-executes 28,672 known-prefix environment decisions, which are
not independent new experience. It performs 832 prefix optimizer updates, 64
oracle updates and 320 counterfactual gradient calculations. Those branches do
not supply checkpoints for further training, and the main PPO stage never reads
their data. Runtime overlaps the fixed pilot and is not an isolated benchmark.

[Machine-readable evidence](analysis.json) ·
[Original online pilot](../human_bc_100epoch_ppo_64k/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    manifest = {}
    for name in ['summarize_first_reward_diagnostic.py', 'experiment_utils.py']:
        data = Path(name).read_bytes(); p = root/'report_source_snapshot'/name
        p.parent.mkdir(exist_ok=True); p.write_bytes(data); manifest[name] = hashlib.sha256(data).hexdigest()
    write_json(root/'report_source_manifest.json', manifest)
    print('Reported first-reward update counterfactuals; no gameplay improvement claimed.')


if __name__ == '__main__':
    main()
