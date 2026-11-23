"""Report smaller-step plus KL-stop probes; no counterfactual policy is retained."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('runs/tetris/curriculum_stability_diagnostic')
    a = json.loads((root/'analysis.json').read_text())
    original = json.loads(Path('runs/tetris/curriculum_first_reward_diagnostic/analysis.json').read_text())
    assert json.loads((root/'run_status.json').read_text())['phase'] == 'complete'
    assert a['fast_prefix'] and a['fast_prefix_proof']['full_state_max_error'] == 0
    assert a['original_rewarded_update_metrics_match'] and not a['policies_exported']
    assert a['branches']['original'] == original['branches']['original']
    for name, digest in json.loads((root/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    labels = {'original': 'Original .0005, no guard',
        'learning_rate_10x_smaller_and_kl_stop': '.00005 + exact minibatch KL stop .015',
        'learning_rate_100x_smaller_and_kl_stop': '.000005 + exact minibatch KL stop .015'}
    rows = '\n'.join(f'| {labels[k]} | {r["accepted_optimizer_steps"]} | {r["exact_full_batch_kl"]:.6f} | {r["encoder_actor_change_l2"]:.6f} |'
        for k, r in a['branches'].items())
    (root/'RESULT.md').write_text(f'''# Smaller PPO steps with KL stopping

This follow-up examines the same first-reward batch after the original diagnostic
showed that one .0005 step could already overshoot before an early-stop check.
It exports no candidate policy and performs no gameplay evaluation.

| Update profile | Accepted updates | Final exact full-batch KL | Encoder/actor L2 change |
| --- | ---: | ---: | ---: |
{rows}

The .000005 profile is the next proposed **fresh-run** stability experiment:
it accepts all 64 updates with controlled divergence on this batch, while the
.00005 profile stops after one update and slightly exceeds the nominal threshold.
The guard did not activate in the .000005 probe, so that result does not establish
an additional benefit from the guard; it remains protection to test on later
batches. This choice is based on local update diagnostics, not a demonstrated
scoring gain. A full run must compare against its exact offline initialization,
the failed original-rate PPO pilot and random play. No diagnostic state is used
to initialize that run.

The current-minibatch pre-update stopping check is not a hard full-rollout
constraint and cannot undo an already accepted overshoot. The follow-up retains
this limitation explicitly. The production implementation must stop gradient
work after the guard fires; these instrumented probes still calculate rejected
proposals for inspection. Report actual accepted updates separately from the
fixed environment-decision budget.

## Faster reconstruction, validated against the ordinary updater

For the known reward-free prefix only, verify zero rewards/values, zero entropy
coefficient, finite action log probabilities and exactly zero Adam moments.
Advancing the sole optimizer clock, optimizer-step count and four epoch RNG
splits is then equivalent to 64 zero-gradient updates. The first such advance is
checked against the ordinary updater's entire resulting state with zero error.
Every reconstructed rollout still matches recorded rewards/actions/metrics;
the rewarded update and all original-branch results exactly match the earlier
ordinary reconstruction. This shortcut is diagnostic-only and is not added to
the PPO trainer.

This follow-up re-executes another 28,672 known-prefix decisions, not new
independent experience. It advances 832 virtual zero-gradient optimizer steps,
uses 64 ordinary updates for the equivalence proof, 64 rewarded oracle updates,
and 192 counterfactual gradient calculations. Count these separately from the
first diagnostic and from the unchanged online pilot. The raw human archive is
not read; the starting artifact is the saved fresh-PPO initial checkpoint.

[Evidence](analysis.json) · [Original diagnostic](../curriculum_first_reward_diagnostic/RESULT.md) ·
[Completed PPO pilot](../human_bc_100epoch_ppo_64k/RESULT.md) ·
[Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    data = Path('summarize_stability_followup.py').read_bytes()
    (root/'report_source_snapshot').mkdir(exist_ok=True)
    (root/'report_source_snapshot/summarize_stability_followup.py').write_bytes(data)
    (root/'report_source_manifest.json').write_text(json.dumps({'summarize_stability_followup.py': hashlib.sha256(data).hexdigest()}, indent=2)+'\n')
    print('Reported stability probes; proposed fresh learning rate .000005 with KL stop .015, not yet a gameplay result.')


if __name__ == '__main__':
    main()
