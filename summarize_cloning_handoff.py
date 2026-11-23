"""Report a completed, separately executed cloning and PPO-handoff investigation."""
import hashlib
import json
from pathlib import Path
import shutil

from experiment_utils import write_json
from round_card import write_round_card


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path('runs/tetris/human_bc_seed1')
    pre = root/'pretrain'
    diag = read(root/'handoff_diagnostic/diagnostic.json')
    fit = read(pre/'summary.json')
    evaluation = read(pre/'heldout/comparison.json')
    smokes = [Path('runs/tetris/human_bc_handoff_smoke_seed1'),
        Path('runs/tetris/human_bc_handoff_entropy_zero_seed1')]
    for run in [pre, *smokes]:
        assert read(run/'run_status.json')['phase'] == 'complete'
        for name, digest in read(run/'source_manifest.json').items():
            assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    assert evaluation['monitoring_arrays_exact']
    assert diag['audits']['initial_evaluation_arrays_exact']
    assert diag['audits']['zero_entropy_final_evaluation_arrays_exact']
    assert diag['entropy_zero']['encoder_actor_arrays_exact']
    assert fit['full_cloning_checkpoint_roundtrip_exact'] and fit['weights_roundtrip_exact']
    totals = {mode: sum(x > 0 for x in evaluation[mode]['episodes']['returns'])
        for mode in ['policy', 'random', 'noop']}
    analysis = dict(conclusion='The independent weights-only boundary works; this cloned policy has not established useful play.',
        scores=totals, evaluation_games=128, seed_start=50000, training_seed=1,
        pretraining=fit, diagnostic=diag, full_curriculum_ppo_decisions=0,
        smoke_decisions_each=2048, smoke_decisions_total=4096,
        source_snapshot_audits=True, reserved_confirmation_bank_used=False)
    write_json(root/'analysis.json', analysis)
    (root/'RESULT.md').write_text('''# Human cloning and independent PPO handoff

**The process boundary works, but useful gameplay has not been established.**
The pretrained stochastic policy scores in 0/128 original-score games, versus
random 3/128 and no-op 0/128. These are exploratory games 50000–50127, reused
across investigations; no fresh confirmation games were consumed. This is one
training seed and one human prefix, not a multi-seed curriculum result.

## Separate processes and actual handoff

`pretrain_human_policy.py` exited after writing a weights-only file. Two fresh
`train_pretrained_ppo.py` processes independently read that file, initialized
fresh optimizer/critic/RNG/environment state, and collected current-policy
rollouts. Neither PPO process imports the demonstration loader. Offline cloning
state, observations and actions never cross the handoff. The source files and
checksummed handoff artifacts are retained with each run.

Both smokes exactly reproduced the pretrained policy's first two monitoring
games, including every saved episode statistic. The matched configuration
differs only in entropy coefficient and output directory. First-rollout action
counts, rewards, completion counts and advantage statistics match; raw rollout
arrays were not saved, so this is a summary comparison rather than a bytewise
rollout comparison. Source-snapshot and checkpoint-roundtrip audits pass.

## Offline fit is incomplete and does not imply control

The fixed human prefix contains 2,240 decisions and 11 points, with
`complete=false`. All transitions and a middle-state restore were independently
replayed exactly before training. It is not a completed game or an independent
validation set. The original recording remains untouched.

The raw human archive is retained locally and excluded from GitHub publication.
Reproducing offline training or the training-prefix diagnostic therefore requires
that separately supplied archive with the recorded SHA-256. Published weights,
configs and source snapshots support independent gameplay evaluation and PPO
handoff reproduction without that archive. Automatic approval review required
this exclusion because publication of the raw recording was not authorized.

Twenty epochs, 360 Adam updates and 44,800 example presentations reduce training
cross-entropy from 1.5820 to 0.4201, below the empirical constant-distribution
baseline 0.8452. Offline training took 96.82 seconds while other work overlapped;
this is not an isolated performance benchmark. The declared fit gate passed.

| Training-prefix diagnostic | Result |
| --- | ---: |
| No-op label share / majority-class accuracy | 76.56% |
| Cloned argmax action accuracy | 86.07% |
| Rotation recall among 47 labels | 0.00% |
| Left / right / down recall | 33.62% / 70.16% / 69.01% |
| No-op recall | 95.45% |
| Unique four-frame pixel observations | 1,323 |
| Observations with conflicting action labels | 150 |
| Empirical identical-observation CE lower bound | 0.12386 |
| Empirical identical-observation accuracy upper bound | 92.95% |

These bounds describe fitting this dataset with this observation representation,
not achievable gameplay performance. Some label ambiguity is present, but it
does not fully explain the remaining training error. Zero argmax rotation recall
does not mean the stochastic policy never rotates. In actual exploratory games,
rotation accounts for about 0.253% of its sampled actions. Mean game length is
617.45 decisions versus random 1,273.88; longer survival alone is not the objective.

## Entropy can erase an initialization before task feedback arrives

Each independent handoff smoke collects 2,048 new decisions and executes 64 PPO
optimizer updates. Both have zero original reward, zero task advantage and zero
value loss. The ordinary entropy coefficient 0.01 is therefore the source of
the actor gradient in this smoke. Its unclipped entropy term pushes the policy
toward uniform actions despite PPO clipping the reward surrogate.

| Read-only diagnostic on the same human pixels | Before PPO | After entropy .01 | After entropy 0 |
| --- | ---: | ---: | ---: |
| Action cross-entropy | 0.4201 | 1.6033 | 0.4201 |
| Policy entropy (nats) | 0.5341 | 1.6091 | 0.5341 |
| Argmax accuracy | 86.07% | 14.15% | 86.07% |
| Encoder/actor parameters unchanged | — | No | Exactly |

The entropy-.01 smoke reports mean approximate KL 0.8327 and parameter L2
change 2.9743. The entropy-zero smoke has exactly zero gradients/loss/KL and
preserves every encoder/actor parameter and monitoring episode array. This
isolates preference loss during optimization from an incorrect disk handoff.
It does not make the original cloned policy useful, nor establish that disabling
entropy generally improves exploration. The separate offline diagnostic performs
no updates and executes no environment decisions.

## Accounting, limitations and next investigation

Keep 2,240 original human decisions, 2,241 diagnostic replay transitions per
validation, 44,800 offline example presentations, 4,096 total smoke decisions,
and evaluation decisions distinct. The standalone actor evaluation additionally
repeats eight monitoring games and evaluates 128 games each for policy/random/
no-op; its illustration is a separate fixed seed. All episode lengths are saved.
No full 64K curriculum PPO run has started. Neither smoke is a full learning run.

The next useful experiment is improving offline action fit and assessing data
coverage before a long PPO run: rotation is underfit even on the demonstration.
A separately declared longer-fit experiment can test that without changing the
architecture or PPO. Keep any new offline experiment separate from these fixed
results, and evaluate it before handoff. Architecture changes, including
attention, remain allowed future experiments. Continue to account for gains
from initialization separately from gains added by fresh on-policy RL.

The displayed trajectory is the pretrained policy on preselected seed 30016,
497 decisions and score zero. It is illustrative, not aggregate evidence.
See [the compact card](result_card.html), [machine-readable evidence](analysis.json),
[offline diagnostic](handoff_diagnostic/diagnostic.json),
[recipe](../../../recipes/pixel_curriculum/README.md) and
[implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    (root/'visualizations').mkdir(exist_ok=True)
    shutil.copyfile(pre/'heldout/policy_trajectory.npz', root/'visualizations/policy_trajectory.npz')
    write_round_card(root, 'Cloning → independent PPO', analysis['conclusion'],
        [('Pretrained / random scoring games', '0 / 3 out of 128'),
         ('Offline fit cross-entropy', '1.582 → 0.420'),
         ('Entropy .01: post-smoke fit CE', '1.603'),
         ('Entropy 0: transferred weights', 'Exactly preserved'),
         ('Fresh PPO smoke decisions', '2 × 2,048'),
         ('Full 64K curriculum run', 'Not started')],
        [('Report', 'RESULT.md'), ('Evidence', 'analysis.json'),
         ('Recipe', '../../../recipes/pixel_curriculum/README.md'),
         ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    manifest = {}
    for name in ['summarize_cloning_handoff.py', 'round_card.py']:
        content = Path(name).read_bytes(); target = root/'report_source_snapshot'/name
        target.parent.mkdir(exist_ok=True); target.write_bytes(content)
        manifest[name] = hashlib.sha256(content).hexdigest()
    write_json(root/'report_source_manifest.json', manifest)
    print(analysis['conclusion'], totals)


if __name__ == '__main__':
    main()
