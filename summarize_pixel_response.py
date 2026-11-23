"""Report frozen-policy pixel response and deterministic inference checks."""
import hashlib
import json
from pathlib import Path

from flax import serialization
import numpy as np

from diagnose_pixel_response import ROOT, RUN, CONTROL, OUT
from evaluate_greedy_diagnostic import GreedyAgent
from experiment_utils import environment_from_config
from octax.agents.ppo import Agent
from round_card import write_round_card
from summarize_credit_ablation import compare
from visualize_run import capture_episode


def load(path):
    return json.loads(path.read_text())


def main():
    response=load(OUT/'analysis.json')
    greedy=load(OUT/'greedy.json')
    assert set(greedy['policies'])=={'untrained','long_trace','credit_control'}
    assert greedy['stochastic_monitoring_reproduced_exactly']
    assert greedy['argmax_matches_actual_policy_log_probabilities']
    assert response['checkpoint_sha256']==greedy['checkpoint_sha256']
    for name,path in [('long_trace',RUN),('credit_control',CONTROL)]:
        assert hashlib.sha256((path/'checkpoint.msgpack').read_bytes()).hexdigest()==response['checkpoint_sha256'][name]
    with np.load(OUT/'observations.npz') as data:
        assert hashlib.sha256(data['packed'].tobytes()).hexdigest()==response['observation_sha256']
        assert data['shape'].tolist()==[2048,4,32,64]
    stochastic=load(RUN/'analysis.json')['raw_policies']
    assert greedy['seed_start']==50000 and greedy['episodes_per_policy']==128
    counts={name:sum(x>0 for x in policy['episodes']['returns']) for name,policy in greedy['policies'].items()}
    comparisons={name:{'random':compare(policy,stochastic['random']),
                       'untrained_greedy':compare(policy,greedy['policies']['untrained']),
                       'own_stochastic':compare(policy,stochastic[name])}
                 for name,policy in greedy['policies'].items() if name!='untrained'}
    candidates=[name for name,pairs in comparisons.items() if all(pairs[k]['mean_return_difference']>0 for k in ['random','untrained_greedy'])]
    conclusion=('A greedy policy exceeds the exploratory controls; it needs frozen-policy confirmation.' if candidates else
                'Neither trained policy establishes a scoring gain with greedy inference. Pixel response remains present; a completely inactive network is not the explanation on this sample.')
    result=dict(conclusion=conclusion,greedy_scoring_games=counts,paired_comparisons=comparisons,
                candidates_for_confirmation=candidates,
                limitations=['Exploratory reused game bank, selected training seed, multiple unadjusted comparisons.',
                             'Inactive on this observation sample does not mean permanently dead; no universal representation claim.',
                             'Read-only diagnosis; no new optimizer steps, auxiliary rewards or training policy.'])
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    table=[]
    for name in ['untrained','long_trace','credit_control']:
        item=response['results'][name]
        stochastic_count=sum(x>0 for x in stochastic[name]['episodes']['returns'])
        layer=item['layers']['actor_hidden']
        table.append(f"| {name.replace('_',' ')} | {item['mean_kl_to_mean_policy']:.6f} | {layer['channels_never_positive']}/256 | {stochastic_count}/128 | {counts[name]}/128 |")
    (OUT/'RESULT.md').write_text(f'''# Pixel response and greedy inference

{conclusion}

## What was checked

The same 2,048 observed four-frame pixel stacks were supplied to the exact untrained seed-1 network and the two completed 512K PPO networks. They contain {response['unique_pixel_stacks']} unique stacks. Collection repeats the first 2,048 decisions of the preceding long-trace gradient diagnostic, with zero rewards; these are additional executed diagnostic interactions, not independent new experience. No network is trained on this data.

Instrumented action probabilities and values match the actual PPO network for all five actions on the checked minibatch. All convolutional channels activate somewhere in the sample. The long-trace actor has more hidden units that never activate on this sample, but it still changes its probabilities with the observed pixels. These measurements do not show complete representation collapse, establish useful features or identify why learning failed.

## Same-bank inference comparison

| Network | Mean KL to its mean action distribution | Actor units never active on sample | Stochastic scoring | Greedy scoring |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

KL is in nats and measures state-dependent action variation on these particular observations. It is not a game-skill metric. Greedy inference selects the largest logit (ties choose the lowest action index); stochastic inference samples the learned probabilities. The greedy adapter uses the existing environment evaluator, and its argmax matches the actual policy log probabilities. The saved stochastic monitoring evaluation reproduces exactly before changing action selection.

All final comparisons reuse seeds 50000–50127. The shared random control scores 3/128. Paired comparisons in `comparison.json` are exploratory, unadjusted tests of scoring probability. The same counts need not represent the same scoring games. Current evidence does not justify using the unused confirmation bank or claiming a policy has learned Tetris.

The animation is the **long-trace final checkpoint with greedy inference**, preselected illustrative seed 30016. It is newly captured from actual gameplay; it does not represent an aggregate evaluation score or a newly trained checkpoint.

## Reproduction

From the repository root with the saved run artifacts installed:

```sh
JAX_PLATFORMS=cpu .venv/bin/python diagnose_pixel_response.py
JAX_PLATFORMS=cpu .venv/bin/python evaluate_greedy_diagnostic.py
JAX_PLATFORMS=cpu .venv/bin/python summarize_pixel_response.py
```

Saved checkpoints remain byte-identical. Observation bytes, model hashes and the exact diagnostic sources accompany the report. [Completed long-trace trial](../v2_long_trace_512k_seed1/RESULT.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    payload=serialization.msgpack_restore((RUN/'checkpoint.msgpack').read_bytes())
    env,_=environment_from_config(payload['config'])
    agent=GreedyAgent(Agent(action_dim=env.num_actions))
    trace=capture_episode(env,agent,payload['train_state']['agent_ts']['params'],'policy',30016)
    (OUT/'visualizations').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'visualizations/policy_trajectory.npz',**trace)
    write_round_card(OUT,'Read-only diagnosis · greedy PPO',conclusion,
        [('Long trace · greedy / stochastic',f"{counts['long_trace']} / 2 of 128"),
         ('Credit control · greedy / stochastic',f"{counts['credit_control']} / 1 of 128"),
         ('Random scored','3 of 128')],
        [('Full diagnosis','RESULT.md'),('Training result','../v2_long_trace_512k_seed1/RESULT.md'),('Implementation log','../../../IMPLEMENTATION_LOG.md')])
    manifest={}
    names=set(load(RUN/'source_manifest.json'))|{'diagnose_pixel_response.py','evaluate_greedy_diagnostic.py',
           'summarize_pixel_response.py','evaluate_initial_policy.py','round_card.py',
           'summarize_credit_ablation.py','summarize_v2_baseline.py'}
    for name in sorted(names):
        data=(ROOT/name).read_bytes();manifest[name]=hashlib.sha256(data).hexdigest()
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    heading='## 2026-11-23 — Frozen-policy pixel response and greedy inference'
    log=ROOT/'IMPLEMENTATION_LOG.md'
    if heading not in log.read_text():
        with log.open('a') as stream:
            stream.write(f"\n{heading}\n\n{conclusion} On the same 2,048 pixel observations, all convolutional channels activate somewhere; the long-trace actor has 171/256 hidden units never active in this sample, compared with 28 initially, but probabilities remain pixel-dependent. These are sampled-activation measurements, not proof of permanently dead units. The diagnostic repeats the first 2,048 decisions of the prior gradient measurement; no weights are updated. Greedy scoring: long trace {counts['long_trace']}/128, prior PPO {counts['credit_control']}/128 and initial network {counts['untrained']}/128, versus random 3/128. Network-probability/critic equivalence, exact stochastic monitoring reproduction, argmax selection and unchanged checkpoint hashes passed. Actual greedy-policy animation and evidence: `runs/tetris/pixel_response_diagnostic/RESULT.md`.\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
