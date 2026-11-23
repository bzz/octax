"""Compare the completed long-trace trial with the on-policy 512K control."""
import csv
import hashlib
import json
from pathlib import Path

from flax import serialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import seaborn as sns

from round_card import write_round_card
from summarize_credit_ablation import compare
from summarize_v2_baseline import wilson

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/tetris/v2_long_trace_512k_seed1'
CONTROL = ROOT/'runs/tetris/v2_credit_horizon_512k_seed1'
PARENT = ROOT/'runs/tetris/v2_credit_horizon_seed1'


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    status = load(RUN/'run_status.json')
    assert status['phase'] == 'complete' and status['total_timesteps'] == 524288
    cfg = OmegaConf.to_container(OmegaConf.load(RUN/'config.yaml'))
    old = OmegaConf.to_container(OmegaConf.load(CONTROL/'config.yaml'))
    changes = {k:[old.get(k),cfg.get(k)] for k in old.keys()|cfg.keys() if old.get(k)!=cfg.get(k)}
    assert set(changes) == {'num_steps','num_minibatches','gamma','gae_lambda','run_dir','resume'}
    assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None
    assert cfg['seed'] == 1 and cfg['total_timesteps'] == 524288
    assert (cfg['num_steps'],cfg['gamma'],cfg['gae_lambda'],cfg['ent_coef']) == (1024,.999,1.,.01)
    manifest = load(RUN/'source_manifest.json')
    for name,digest in manifest.items():
        assert hashlib.sha256((RUN/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    for name in ['train.py','octax/agents/ppo.py','octax/env.py','octax/wrappers.py']:
        assert manifest[name] == load(CONTROL/'source_manifest.json')[name]
    current = rows(RUN/'metrics.csv')
    previous = rows(ROOT/'runs/tetris/learning_512k_seed1/combined_credit_metrics.csv')
    assert [int(r['step']) for r in current] == list(range(16384,524289,16384))
    assert all(np.isfinite(float(v)) for r in current for v in r.values())
    updates = len(current)*cfg['num_epochs']*cfg['num_minibatches']
    payload = serialization.msgpack_restore((RUN/'checkpoint.msgpack').read_bytes())
    assert int(payload['train_state']['global_step']) == 524288
    assert int(payload['train_state']['agent_ts']['step']) == updates == 16384
    evaluations = load(RUN/'evaluation.json')
    assert [e['step'] for e in evaluations] == list(range(0,524289,65536))
    assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
    assert evaluations[0]['episodes'] == load(PARENT/'evaluation.json')[0]['episodes']
    final, control = load(RUN/'heldout/comparison.json'), load(CONTROL/'heldout/comparison.json')
    initial = load(ROOT/'runs/tetris/v2_replication_64k/initial_seed1.json')
    preflight = load(ROOT/'runs/tetris/entropy_diagnostic/long_trace_preflight.json')
    assert preflight['initial_parameters_match_exactly']
    assert initial['initial_parameter_sha256'] == preflight['initial_parameter_sha256']
    assert initial['training_seed'] == cfg['seed'] and initial['initial_development_arrays_exact']
    for data in [final, control, initial]:
        assert data['seed_start'] == 50000 and data['episodes_per_policy'] == 128
    for mode in ['random','noop']:
        assert final[mode]['episodes'] == control[mode]['episodes']
    policies = {'long_trace':final['policy'], 'credit_control':control['policy'],
                'untrained':initial['policy'], 'random':final['random']}
    counts = {}
    for name, data in policies.items():
        n = len(data['episodes']['returns'])
        k = sum(x>0 for x in data['episodes']['returns'])
        assert n == 128
        counts[name] = dict(scoring_games=k, games=n,
                           scoring_rate_ci95_wilson=wilson(k,n), **data['summary'])
    comparisons = {k:compare(final['policy'],v) for k,v in policies.items() if k!='long_trace'}
    candidate = all(comparisons[k]['mean_return_difference']>0 for k in ['untrained','random'])
    conclusion = ('The long-trace policy exceeds random and its initialization on the exploratory bank; freeze it and test unused game seeds before claiming learning.'
                  if candidate else 'The long-trace policy does not exceed both random and its initialization; useful on-policy learning remains unestablished.')
    train = dict(steps=524288, rollouts=len(current), optimizer_updates=updates,
        positive_reward_events=sum(int(r['positive_reward_steps']) for r in current),
        reward_sum=sum(float(r['reward_sum']) for r in current),
        completed_episodes=sum(int(r['done_count']) for r in current),
        event_rollouts=[dict(rollout_end_step=int(r['step']), events=int(r['positive_reward_steps'])) for r in current if int(r['positive_reward_steps'])],
        final_entropy=float(current[-1]['entropy']), **load(RUN/'summary.json'))
    prior = load(ROOT/'runs/tetris/learning_512k_seed1/analysis.json')['records']['credit']
    random_budget = load(CONTROL/'random_training_budget.json')
    assert random_budget['total_steps'] == train['steps']
    analysis = dict(conclusion=conclusion,candidate_for_confirmation=candidate,
        config_differences=changes,training=train,control_training=prior['training'],
        policies=counts,paired_comparisons=comparisons,raw_policies=policies,
        random_training_budget=random_budget,
        checks='Source snapshots, unchanged PPO implementation, fresh initialization, complete step sequence, actual optimizer counter, checkpoint roundtrips and repeated controls passed.',
        limitations=['One selected training seed; independent training-seed replication is still needed.',
                     'Reused 128-game evaluation bank; comparisons are exploratory. Paired p-values are unadjusted scoring-probability tests, not tests of score magnitude.',
                     'Rollout length, gamma and lambda change together, with different RNG/update scheduling.',
                     'No historical replay or auxiliary objectives are used by either PPO arm.'])
    (RUN/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    sns.set_theme(style='whitegrid')
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for label,records,dev,color in [('long trace',current,evaluations,'#205c4f'),
            ('credit control',previous,prior['cumulative_development_evaluations'],'#8066b3')]:
        steps=[int(r['step']) for r in records]
        axes[0,0].step([0,*steps],[0,*np.cumsum([int(r['positive_reward_steps']) for r in records])],where='post',label=label,color=color)
        axes[0,1].plot([e['step'] for e in dev],[e['summary']['mean_return'] for e in dev],'o-',color=color)
        axes[1,0].plot(steps,[float(r['entropy']) for r in records],color=color)
        axes[1,1].plot(steps,[float(r['approx_kl']) for r in records],color=color)
    for ax,title in zip(axes.flat,['Cumulative training reward events','Development score · same 8 games','Policy entropy · 0 to log(5)','PPO approximate KL · training diagnostic']):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Environment decisions');ax.set_xlim(0,524288);ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[1,0].set_ylim(0,np.log(5)*1.02);axes[0,0].legend()
    fig.suptitle('Strictly on-policy PPO · seed 1 · matched 512K budgets')
    fig.savefig(RUN/'comparison_curves.png',dpi=150);plt.close(fig)
    table='\n'.join(f"| {k.replace('_',' ')} | {v['scoring_games']}/128 | {v['mean_return']:.5f} | {v['mean_length']:.1f} |" for k,v in counts.items())
    (RUN/'RESULT.md').write_text(f'''# Strictly on-policy PPO · long trace · 512K

{conclusion}

## Matched method

Fresh seed 1, four monochrome frames, unchanged CNN and PPO implementation, original score reward. Rollout length 1024, gamma 0.999 and lambda 1.0 are compared with the earlier on-policy control's 128, 0.995 and 0.98. Both use minibatches of 128, four epochs, learning rate 0.0005 and entropy coefficient 0.01. Both receive exactly 524,288 decisions and 16,384 optimizer updates; long trace has 32 larger rollouts, control 256 smaller ones. The control includes its 64K parent and continuation. This is a bundled credit-assignment test with changed update/RNG timing, not a single-factor causal result.

## Original-score results

| Policy | Scoring games | Mean score | Mean decisions per game |
| --- | ---: | ---: | ---: |
{table}

Long trace found {train['positive_reward_events']} training reward events in {train['completed_episodes']} completed games; control found {prior['training']['reward_events']} in {prior['training']['completed_episodes']}. The reused equal-budget random rollout found {random_budget['positive_reward_events']} in {random_budget['completed_episodes']}. Discovery is distinct from learned policy skill. Measured long-trace training throughput: {train['training_steps_per_second']:.1f} decisions/s.

![Matched learning curves](comparison_curves.png)

The eight development games are repeated monitoring data. The 128 final games, seeds 50000–50127, have also been inspected in earlier rounds. Paired comparisons in `analysis.json` are exploratory; they do not substitute for unused-seed confirmation and independent training-seed replication. Fresh initial-policy inference matched exactly and repeated random/no-op controls reproduced their episode arrays.

The actual-gameplay card uses preselected game seed 30016. No replay, demonstrations, auxiliary objectives or shaped rewards enter this trial. Longer lambda/rollout settings do not guarantee complete-episode Monte Carlo targets: episode boundaries stop credit and unfinished rollouts bootstrap. The on-policy learning objective remains actual scoring improvement.

[Protocol](../../../ON_POLICY_CREDIT.md) · [diagnostic rationale](../entropy_diagnostic/RESULT.md) · [implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    write_round_card(RUN,'On-policy PPO · long-trace result',conclusion,
        [('Long trace / control scored',f"{counts['long_trace']['scoring_games']} / {counts['credit_control']['scoring_games']} of 128"),
         ('Untrained / random scored',f"{counts['untrained']['scoring_games']} / {counts['random']['scoring_games']} of 128"),
         ('Training rewards',train['positive_reward_events'])],
        [('Full result','RESULT.md'),('Protocol','../../../ON_POLICY_CREDIT.md'),('Implementation log','../../../IMPLEMENTATION_LOG.md')])
    heading=f"## {status['updated_at'][:10]} — On-policy long-trace 512K trial completed"
    log=ROOT/'IMPLEMENTATION_LOG.md'
    if heading not in log.read_text():
        with log.open('a') as stream:
            stream.write(f"\n{heading}\n\n{conclusion} Long trace found {train['positive_reward_events']} training reward events and scored {counts['long_trace']['scoring_games']}/128, compared with control {counts['credit_control']['scoring_games']}/128, untrained {counts['untrained']['scoring_games']}/128 and random {counts['random']['scoring_games']}/128. Exact step/optimizer counts, fresh initialization, source snapshots, checkpoint restoration and controls passed. Both arms are strictly on-policy with original rewards. Full evidence and actual-gameplay card: `runs/tetris/v2_long_trace_512k_seed1/RESULT.md`.\n")
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    main()
