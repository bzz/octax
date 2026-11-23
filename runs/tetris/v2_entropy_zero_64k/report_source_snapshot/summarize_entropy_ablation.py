"""Compare the fixed two-seed score-only PPO entropy ablation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

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

ROOT=Path(__file__).resolve().parent
SERIES=ROOT/'runs/tetris/v2_entropy_zero_64k'


def load(path):
    return json.loads(path.read_text())


def metrics(run):
    with (run/'metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert [int(r['step']) for r in rows]==list(range(2048,65537,2048))
    assert all(np.isfinite(float(v)) for row in rows for v in row.values())
    return rows


def training(run):
    rows=metrics(run)
    summary=load(run/'summary.json')
    return dict(reward_events=sum(int(r['positive_reward_steps']) for r in rows),
        completed_games=sum(int(r['done_count']) for r in rows),
        event_rollouts=[int(r['step']) for r in rows if int(r['positive_reward_steps'])],
        final_entropy=float(rows[-1]['entropy']),**summary)


def main():
    status=load(SERIES/'run_status.json')
    assert status['phase']=='complete' and status['total_new_training_steps']==131072
    preflight=load(SERIES/'preflight.json')
    records={}
    shared_random=None
    sns.set_theme(style='whitegrid')
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for column,seed in enumerate([1,2]):
        run=ROOT/f'runs/tetris/v2_entropy_zero_seed{seed}'
        control=ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}'
        assert load(run/'run_status.json')['phase']=='complete'
        cfg=OmegaConf.to_container(OmegaConf.load(run/'config.yaml'))
        old=OmegaConf.to_container(OmegaConf.load(control/'config.yaml'))
        differences={k:[old.get(k),cfg.get(k)] for k in cfg.keys()|old.keys() if cfg.get(k)!=old.get(k)}
        assert set(differences)=={'ent_coef','run_dir'} and cfg['ent_coef']==0
        assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None and cfg['seed']==seed
        assert preflight[str(seed)]['config_differences']==differences
        manifest=load(run/'source_manifest.json')
        for name,digest in manifest.items():
            assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
        for name in ['train.py','octax/agents/ppo.py','octax/env.py','octax/wrappers.py']:
            assert manifest[name]==load(control/'source_manifest.json')[name]
        checkpoint=serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
        assert int(checkpoint['train_state']['global_step'])==65536
        assert int(checkpoint['train_state']['agent_ts']['step'])==2048
        evaluations=load(run/'evaluation.json')
        assert [e['step'] for e in evaluations]==[0,65536]
        assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
        assert evaluations[0]['episodes']==load(control/'evaluation.json')[0]['episodes']
        initial=load(ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json')
        assert preflight[str(seed)]['initial_parameter_sha256']==initial['initial_parameter_sha256']
        final,old_final=load(run/'heldout/comparison.json'),load(control/'heldout/comparison.json')
        assert final['seed_start']==old_final['seed_start']==initial['seed_start']==50000
        assert final['episodes_per_policy']==old_final['episodes_per_policy']==initial['episodes_per_policy']==128
        for mode in ['random','noop']:
            assert final[mode]['episodes']==old_final[mode]['episodes']
        if shared_random is not None:
            assert final['random']['episodes']==shared_random['episodes']
        shared_random=final['random']
        current=metrics(run)
        first=next((i for i,r in enumerate(current) if int(r['positive_reward_steps'])),len(current))
        assert all(float(r[k])==0 for r in current[:first] for k in ['grad_norm','value_loss','advantage_std'])
        train=training(run)
        if train['reward_events']==0:
            assert train['parameter_change_l2']==0
            assert final['policy']['episodes']==initial['policy']['episodes']
        policies={'entropy_zero':final['policy'],'control':old_final['policy'],
                  'untrained':initial['policy'],'random':final['random']}
        counts={}
        for name,policy in policies.items():
            k=sum(x>0 for x in policy['episodes']['returns'])
            counts[name]=dict(scoring_games=k,scoring_rate_ci95_wilson=wilson(k,128),**policy['summary'])
        comparisons={name:compare(final['policy'],policy) for name,policy in policies.items() if name!='entropy_zero'}
        candidate=all(comparisons[name]['mean_return_difference']>0 for name in ['random','untrained'])
        random_budget=load(ROOT/f'runs/tetris/v2_baseline_seed{seed}/random_training_budget.json')
        records[str(seed)]=dict(training=train,control_training=training(control),
            random_budget=random_budget,policies=counts,raw_policies=policies,paired_comparisons=comparisons,
            candidate_for_confirmation=candidate,pre_reward_zero_gradient_check=True)
        for name,path,color in [('entropy 0',run,'#205c4f'),('entropy 0.01',control,'#8066b3')]:
            rows=metrics(path);steps=[int(r['step']) for r in rows]
            axes[0,column].step([0,*steps],[0,*np.cumsum([int(r['positive_reward_steps']) for r in rows])],where='post',label=name,color=color)
            axes[1,column].plot(steps,[float(r['entropy']) for r in rows],color=color)
        axes[0,column].set_title(f'Seed {seed} · cumulative training rewards')
        axes[1,column].set_title(f'Seed {seed} · policy entropy')
        axes[1,column].set_ylim(0,np.log(5)*1.02)
        write_round_card(run,f'PPO entropy ablation · seed {seed}',
            'Exploratory score comparison; two-seed interpretation is in the full report.',
            [('Entropy 0 / control scored',f"{counts['entropy_zero']['scoring_games']} / {counts['control']['scoring_games']} of 128"),
             ('Untrained / random scored',f"{counts['untrained']['scoring_games']} / {counts['random']['scoring_games']} of 128"),
             ('Training reward events',train['reward_events'])],
            [('Two-seed result','../v2_entropy_zero_64k/RESULT.md'),('Protocol','../../../ENTROPY_ABLATION.md'),('Implementation log','../../../IMPLEMENTATION_LOG.md')])
    for ax in axes.flat:
        ax.set_xlim(0,65536);ax.set_xlabel('Environment decisions')
    axes[0,0].legend();fig.suptitle('Strictly on-policy PPO · entropy coefficient 0 versus 0.01')
    fig.savefig(SERIES/'comparison_curves.png',dpi=150);plt.close(fig)
    passes=all(r['candidate_for_confirmation'] for r in records.values())
    conclusion=('Both seeds exceed random and their initial networks on the exploratory bank; freeze the policies before unused-seed confirmation.' if passes else
                'Removing entropy regularization does not pass the two-seed scoring screen; useful on-policy learning remains unestablished.')
    analysis=dict(conclusion=conclusion,candidate_for_confirmation=passes,records=records,
        checks='Exact initialization, source snapshots, unchanged PPO implementation, complete step/optimizer counts, zero pre-reward gradients, checkpoint roundtrips and repeated controls passed.',
        limitations=['Two training seeds; the chosen 128-game bank is reused and exploratory.',
                     'Paired scoring tests are unadjusted; repeated random/no-op controls are shared, not additional independent evidence.',
                     'Reward discovery can differ because the entropy coefficient changes policy actions; all other configuration is matched.',
                     'A reward-free seed cannot test whether entropy erases previously learned successful behavior.',
                     'Changing observed entropy can reflect different visited pixels even when parameters remain unchanged.'])
    (SERIES/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    table='\n'.join(f"| {seed} | {r['training']['reward_events']} / {r['control_training']['reward_events']} | {r['policies']['entropy_zero']['scoring_games']}/128 | {r['policies']['control']['scoring_games']}/128 | {r['policies']['untrained']['scoring_games']}/128 | 3/128 |" for seed,r in records.items())
    (SERIES/'RESULT.md').write_text(f'''# On-policy PPO entropy ablation · two seeds at 64K

{conclusion}

Fresh seeds 1 and 2 each received 65,536 decisions, 32 rollouts and 2,048 optimizer updates. Entropy coefficient 0 is compared with the saved 0.01 control. Resolved configurations differ only in that coefficient and output path. Both use the original 128-step credit setup, original game score, the same four-frame CNN and minibatch/update settings. No replay, demonstrations or novelty bonus is used.

| Seed | Training rewards: entropy 0 / control | Entropy 0 scoring | Control scoring | Untrained scoring | Shared random scoring |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

All final scoring uses the reused 128-game bank, seeds 50000–50127. Random is one shared control, not 256 independent games. Confidence intervals, paired unadjusted comparisons, survival/action statistics and seed-specific equal-budget random-rollout counts are in `analysis.json`. These comparisons are exploratory and cannot establish robust learning without fresh games and further independent training evidence.

With zero-initialized values and entropy disabled, all recorded gradients before the first reward are zero. Both seeds encountered feedback and their parameters changed: L2 changes {records['1']['training']['parameter_change_l2']:.3f} and {records['2']['training']['parameter_change_l2']:.3f} for seeds 1 and 2. Final training entropy was {records['1']['training']['final_entropy']:.3f} and {records['2']['training']['final_entropy']:.3f}, versus log(5) ≈ 1.609 for uniform choices. The resulting stronger preferences did not improve scoring. A moving entropy curve can also reflect changing visited observations even with fixed weights; entropy alone is not learning evidence.

A seed that never encounters reward provides no test of retaining successful behavior after learning from it. This short-budget ablation measures the overall effect of the coefficient under these seeds; a failed scoring screen does not by itself resolve the long-run forgetting hypothesis.

![Matched training curves](comparison_curves.png)

Fresh parameter hashes, original development evaluations, all source snapshots, 65,536-step checkpoints with 2,048 optimizer updates, repeated random/no-op arrays and zero pre-reward gradients passed validation. The aggregate card uses preselected training seed 1 and game seed 30016; each run also has its own actual-gameplay card.

[Protocol](../../../ENTROPY_ABLATION.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md).
''')
    (SERIES/'visualizations').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'runs/tetris/v2_entropy_zero_seed1/visualizations/policy_trajectory.npz',SERIES/'visualizations/policy_trajectory.npz')
    write_round_card(SERIES,'On-policy PPO · entropy ablation',conclusion,
        [('Entropy 0 scored · seeds 1 / 2',' / '.join(str(r['policies']['entropy_zero']['scoring_games']) for r in records.values())+' of 128'),
         ('Control scored · seeds 1 / 2',' / '.join(str(r['policies']['control']['scoring_games']) for r in records.values())+' of 128'),
         ('Shared random scored','3 of 128')],
        [('Full comparison','RESULT.md'),('Protocol','../../../ENTROPY_ABLATION.md'),('Implementation log','../../../IMPLEMENTATION_LOG.md')])
    heading='## 2026-11-23 — Two-seed entropy ablation completed'
    log=ROOT/'IMPLEMENTATION_LOG.md'
    if heading not in log.read_text():
        with log.open('a') as stream:
            stream.write('\n'+heading+'\n\n'+conclusion+' Entropy-zero scoring games by seed: '+str({s:r['policies']['entropy_zero']['scoring_games'] for s,r in records.items()})+'. Original score, pixels and on-policy PPO remain unchanged. Exact initialization, source/step/optimizer checks, checkpoint roundtrips and shared controls passed. Full evidence and actual-policy cards: `runs/tetris/v2_entropy_zero_64k/RESULT.md`.\n')
    print(json.dumps(analysis,indent=2),flush=True)


if __name__=='__main__':
    main()
