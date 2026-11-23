"""Fixed 64K entropy-coefficient ablation, fresh seeds 1 and 2."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from flax import serialization
import jax
from omegaconf import OmegaConf

from evaluate_initial_policy import initial_state

ROOT=Path(__file__).resolve().parent
SERIES=ROOT/'runs/tetris/v2_entropy_zero_64k'
SEEDS=(1,2)
SOURCES=['run_entropy_ablation.py','train.py','octax/agents/ppo.py','octax/env.py',
         'octax/checkpoints.py','octax/wrappers.py','experiment_utils.py',
         'evaluate_checkpoint.py','evaluate_initial_policy.py','visualize_run.py',
         'conf/config.yaml','conf/credit_horizon.yaml','conf/entropy_zero.yaml',
         'ENTROPY_ABLATION.md','requirements-macos.lock.txt','roms/Tetris [Fran Dachille, 1991].ch8']


def load(path):
    return json.loads(path.read_text())


def status(directory,phase,**extra):
    temp=directory/'run_status.tmp'
    temp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
    temp.replace(directory/'run_status.json')


def main():
    os.chdir(ROOT)
    if (SERIES/'run_status.json').exists():
        raise FileExistsError('Series exists; refusing duplicate launch.')
    for seed in SEEDS:
        if (ROOT/f'runs/tetris/v2_entropy_zero_seed{seed}').exists():
            raise FileExistsError(f'Run for seed {seed} exists; refusing overwrite.')
    SERIES.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,JAX_PLATFORMS='cpu',MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR',str(ROOT/'.matplotlib')))
    sources={name:(ROOT/name).read_bytes() for name in SOURCES}
    preflight={}
    commands={}
    for seed in SEEDS:
        relative=Path(f'runs/tetris/v2_entropy_zero_seed{seed}')
        command=[sys.executable,'-u','train.py','--config-name','entropy_zero',f'seed={seed}',
                 'num_seeds=1',f'run_dir={relative}','resume=null','encoder_checkpoint=null']
        resolved=subprocess.check_output([*command,'--cfg','job'],env=env,text=True)
        cfg=OmegaConf.to_container(OmegaConf.create(resolved));cfg.pop('num_seeds')
        control=ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}'
        old=OmegaConf.to_container(OmegaConf.load(control/'config.yaml'))
        differences={k:[old.get(k),cfg.get(k)] for k in old.keys()|cfg.keys() if old.get(k)!=cfg.get(k)}
        assert set(differences)=={'ent_coef','run_dir'}
        assert cfg['ent_coef']==0 and cfg['seed']==seed and cfg['total_timesteps']==65536
        assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None
        for name in ['train.py','octax/agents/ppo.py','octax/env.py','octax/wrappers.py']:
            assert hashlib.sha256(sources[name]).hexdigest()==load(control/'source_manifest.json')[name]
        _,_,initial=initial_state(cfg)
        data=serialization.msgpack_serialize(jax.device_get(serialization.to_state_dict(initial.agent_ts.params)))
        digest=hashlib.sha256(data).hexdigest()
        assert digest==load(ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json')['initial_parameter_sha256']
        preflight[str(seed)]=dict(config_differences=differences,initial_parameter_sha256=digest,
                                  initial_parameters_match_exactly=True,optimizer_updates=2048)
        commands[seed]=command
    (SERIES/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    completed=[]
    try:
        for seed in SEEDS:
            relative=Path(f'runs/tetris/v2_entropy_zero_seed{seed}')
            run=ROOT/relative;run.mkdir()
            manifest={}
            for name,data in sources.items():
                target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                manifest[name]=hashlib.sha256(data).hexdigest()
            (run/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            phases=[('training',commands[seed]),
                    ('matched_evaluation',[sys.executable,'-u','evaluate_checkpoint.py',str(relative/'checkpoint.msgpack'),
                                            '--episodes','128','--seed-start','50000','--out',str(relative/'heldout')]),
                    ('visualization',[sys.executable,'-u','visualize_run.py','--run',str(relative),'--seed','30016'])]
            with (run/'job.log').open('w') as log:
                for phase,command in phases:
                    assert all((ROOT/name).read_bytes()==data for name,data in sources.items()),'Frozen source changed'
                    status(SERIES,phase,run=str(relative),completed=completed)
                    status(run,phase)
                    print(seed,phase,flush=True)
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    if phase=='training':
                        evaluations=load(run/'evaluation.json')
                        assert evaluations[0]['episodes']==load(ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}/evaluation.json')[0]['episodes']
                        assert [x['step'] for x in evaluations]==[0,65536]
                        assert all(x['checkpoint_roundtrip_exact'] for x in evaluations)
                        assert load(run/'summary.json')['step']==65536
            status(run,'complete',total_timesteps=65536)
            completed.append(str(relative))
        status(SERIES,'complete',completed=completed,total_new_training_steps=131072)
    except Exception as exc:
        status(SERIES,'failed',completed=completed,error=str(exc));raise


if __name__=='__main__':
    main()
