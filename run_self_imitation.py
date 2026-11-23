"""Fresh 64K score-replay mechanism test against longer-credit seed 1."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/tetris/v3_self_imitation_seed1'


def status(phase,**extra):
    temp=RUN/'run_status.tmp'
    temp.write_text(json.dumps(dict(phase=phase,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
    temp.replace(RUN/'run_status.json')


if __name__=='__main__':
    os.chdir(ROOT)
    if json.loads((ROOT/'runs/tetris/v2_replication_64k/run_status.json').read_text())['phase']!='complete':
        raise RuntimeError('Complete the fixed replication before starting the new objective.')
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'config.yaml').exists():
        raise FileExistsError('Score-replay run exists; refusing duplicate launch')
    names=['train_self_imitation.py','self_imitation.py','run_self_imitation.py',
           'octax/agents/ppo.py','octax/env.py','octax/checkpoints.py','octax/wrappers.py',
           'experiment_utils.py','conf/config.yaml','conf/credit_horizon.yaml','conf/self_imitation.yaml',
           'evaluate_checkpoint.py','visualize_run.py','requirements-macos.lock.txt',
           'tests/test_self_imitation.py','SELF_IMITATION.md','roms/Tetris [Fran Dachille, 1991].ch8']
    manifest={}
    for name in names:
        data=(ROOT/name).read_bytes();manifest[name]=hashlib.sha256(data).hexdigest()
        target=RUN/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (RUN/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,JAX_PLATFORMS='cpu',MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR',str(ROOT/'.matplotlib')))
    relative=RUN.relative_to(ROOT)
    commands=[
        ('training',[sys.executable,'-u','train_self_imitation.py','total_timesteps=65536',
                     'seed=1','num_seeds=1','resume=null','encoder_checkpoint=null']),
        ('matched_evaluation',[sys.executable,'-u','evaluate_checkpoint.py',str(relative/'checkpoint.msgpack'),
                              '--episodes','128','--seed-start','50000','--out',str(relative/'heldout')]),
        ('visualization',[sys.executable,'-u','visualize_run.py','--run',str(relative),'--seed','30016']),
    ]
    try:
        for phase,command in commands:
            status(phase,command=command);print('PHASE',phase,flush=True)
            subprocess.run(command,env=env,check=True)
        status('complete',total_timesteps=65536)
        print('SCORE REPLAY RUN COMPLETE',flush=True)
    except Exception as exc:
        status('failed',error=str(exc));raise
