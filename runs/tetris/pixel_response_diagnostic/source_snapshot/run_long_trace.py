"""Fresh, strictly on-policy PPO credit-horizon experiment at 512K steps."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/tetris/v2_long_trace_512k_seed1'
SOURCES = ['run_long_trace.py','train.py','octax/agents/ppo.py','octax/env.py',
           'octax/checkpoints.py','octax/wrappers.py','experiment_utils.py',
           'conf/config.yaml','conf/credit_horizon.yaml','conf/long_trace.yaml',
           'evaluate_checkpoint.py','visualize_run.py','requirements-macos.lock.txt',
           'ON_POLICY_CREDIT.md','roms/Tetris [Fran Dachille, 1991].ch8']


def load(path):
    return json.loads(path.read_text())


def status(phase, **extra):
    temp = RUN/'run_status.tmp'
    temp.write_text(json.dumps(dict(phase=phase, pid=os.getpid(),
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra),indent=2)+'\n')
    temp.replace(RUN/'run_status.json')


def main():
    os.chdir(ROOT)
    assert load(ROOT/'runs/tetris/learning_512k_seed1/run_status.json')['phase'] == 'complete'
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'config.yaml').exists() or (RUN/'run_status.json').exists():
        raise FileExistsError('Long-trace trial exists; refusing duplicate launch.')
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR', str(ROOT/'.matplotlib')))
    resolved = subprocess.check_output([sys.executable, 'train.py', '--config-name', 'long_trace', '--cfg', 'job'], env=env, text=True)
    cfg = OmegaConf.to_container(OmegaConf.create(resolved))
    old = OmegaConf.to_container(OmegaConf.load(ROOT/'runs/tetris/v2_credit_horizon_seed1/config.yaml'))
    differences = {k for k in cfg.keys()|old.keys() if cfg.get(k) != old.get(k)}
    assert differences == {'num_steps','num_minibatches','gamma','gae_lambda','total_timesteps','run_dir','num_seeds'}
    assert cfg['num_envs']*cfg['num_steps']//cfg['num_minibatches'] == 128
    assert cfg['total_timesteps']//(cfg['num_envs']*cfg['num_steps'])*cfg['num_epochs']*cfg['num_minibatches'] == 16384
    assert cfg['resume'] is None and cfg['encoder_checkpoint'] is None and cfg['seed'] == 1
    assert cfg['run_dir'] == str(RUN.relative_to(ROOT)) and cfg['total_timesteps'] == 524288
    assert (cfg['num_steps'], cfg['num_minibatches'], cfg['gamma'], cfg['gae_lambda']) == (1024,128,.999,1.)
    manifest = {}
    for name in SOURCES:
        data = (ROOT/name).read_bytes(); manifest[name] = hashlib.sha256(data).hexdigest()
        target = RUN/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    (RUN/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (RUN/'resolved_preflight.yaml').write_text(resolved)
    relative = RUN.relative_to(ROOT)
    commands = [
        ('training', [sys.executable,'-u','train.py','--config-name','long_trace',
                      'seed=1','num_seeds=1','resume=null','encoder_checkpoint=null']),
        ('matched_evaluation', [sys.executable,'-u','evaluate_checkpoint.py',str(relative/'checkpoint.msgpack'),
                                '--episodes','128','--seed-start','50000','--out',str(relative/'heldout')]),
        ('visualization', [sys.executable,'-u','visualize_run.py','--run',str(relative),'--seed','30016']),
    ]
    try:
        for phase, command in commands:
            assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name,digest in manifest.items())
            status(phase, command=command); print('PHASE',phase,flush=True)
            subprocess.run(command, env=env, check=True)
            if phase == 'training':
                evaluations = load(RUN/'evaluation.json')
                assert evaluations[0]['episodes'] == load(ROOT/'runs/tetris/v2_credit_horizon_seed1/evaluation.json')[0]['episodes']
                assert all(e['checkpoint_roundtrip_exact'] for e in evaluations)
                assert load(RUN/'summary.json')['step'] == 524288
        status('complete', total_timesteps=524288)
        print('ON-POLICY LONG-TRACE TRIAL COMPLETE',flush=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
