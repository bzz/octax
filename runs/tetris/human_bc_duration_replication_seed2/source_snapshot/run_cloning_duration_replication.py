"""Fixed seed-2 replication: independent 20/100-epoch processes, then evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiment_utils import write_json

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    out = ROOT/'runs/tetris/human_bc_duration_replication_seed2'
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: (ROOT/name).read_bytes() for name in
        ['run_cloning_duration_replication.py', 'pretrain_human_policy.py', 'evaluate_policy_weights.py',
         'CLONING_DURATION_EXPERIMENT.md']}
    for name, content in sources.items():
        p = out/'source_snapshot'/name; p.parent.mkdir(exist_ok=True); p.write_bytes(content)
    write_json(out/'source_manifest.json', {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()})
    write_json(out/'protocol.json', dict(seed=2, selected_after='seed1 passed the declared 100-epoch exploratory screen',
        epochs=[20, 100], total_offline_updates=2160, total_example_presentations=268800,
        dataset_decisions=2240, dataset_shared_with_seed1=True,
        evaluation_seed_start=50000, evaluation_games_each=128, ppo_decisions=0))
    def status(phase, **kwargs):
        write_json(out/'run_status.json', dict(phase=phase, pid=os.getpid(),
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs))
    def execute(stage, args):
        assert all((ROOT/k).read_bytes() == v for k, v in sources.items()), 'Frozen launcher source changed'
        status(stage)
        with (out/f'{stage}.log').open('w') as stream:
            subprocess.run([sys.executable, *args], check=True, stdout=stream, stderr=subprocess.STDOUT)
    try:
        short = 'runs/tetris/human_bc_seed2/pretrain'
        long = 'runs/tetris/human_bc_100epoch_seed2/pretrain'
        dataset = 'runs/tetris/human_bc_seed1/pretrain/dataset.npz'
        execute('pretrain_20', ['pretrain_human_policy.py', '--seed', '2', '--epochs', '20',
            '--dataset', dataset, '--run', short])
        execute('pretrain_100', ['pretrain_human_policy.py', '--seed', '2', '--epochs', '100',
            '--dataset', dataset, '--reference-20-epoch', short+'/cloning_checkpoint.msgpack', '--run', long])
        for label, run in [('evaluate_20', short), ('evaluate_100', long)]:
            summary = json.loads((ROOT/run/'summary.json').read_text())
            execute(label, ['evaluate_policy_weights.py', '--weights', run+'/policy_weights.msgpack',
                '--weights-sha256', summary['policy_weights_sha256'], '--config', run+'/config.yaml',
                '--verify-monitoring', run+'/pretrained_evaluation.json', '--out', run+'/heldout'])
        status('complete', offline_updates=2160, example_presentations=268800, ppo_decisions=0)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
