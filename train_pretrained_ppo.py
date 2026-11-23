"""Independent score-only PPO stage: reads pretrained weights, never a dataset.

This entry point cannot resume a pretraining checkpoint. It creates fresh PPO
state through the ordinary trainer and supplies a checksum-pinned weights file.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

from experiment_utils import write_json
from train import run

ROOT = Path(__file__).resolve().parent
SOURCES = ['train_pretrained_ppo.py', 'train.py', 'policy_initialization.py',
    'experiment_utils.py', 'evaluate_checkpoint.py', 'visualize_run.py',
    'requirements-macos.lock.txt', 'roms/Tetris [Fran Dachille, 1991].ch8']
SOURCES += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'octax').rglob('*.py'))]


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('Expected a positive finite number')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=[1, 2], required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--smoke', action='store_true', help='Separate 2048-step handoff smoke')
    parser.add_argument('--entropy-coef', type=float, choices=[0., .01], default=.01)
    parser.add_argument('--learning-rate', type=positive_float)
    parser.add_argument('--target-kl', type=positive_float)
    args = parser.parse_args()
    os.chdir(ROOT)
    weights = Path(args.weights).read_bytes()
    if hashlib.sha256(weights).hexdigest() != args.weights_sha256:
        raise ValueError('Pretrained weights checksum mismatch')
    out = Path(args.run).resolve(); relative = str(out.relative_to(ROOT))
    out.mkdir(parents=True, exist_ok=False)

    def status(phase, **extra):
        temp = out/'status.tmp'
        write_json(temp, dict(phase=phase, pid=os.getpid(),
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)); temp.replace(out/'run_status.json')

    try:
        status('fresh_ppo')
        # The copied handoff artifact contains weights and provenance only.
        local_weights = out/'initialization_weights.msgpack'; local_weights.write_bytes(weights)
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT/f'runs/tetris/v2_credit_horizon_seed{args.seed}/config.yaml'))
        cfg.update(run_dir=relative, resume=None, encoder_checkpoint=None,
            initialization='pretrained_policy_weights', initialization_weights_sha256=args.weights_sha256,
            purpose='handoff_smoke' if args.smoke else 'curriculum_ppo', ent_coef=args.entropy_coef)
        if args.learning_rate is not None:
            cfg['learning_rate'] = args.learning_rate
        if args.target_kl is not None:
            cfg['target_kl'] = args.target_kl
        if args.smoke: cfg.update(total_timesteps=2048, eval_freq=2048, eval_episodes=2)
        sources = {name: (ROOT/name).read_bytes() for name in SOURCES}
        for name, data in sources.items():
            dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
        write_json(out/'source_manifest.json', {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
        run(cfg, policy_weights_path=local_weights, policy_weights_sha256=args.weights_sha256)
        assert all((ROOT/name).read_bytes() == data for name, data in sources.items()), 'Frozen PPO source changed'
        summary = json.loads((out/'summary.json').read_text())
        assert summary['step'] == cfg['total_timesteps']
        status('complete', step=summary['step'])
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
