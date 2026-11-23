"""Read-only audit of reward-bearing targets in a saved replay buffer."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def audit(run, out=None):
    run = Path(run)
    manifest = json.loads((run/'checkpoint_pair.json').read_text())
    data = (run/'replay.npz').read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest['sha256']['replay.npz']:
        raise RuntimeError('Replay checkpoint is being written or has an invalid hash; retry the audit.')
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays['metadata']))
        targets, priorities = arrays['returns'], arrays['priorities']
        assert metadata['step'] == manifest['step'] and len(targets) == metadata['size']
        assert np.isfinite(targets).all() and np.isfinite(priorities).all()
        mass = np.maximum(priorities, metadata['epsilon']) ** metadata['alpha']
        result = dict(step=metadata['step'], buffer_size=metadata['size'],
            completed_scoring_episodes=metadata['positive_episodes'],
            positive_return_transitions=int((targets > 0).sum()),
            transitions_with_return_over_point01=int((targets > .01).sum()),
            max_return=float(targets.max()) if len(targets) else 0.,
            sampling_mass_on_positive_returns=float(mass[targets > 0].sum()/mass.sum()) if len(targets) else 0.,
            replay_sha256=digest)
    if json.loads((run/'checkpoint_pair.json').read_text()) != manifest:
        raise RuntimeError('Checkpoint changed during audit; retry.')
    if out is not None:
        out = Path(out)
        previous = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        previous = [record for record in previous if record['step'] != result['step']]
        previous.append(result); previous.sort(key=lambda record: record['step'])
        out.write_text(''.join(json.dumps(record)+'\n' for record in previous))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.out), indent=2))
