"""Experimental replacement rule; not used by the fixed FIFO 512K run."""
import json
from pathlib import Path

import numpy as np

from self_imitation import EpisodeReplay


class ProtectedEpisodeReplay(EpisodeReplay):
    """Nonpositive targets cannot evict positive-return transitions.

    Keep capacity, array indices, priorities and sampling unchanged. New positive
    targets can replace old entries when full, so memory remains bounded even if
    every slot has a positive target. This is a storage ablation, not a claim of
    better learning or an upstream bug fix.
    """
    retention_mode = 'positive_return_protected'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_slots = 0

    def _insert(self, episode, discounted):
        for (observation, action, _), target in zip(episode, discounted):
            if self.size == self.capacity and target <= 0:
                if self.positive_slots == self.capacity:
                    continue
                while self.returns[self.position] > 0:
                    self.position = (self.position+1) % self.capacity
            i = self.position
            self.positive_slots += int(target > 0) - int(self.returns[i] > 0)
            self.observations[i] = observation
            self.actions[i] = action
            self.returns[i] = target
            self.priorities[i] = max(float(target), self.epsilon)
            self.position = (i+1) % self.capacity
            self.size = min(self.size+1, self.capacity)

    def save(self, path, step):
        # Reuse the existing complete checkpoint schema, adding an explicit mode
        # marker. The trainer binds the final file hash only after this returns.
        path = Path(path)
        super().save(path, step)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(arrays['metadata']))
        metadata['retention_mode'] = self.retention_mode
        arrays['metadata'] = np.array(json.dumps(metadata))
        temp = path.with_suffix('.retention.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)

    @classmethod
    def _load_mode(cls, path, expected_step, required_mode):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
        if metadata.get('retention_mode', 'fifo') != required_mode:
            raise ValueError('Replay retention mode differs; use an explicit experiment fork.')
        replay = super().load(path, expected_step)
        replay.positive_slots = int((replay.returns[:replay.size] > 0).sum())
        return replay

    @classmethod
    def load(cls, path, expected_step=None):
        return cls._load_mode(path, expected_step, cls.retention_mode)

    @classmethod
    def fork_fifo(cls, path, expected_step=None):
        """Explicit storage-method fork with identical arrays, pending data and RNG."""
        return cls._load_mode(path, expected_step, 'fifo')
