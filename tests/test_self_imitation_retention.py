import hashlib

import numpy as np
import pytest

from self_imitation import EpisodeReplay
from self_imitation_retention import ProtectedEpisodeReplay


def episode(replay, reward, action=1, done=True):
    replay.add_rollout(np.ones((1, 1, 1), np.float32), np.array([[action]]),
                       np.array([[reward]], np.float32), np.array([[done]]))


def same_arrays(a, b):
    for field in ['observations', 'actions', 'returns', 'priorities']:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert (a.size, a.position, a.completed_episodes, a.positive_episodes) == (
        b.size, b.position, b.completed_episodes, b.positive_episodes)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_nonpositive_flood_preserves_positive_target_and_capacity():
    replay = ProtectedEpisodeReplay((1,), 1, capacity=4)
    episode(replay, 1, action=7)
    for i in range(30):
        episode(replay, 0, action=i+10)
    assert replay.size == 4 and replay.positive_slots == 1
    np.testing.assert_array_equal(replay.actions[replay.returns > 0], [7])
    assert replay.actions.max() == 39


def test_full_positive_buffer_rejects_zero_and_accepts_new_success():
    replay = ProtectedEpisodeReplay((1,), 1, capacity=2)
    episode(replay, 1, action=1); episode(replay, 1, action=2)
    before = replay.actions.copy(); position = replay.position
    episode(replay, 0, action=3)
    np.testing.assert_array_equal(replay.actions, before)
    assert replay.position == position
    episode(replay, 2, action=4)
    assert 4 in replay.actions and replay.size == replay.positive_slots == 2


def test_matches_fifo_before_a_positive_slot_would_be_evicted():
    fifo = EpisodeReplay((1,), 1, capacity=4, seed=21)
    protected = ProtectedEpisodeReplay((1,), 1, capacity=4, seed=21)
    for reward, action in [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)]:
        for replay in [fifo, protected]:
            episode(replay, reward, action)
        same_arrays(fifo, protected)
        if fifo.ready:
            a, b = fifo.sample(8), protected.sample(8)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    episode(fifo, 0, 6); episode(protected, 0, 6)
    assert not np.any(fifo.returns > 0) and protected.positive_slots == 1


def test_explicit_fork_and_checkpoint_preserve_sampling_and_pending_episode(tmp_path):
    fifo = EpisodeReplay((1,), 1, capacity=4, seed=21)
    episode(fifo, 1); episode(fifo, 0, done=False)
    original = tmp_path/'fifo.npz'; fifo.save(original, 65536)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='retention mode'):
        ProtectedEpisodeReplay.load(original, 65536)
    fork = ProtectedEpisodeReplay.fork_fifo(original, 65536)
    same_arrays(fifo, fork)
    a, b = fifo.sample(8), fork.sample(8)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    saved = tmp_path/'protected.npz'; fork.save(saved, 65536)
    restored = ProtectedEpisodeReplay.load(saved, 65536)
    same_arrays(fork, restored)
    episode(fork, 1); episode(restored, 1)
    same_arrays(fork, restored)
    assert fork.positive_slots == restored.positive_slots == 3
    with pytest.raises(ValueError, match='retention mode'):
        ProtectedEpisodeReplay.fork_fifo(saved, 65536)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
