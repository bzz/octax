import json
import numpy as np
import pytest

from human_demonstrations import HumanRecorder, validate_demonstration


def test_recorded_pre_action_pixels_and_full_state_replay_exactly(tmp_path):
    recorder = HumanRecorder(tmp_path/'session', autosave_steps=8, source='automated_test')
    initial = recorder.start('tester')
    for index, action in enumerate([0, 1, 2, 3, 4]*5):
        before = recorder.obs.copy()
        recorder.step(action, index, 'tester', initial['episode_id'], .5 if index % 2 else 1.)
        np.testing.assert_array_equal(recorder.observations[index], before)
    # A complete atomic autosave remains available before manual saving.
    path = recorder.directory/(initial['episode_id']+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive['actions']) == 24
        assert json.loads(str(archive['metadata']))['end_reason'] == 'in_progress'
    recorder.finish(owner='tester')
    result = validate_demonstration(path)
    assert result['steps'] == 25 and result['source'] == 'automated_test'
    assert result['exact_replay'] and result['mid_episode_restore'] and not result['complete']
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['actions'], [0, 1, 2, 3, 4]*5)
        assert archive['observations'].shape == (26, 4, 32, 64)
    with pytest.raises(FileExistsError):
        HumanRecorder(tmp_path/'session')


def test_stale_duplicate_foreign_and_invalid_actions_never_advance_game(tmp_path):
    recorder = HumanRecorder(tmp_path, source='automated_test')
    game = recorder.start('tester')
    recorder.step(1, 0, 'tester', game['episode_id'])
    for args in [(2, 0, 'tester', game['episode_id']), (2, 1, 'other', game['episode_id']),
                 (5, 1, 'tester', game['episode_id']), (1.5, 1, 'tester', game['episode_id']),
                 (True, 1, 'tester', game['episode_id']), (2, 1, 'tester', 'stale-game')]:
        with pytest.raises(ValueError):
            recorder.step(*args)
        assert len(recorder.actions) == 1
    with pytest.raises(ValueError):
        recorder.start('tester')
    recorder.finish(owner='tester')
    new_game = recorder.start('tester')
    assert new_game['episode_id'] != game['episode_id'] and recorder.seed == 200001


def test_real_terminal_transition_is_retained_and_automatically_saved(tmp_path):
    recorder = HumanRecorder(tmp_path, source='automated_test')
    game = recorder.start('tester')
    while recorder.active:
        recorder.step(4, len(recorder.actions), 'tester', game['episode_id'])
    path = tmp_path/(game['episode_id']+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        assert metadata['complete']
        assert archive['terminated'][-1] or archive['truncated'][-1]
        assert not np.any(archive['terminated'][:-1] | archive['truncated'][:-1])
        assert len(archive['observations']) == len(archive['actions'])+1
    assert validate_demonstration(path)['exact_replay']


def test_misaligned_pixel_or_action_data_is_detected(tmp_path):
    recorder = HumanRecorder(tmp_path/'original', source='automated_test')
    game = recorder.start('tester')
    recorder.step(4, 0, 'tester', game['episode_id'])
    path = recorder.finish(owner='tester')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['observations'][0, 0, 0, 0] ^= True
    corrupt = tmp_path/'corrupt.npz'; np.savez_compressed(corrupt, **arrays)
    with pytest.raises(AssertionError):
        validate_demonstration(corrupt)
