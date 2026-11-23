"""Human pixel/action recordings with exact, separately named emulator snapshots."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util

from octax.environments import create_environment

ROOT = Path(__file__).resolve().parent
ACTIONS = ['Rotate', 'Left', 'Right', 'Down', 'No-op']
ENV_KWARGS = dict(timer_mode='frame', frame_skip=4, fps=60,
                  instruction_frequency=700, max_num_steps_per_episodes=4500)


def state_arrays(state):
    return {'state__'+'__'.join(key): np.asarray(value).copy()
            for key, value in traverse_util.flatten_dict(serialization.to_state_dict(jax.device_get(state))).items()}


def restore_state(archive, index, template):
    flat = {tuple(key[len('state__'):].split('__')): jnp.asarray(archive[key][index])
            for key in archive.files if key.startswith('state__')}
    return serialization.from_state_dict(template, traverse_util.unflatten_dict(flat))


class HumanRecorder:
    def __init__(self, directory, seed_start=200000, autosave_steps=64, source='human'):
        if autosave_steps < 1 or source not in ('human', 'automated_test'):
            raise ValueError('Invalid autosave cadence or recording source.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if list(self.directory.glob('episode_*.npz')):
            raise FileExistsError('Use a fresh session directory; existing demonstrations are never overwritten.')
        self.env, _ = create_environment('tetris', **ENV_KWARGS)
        self.seed_start, self.autosave_steps, self.source = seed_start, autosave_steps, source
        self.index = 0
        self.active = False
        self.owner = None
        self.saved = []
        self.episode_id = None
        self.state = None
        self.obs = None
        self.actions = []
        self.last_saved = None
        self.provenance = dict(
            format_version=1, source=source, environment='tetris', env_kwargs=ENV_KWARGS,
            observation_shape=[4, 32, 64], observation_axes='time,stack,height,width',
            action_names=ACTIONS, chip8_action_keys=[4, 5, 6, 7, None],
            rom_sha256=hashlib.sha256(Path(self.env.rom_path).read_bytes()).hexdigest(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            modern_mode=True,
            state_usage='Debugging/reproducibility only. Behavioral cloning inputs must be observations[:-1] only.')
        files = ['human_demonstrations.py', 'record_human.py', 'human_recorder.html']
        files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'octax').rglob('*.py'))]
        self.provenance['source_sha256'] = {}
        for name in files:
            if (ROOT/name).exists():
                data = (ROOT/name).read_bytes()
                self.provenance['source_sha256'][name] = hashlib.sha256(data).hexdigest()
                target = self.directory/'source_snapshot'/name
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        (self.directory/'session.json').write_text(json.dumps(self.provenance, indent=2)+'\n')

    def warmup(self):
        state, _, _ = self.env.reset(jax.random.PRNGKey(self.seed_start))
        jax.block_until_ready(self.env.step(state, jnp.int32(4)))

    def start(self, owner):
        if self.active:
            raise ValueError('A game is already recording. Save and end it before starting another.')
        self.owner = owner
        self.seed = self.seed_start+self.index
        self.episode_id = f'episode_{self.index:03d}_seed{self.seed}'
        self.index += 1
        self.state, obs, _ = self.env.reset(jax.random.PRNGKey(self.seed))
        self.obs = np.asarray(obs.transpose(0, 2, 1), dtype=np.bool_)
        self.observations = [self.obs.copy()]
        self.states = [state_arrays(self.state)]
        self.actions, self.rewards, self.terminated, self.truncated = [], [], [], []
        self.wall_times, self.playback_speeds = [], []
        self.started = time.monotonic()
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.active = True
        self.last_saved = None
        return self.status(owner)

    def step(self, action, expected_step, owner, episode_id, playback_speed=1.):
        if not self.active or owner != self.owner or episode_id != self.episode_id:
            raise ValueError('This tab does not own an active recording.')
        if expected_step != len(self.actions):
            raise ValueError('Stale or duplicate step request; recording paused to preserve alignment.')
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError('Action must be an integer from 0 to 4.')
        if playback_speed not in (.5, 1.):
            raise ValueError('Playback pace must be 0.5 or 1.')
        wall_time = time.monotonic()-self.started
        state, obs, reward, term, trunc, _ = self.env.step(self.state, jnp.int32(action))
        state, obs, reward, term, trunc = jax.device_get((state, obs, reward, term, trunc))
        self.state = state
        self.obs = np.asarray(obs.transpose(0, 2, 1), dtype=np.bool_)
        self.actions.append(action); self.rewards.append(float(reward))
        self.terminated.append(bool(term)); self.truncated.append(bool(trunc))
        self.wall_times.append(wall_time); self.playback_speeds.append(playback_speed)
        self.observations.append(self.obs.copy()); self.states.append(state_arrays(state))
        if term or trunc:
            self.finish('terminated' if term else 'time_limit', owner)
        elif len(self.actions) % self.autosave_steps == 0:
            self.save('in_progress')
        return self.status(owner)

    def save(self, reason):
        if not self.actions:
            return None
        metadata = dict(self.provenance, episode_id=self.episode_id, reset_seed=self.seed,
            started_utc=self.started_utc, steps=len(self.actions), return_sum=sum(self.rewards),
            end_reason=reason, complete=reason in ('terminated', 'time_limit'),
            simulated_seconds=len(self.actions)*4/60, wall_seconds=time.monotonic()-self.started,
            observation_alignment='observations[t] and state__[t] precede actions[t]; rewards[t] and observations[t+1] follow it.')
        arrays = dict(observations=np.stack(self.observations), actions=np.asarray(self.actions, np.int32),
            rewards=np.asarray(self.rewards, np.float32), terminated=np.asarray(self.terminated, np.bool_),
            truncated=np.asarray(self.truncated, np.bool_), decision_wall_times=np.asarray(self.wall_times),
            playback_speeds=np.asarray(self.playback_speeds, np.float32), metadata=np.array(json.dumps(metadata)))
        arrays.update({key: np.stack([state[key] for state in self.states]) for key in self.states[0]})
        path = self.directory/(self.episode_id+'.npz')
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)  # Each autosave is a complete, atomic, self-contained archive.
        summary = dict(metadata, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        temp = path.with_suffix('.json.tmp'); temp.write_text(json.dumps(summary, indent=2)+'\n')
        temp.replace(path.with_suffix('.json'))
        self.last_saved = path.name
        return path

    def finish(self, reason='manual_stop', owner=None):
        if not self.active:
            return None
        if owner is not None and owner != self.owner:
            raise ValueError('This tab does not own the recording.')
        path = self.save(reason)
        self.active = False
        if path:
            self.saved.append(dict(file=path.name, score=sum(self.rewards), steps=len(self.actions), reason=reason))
            print(f'SAVED {path} | score {sum(self.rewards):g} | {len(self.actions)} decisions | {reason}', flush=True)
        return path

    def status(self, owner=None):
        return dict(active=self.active, owned=self.owner in (None, owner), episode_id=self.episode_id,
            step=len(self.actions), score=float(self.state.current_score) if self.state is not None else 0.,
            frames=self.obs.astype(np.uint8).tolist() if self.obs is not None else None,
            saved=self.saved, last_saved=self.last_saved, directory=str(self.directory.resolve()))


def validate_demonstration(path):
    """Replay every recorded action, checking pixels, reward, done and full state."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['format_version'] != 1 or metadata['observation_shape'] != [4, 32, 64]:
            raise ValueError('Unsupported demonstration format.')
        env, _ = create_environment(metadata['environment'], **metadata['env_kwargs'])
        assert hashlib.sha256(Path(env.rom_path).read_bytes()).hexdigest() == metadata['rom_sha256']
        actions = archive['actions']; n = len(actions)
        assert archive['observations'].shape == (n+1, 4, 32, 64)
        assert np.all((actions >= 0) & (actions < 5))
        assert all(len(archive[k]) == n for k in ['rewards', 'terminated', 'truncated', 'decision_wall_times', 'playback_speeds'])
        assert not np.any(archive['terminated'][:-1] | archive['truncated'][:-1])
        state, obs, _ = env.reset(jax.random.PRNGKey(metadata['reset_seed']))
        for i in range(n+1):
            np.testing.assert_array_equal(np.asarray(obs).transpose(0, 2, 1), archive['observations'][i])
            for key, value in state_arrays(state).items():
                np.testing.assert_array_equal(value, archive[key][i], err_msg=f'{key} at transition {i}')
            if i == n:
                break
            state, obs, reward, term, trunc, _ = env.step(state, jnp.int32(actions[i]))
            np.testing.assert_equal(float(reward), archive['rewards'][i])
            assert bool(term) == bool(archive['terminated'][i]) and bool(trunc) == bool(archive['truncated'][i])
        # Verify an independently restored mid-episode state too.
        if n:
            middle = n//2
            restored = restore_state(archive, middle, env.reset(jax.random.PRNGKey(metadata['reset_seed']))[0])
            _, obs, reward, _, _, _ = env.step(restored, jnp.int32(actions[middle]))
            np.testing.assert_array_equal(np.asarray(obs).transpose(0, 2, 1), archive['observations'][middle+1])
            np.testing.assert_equal(float(reward), archive['rewards'][middle])
        assert metadata['steps'] == n and metadata['return_sum'] == float(archive['rewards'].sum())
        return dict(file=str(path), steps=n, score=metadata['return_sum'], source=metadata['source'],
                    complete=metadata['complete'], exact_replay=True, mid_episode_restore=True)


def new_session_path():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return ROOT/'runs/tetris/human_demos'/f'{stamp}_{uuid.uuid4().hex[:6]}'
