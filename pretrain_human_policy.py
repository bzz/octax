"""Independent 20-epoch cloning stage; exports only CNN/actor weights for PPO."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from flax import serialization
import jax
import numpy as np
from omegaconf import OmegaConf

from evaluate_initial_policy import initial_state
from experiment_utils import evaluate, make_evaluator, system_info, write_json
from human_demonstrations import validate_demonstration
from octax.checkpoints import save_checkpoint, restore_checkpoint
from pixel_pretraining import PixelCloner, load_human_pairs
from policy_initialization import save_policy_weights, read_policy_weights

ROOT = Path(__file__).resolve().parent
DATA_SHA256 = '7a5a60ca8632cd751a23a2553d21762254b8aa208a94f658d835f6abc3c05cec'
EPOCHS = 20
SOURCES = ['pretrain_human_policy.py', 'pixel_pretraining.py', 'policy_initialization.py', 'human_demonstrations.py',
    'evaluate_initial_policy.py', 'experiment_utils.py', 'train.py', 'CURRICULUM.md',
    'requirements-macos.lock.txt', 'roms/Tetris [Fran Dachille, 1991].ch8']
SOURCES += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'octax').rglob('*.py'))]


def exact(a, b):
    a, b = serialization.to_state_dict(a), serialization.to_state_dict(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def run(directory, dataset, seed):
    os.chdir(ROOT)
    data_bytes = Path(dataset).read_bytes()
    assert hashlib.sha256(data_bytes).hexdigest() == DATA_SHA256, 'Pinned dataset changed'
    out = Path(directory).resolve(); relative = str(out.relative_to(ROOT))
    out.mkdir(parents=True, exist_ok=False)

    def status(phase, **extra):
        temp = out/'status.tmp'
        write_json(temp, dict(phase=phase, pid=os.getpid(),
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)); temp.replace(out/'run_status.json')

    try:
        status('validating_dataset')
        snapshot = out/'dataset.npz'; snapshot.write_bytes(data_bytes)
        validation = validate_demonstration(snapshot)
        validation['file'] = 'dataset.npz'
        validation['diagnostic_replay_transitions'] = validation['steps'] + 1
        write_json(out/'dataset_validation.json', validation)
        pairs = load_human_pairs(snapshot, DATA_SHA256)
        counts = np.bincount(pairs.actions, minlength=5)
        probs = counts/counts.sum()
        constant_ce = -float(np.sum(probs[probs > 0]*np.log(probs[probs > 0])))
        write_json(out/'dataset_manifest.json', dict(**pairs.provenance,
            action_counts=counts.tolist(), action_proportions=probs.tolist(),
            majority_action_accuracy=float(probs.max()), constant_action_cross_entropy=constant_ce,
            independent_trajectories=1, held_out_action_validation=False,
            dataset_collection_decisions=2240, diagnostic_replay_transitions=2241))
        control = ROOT/f'runs/tetris/v2_credit_horizon_seed{seed}'
        cfg = OmegaConf.to_container(OmegaConf.load(control/'config.yaml'))
        cfg.update(run_dir=relative, initialization='human_behavior_cloning',
            pretraining=dict(dataset_sha256=DATA_SHA256, epochs=EPOCHS, batch_size=128,
                             learning_rate=.0005, max_grad_norm=.5, source='human_prefix'))
        OmegaConf.save(OmegaConf.create(cfg), out/'config.yaml')
        sources = {name: (ROOT/name).read_bytes() for name in SOURCES}
        for name, data in sources.items():
            path = out/'source_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        write_json(out/'source_manifest.json', {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
        write_json(out/'system.json', system_info())
        raw, agent, fresh = initial_state(cfg)
        original_hash = hashlib.sha256(serialization.msgpack_serialize(jax.device_get(
            serialization.to_state_dict(fresh.agent_ts.params)))).hexdigest()
        assert original_hash == json.loads((ROOT/f'runs/tetris/v2_replication_64k/initial_seed{seed}.json').read_text())['initial_parameter_sha256']
        evaluator = make_evaluator(raw, agent, 8, 10000)
        initial_eval = evaluate(evaluator, fresh.agent_ts.params, 'policy')
        assert initial_eval['episodes'] == json.loads((control/'evaluation.json').read_text())[0]['episodes']
        write_json(out/'initial_evaluation.json', initial_eval)
        cloner = PixelCloner(agent)
        state = cloner.init(fresh.agent_ts.params, jax.random.fold_in(jax.random.PRNGKey(seed), 52001))
        before = float(cloner.fit_metrics(state, pairs.observations, pairs.actions)['cross_entropy'])
        total_seconds = 0.
        with (out/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, ['epoch', 'loss_during_epoch', 'entropy_during_epoch',
                'examples_seen', 'optimizer_updates', 'seconds']); writer.writeheader()
            for epoch in range(1, EPOCHS+1):
                status('pretraining', epoch=epoch-1)
                assert all((ROOT/n).read_bytes() == d for n, d in sources.items()), 'Frozen source changed'
                start = time.perf_counter()
                state, metrics = cloner.epoch(state, pairs.observations, pairs.actions)
                jax.block_until_ready((state, metrics)); seconds = time.perf_counter()-start; total_seconds += seconds
                assert all(np.isfinite(x).all() for x in jax.tree.leaves(jax.device_get(state)))
                assert int(metrics['examples']) == 2240
                row = dict(epoch=epoch, loss_during_epoch=float(metrics['loss']), entropy_during_epoch=float(metrics['entropy']),
                    examples_seen=int(state.examples_seen), optimizer_updates=int(state.policy.step), seconds=seconds)
                writer.writerow(row); stream.flush(); print('PRETRAIN', row, flush=True)
        after = float(cloner.fit_metrics(state, pairs.observations, pairs.actions)['cross_entropy'])
        save_checkpoint(out/'cloning_checkpoint.msgpack', state, cfg)
        restored, _ = restore_checkpoint(out/'cloning_checkpoint.msgpack', cloner.init(fresh.agent_ts.params, jax.random.PRNGKey(999)))
        exact(state, restored)
        weights_sha = save_policy_weights(out/'policy_weights.msgpack', restored.policy.params,
            dict(dataset_sha256=DATA_SHA256, epochs=EPOCHS, optimizer_updates=int(state.policy.step),
                seed=seed, observation_shape=[4,32,64], actions=5, architecture='octax_cnn_v2'))
        weights, _ = read_policy_weights(out/'policy_weights.msgpack', weights_sha)
        exact(restored.policy.params, weights)
        # A critic is needed only by the existing evaluation API. It remains
        # local to this process and is explicitly absent from the exported file.
        evaluation_params = {'params': dict(fresh.agent_ts.params['params'], **weights['params'])}
        status('pretrained_evaluation')
        final_eval = evaluate(evaluator, evaluation_params, 'policy')
        write_json(out/'pretrained_evaluation.json', final_eval)
        # This is a training-fit gate, not a gameplay/generalization claim.
        fit_gate = after < before and after < constant_ce
        summary = dict(seed=seed, epochs=EPOCHS, examples_seen=int(state.examples_seen),
            optimizer_updates=int(state.policy.step), training_seconds=total_seconds,
            initial_training_cross_entropy=before, final_training_cross_entropy=after,
            constant_action_cross_entropy=constant_ce, training_fit_gate=fit_gate,
            full_cloning_checkpoint_roundtrip_exact=True, weights_roundtrip_exact=True,
            dataset_sha256=DATA_SHA256, dataset_collection_decisions=2240,
            diagnostic_replay_transitions=2241,
            evaluation_decisions=sum(initial_eval['episodes']['lengths'])+sum(final_eval['episodes']['lengths']),
            policy_weights_sha256=weights_sha,
            initial_parameter_sha256=original_hash, final_evaluation=final_eval['summary'])
        assert summary['examples_seen'] == 44800 and summary['optimizer_updates'] == 360
        write_json(out/'summary.json', summary)
        status('complete', training_fit_gate=fit_gate, ppo_training_started=False)
        print('PRETRAINING COMPLETE', summary, flush=True)
    except Exception as exc:
        status('failed', error=str(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    run(args.run, args.dataset, args.seed)
