"""Read-only fit and entropy-handoff audit; never collects data or updates weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from experiment_utils import write_json
from octax.agents.ppo import Agent
from pixel_pretraining import load_human_pairs
from policy_initialization import read_policy_weights


def action_log_probabilities(module, observations):
    features = module.encode(observations)
    # Public log-probability method, evaluated for every action without sampling.
    return jnp.stack([module.actor.log_prob_entropy(features,
        jnp.full((len(observations),), action, dtype=jnp.int32))[0]
        for action in range(5)], axis=-1)


def fit_metrics(agent, params, observations, actions):
    infer = jax.jit(lambda obs: agent.apply(params, obs, method=action_log_probabilities))
    logs = np.concatenate([np.asarray(infer(observations[i:i+128]))
        for i in range(0, len(actions), 128)])
    assert np.isfinite(logs).all()
    probabilities = np.exp(logs)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    predictions = logs.argmax(axis=1)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (actions, predictions), 1)
    return dict(cross_entropy=float(-logs[np.arange(len(actions)), actions].mean()),
        entropy=float(-(probabilities*logs).sum(axis=1).mean()),
        argmax_accuracy=float((predictions == actions).mean()),
        per_action_recall=(confusion.diagonal()/confusion.sum(axis=1)).tolist(),
        confusion_true_rows_predicted_columns=confusion.tolist(),
        mean_action_probabilities=probabilities.mean(axis=0).tolist())


def exact(a, b):
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain', type=Path, required=True)
    parser.add_argument('--entropy-smoke', type=Path, required=True)
    parser.add_argument('--zero-smoke', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.pretrain/'config.yaml'))
    data = load_human_pairs(args.pretrain/'dataset.npz', cfg['pretraining']['dataset_sha256'])
    weight_path = args.pretrain/'policy_weights.msgpack'
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    weights, provenance = read_policy_weights(weight_path, weight_hash)
    agent = Agent(action_dim=5)
    counts = np.bincount(data.actions, minlength=5)
    probabilities = counts/counts.sum()
    _, inverse = np.unique(np.packbits(data.observations.reshape(len(data.actions), -1), axis=1),
        axis=0, return_inverse=True)
    label_counts = np.zeros((inverse.max()+1, 5), dtype=np.int64)
    np.add.at(label_counts, (inverse, data.actions), 1)
    conditional = label_counts/label_counts.sum(axis=1, keepdims=True)
    conditional_entropy = -np.sum(label_counts*np.log(np.maximum(conditional, 1e-30)))/len(data.actions)
    result = dict(kind='offline_read_only_diagnostic', additional_environment_decisions=0,
        optimization_updates=0, dataset=data.provenance, weights=provenance,
        action_counts=counts.tolist(), constant_action_cross_entropy=float(-(probabilities*np.log(probabilities)).sum()),
        constant_argmax_accuracy=float(probabilities.max()), unique_observations=len(label_counts),
        ambiguous_observations=int(((label_counts > 0).sum(axis=1) > 1).sum()),
        empirical_observation_conditional_ce_lower_bound=float(conditional_entropy),
        empirical_observation_argmax_accuracy_upper_bound=float(label_counts.max(axis=1).sum()/len(data.actions)),
        pretrained=fit_metrics(agent, weights, data.observations, data.actions))
    smoke_configs, smoke_metrics, smoke_evaluations = {}, {}, {}
    for name, path in [('entropy_001', args.entropy_smoke), ('entropy_zero', args.zero_smoke)]:
        payload = serialization.msgpack_restore((path/'checkpoint.msgpack').read_bytes())
        full = payload['train_state']['agent_ts']['params']
        params = {'params': {key: full['params'][key] for key in ('features', 'actor')}}
        smoke_configs[name] = OmegaConf.to_container(OmegaConf.load(path/'config.yaml'))
        smoke_metrics[name] = list(csv.DictReader((path/'metrics.csv').read_text().splitlines()))
        smoke_evaluations[name] = json.loads((path/'evaluation.json').read_text())
        result[name] = dict(fit_metrics(agent, params, data.observations, data.actions),
            encoder_actor_arrays_exact=exact(weights, params),
            checkpoint_sha256=hashlib.sha256((path/'checkpoint.msgpack').read_bytes()).hexdigest(),
            global_step=int(payload['train_state']['global_step']))
    a, b = smoke_configs.values()
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert set(differences) <= {'run_dir', 'ent_coef'}, differences
    assert result['entropy_zero']['encoder_actor_arrays_exact']
    assert not result['entropy_001']['encoder_actor_arrays_exact']
    a, b = [rows[0] for rows in smoke_metrics.values()]
    rollout_keys = ['step', 'reward_sum', 'positive_reward_steps', 'done_count', 'advantage_std'] + [f'action_{i}_fraction' for i in range(5)]
    assert all(a[k] == b[k] for k in rollout_keys)
    assert all(float(b[k]) == 0 for k in ['loss', 'grad_norm', 'approx_kl', 'value_loss', 'advantage_std'])
    a, b = smoke_evaluations.values()
    expected = json.loads((args.pretrain/'pretrained_evaluation.json').read_text())['episodes']
    expected = {key: value[:2] for key, value in expected.items()}
    assert a[0]['episodes'] == b[0]['episodes'] == expected
    assert b[0]['episodes'] == b[-1]['episodes']
    result['audits'] = dict(config_differences=differences, first_rollout_summary_exact=True,
        zero_entropy_zero_gradients=True, initial_evaluation_arrays_exact=True,
        zero_entropy_final_evaluation_arrays_exact=True,
        note='Rollout summaries match; raw rollout arrays were not saved.')
    result['limitations'] = ['One human prefix: all fit metrics are in-sample, not generalization.',
        'Empirical identical-observation bounds apply to deterministic inference on this fixed input representation and dataset.',
        'No old observations were used in PPO updates; this separate diagnostic only reads completed artifacts.']
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'diagnostic.json', result)
    sources = ['diagnose_cloning_handoff.py', 'pixel_pretraining.py', 'policy_initialization.py',
        'experiment_utils.py', 'octax/agents/ppo.py']
    manifest = {}
    for name in sources:
        contents = Path(name).read_bytes(); target = args.out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(contents)
        manifest[name] = hashlib.sha256(contents).hexdigest()
    write_json(args.out/'source_manifest.json', manifest)
    print(json.dumps({k: v for k, v in result.items() if k not in ('weights', 'dataset')}, indent=2))


if __name__ == '__main__':
    main()
