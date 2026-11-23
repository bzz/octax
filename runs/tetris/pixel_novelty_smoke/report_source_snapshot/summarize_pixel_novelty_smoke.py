"""Report smoke correctness/runtime, with actual gameplay but no learning claim."""
import json
from pathlib import Path

from flax import serialization
import numpy as np

from experiment_utils import environment_from_config
from octax.agents.ppo import Agent
from round_card import write_round_card
from run_pixel_novelty import smoke_gates
from visualize_run import capture_episode

ROOT = Path(__file__).resolve().parent


def main():
    result = smoke_gates()
    run = ROOT/'runs/tetris/pixel_novelty_smoke'
    payload = serialization.msgpack_restore((run/'checkpoint.msgpack').read_bytes())
    env, _ = environment_from_config(payload['config']); agent = Agent(action_dim=env.num_actions)
    trace = capture_episode(env, agent, payload['train_state']['agent_ts']['params'], 'policy', 30016)
    out = run/'visualizations'; out.mkdir(exist_ok=True)
    np.savez_compressed(out/'policy_trajectory.npz', **trace)
    result['illustration'] = {key: trace[key] for key in ['seed', 'length', 'final_score']}
    (run/'smoke_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    base, novel = result['control']['summary'], result['novelty']['summary']
    rows = [('Training decisions per arm', '8,192'),
        ('Original rewards: disabled / enabled', f"{base['training_reward_events']} / {novel['training_reward_events']}"),
        ('Warmed decisions/s: disabled / enabled', f"{base['steady_training_steps_per_second']:.1f} / {novel['steady_training_steps_per_second']:.1f}"),
        ('Checkpoint and source checks', 'Passed')]
    write_round_card(run, 'Pixel novelty · CPU smoke only',
        'The on-policy integration passes its smoke checks. This small diagnostic does not establish better Tetris play.', rows,
        [('Smoke report', 'SMOKE_REPORT.md'), ('Protocol', '../../../PIXEL_EXPLORATION_PROPOSAL.md'), ('Implementation log', '../../../IMPLEMENTATION_LOG.md')])
    (run/'SMOKE_REPORT.md').write_text(f'''# Pixel-novelty CPU smoke comparison

Both matched seed-101 runs completed 8,192 decisions and 256 PPO updates. All states/metrics were finite, the fixed target stayed unchanged and full checkpoints round-tripped exactly. The disabled arm had zero auxiliary optimizer updates; the enabled arm had 256 predictor and intrinsic-head updates. Initial policy hashes and evaluation arrays matched exactly. The only resolved differences were output path and novelty coefficient 0 versus .01. No replay or demonstrations were used.

| Diagnostic | Disabled | Enabled |
| --- | ---: | ---: |
| Original training rewards | {base['training_reward_events']} | {novel['training_reward_events']} |
| All-rollout decisions/s | {base['training_steps_per_second']:.2f} | {novel['training_steps_per_second']:.2f} |
| Warmed decisions/s (three rollouts) | {base['steady_training_steps_per_second']:.2f} | {novel['steady_training_steps_per_second']:.2f} |
| Monitoring evaluation decisions | {base['evaluation_decisions']} | {novel['evaluation_decisions']} |

These are separately counted smoke diagnostics, not pilot-training experience or evidence of improvement. The two-game initial/final evaluation checks have no useful power to assess learning. The disabled control still executes the extended collector's auxiliary-value inference, so this is not a benchmark of the untouched baseline runner. Throughput covers collection and updates, excluding checkpoint I/O and evaluation. Zero normalization warm-up decisions were used. The illustration adds {trace['length']} evaluation decisions on preselected seed 30016; it is not used for updates or policy selection.

The next frozen experiment is fresh seeds 1 and 2 at 65,536 decisions each, compared on original game score against the saved score-only PPO, exact untrained networks and random play. Coefficient .01 and other settings remain fixed. No scoring improvement is claimed by this smoke test.

[Actual smoke-policy card](result_card.html) · [Protocol](../../../PIXEL_EXPLORATION_PROPOSAL.md) · [Implementation log](../../../IMPLEMENTATION_LOG.md)
''')
    print(json.dumps(dict(control_steps_per_second=base['steady_training_steps_per_second'],
        novelty_steps_per_second=novel['steady_training_steps_per_second'], illustration=result['illustration']), indent=2))


if __name__ == '__main__':
    main()
