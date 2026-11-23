# Staged pixel pretraining followed by on-policy PPO

The user explicitly clarified that a curriculum may include rejection-sampling fine-tuning or visual pretraining before online, on-policy RL. The on-policy requirement applies to the subsequent RL stage; it does not prohibit an earlier fixed-data stage. This supersedes the earlier interpretation that all learning must start from scratch. The separate pixel-novelty pilot is complete and failed its scoring screen. The first independent cloning/handoff investigation is also complete: the boundary works, but useful gameplay remains unestablished. [Results and policy card](runs/tetris/human_bc_seed1/RESULT.md).

## Stage boundary

1. Freeze a separately accounted pretraining dataset and its provenance. Only pixels enter the network. Actions may be supervised targets; original game scores may select successful trajectories. Neither emulator state nor board features are model inputs.
2. Fit the initialization offline. Reusing this fixed dataset belongs to this stage only. Save its parameters, optimizer, preprocessing, data hashes and training budget.
3. Evaluate that initialization in actual games before PPO. Good action prediction or reconstruction alone is not a gameplay result.
4. Initialize PPO with the selected parameters, a fresh PPO optimizer, fresh environment/RNG state and a freshly initialized critic. Collect every PPO rollout under the current policy. Remove the dataset loader, replay buffer and auxiliary pretraining loss from this stage. Initialize PPO step counts at zero and report pretraining work separately.
5. Evaluate original game score after matched PPO budgets, against random, scratch PPO and the exact pretrained initialization. This distinguishes gains due to pretraining from gains added during RL.

The first curriculum comparison retains the existing feed-forward CNN, four monochrome 32×64 frames and five keypress actions. The user subsequently clarified that architecture experiments, including attention, are allowed; the old preference for a simple CNN is not a permanent prohibition. There is no architecture change in this first curriculum comparison. Pixel-based control remains required, without geometry features or a placement planner. Use original-score PPO without simultaneously adding the novelty reward, to keep the source of any gain identifiable.

## Distinct candidates

| Initialization | Offline objective and transferred parameters | Key question |
| --- | --- | --- |
| Successful-trajectory cloning | Filter a fixed set of agent trajectories by positive original return; fit pixel-to-action cross-entropy; transfer CNN and actor | Can the rare successes provide a better starting action policy? |
| Human behavior cloning | Fit pixel-to-action cross-entropy on validated human observations/actions; transfer CNN and actor | Does demonstrated control bootstrap reward discovery? |
| Visual pretraining | Fit a deterministic reconstruction/denoising objective on fixed pixel stacks; transfer CNN only, with fresh actor and critic | Is weak visual representation a bottleneck? |
| Combined curriculum, later | Visual pretraining, then cloning, then fresh on-policy PPO | Does combining stages improve on the individually tested stages? |

“RFT” here means the user's rejection-sampling fine-tuning proposal. It is related to the prior SIL experiment, but not identical. Our SIL loss weights old actions by positive return advantage and updates a value estimate; ordinary accepted-trajectory cloning uses cross-entropy on selected examples. The old experiment interleaved replay with PPO. A separate offline stage followed by PPO does not reproduce that methodological issue. [Original SIL paper](https://proceedings.mlr.press/v80/oh18b.html).

## Why visual pretraining is plausible, and what it does not prove

ATC studies separating representation learning from RL and includes offline encoder pretraining followed by RL with frozen encoders. Its Atari experiments use feed-forward PPO. Its online variant uses a recent-observation replay buffer, so that variant should not be copied into our replay-free PPO stage. This supports investigating the staged approach, not a claim that it will work at our small budget or on Tetris. [ATC paper](https://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf).

Image reconstruction has also helped pixel-based RL, but SAC+AE studied an off-policy setting. It is motivation for an auxiliary representation objective, not a direct validation of our proposed PPO curriculum. [SAC+AE paper](https://arxiv.org/abs/1910.01741).

For a first reconstruction baseline, keep the existing encoder and add a temporary small bottleneck/decoder, discarded before PPO. A deterministic denoising autoencoder is simpler to diagnose than a stochastic VAE. Use full binary frames, no game-specific crop, and report reconstruction against trivial constant/static-image baselines: much of the display is empty or unchanged. This empty-background failure mode is a local hypothesis to measure, not a result yet. Reconstruction metrics should include how well changed pixels are retained; frame-difference diagnostics do not become actor inputs or rewards. Record CNN training time rather than assuming pretraining is computationally free.

If reconstruction fails to help actual control, temporal contrast is another candidate. ATC's own analysis cautions against equating contrastive-task accuracy with downstream RL performance, and reports mixed effects from image shifts. We should not blindly transplant large pixel augmentations to a 32×64 game. [ATC analysis and ablations](https://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf).

## Data currently available

Four archived agent scoring episodes contain 6,551 decisions and one point each: lengths 2,021, 1,481, 1,187 and 1,862. They were selected from the previous 524,288-decision self-imitation collection history, including its 64K parent, not collected for only 6,551 decisions. Collection began with score-only PPO and later included SIL-assisted behavior after replay activation; that is acceptable offline-data provenance but must be labeled. The archives contain packed pre-action pixels and actions. Their historical exporter records completed episodes; they lack reset/emulator snapshots for an independent exact replay of each whole game. Preserve these limits when auditing them.

A human recording also exists: `runs/tetris/human_demos/20261123T155105Z_ec4cb0/episode_000_seed200000.npz`. The inspected, checksum-matched version has 2,240 decisions and score 11, and is marked `complete=false`, `end_reason=in_progress`. It was copied to a fixed scratch snapshot without modifying the recorder's original. `validate_human_demos.py` reproduced every transition, pixel, reward and full emulator state exactly and verified independent mid-episode restoration. This is a valid supervised prefix, not a completed episode or a held-out dataset. The attempt to inspect whether the recorder process was still live was unavailable in the sandbox; no claim is made that recording had stopped. The fixed copy is now retained at `runs/tetris/human_bc_seed1/pretrain/dataset.npz` with the completed offline experiment. The original recording remains untouched.

The fixed prefix has SHA-256 `7a5a60ca8632cd751a23a2553d21762254b8aa208a94f658d835f6abc3c05cec`. The validation executes 2,240 full-replay transitions plus one mid-episode-restoration transition: 2,241 diagnostic transitions in total. These replay existing actions; they are not new independent experience or policy training. Reproduce on that file version with `JAX_PLATFORMS=cpu .venv/bin/python validate_human_demos.py <episode.npz>`. The original recording remains untouched.

Do not split neighboring frames from one game between train and validation and call that generalization. Split by whole trajectories/recording sessions when sufficient data exists. With only one human prefix, training-fit metrics cannot supply independent action-validation evidence; fresh complete gameplay evaluation is required. Cloning all actions in a lucky scoring episode can also imitate its mistakes, so successful filtering alone does not certify expert demonstrations.

## Implementation next steps

The existing `train.py` encoder checkpoint option copies only convolutional features and resets the actor. It is appropriate for a visual-encoder initialization, but insufficient for cloning: the cloning path must transfer both CNN and actor, leave the target seed's critic fresh, reset Adam state, and preserve target collection RNG. Verify parameter subsets and an exact first fresh rollout before a learning run.

Implement one simple initialization experiment at a time, then evaluate pretraining alone and after the same 64K PPO budget with independent training seeds. Keep the original-score development bank exploratory, reserve unused game seeds for confirmation, record all offline data/optimization/collection costs, and publish actual-policy animation cards with the implementation log. No gameplay benefit from pretraining is established by the completed first investigation.

## Independent process boundary implemented

Following the user's design guidance, the curriculum has two independent entry points. `pretrain_human_policy.py` writes `policy_weights.msgpack`; `train_pretrained_ppo.py` creates its own fresh PPO state and reads that file. The handoff schema contains only the declared encoder/actor weights and provenance. It excludes critic, optimizers, RNG, environment state, observations and actions; full training checkpoints are rejected. The offline process may retain its own resumable cloning checkpoint, but PPO cannot consume that checkpoint through the weights-only loader. `policy_initialization.py` has no dataset-loader or pretraining imports. The small optional initialization path in the existing PPO runner leaves its default behavior unchanged.

The organization takes inspiration from Tinker Cookbook's separate supervised/RL entry points and documented recipes. Our [pixel curriculum recipe](recipes/pixel_curriculum/README.md) lists the stage commands and experiment budgets; the existing result directories retain configurations, sources and metrics. [Cookbook recipe structure](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/README.md).

The final targeted suite passed six tests in 24.85 seconds, covering pixel/action alignment and hash/source/timing rejection, supervised loss reduction with partial minibatches, exact cloning checkpoint continuation, weights-file handoff with fresh critic/optimizer/environment/RNG, a newly collected ordinary PPO update, encoder-only transfer and rejection of full training checkpoints. The strict schema permits only the declared `features` and optional `actor` parameter collections. Existing dependency/emulator warnings remain. Reproduce with `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_pixel_pretraining.py`.

Real 20-epoch pretraining and two independent 2,048-decision PPO smokes subsequently completed. Offline CE fell to 0.4201, but the clone scored 0/128 versus random 3/128. Both processes exactly reproduced initial monitoring arrays. With no task reward or advantage, entropy .01 erased the fitted preferences; entropy zero preserved all encoder/actor parameters exactly. Training-prefix rotation argmax recall was zero even before PPO. These findings justify investigating offline fit/coverage before a full 64K handoff; preserving the weak clone alone does not establish learning. [Full evidence](runs/tetris/human_bc_seed1/RESULT.md).
