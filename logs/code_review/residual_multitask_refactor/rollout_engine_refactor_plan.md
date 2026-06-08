# Residual Rollout Engine Refactor Plan

## Current Step

This step introduces module boundaries without changing rollout behavior:

- `residual_actor_runtime.py`: actor and `ResidualEEInterventionRunner` construction.
- `episode_logger.py`: hybrid logs, intervention records, JSON summaries, and compact summary printing.
- `rollout_engine.py`: a minimal `ResidualRolloutEngine` shell that can call the current episode function and produce the existing aggregate summary shape.

The large per-step rollout loop still lives in `scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py`.

## Collect-Replay Path

`collect_residual_replay.py` should eventually stop assembling a partial `SimpleNamespace` and instead build a `ResidualRolloutEngine` from:

- task config and seed resolver,
- env/model/gate runtime factories,
- residual actor runtime,
- an episode logger instance.

The replay builder can then consume `engine.run(...)` summaries and intervention record paths exactly as it does today, preserving `replay.npz`.

## Train-TD3 Path

`train_residual_td3_from_replay.py` remains an offline NPZ trainer for now.  The future RLinf-compatible path should:

- keep `NpzReplayDataset` as the offline compatibility adapter,
- add a runner/worker wrapper that logs through `MetricLogger`,
- save actor/critic checkpoints under RLinf checkpoint roots,
- later adapt official replay batches into the same TD3 update function.

## Pipeline Evolution

`scripts/residual_td3_pipeline.py` should stay a thin command router while the rollout engine stabilizes.  The migration order should be:

1. route `rollout-eval` to `ResidualRolloutEngine` directly;
2. route `collect-replay` to the same engine and replay builder;
3. move run metadata/config snapshots into a small pipeline utility module;
4. add Hydra config groups once the same task config can reproduce current handover smoke;
5. add RLinf `EmbodiedEvalRunner` integration only after rollout behavior is fully covered by smoke tests.
