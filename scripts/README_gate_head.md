# Frozen VLA Feature GateHead

This mini project trains only a lightweight binary MLP gate head on top of
frozen OpenPI/pi05 features. It does not train pi05, the action head, TD3, or a
residual actor.

## Data

Use the gated LeRobot copy:

```bash
/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/
```

It contains:

- `observation.state`: joint14
- `action`: joint14
- `observation.residual_gate`: float32 `[1]`, values `{0, 1}`

## Feature Extraction

The preferred feature source is `action_head_hidden`: pi05 `suffix_out` right
before `action_out_proj`, mean-pooled over `action_chunk`.

```bash
cd /home/user/wyw/RLinf
source scripts/setup_master40_robotwin_env.sh

CUDA_VISIBLE_DEVICES=2 python scripts/extract_pi05_features_for_gate.py \
  --pi05-config pi05_aloha_robotwin_handover \
  --pi05-checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors \
  --dataset-dir /nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/ \
  --output-dir logs/gate_head_vla_feature/full \
  --feature-source action_head_hidden \
  --device cuda \
  --batch-size 16
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/extract_pi05_features_for_gate.py \
  --pi05-config pi05_aloha_robotwin_handover \
  --pi05-checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors \
  --dataset-dir /nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/ \
  --output-dir logs/gate_head_vla_feature/smoke \
  --feature-source action_head_hidden \
  --limit-episodes 4 \
  --max-frames 512 \
  --device cuda
```

CPU smoke is also supported, though full feature extraction will be slow:

```bash
python scripts/extract_pi05_features_for_gate.py \
  --pi05-config pi05_aloha_robotwin_handover \
  --pi05-checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors \
  --dataset-dir /nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/ \
  --output-dir logs/gate_head_vla_feature/cpu_smoke \
  --feature-source action_head_hidden \
  --limit-episodes 1 \
  --max-frames 2 \
  --device cpu \
  --batch-size 1
```

Outputs:

- `features/features_train.npz`
- `features/features_val.npz`
- `features/feature_config.json`

## Train GateHead

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_gate_head_from_features.py \
  --feature-dir logs/gate_head_vla_feature/full/features \
  --output-dir logs/gate_head_vla_feature/full/gate_head \
  --epochs 20 \
  --batch-size 1024 \
  --device cuda \
  --use-pos-weight
```

Outputs:

- `gate_head.pt`
- `gate_head_config.json`
- `metrics.json`
- `threshold_sweep.json`
- `val_predictions.csv`

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_gate_head.py \
  --feature-dir logs/gate_head_vla_feature/full/features \
  --checkpoint logs/gate_head_vla_feature/full/gate_head/gate_head.pt \
  --output-dir logs/gate_head_vla_feature/full/eval \
  --device cuda
```

Plots are written to `eval/plots/`.

## Runtime Loading

```python
import torch
from rlinf.algorithms.residual_td3.gate_head import GateHeadRuntime

runtime = GateHeadRuntime.load_from_checkpoint(
    "logs/gate_head_vla_feature/full/gate_head/gate_head.pt",
    device="cuda",
)
z_t = torch.randn(8, runtime.cfg.feature_dim, device="cuda")
prob, gate_binary = runtime.predict_from_feature(z_t)
```

The runtime only consumes already-extracted `z_t`; it does not extract VLA
features or control the environment.
