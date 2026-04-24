# RLinf Agilex RealWorld Runbook

这份文档整理了 Agilex RealWorld 在 RLinf 中的环境准备、dummy 链路验证和真机联调步骤，便于自用和团队复用。

## 1. 环境配置

```bash
cd ~/RLinf
uv venv .venv --python 3.11.14
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync --active --extra embodied --extra realworld_agilex
uv pip install git+https://github.com/RLinf/openpi
```

## 2. 资产下载（两台服务器都执行）

```bash
bash requirements/embodied/download_assets.sh --assets openpi
```

## 3. 常见问题：`transformers_replace is not installed correctly`

如出现该报错，执行以下修复：

```bash
uv pip install transformers==4.53.2

PY_MM=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE="$VIRTUAL_ENV/lib/python${PY_MM}/site-packages"

# 1) 确认 openpi 的替换文件在
ls "$SITE/openpi/models_pytorch/transformers_replace" >/dev/null

# 2) 覆盖到 transformers
cp -rv "$SITE/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"
```

## 4. PyTorch 权重路径

注意：`Norm stats file` 需要放在对应 `repo_id/` 目录下。

- 167 服务器：`/nfs1/checkpoints/pi05_agilex_realworld`
- 40 服务器：`/data4/nfs_data/checkpoints/pi05_agilex_realworld`

## 5. 隔离一个 Ray 实例（示例）

```bash
cd ~/wyw/RLinf
source .venv/bin/activate

# 1) 清掉可能继承的 Ray 连接信息
unset RAY_ADDRESS
unset CLUSTER_NAMESPACE

# 2) 只暴露你要用的 4 张卡（改成你实际空闲卡号）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3) 单机 rank
export RLINF_NODE_RANK=0

# 4) 启动你自己的 Ray（独立端口 + 独立临时目录）
export MY_RAY_PORT=26379
export MY_RAY_TMP=/tmp/ray_${USER}_rlinf_dummy_${MY_RAY_PORT}
ray start --head \
  --node-ip-address=172.18.41.40 \
  --port=${MY_RAY_PORT} \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=28265 \
  --temp-dir=${MY_RAY_TMP}

# 5) 强制 RLinf 连接到你这套 Ray
export RAY_ADDRESS=172.18.41.40:${MY_RAY_PORT}
ray status --address=${RAY_ADDRESS}
```

## 6. 不连真机：dummy 链路测试

### 6.1 train 链路 dummy 测试

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name realworld_agilexdesk_dagger_openpi \
  +env.train.override_cfg.is_dummy=True \
  +env.eval.override_cfg.is_dummy=True \
  runner.max_epochs=1 \
  env.train.max_steps_per_rollout_epoch=10 \
  env.eval.max_steps_per_rollout_epoch=10 \
  actor.model.model_path=/data4/nfs_data/checkpoints/pi05_agilex_realworld \
  rollout.model.model_path=/data4/nfs_data/checkpoints/pi05_agilex_realworld
```

### 6.2 eval 链路 dummy 测试

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name realworld_agilexdesk_dagger_openpi \
  +env.train.override_cfg.is_dummy=True \
  +env.eval.override_cfg.is_dummy=True \
  algorithm.eval_rollout_epoch=1 \
  env.eval.max_steps_per_rollout_epoch=10 \
  env.eval.max_episode_steps=10 \
  actor.model.model_path=/data4/nfs_data/checkpoints/pi05_agilex_realworld \
  rollout.model.model_path=/data4/nfs_data/checkpoints/pi05_agilex_realworld
```

## 7. 连真机

### 7.1 policy-side

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name realworld_agilexdesk_dagger_openpi \
  +env.train.override_cfg.is_dummy=False \
  +env.eval.override_cfg.is_dummy=False \
  algorithm.eval_rollout_epoch=1 \
  env.eval.max_steps_per_rollout_epoch=100 \
  env.eval.max_episode_steps=100
```

### 7.2 robot-side

```bash
cd ~/cobot_magic/Piper_ros_private-ros-noetic
bash can_config.sh
```

```bash
cd ~/cobot_magic/infer
bash infer.sh
```

```bash
python -m rlinf.envs.realworld.agilex.robot_side_bridge \
  --host 0.0.0.0 \
  --port 10001
```

## 8. 备注

- `policy-side` 配置中的 `robot_port` 必须与 `robot_side_bridge --port` 一致（例如都为 `10001`）。
- 若 `policy-side` 与 `robot-side` 网络不通，先检查端口连通性与防火墙策略。
