# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Residual TD3 schema and action-space adapters."""

from .action_adapter import ResidualActionAdapter, ResidualActionSpec
from .config import build_residual_action_spec, validate_residual_td3_cfg
from .endpose_action_pipeline import (
    EndposeActionPipeline,
    EndposeActionPipelineConfig,
    EndposeActionPipelineOutput,
)
from .endpose_bridge import (
    EndposeBridge,
    EndposeBridgeMode,
    EndposeBridgeSpec,
    normalize_gripper_column,
    require_obs_tensor,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from .fk_bridge import AlohaFKBridge, AlohaFKBridgeConfig
from .residual_actor import ResidualActorConfig, ZeroInitResidualActorMLP
from .residual_bc_dataset import ResidualBCTargetBuilder, ResidualBCTargetConfig
from .residual_bc_training import (
    ResidualBCDatasetConfig,
    ResidualBCNpzDataset,
    build_residual_actor_obs_features,
    build_residual_actor_obs_features_from_endpose,
    residual_actor_obs_features_to_tensor,
    split_by_episode,
)
from .residual_ee_intervention import (
    BCResidualActor,
    ConstantResidualActor,
    RandomNoiseResidualActor,
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    ZeroInitResidualActor,
    ZeroResidualActor,
)
from .residual_replay import (
    ReplayRewardConfig,
    build_replay_from_rollout,
    episode_reward,
    load_baseline_success_by_seed,
    save_replay_artifacts,
)
from .schema import (
    ExpertResidualSample,
    ResidualBaseActionSpace,
    ResidualFrame,
    ResidualMode,
    ResidualObs,
    ResidualTransition,
)

__all__ = [
    "ExpertResidualSample",
    "ResidualActionAdapter",
    "ResidualActionSpec",
    "ResidualActorConfig",
    "ResidualBCTargetBuilder",
    "ResidualBCTargetConfig",
    "ResidualBCDatasetConfig",
    "ResidualBCNpzDataset",
    "ResidualBaseActionSpace",
    "build_residual_actor_obs_features",
    "build_residual_actor_obs_features_from_endpose",
    "EndposeBridge",
    "EndposeBridgeMode",
    "EndposeBridgeSpec",
    "EndposeActionPipeline",
    "EndposeActionPipelineConfig",
    "EndposeActionPipelineOutput",
    "AlohaFKBridge",
    "AlohaFKBridgeConfig",
    "BCResidualActor",
    "ConstantResidualActor",
    "RandomNoiseResidualActor",
    "ReplayRewardConfig",
    "ResidualEEInterventionConfig",
    "ResidualEEInterventionRunner",
    "ResidualFrame",
    "ResidualMode",
    "ResidualObs",
    "ResidualTransition",
    "ZeroInitResidualActor",
    "ZeroInitResidualActorMLP",
    "ZeroResidualActor",
    "build_residual_action_spec",
    "build_replay_from_rollout",
    "episode_reward",
    "load_baseline_success_by_seed",
    "normalize_gripper_column",
    "residual_actor_obs_features_to_tensor",
    "require_obs_tensor",
    "save_replay_artifacts",
    "split_by_episode",
    "validate_residual_td3_cfg",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
