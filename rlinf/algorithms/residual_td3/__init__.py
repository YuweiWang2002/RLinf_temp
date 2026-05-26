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
from .residual_ee_intervention import (
    ConstantResidualActor,
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    ZeroInitResidualActor,
    ZeroResidualActor,
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
    "ResidualBaseActionSpace",
    "EndposeBridge",
    "EndposeBridgeMode",
    "EndposeBridgeSpec",
    "EndposeActionPipeline",
    "EndposeActionPipelineConfig",
    "EndposeActionPipelineOutput",
    "AlohaFKBridge",
    "AlohaFKBridgeConfig",
    "ConstantResidualActor",
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
    "normalize_gripper_column",
    "require_obs_tensor",
    "validate_residual_td3_cfg",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
