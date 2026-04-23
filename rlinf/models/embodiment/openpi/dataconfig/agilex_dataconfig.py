import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import agilex_policy


@dataclasses.dataclass(frozen=True)
class LeRobotAgilexDataConfig(DataConfigFactory):
    """Data configuration for Agilex-style dual-arm OpenPI datasets."""

    default_prompt: str | None = None
    extra_delta_transform: bool = False
    use_left_arm: bool = True
    use_right_arm: bool = True

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    def generate_observations(
        self, image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                agilex_policy.AgilexInputs(
                    use_left_arm=self.use_left_arm,
                    use_right_arm=self.use_right_arm,
                )
            ],
            outputs=[
                agilex_policy.AgilexOutputs(
                    use_left_arm=self.use_left_arm,
                    use_right_arm=self.use_right_arm,
                )
            ],
        )

        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=bool,
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
