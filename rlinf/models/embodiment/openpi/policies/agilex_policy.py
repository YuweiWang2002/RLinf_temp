import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_agilex_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    use_left_arm: bool = True
    use_right_arm: bool = True

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, use_left_arm=self.use_left_arm, use_right_arm=self.use_right_arm)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        images = {
            "base_0_rgb": in_images["cam_high"],
            "left_wrist_0_rgb": in_images["cam_left_wrist"],
            "right_wrist_0_rgb": in_images["cam_right_wrist"],
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, self.use_left_arm, self.use_right_arm)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    use_left_arm: bool = True
    use_right_arm: bool = True

    def __call__(self, data: dict) -> dict:
        if not self.use_left_arm or not self.use_right_arm:
            # Only return the first 7 dims.
            actions= _encode_actions(np.asarray(data["actions"]))[:, :7]
        else:
            # Only return the first 14 dims.
            actions = _encode_actions(np.asarray(data["actions"]))[:, :14]
        return {"actions": actions}
    
def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def _gripper_to_angular(value):
    return 1 - np.clip(_normalize(value, min_val=0, max_val=0.0731), 0, 1)

def _gripper_from_angular(value):
    return _unnormalize(np.clip(1 - value, 0, 1), min_val=0, max_val=0.0731)

def _gripper_from_angular_inv(value):
    return 1 - np.clip(_normalize(value, min_val=0, max_val=0.0731), 0, 1)

def _decode_aloha(data: dict, use_left_arm: bool = True, use_right_arm: bool = True) -> dict:
    def convert_image(img):
        img = np.array(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)

        if img.ndim == 3:
            if img.shape[0] == 3:
                # [C, H, W] -> [H, W, C]
                return einops.rearrange(img, "c h w -> h w c")
            if img.shape[2] == 3:
                # [H, W, C]
                return img
        elif img.ndim == 4:
            if img.shape[1] == 3:
                # [N, C, H, W] -> [N, H, W, C]
                return einops.rearrange(img, "n c h w -> n h w c")
            if img.shape[3] == 3:
                # [N, H, W, C]
                return img
        raise ValueError(
            f"Unexpected image shape {img.shape}. Expected [C,H,W]/[H,W,C] "
            "or batched [N,C,H,W]/[N,H,W,C]."
        )

    # Support both online RLinf env format and offline dataset format.
    if "observation/state" in data:
        state = np.asarray(data["observation/state"])
        base_image = convert_image(data["observation/image"])

        wrist_images = None
        if "observation/wrist_image" in data:
            wrist_images = convert_image(data["observation/wrist_image"])
        elif "observation/extra_view_image" in data:
            wrist_images = convert_image(data["observation/extra_view_image"])

        if wrist_images is None:
            raise KeyError(
                "Missing wrist views for Agilex inputs. Expected "
                "'observation/wrist_image' or 'observation/extra_view_image'."
            )
        if wrist_images.ndim != 4 or wrist_images.shape[0] < 2:
            raise ValueError(
                "Agilex wrist images must have shape [N, H, W, C] with N>=2. "
                f"Got {wrist_images.shape}."
            )
        images_dict = {
            "cam_high": base_image,
            "cam_left_wrist": wrist_images[0, ...],
            "cam_right_wrist": wrist_images[1, ...],
        }
    else:
        # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
        # dim sizes: [6, 1, 6, 1]
        state = np.asarray(data["state"])
        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}

    state = _decode_state(state, use_left_arm, use_right_arm)

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, use_left_arm: bool = True, use_right_arm: bool = True) -> np.ndarray:
    if not use_left_arm:
        state[13] = _gripper_to_angular(state[13])
        return state[7:]
    if not use_right_arm:
        state[6] = _gripper_to_angular(state[6])
        return state[:7]
    state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray) -> np.ndarray:
    actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, use_left_arm: bool = True, use_right_arm: bool = True) -> np.ndarray:
    if not use_left_arm:
        actions[:, 13] = _gripper_from_angular_inv(actions[:, 13])
        return actions[:, 7:]
    if not use_right_arm:
        actions[:, 6] = _gripper_from_angular_inv(actions[:, 6])
        return actions[:, :7]
    actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
