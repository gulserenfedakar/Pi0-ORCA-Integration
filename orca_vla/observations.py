from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import mujoco
from .cameras import VLACameraRenderer


def _mat_to_axisangle(mat_flat: np.ndarray) -> np.ndarray:
    """Converts a 9-element MuJoCo rotation matrix to a 3D axis-angle vector (matching robosuite/LIBERO)."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat_flat, dtype=np.float64))
    # MuJoCo quat is (w, x, y, z); robosuite quat is (x, y, z, w)
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    return ((xyz * 2.0 * math.acos(w)) / den).astype(np.float32)


@dataclass
class OrcaVLAObservationConfig:
    external_cam: str = "palm_front"
    wrist_cam: str = "palm_top"
    num_joints: int = 17
    cube_body_name: str = "task_cube"


class OrcaVLAObservationBuilder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        renderer: VLACameraRenderer,
        config: OrcaVLAObservationConfig | None = None,
    ):
        self.model = model
        self.data = data
        self.renderer = renderer
        self.config = config if config is not None else OrcaVLAObservationConfig()

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        """Renders (external_img, wrist_img) as uint8 (H, W, 3) RGB arrays."""
        try:
            ext_img = self.renderer.render(self.config.external_cam)
        except ValueError:
            ext_img = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

        try:
            wrist_img = self.renderer.render(self.config.wrist_cam)
        except ValueError:
            wrist_img = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

        return ext_img, wrist_img

    def build_libero_eef_state(self, gripper_close_ratio: float = 0.0) -> np.ndarray:
        """Builds the 8-dim state expected by pi0_libero:
        [cube_pos (3), cube_axisangle (3), hand_closure_qpos (2)].
        """
        mujoco.mj_kinematics(self.model, self.data)
        try:
            cube_id = self.model.body(self.config.cube_body_name).id
            eef_pos = self.data.xpos[cube_id].astype(np.float32)
            eef_axisangle = _mat_to_axisangle(self.data.xmat[cube_id])
        except KeyError:
            eef_pos = np.zeros(3, dtype=np.float32)
            eef_axisangle = np.zeros(3, dtype=np.float32)

        opening = float(0.04 * (1.0 - np.clip(gripper_close_ratio, 0.0, 1.0)))
        gripper_qpos = np.array([opening, -opening], dtype=np.float32)
        return np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)

    def build_droid_observation(self, instruction: str, gripper_close_ratio: float = 0.0) -> Dict[str, Any]:
        """Builds the exact observation dictionary expected by `pi0_droid` (DroidInputs):
        - observation/exterior_image_1_left: (224, 224, 3) uint8
        - observation/wrist_image_left: (224, 224, 3) uint8
        - observation/joint_position: (7,) float32 (7D proprioceptive state)
        - observation/gripper_position: (1,) float32 ([0=open, 1=closed])
        - prompt: str
        """
        ext_img, wrist_img = self.render_cameras()
        joint_pos = np.asarray(self.data.qpos[:7], dtype=np.float32).copy()
        gripper_pos = np.array([np.clip(gripper_close_ratio, 0.0, 1.0)], dtype=np.float32)
        return {
            "observation/exterior_image_1_left": ext_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_pos,
            "prompt": instruction,
        }

    def build_observation(
        self,
        instruction: str,
        *,
        use_cartesian_eef_state: bool = False,
        gripper_close_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """Builds the observation dictionary for Libero / raw pi0 models."""
        ext_img, wrist_img = self.render_cameras()
        left_wrist_img = np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

        if use_cartesian_eef_state:
            state = self.build_libero_eef_state(gripper_close_ratio=gripper_close_ratio)
        else:
            state = np.array(self.data.qpos[: self.config.num_joints], dtype=np.float32)

        return {
            "observation/image": ext_img,
            "observation/wrist_image": wrist_img,
            "observation/left_wrist_image": left_wrist_img,
            "observation/state": state,
            "prompt": instruction,
        }

