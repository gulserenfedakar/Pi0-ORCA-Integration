from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np

@dataclass(frozen=True)
class OrcaVLAActionConfig:
    expected_num_actuators: int = 17
    model_action_dim: int = 32
    max_finger_curl: float = 0.65
    thumb_cmc: float = 0.0
    thumb_abd: float = 0.0


class OrcaVLAActionAdapter:
    def __init__(self, model: mujoco.MjModel, config: OrcaVLAActionConfig | None = None) -> None:
        self.model = model
        self.config = config if config is not None else OrcaVLAActionConfig()
        if self.model.nu < self.config.expected_num_actuators:
            raise ValueError(
                f"Unexpected number of actuators: expected at least {self.config.expected_num_actuators}, got {self.model.nu}"
            )

        ctrl_range = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        self.num_actuators_to_control = self.config.expected_num_actuators
        self.hand_act_indices = np.arange(0, 17)
        self.hand_joint_names = [
            self.model.joint(self.model.actuator_trnid[i, 0]).name for i in self.hand_act_indices
        ]

        # Bounds for the 17 ORCA hand actuators
        self.action_low = ctrl_range[self.hand_act_indices, 0].copy()
        self.action_high = ctrl_range[self.hand_act_indices, 1].copy()

    @property
    def num_actuators(self) -> int:
        return self.num_actuators_to_control

    def validate_model_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape (D,) or (T, D), got {actions.shape}")
        if actions.shape[1] < self.num_actuators:
            raise ValueError(f"Model action dimension too small: need {self.num_actuators}, got {actions.shape[1]}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Action chunk contains NaN or infinite values.")
        return actions

    def to_orca_actions(self, model_actions: np.ndarray) -> np.ndarray:
        model_actions = self.validate_model_actions(model_actions)
        orca_actions = model_actions[:, : self.num_actuators].copy()
        orca_actions = np.clip(orca_actions, self.action_low, self.action_high)
        return orca_actions.astype(np.float32, copy=False)

    def hand_synergy_pose(
        self,
        curl: float = 0.0,
        t_cmc: float | None = None,
        t_abd: float | None = None,
        t_curl: float | None = None,
    ) -> np.ndarray:
        """Parametric 17-DoF ORCA hand synergy pose.
        Drives finger MCP/PIP joints and thumb flexion to `curl` fraction of their upper limit,
        while holding thumb CMC and abduction at stable opposition angles.
        """
        t_cmc = self.config.thumb_cmc if t_cmc is None else t_cmc
        t_abd = self.config.thumb_abd if t_abd is None else t_abd
        q = np.zeros(17, dtype=np.float32)
        for i, name in enumerate(self.hand_joint_names):
            lo, hi = self.action_low[i], self.action_high[i]
            if name in ("right_t-cmc", "right_thumb_mcp"):
                q[i] = t_cmc
            elif name in ("right_t-abd", "right_thumb_abd"):
                q[i] = t_abd
            elif name.startswith("right_t-") or name.startswith("right_thumb_"):
                q[i] = hi * (curl if t_curl is None else t_curl)
            elif "mcp" in name or "pip" in name:
                q[i] = hi * curl
            q[i] = np.clip(q[i], lo, hi)
        return q

    def gripper_to_hand_ctrl(self, gripper_action: float, mode: str = "droid") -> tuple[np.ndarray, float]:
        """Maps a 1D VLA gripper output to a 17-DOF ORCA hand control vector using BEST_GRASP synergy.
        Returns (hand_ctrl_17d, normalized_close_amount).
        """
        val = float(np.asarray(gripper_action).squeeze())
        if mode == "droid":
            # In DROID, gripper action is in [0, 1] where > 0.5 means close
            close_ratio = np.clip(val, 0.0, 1.0)
        else:
            # In Libero, gripper action is in [-1, 1]
            close_ratio = np.clip((val + 1.0) / 2.0, 0.0, 1.0)

        target_curl = close_ratio * self.config.max_finger_curl
        return self.hand_synergy_pose(curl=target_curl), float(close_ratio)

    def in_hand_synergy_delta(self, vla_action: np.ndarray) -> np.ndarray:
        """Maps a 7D VLA action [dx, dy, dz, drx, dry, drz, grip] into a 17-DoF normalized
        relative joint delta in [-1, 1]^17 for OrcaHandRightCubeOrientation (RelativeJointTargetAction).
        """
        act = np.asarray(vla_action, dtype=np.float64).squeeze()
        dx, dy, dz = np.clip(act[0:3], -1.0, 1.0)
        drx, dry, drz = np.clip(act[3:6], -1.0, 1.0)
        grip = float(np.clip(act[6], -1.0, 1.0))

        # Combine translational and rotational cues into 6 anatomical in-hand synergies:
        # 1) Wrist pitch (safe upward/cradle bias so cube never slides off fingertips)
        wrist_delta = np.clip(0.35 * dry - 0.25 * dx + 0.15, -0.35, 0.75)
        # 2) Uniform MCP flexion (cradle wall + pitch roll)
        mcp_flex = np.clip(0.45 * (grip + 0.5) + 0.35 * dry - 0.30 * dx, -0.6, 0.9)
        # 3) Uniform PIP flexion (distal finger roll)
        pip_flex = np.clip(0.40 * (grip + 0.4) + 0.30 * dz, -0.6, 0.9)
        # 4) Radial-ulnar finger wave (index vs pinky differential flexion for Y-axis roll)
        wave = np.clip(0.65 * drx + 0.45 * dy, -0.9, 0.9)
        # 5) Thumb opposition (CMC + abduction)
        thumb_opp = np.clip(0.40 * drz + 0.35 * (grip + 0.4), -0.6, 0.85)
        # 6) Thumb flexion (MCP + PIP)
        thumb_flex = np.clip(0.45 * (grip + 0.3) - 0.35 * dy, -0.6, 0.85)

        delta = np.zeros(17, dtype=np.float32)
        delta[0] = wrist_delta
        # Pinky (1: abd, 2: mcp, 3: pip)
        delta[1] = np.clip(-0.4 * wave, -1.0, 1.0)
        delta[2] = np.clip(mcp_flex - 0.7 * wave, -1.0, 1.0)
        delta[3] = np.clip(pip_flex - 0.7 * wave, -1.0, 1.0)
        # Ring (4: abd, 5: mcp, 6: pip)
        delta[4] = np.clip(-0.2 * wave, -1.0, 1.0)
        delta[5] = np.clip(mcp_flex - 0.3 * wave, -1.0, 1.0)
        delta[6] = np.clip(pip_flex - 0.3 * wave, -1.0, 1.0)
        # Middle (7: abd, 8: mcp, 9: pip)
        delta[7] = np.clip(0.2 * wave, -1.0, 1.0)
        delta[8] = np.clip(mcp_flex + 0.3 * wave, -1.0, 1.0)
        delta[9] = np.clip(pip_flex + 0.3 * wave, -1.0, 1.0)
        # Index (10: abd, 11: mcp, 12: pip)
        delta[10] = np.clip(0.4 * wave, -1.0, 1.0)
        delta[11] = np.clip(mcp_flex + 0.7 * wave, -1.0, 1.0)
        delta[12] = np.clip(pip_flex + 0.7 * wave, -1.0, 1.0)
        # Thumb (13: cmc, 14: abd, 15: mcp, 16: pip)
        delta[13] = thumb_opp
        delta[14] = thumb_opp
        delta[15] = thumb_flex
        delta[16] = thumb_flex
        return np.clip(delta, -1.0, 1.0)


