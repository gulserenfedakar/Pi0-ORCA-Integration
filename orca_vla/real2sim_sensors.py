"""Real2Sim Multi-Modal Sensor Calibration and Kinematic Alignment Wrappers for the 17-DoF ORCA Hand.

Directly integrates Joe's `real2sim-cameras` branch (`21f7aec`) from `SEP730-Dexterous-Hand-Simulation`:
  - Scene: `src/orca_sim/scenes/v2/scene_right_cube_orientation_real2sim.xml` (`real2sim_bench` at z=0.092 m,
    `drop_height=0.13 m`, calibrated `<camera name="top">` & `<camera name="side">`, and 25 ultrasonic
    rangefinders `<site name="us_0..us_24">` + `<rangefinder name="us_0..us_24">`).
  - Camera Calibration & AprilTag Layout: exact `T_world_cam_opencv` and intrinsics `K` from
    `real2sim/cameras/calib/cameras.json` (`top` and `side` Logitech Brio 100 at 1920x1080) and
    exact 6-face AprilTag transforms `T_cube_tag` (`id 0..5`, `effective_tag_mm = 26.25 mm`) from
    `real2sim/cameras/calib/cube_tags_layout.json`.
  - Dual-Camera (`top` + `side`) Pose Estimation & Fusion (`real2sim/cube/cube_tags.py`):
    At 30 Hz, checks per-camera AprilTag visibility and finger line-of-sight occlusion (`mj_ray` from
    `top` and `side` cameras to each tag on the cube), computes per-camera PnP/corner pose estimates,
    and fuses `top` and `side` views using Joe's `cube_pose_fused` (`depth_weight = 0.1` along each
    camera's viewing ray, weight `1.0` across the image plane, and SVD rotation averaging).
  - 25-Ray Specular Ultrasonic Digital Twin (`real2sim/cameras/camlib.py` & `real2sim/recording/twin.py`):
    Reads `us_0..us_24` sites from the `_real2sim` scene, computes surface normals via two `1 mm` offset
    neighbor rays (exact `ultrasonic_reading_specular` / `twin.py:echo` geometry with `max_incidence_deg = 25.0`,
    `range_offset_m = +0.0029`, `min_range_m = 0.020`), and passes hits through the calibrated 10 Hz
    HC-SR04 dead-zone, bistatic tilt dropout, curved-finger deflection, and 3-tap median + 150 mm filter.
  - Hardware-Aligned Joint Mapping & Locked Wrist (`real2sim/hand/joint_map.py` & `policy_run08.yaml`):
    * Locks the wrist at physical neutral `-9.6 deg` (`wrist_lock_deg = -9.6`).
    * Applies `joint_map.py` offsets (`ring_abd: -10.0 deg`, `pinky_abd: -14.0 deg`) and physical
      `config.yaml` ROM limits (`t-mcp` [-5, +40] deg, `t-abd` [-20, +35] deg, `p-abd` [-30, +30] core deg
      -> [-44, +16] sim deg) BEFORE `RelativeJointTargetAction` caches `_low`, `_high`, `_span`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------------------
# Exact calibrated camera matrices from `real2sim/cameras/calib/cameras.json`
# ------------------------------------------------------------------------------
T_WORLD_CAM_OPENCV: Dict[str, np.ndarray] = {
    "top": np.array(
        [
            [-0.2602884642186061, -0.9641421165318970, -0.0517676976898836, 0.1969233179238729],
            [-0.9653938268237223, 0.2589737717160663, 0.0307789651190898, -0.0139225264433774],
            [-0.0162688206507744, 0.0579876253397757, -0.9981847327935273, 0.6635584483571063],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "side": np.array(
        [
            [-0.9975729655335511, 0.0142150934688828, 0.0681623763836580, 0.1899048561978239],
            [-0.0693280592105924, -0.1118833570659484, -0.9913000225046636, 0.5776923338081502],
            [-0.0064651869802130, -0.9936196684493143, 0.1125973171438215, 0.1403402517360242],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

CAM_K: Dict[str, np.ndarray] = {
    "top": np.array(
        [
            [1960.1090888664764, 0.0, 981.4748761624927],
            [0.0, 1956.5281951347308, 525.5506183789169],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "side": np.array(
        [
            [1935.7135848064138, 0.0, 945.9318968135051],
            [0.0, 1931.2040655228334, 502.0159271289673],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

# ------------------------------------------------------------------------------
# Exact measured 6-face AprilTag layout from `real2sim/cameras/calib/cube_tags_layout.json`
# (`tag_mm = 27.0`, `effective_tag_mm = 26.25`, `cube_mm = 35.0`)
# ------------------------------------------------------------------------------
EFFECTIVE_TAG_M: float = 0.02625
T_CUBE_TAG: Dict[int, np.ndarray] = {
    0: np.array(
        [
            [0.0010204754261235, -0.9999994793148168, 0.0, 0.0013227287715824],
            [0.9999994793148168, 0.0010204754261235, 0.0, 0.0011316536165275],
            [0.0, 0.0, 1.0, 0.0180434755411925],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    1: np.array(
        [
            [-0.9983509024690724, -0.0433099398199888, 0.0376792336968038, -0.0027507080141967],
            [-0.0408396513609005, 0.9971120415504823, 0.0640288955999707, -0.0018950792166578],
            [-0.0403435052506526, 0.0623844989386056, -0.9972364693874121, -0.0167141769583370],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    2: np.array(
        [
            [-0.0040404128779152, 0.0139082444142148, -0.9998951123998412, -0.0178404716650963],
            [-0.9997881830961141, -0.0202351798366825, 0.0037585154449000, 0.0016188013950380],
            [-0.0201807830657883, 0.9996985036671264, 0.0139870568941431, 0.0000583845038186],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    3: np.array(
        [
            [-0.0077513639748083, 0.0034756935147905, 0.9999639173045802, 0.0185993353900967],
            [0.9998570455615744, -0.0150002241807887, 0.0078026736063250, 0.0000049639360220],
            [0.0150268026343200, 0.9998814493874352, -0.0033589244516470, -0.0004004177783366],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    4: np.array(
        [
            [0.0049637499778917, -0.9999875514985259, -0.0005079706093749, 0.0013854487573171],
            [0.0052895285539999, 0.0005342259781099, -0.9999858676452791, -0.0177982480277290],
            [0.9999736906908271, 0.0049609929033735, 0.0052921144714827, 0.0006656524697652],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    5: np.array(
        [
            [-0.9995143925111700, 0.0309592792519613, 0.0035358437785215, -0.0000027547837161],
            [0.0034502828036047, -0.0028166478051353, 0.9999900809726650, 0.0182608085114245],
            [0.0309689313926422, 0.9995166779615740, 0.0027084617510530, -0.0012785535907443],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

# ------------------------------------------------------------------------------
# Exact joint mapping from `real2sim/hand/joint_map.py` (`fullcheck01`)
# ------------------------------------------------------------------------------
JOINT_MAP_FIX_DEG: Dict[str, Dict[str, float]] = {
    "ring_abd": {"sign": 1.0, "offset_deg": -10.0},
    "pinky_abd": {"sign": 1.0, "offset_deg": -14.0},
}


def core_to_sim_deg(joint: str, core_deg: float) -> float:
    """Convert physical `orca_core` joint angle (deg) to MuJoCo simulation angle (deg) per `joint_map.py`."""
    if joint in JOINT_MAP_FIX_DEG:
        f = JOINT_MAP_FIX_DEG[joint]
        return f["sign"] * core_deg + f["offset_deg"]
    return -core_deg if joint == "index_abd" else core_deg


def sim_to_core_deg(joint: str, sim_deg: float) -> float:
    """Convert MuJoCo simulation joint angle (deg) to physical `orca_core` angle (deg) per `joint_map.py`."""
    if joint in JOINT_MAP_FIX_DEG:
        f = JOINT_MAP_FIX_DEG[joint]
        return (sim_deg - f["offset_deg"]) / f["sign"]
    return -sim_deg if joint == "index_abd" else sim_deg


def average_rotations(Rs: List[np.ndarray]) -> np.ndarray:
    """SVD rotation average matching `real2sim/cameras/camlib.py:average_rotations`."""
    u, _, vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


@dataclass
class UltrasonicReal2SimConfig:
    """Empirically fitted parameters for the palm-base 40 kHz HC-SR04 ultrasonic sensor (`rig.yaml`)."""

    range_offset_mm: float = 2.9
    input_includes_offset: bool = True
    dead_zone_cutoff_mm: float = 22.0
    max_valid_palm_mm: float = 150.0
    quantization_step_mm: float = 3.2
    control_hz: int = 100
    ultrasonic_hz: int = 10
    sigma_step_mm: float = 1.60
    sigma_episode_bias_mm: float = 2.50
    hard_incidence_limit_deg: float = 25.0
    use_stochastic_logistic_dropout: bool = True
    logistic_theta50_far_deg: float = 26.50
    logistic_theta50_near_deg: float = 18.50
    logistic_slope_per_deg: float = 0.38
    base_dropout_prob_in_cone: float = 0.019
    hand_echo_survival_prob: float = 0.015
    prob_wall_jump_on_miss: float = 0.900
    wall_jump_mode1_mm: Tuple[float, float] = (584.0, 95.0)
    wall_jump_mode2_mm: Tuple[float, float] = (1289.0, 110.0)
    prob_mode1_vs_mode2: float = 0.55


@dataclass
class CameraAprilTagReal2SimConfig:
    """Parameters for Joe's dual-camera (`top` + `side`) AprilTag `cube_pose_fused` pipeline."""

    control_hz: int = 100
    camera_hz: int = 30
    pos_episode_bias_sigma_m: float = 0.0045
    pos_step_jitter_sigma_m: float = 0.0004
    depth_step_jitter_sigma_m: float = 0.0012
    depth_weight: float = 0.10
    max_tag_incidence_deg: float = 68.0
    rot_episode_bias_deg: float = 2.50
    rot_step_jitter_deg: float = 0.70
    vel_lin_jitter_sigma_m_s: float = 0.015
    vel_ang_jitter_sigma_rad_s: float = 0.45
    single_tag_ippe_flip_prob: float = 0.010
    baseline_dropout_prob: float = 0.008
    finger_occlusion_dropout_prob: float = 0.040


class CalibratedUltrasonicSensor:
    """Real2Sim conical ultrasonic sensor model and RL observation filter."""

    def __init__(self, config: Optional[UltrasonicReal2SimConfig] = None, seed: Optional[int] = None):
        self.cfg = config or UltrasonicReal2SimConfig()
        self.rng = random.Random(seed)
        self._history: List[float] = []
        self._episode_bias_mm: float = 0.0
        self._step_counter: int = 0
        self._held_raw_mm: Optional[float] = None
        self._held_obs: Tuple[float, float] = (0.0, 0.0)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self._history.clear()
        self._episode_bias_mm = self.rng.gauss(0.0, self.cfg.sigma_episode_bias_mm)
        self._step_counter = 0
        self._held_raw_mm = None
        self._held_obs = (0.0, 0.0)

    def _quantize_mm(self, val_mm: float) -> float:
        q = self.cfg.quantization_step_mm
        return round(val_mm / q) * q if q > 0 else val_mm

    def echo_probability(self, true_dist_mm: float, incidence_deg: Optional[float]) -> float:
        if incidence_deg is None or math.isnan(incidence_deg):
            return 0.0
        if not self.cfg.use_stochastic_logistic_dropout:
            return (1.0 - self.cfg.base_dropout_prob_in_cone) if incidence_deg <= self.cfg.hard_incidence_limit_deg else 0.0

        alpha = max(0.0, min(1.0, (true_dist_mm - 35.0) / (74.0 - 35.0)))
        theta_50 = self.cfg.logistic_theta50_near_deg + alpha * (
            self.cfg.logistic_theta50_far_deg - self.cfg.logistic_theta50_near_deg
        )
        p_spec = 1.0 / (1.0 + math.exp(self.cfg.logistic_slope_per_deg * (incidence_deg - theta_50)))
        return max(0.0, min(1.0, (1.0 - self.cfg.base_dropout_prob_in_cone) * p_spec))

    def sample_jump_or_timeout(self) -> Optional[float]:
        if self.rng.random() < self.cfg.prob_wall_jump_on_miss:
            mu, sig = (
                self.cfg.wall_jump_mode1_mm
                if self.rng.random() < self.cfg.prob_mode1_vs_mode2
                else self.cfg.wall_jump_mode2_mm
            )
            return self._quantize_mm(max(161.0, self.rng.gauss(mu, sig)))
        return None

    def simulate_raw_reading_mm(
        self,
        sim_cone_nearest_mm: Optional[float],
        face_incidence_deg: Optional[float] = None,
        hit_body: str = "cube",
        include_episode_bias: bool = True,
    ) -> Optional[float]:
        if sim_cone_nearest_mm is None or sim_cone_nearest_mm <= 0 or hit_body == "none":
            return self.sample_jump_or_timeout()

        d_cal = float(sim_cone_nearest_mm)
        if not self.cfg.input_includes_offset:
            d_cal += self.cfg.range_offset_mm

        if d_cal <= self.cfg.dead_zone_cutoff_mm:
            if self.rng.random() < 0.08:
                return None
            return self.rng.choice([17.0, 20.0, 20.0, 20.0])

        if hit_body == "hand":
            if self.rng.random() > self.cfg.hand_echo_survival_prob:
                return self.sample_jump_or_timeout()
        elif hit_body == "cube":
            p_echo = self.echo_probability(d_cal, face_incidence_deg)
            if self.rng.random() > p_echo:
                return self.sample_jump_or_timeout()

        bias = self._episode_bias_mm if include_episode_bias else 0.0
        noisy_mm = d_cal + bias + self.rng.gauss(0.0, self.cfg.sigma_step_mm)
        return self._quantize_mm(max(17.0, noisy_mm))

    def filter_for_rl_observation(self, raw_us_mm: Optional[float]) -> Tuple[float, float]:
        if raw_us_mm is None or math.isnan(raw_us_mm):
            self._history.clear()
            return (0.0, 0.0)

        val = float(raw_us_mm)
        if val <= self.cfg.dead_zone_cutoff_mm or val > self.cfg.max_valid_palm_mm:
            self._history.clear()
            return (0.0, 0.0)

        self._history.append(val)
        if len(self._history) > 3:
            self._history.pop(0)
        sorted_h = sorted(self._history)
        med_mm = sorted_h[len(sorted_h) // 2]
        return (med_mm / self.cfg.max_valid_palm_mm, 1.0)

    def step_100hz(
        self,
        sim_cone_nearest_mm: Optional[float],
        face_incidence_deg: Optional[float] = None,
        hit_body: str = "cube",
    ) -> Tuple[Optional[float], Tuple[float, float]]:
        hold_steps = max(1, self.cfg.control_hz // self.cfg.ultrasonic_hz)
        if self._step_counter % hold_steps == 0:
            self._held_raw_mm = self.simulate_raw_reading_mm(sim_cone_nearest_mm, face_incidence_deg, hit_body)
            self._held_obs = self.filter_for_rl_observation(self._held_raw_mm)
        self._step_counter += 1
        return self._held_raw_mm, self._held_obs


class CalibratedAprilTagCameraWrapper:
    """Implements Joe's `top` + `side` dual-camera AprilTag pose estimation and `cube_pose_fused` fusion."""

    TOP_CAM_POS_M = T_WORLD_CAM_OPENCV["top"][:3, 3].copy()
    SIDE_CAM_POS_M = T_WORLD_CAM_OPENCV["side"][:3, 3].copy()

    def __init__(self, config: Optional[CameraAprilTagReal2SimConfig] = None, seed: Optional[int] = None):
        self.cfg = config or CameraAprilTagReal2SimConfig()
        self.rng = random.Random(None if seed is None else seed + 100_003)
        self._pos_bias = np.zeros(3, dtype=np.float64)
        self._top_pos_bias = np.zeros(3, dtype=np.float64)
        self._side_pos_bias = np.zeros(3, dtype=np.float64)
        self._rot_bias_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._top_rot_bias_mat = np.eye(3, dtype=np.float64)
        self._side_rot_bias_mat = np.eye(3, dtype=np.float64)
        self._last_cube_pos = np.array([0.17, -0.01, 0.18], dtype=np.float64)
        self._last_cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_cube_qvel = np.zeros(6, dtype=np.float64)
        self._last_red_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._step_counter: int = 0
        self._occluded_streak: int = 0
        self._tag_visible: bool = True
        self._last_tags_top: List[int] = [0]
        self._last_tags_side: List[int] = [5]

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat / max(1e-8, float(np.linalg.norm(quat)))
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _mat_to_quat(R: np.ndarray) -> np.ndarray:
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, R.flatten())
        return q / max(1e-8, float(np.linalg.norm(q)))

    @staticmethod
    def _quat_to_red_normal(quat: np.ndarray) -> np.ndarray:
        norm = max(1e-8, float(np.linalg.norm(quat)))
        w, x, y, z = quat / norm
        return np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )

    def _sample_small_rot_mat(self, sigma_deg: float) -> np.ndarray:
        half = 0.5 * math.radians(sigma_deg)
        q = np.array(
            [1.0, self.rng.gauss(0.0, half), self.rng.gauss(0.0, half), self.rng.gauss(0.0, half)],
            dtype=np.float64,
        )
        return self._quat_to_mat(q)

    def reset(
        self,
        seed: Optional[int] = None,
        init_pos: Optional[np.ndarray] = None,
        init_quat: Optional[np.ndarray] = None,
    ) -> None:
        if seed is not None:
            self.rng.seed(seed + 100_003)
        self._pos_bias = np.array(
            [self.rng.gauss(0.0, self.cfg.pos_episode_bias_sigma_m) for _ in range(3)],
            dtype=np.float64,
        )
        self._top_pos_bias = np.array(
            [self.rng.gauss(0.0, self.cfg.pos_episode_bias_sigma_m) for _ in range(3)],
            dtype=np.float64,
        )
        self._side_pos_bias = np.array(
            [self.rng.gauss(0.0, self.cfg.pos_episode_bias_sigma_m) for _ in range(3)],
            dtype=np.float64,
        )
        half_bias = 0.5 * math.radians(self.cfg.rot_episode_bias_deg)
        rb = np.array(
            [1.0, self.rng.gauss(0.0, half_bias), self.rng.gauss(0.0, half_bias), self.rng.gauss(0.0, half_bias)],
            dtype=np.float64,
        )
        self._rot_bias_quat = rb / max(1e-8, float(np.linalg.norm(rb)))
        self._top_rot_bias_mat = self._sample_small_rot_mat(self.cfg.rot_episode_bias_deg)
        self._side_rot_bias_mat = self._sample_small_rot_mat(self.cfg.rot_episode_bias_deg)
        self._step_counter = 0
        self._occluded_streak = 0
        self._tag_visible = True
        self._last_tags_top = [0]
        self._last_tags_side = [5]
        if init_pos is not None:
            self._last_cube_pos = np.asarray(init_pos, dtype=np.float64).copy()
        if init_quat is not None:
            q = np.asarray(init_quat, dtype=np.float64).copy()
            q /= max(1e-8, float(np.linalg.norm(q)))
            self._last_cube_quat = q
            self._last_red_normal = self._quat_to_red_normal(q)
        self._last_cube_qvel = np.zeros(6, dtype=np.float64)

    def _fuse_top_and_side_cameras(
        self,
        true_pos: np.ndarray,
        true_quat: np.ndarray,
        tags_top: List[int],
        tags_side: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Joe's `cube_pose_fused` (`real2sim/cube/cube_tags.py` lines 262-291):
        Fuses per-camera `T_world_cube` estimates from `top` and `side` cameras using
        weighted least-squares (`W = I - (1 - depth_weight) * outer(ray, ray)`, `depth_weight = 0.1`)
        for position and SVD `average_rotations(Rs)` for orientation.
        """
        R_true = self._quat_to_mat(true_quat)
        Rs: List[np.ndarray] = []
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        dw = self.cfg.depth_weight

        for cam_pos, pos_bias, rot_bias_mat, vis_ids in (
            (self.TOP_CAM_POS_M, self._top_pos_bias, self._top_rot_bias_mat, tags_top),
            (self.SIDE_CAM_POS_M, self._side_pos_bias, self._side_rot_bias_mat, tags_side),
        ):
            n_tags = len(vis_ids)
            if n_tags <= 0:
                continue
            scale = 1.0 / math.sqrt(float(n_tags))
            ray = true_pos - cam_pos
            ray_u = ray / max(1e-8, float(np.linalg.norm(ray)))
            iso_jit = np.array(
                [self.rng.gauss(0.0, self.cfg.pos_step_jitter_sigma_m * scale) for _ in range(3)],
                dtype=np.float64,
            )
            depth_jit = self.rng.gauss(0.0, self.cfg.depth_step_jitter_sigma_m * scale) * ray_u
            p_cam = true_pos + pos_bias + iso_jit + depth_jit

            # Exact `cube_pose_fused` weighting matrix: trusted across viewing plane (1.0), down-weighted along depth (0.1)
            W = np.eye(3, dtype=np.float64) - (1.0 - dw) * np.outer(ray_u, ray_u)
            A += W
            b += W @ p_cam

            rot_jit_deg = self.cfg.rot_step_jitter_deg * scale
            if n_tags == 1 and self.rng.random() < self.cfg.single_tag_ippe_flip_prob:
                rot_jit_deg *= 3.5
            R_cam = R_true @ rot_bias_mat @ self._sample_small_rot_mat(rot_jit_deg)
            Rs.append(R_cam)

        if Rs and np.linalg.det(A) > 1e-9:
            p_fused = np.linalg.solve(A, b)
            R_fused = average_rotations(Rs)
            q_fused = self._mat_to_quat(R_fused)
            if np.dot(q_fused, true_quat) < 0:
                q_fused = -q_fused
            return p_fused, q_fused

        # Fallback if called without per-camera tag lists
        p_fallback = (
            true_pos
            + self._pos_bias
            + np.array([self.rng.gauss(0.0, self.cfg.pos_step_jitter_sigma_m) for _ in range(3)], dtype=np.float64)
        )
        half_ang = 0.5 * math.radians(self.cfg.rot_step_jitter_deg)
        dq = np.array(
            [1.0, self.rng.gauss(0.0, half_ang), self.rng.gauss(0.0, half_ang), self.rng.gauss(0.0, half_ang)],
            dtype=np.float64,
        )
        dq /= max(1e-8, float(np.linalg.norm(dq)))
        q_true_norm = true_quat / max(1e-8, float(np.linalg.norm(true_quat)))
        q_fallback = self._quat_mul(self._quat_mul(q_true_norm, self._rot_bias_quat), dq)
        q_fallback /= max(1e-8, float(np.linalg.norm(q_fallback)))
        if np.dot(q_fallback, true_quat) < 0:
            q_fallback = -q_fallback
        return p_fallback, q_fallback

    def perturb_full_cube_state(
        self,
        true_pos: np.ndarray,
        true_quat: np.ndarray,
        true_qvel: np.ndarray,
        fingers_occluding: bool = False,
        tags_top: Optional[List[int]] = None,
        tags_side: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """Perturb 3D cube pos, 4D cube quat, 6D cube vel, 3D red-face normal, and 1D up alignment at 30 Hz."""
        hold_steps = max(1, self.cfg.control_hz // self.cfg.camera_hz)
        if self._step_counter % hold_steps == 0:
            if tags_top is not None and tags_side is not None:
                self._last_tags_top = list(tags_top)
                self._last_tags_side = list(tags_side)

            if self._occluded_streak > 0:
                self._occluded_streak -= 1
                self._tag_visible = False
            elif tags_top is not None and tags_side is not None and (len(tags_top) + len(tags_side) == 0):
                # Both top and side cameras have all 6 AprilTags occluded by fingers in MuJoCo
                self._tag_visible = False
            else:
                p_drop = self.cfg.finger_occlusion_dropout_prob if fingers_occluding else self.cfg.baseline_dropout_prob
                if self.rng.random() < p_drop:
                    self._occluded_streak = self.rng.randint(0, 1)
                    self._tag_visible = False
                else:
                    self._tag_visible = True

            if self._tag_visible:
                if tags_top is not None and tags_side is not None:
                    noisy_pos, noisy_quat = self._fuse_top_and_side_cameras(true_pos, true_quat, tags_top, tags_side)
                else:
                    noisy_pos, noisy_quat = self._fuse_top_and_side_cameras(true_pos, true_quat, [], [])

                v_noise = [self.rng.gauss(0.0, self.cfg.vel_lin_jitter_sigma_m_s) for _ in range(3)] + [
                    self.rng.gauss(0.0, self.cfg.vel_ang_jitter_sigma_rad_s) for _ in range(3)
                ]
                noisy_qvel = true_qvel + np.array(v_noise, dtype=np.float64)
                self._last_cube_pos = noisy_pos
                self._last_cube_quat = noisy_quat
                self._last_cube_qvel = noisy_qvel
                self._last_red_normal = self._quat_to_red_normal(noisy_quat)
            else:
                self._last_cube_qvel = 0.5 * self._last_cube_qvel

        self._step_counter += 1
        return (
            self._last_cube_pos.copy(),
            self._last_cube_quat.copy(),
            self._last_cube_qvel.copy(),
            self._last_red_normal.copy(),
            float(self._last_red_normal[2]),
            self._tag_visible,
        )


def apply_physical_joint_roms_to_base_env(
    base_env: Any,
    *,
    apply_joint_map_offsets: bool = True,
) -> dict[str, Tuple[float, float]]:
    """Apply physical ORCA hand ROM limits (and `real2sim/hand/joint_map.py` offsets) to `base_env.model`
    BEFORE wrapping with `RelativeJointTargetAction`.

    In `joint_map.py` (`fullcheck01`):
      - `ring_abd`: `sim_deg = core_deg - 10.0` (core [-30, +30] -> sim [-40, +20])
      - `pinky_abd`: `sim_deg = core_deg - 14.0` (core [-30, +30] -> sim [-44, +16])
      - `t-mcp`: `[-5.0, +40.0]` deg
      - `t-abd`: `[-20.0, +35.0]` deg
    """
    applied: dict[str, Tuple[float, float]] = {}
    if not hasattr(base_env, "model"):
        return applied

    m = base_env.model
    if apply_joint_map_offsets:
        rom_overrides_deg = {
            "t-mcp": (-5.0, 40.0),
            "t-abd": (-20.0, 35.0),
            "r-abd": (core_to_sim_deg("ring_abd", -30.0), core_to_sim_deg("ring_abd", 30.0)),
            "p-abd": (core_to_sim_deg("pinky_abd", -30.0), core_to_sim_deg("pinky_abd", 30.0)),
        }
    else:
        rom_overrides_deg = {
            "t-mcp": (-5.0, 40.0),
            "t-abd": (-20.0, 35.0),
            "p-abd": (-30.0, 30.0),
        }

    for a_i in range(m.nu):
        aname = m.actuator(a_i).name
        for key, (lo_deg, hi_deg) in rom_overrides_deg.items():
            if key in aname:
                lo_rad, hi_rad = math.radians(lo_deg), math.radians(hi_deg)
                m.actuator_ctrlrange[a_i, 0] = max(float(m.actuator_ctrlrange[a_i, 0]), lo_rad)
                m.actuator_ctrlrange[a_i, 1] = min(float(m.actuator_ctrlrange[a_i, 1]), hi_rad)
                if hasattr(base_env, "action_low"):
                    base_env.action_low[a_i] = np.float32(m.actuator_ctrlrange[a_i, 0])
                    base_env.action_high[a_i] = np.float32(m.actuator_ctrlrange[a_i, 1])
                applied[aname] = (lo_deg, hi_deg)
    if hasattr(base_env, "action_low"):
        base_env.action_space = spaces.Box(
            low=base_env.action_low,
            high=base_env.action_high,
            dtype=np.float32,
        )
    return applied


class Real2SimSensorAndKinematicWrapper(gym.Wrapper):
    """Gymnasium wrapper implementing Joe's `real2sim-cameras` setup:
    - Scene `us_0..us_24` ultrasonic rangefinders with 1 mm neighbor-ray specular normal check (`camlib.py` / `twin.py`)
    - Calibrated `top` & `side` camera AprilTag visibility (`cube_tags_layout.json`) and `cube_pose_fused` fusion
    - Locked wrist at `-9.6 deg` (or configurable `wrist_lock_deg`) and `joint_map.py` physical ROM limits.
    """

    US_POS_WORLD_M = np.array([0.1069, -0.0107, 0.2010], dtype=np.float64)
    US_DIR_WORLD = np.array([0.965, -0.262, 0.0], dtype=np.float64)

    def __init__(
        self,
        env: gym.Env,
        *,
        lock_wrist: bool = True,
        wrist_lock_deg: float = -9.6,
        real_joint_roms: bool = True,
        apply_joint_map_offsets: bool = True,
        real2sim_noise: bool = True,
        append_ultrasonic_obs: bool = False,
        render_camera_images: bool = False,
        render_width: int = 320,
        render_height: int = 180,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.lock_wrist = lock_wrist
        self.wrist_lock_deg = float(wrist_lock_deg)
        self.real_joint_roms = real_joint_roms
        self.apply_joint_map_offsets = apply_joint_map_offsets
        self.real2sim_noise = real2sim_noise
        self.append_ultrasonic_obs = append_ultrasonic_obs
        self.render_camera_images = render_camera_images
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self._cam_renderer: Any = None

        self.us_sensor = CalibratedUltrasonicSensor(seed=seed)
        self.cam_sensor = CalibratedAprilTagCameraWrapper(seed=seed)

        base_env = env.unwrapped
        self._wrist_act_idx = 0
        self._wrist_qpos_adr: Optional[int] = None
        self._us_site_ids: List[int] = []
        self._top_cam_id: int = -1
        self._side_cam_id: int = -1

        if hasattr(base_env, "model"):
            m = base_env.model
            for a_i in range(m.nu):
                if "wrist" in m.actuator(a_i).name:
                    self._wrist_act_idx = a_i
                    trnid = int(m.actuator_trnid[a_i, 0])
                    if 0 <= trnid < m.njnt:
                        self._wrist_qpos_adr = int(m.jnt_qposadr[trnid])
                    break

            for s_i in range(m.nsensor):
                if m.sensor(s_i).name.startswith("us_"):
                    self._us_site_ids.append(int(m.sensor_objid[s_i]))
            self._top_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "top")
            self._side_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "side")

            if self.lock_wrist and self._wrist_qpos_adr is not None and hasattr(base_env, "_default_hand_qpos"):
                w_rad = float(
                    np.clip(
                        math.radians(self.wrist_lock_deg),
                        m.actuator_ctrlrange[self._wrist_act_idx, 0],
                        m.actuator_ctrlrange[self._wrist_act_idx, 1],
                    )
                )
                base_env._default_hand_qpos[self._wrist_qpos_adr] = w_rad

            if self.real_joint_roms:
                apply_physical_joint_roms_to_base_env(
                    base_env, apply_joint_map_offsets=self.apply_joint_map_offsets
                )
                cur = self.env
                while cur is not None and cur is not base_env:
                    if hasattr(cur, "_low") and hasattr(cur, "_high") and hasattr(cur, "_span"):
                        cur._low = np.asarray(base_env.action_low, dtype=np.float64)
                        cur._high = np.asarray(base_env.action_high, dtype=np.float64)
                        cur._span = np.maximum(cur._high - cur._low, 1e-6)
                    cur = getattr(cur, "env", None)

        # Precompute 6 AprilTag local positions & outward normals in the cube frame from Joe's `cube_tags_layout.json`
        self._tag_ids = sorted(T_CUBE_TAG.keys())
        self._tag_local_pos = np.array([T_CUBE_TAG[i][:3, 3] for i in self._tag_ids], dtype=np.float64)
        self._tag_local_normal = np.array([T_CUBE_TAG[i][:3, 2] for i in self._tag_ids], dtype=np.float64)

        # Fallback 25-ray 15-deg cone (matching `camlib.ultrasonic_rays`) when `us_0..us_24` are not in scene XML
        z_ax = self.US_DIR_WORLD / np.linalg.norm(self.US_DIR_WORLD)
        x_ax = np.cross(np.array([0.0, 0.0, 1.0]), z_ax)
        x_ax /= max(1e-8, float(np.linalg.norm(x_ax)))
        y_ax = np.cross(z_ax, x_ax)
        rays = [z_ax]
        for ring_deg in (5.0, 10.0, 15.0):
            th = math.radians(ring_deg)
            for az_i in range(8):
                ph = az_i * (0.25 * math.pi)
                d_vec = math.cos(th) * z_ax + math.sin(th) * (math.cos(ph) * x_ax + math.sin(ph) * y_ax)
                rays.append(d_vec / np.linalg.norm(d_vec))
        self._cone_rays = np.asarray(rays, dtype=np.float64)
        self._ray_geomgroup = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)

        if self.append_ultrasonic_obs:
            old_space = self.env.observation_space
            low = np.concatenate([old_space.low, np.array([0.0, 0.0], dtype=np.float64)])
            high = np.concatenate([old_space.high, np.array([1.0, 1.0], dtype=np.float64)])
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def _apply_locked_wrist_pose(self) -> None:
        """Lock wrist at `wrist_lock_deg` (default `-9.6 deg` matching `policy_run07/08.yaml`)."""
        if not self.lock_wrist or abs(self.wrist_lock_deg) < 1e-8:
            return
        base_env = self.env.unwrapped
        if not hasattr(base_env, "model") or not hasattr(base_env, "data"):
            return
        m, d = base_env.model, base_env.data
        w_rad = float(
            np.clip(
                math.radians(self.wrist_lock_deg),
                m.actuator_ctrlrange[self._wrist_act_idx, 0],
                m.actuator_ctrlrange[self._wrist_act_idx, 1],
            )
        )
        if self._wrist_qpos_adr is not None:
            d.qpos[self._wrist_qpos_adr] = w_rad
        d.ctrl[self._wrist_act_idx] = w_rad
        mujoco.mj_forward(m, d)

        cur = self.env
        while cur is not None and cur is not base_env:
            if hasattr(cur, "_target") and len(cur._target) > self._wrist_act_idx:
                cur._target[self._wrist_act_idx] = w_rad
            cur = getattr(cur, "env", None)

    def _detect_visible_tags_top_and_side(self) -> Tuple[List[int], List[int]]:
        """Check which AprilTags (`id 0..5` from `cube_tags_layout.json`) are visible in Joe's `top` and `side` cameras."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, "model") or not hasattr(base_env, "data"):
            return [0], [5]

        m, d = base_env.model, base_env.data
        cube_body_id = getattr(base_env, "_cube_body_id", -1)
        if cube_body_id < 0 or not hasattr(base_env, "_cube_quat"):
            return [0], [5]

        cube_pos = d.xpos[cube_body_id].copy()
        R_cube = self.cam_sensor._quat_to_mat(base_env._cube_quat())

        # Transform all 6 measured AprilTag centers & normals (`T_cube_tag`) to world frame
        tag_world_pos = cube_pos + (R_cube @ self._tag_local_pos.T).T
        tag_world_norm = (R_cube @ self._tag_local_normal.T).T

        top_pos = d.cam_xpos[self._top_cam_id].copy() if self._top_cam_id >= 0 else self.cam_sensor.TOP_CAM_POS_M
        side_pos = d.cam_xpos[self._side_cam_id].copy() if self._side_cam_id >= 0 else self.cam_sensor.SIDE_CAM_POS_M
        cos_limit = math.cos(math.radians(self.cam_sensor.cfg.max_tag_incidence_deg))

        vis_per_cam: List[List[int]] = []
        for cam_pos in (top_pos, side_pos):
            vis_ids: List[int] = []
            for idx, tid in enumerate(self._tag_ids):
                t_pos = tag_world_pos[idx]
                t_norm = tag_world_norm[idx]
                to_cam = cam_pos - t_pos
                dist_to_cam = float(np.linalg.norm(to_cam))
                if dist_to_cam < 1e-6:
                    continue
                view_dir = to_cam / dist_to_cam
                if float(np.dot(t_norm, view_dir)) < cos_limit:
                    continue
                # Raycast from camera to tag center in MuJoCo to check finger occlusion
                geomid = np.array([-1], dtype=np.int32)
                hit_d = float(mujoco.mj_ray(m, d, cam_pos, -view_dir, self._ray_geomgroup, 1, -1, geomid))
                if hit_d > 0.0 and geomid[0] >= 0:
                    hit_bid = int(m.geom_bodyid[geomid[0]])
                    if hit_bid == cube_body_id or hit_d >= dist_to_cam - 0.006:
                        vis_ids.append(tid)
            vis_per_cam.append(vis_ids)
        return vis_per_cam[0], vis_per_cam[1]

    def _cast_ultrasonic_cone(self) -> Tuple[Optional[float], Optional[float], str]:
        """Joe's `real2sim/recording/twin.py:echo` & `camlib.py:ultrasonic_reading_specular`:
        Uses `us_0..us_24` sites in `scene_right_cube_orientation_real2sim.xml` (with two 1 mm offset
        neighbor rays per ray to measure surface normal incidence <= 25 deg).
        """
        base_env = self.env.unwrapped
        if not hasattr(base_env, "model") or not hasattr(base_env, "data"):
            return None, None, "none"

        m, d = base_env.model, base_env.data
        cube_body_id = getattr(base_env, "_cube_body_id", -1)
        R_cube: Optional[np.ndarray] = None
        if hasattr(base_env, "_cube_quat"):
            R_cube = self.cam_sensor._quat_to_mat(base_env._cube_quat())

        best_dist_m: Optional[float] = None
        best_inc_deg: Optional[float] = None
        best_body: str = "none"
        spec_limit_deg = self.us_sensor.cfg.hard_incidence_limit_deg

        if self._us_site_ids:
            # Exact `twin.py:echo` loop over `us_0..us_24` sites in `scene_right_cube_orientation_real2sim.xml`
            for sid in self._us_site_ids:
                o = d.site_xpos[sid].copy()
                R = d.site_xmat[sid].reshape(3, 3)
                v = R[:, 2].copy()
                g = np.array([-1], dtype=np.int32)
                t = float(mujoco.mj_ray(m, d, o, v, self._ray_geomgroup, 1, -1, g))
                if t <= 0.0 or g[0] < 0:
                    continue

                hit_bid = int(m.geom_bodyid[g[0]])
                if hit_bid == cube_body_id and R_cube is not None:
                    max_cos = max(abs(float(np.dot(R_cube[:, axis], v))) for axis in range(3))
                    inc_deg = math.degrees(math.acos(min(1.0, max_cos)))
                else:
                    # Two 1 mm offset rays beside the primary ray to measure local surface normal (`camlib.py` line 183)
                    p = o + t * v
                    pts = []
                    for e in (R[:, 0], R[:, 1]):
                        g2 = np.array([-1], dtype=np.int32)
                        t2 = float(mujoco.mj_ray(m, d, o + 1e-3 * e, v, self._ray_geomgroup, 1, -1, g2))
                        if t2 <= 0.0 or g2[0] != g[0]:
                            break
                        pts.append(o + 1e-3 * e + t2 * v)
                    if len(pts) < 2:
                        continue
                    n = np.cross(pts[0] - p, pts[1] - p)
                    n_norm = float(np.linalg.norm(n))
                    if n_norm < 1e-12:
                        continue
                    n /= n_norm
                    inc_deg = math.degrees(math.acos(min(1.0, abs(float(np.dot(n, v))))))

                if inc_deg > spec_limit_deg:
                    continue

                if hit_bid == cube_body_id:
                    hit_body = "cube"
                elif hit_bid == 0 or "bench" in m.geom(g[0]).name or "board" in m.geom(g[0]).name:
                    hit_body = "bench"
                else:
                    hit_body = "hand"

                if best_dist_m is None or t < best_dist_m:
                    best_dist_m = t
                    best_inc_deg = inc_deg
                    best_body = hit_body
        else:
            # Fallback when using plain scene without `us_0..us_24` sites
            for ray_dir in self._cone_rays:
                geomid = np.array([-1], dtype=np.int32)
                dist_m = float(mujoco.mj_ray(m, d, self.US_POS_WORLD_M, ray_dir, self._ray_geomgroup, 1, -1, geomid))
                if dist_m <= 0.0 or geomid[0] < 0:
                    continue
                hit_bid = int(m.geom_bodyid[geomid[0]])
                if hit_bid == cube_body_id:
                    hit_body = "cube"
                    if R_cube is not None:
                        max_cos = max(abs(float(np.dot(R_cube[:, axis], ray_dir))) for axis in range(3))
                        inc_deg = math.degrees(math.acos(min(1.0, max_cos)))
                    else:
                        inc_deg = 0.0
                    if inc_deg > spec_limit_deg:
                        continue
                elif hit_bid == 0:
                    hit_body = "bench"
                    inc_deg = 0.0
                else:
                    hit_body = "hand"
                    inc_deg = 0.0
                if best_dist_m is None or dist_m < best_dist_m:
                    best_dist_m = dist_m
                    best_inc_deg = inc_deg
                    best_body = hit_body

        if best_dist_m is None:
            return None, None, "none"

        dist_mm = best_dist_m * 1000.0 + 2.9
        return dist_mm, best_inc_deg, best_body

    def _process_obs(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        obs = obs.copy()
        surf_dist_mm, inc_deg, hit_body = self._cast_ultrasonic_cone()
        raw_us_mm, (us_d_norm, us_valid) = self.us_sensor.step_100hz(surf_dist_mm, inc_deg, hit_body)
        info["ultrasonic_raw_mm"] = raw_us_mm
        info["ultrasonic_d_norm"] = us_d_norm
        info["ultrasonic_valid"] = us_valid
        info["ultrasonic_hit_body"] = hit_body

        if self.render_camera_images:
            base_env = self.env.unwrapped
            if hasattr(base_env, "model") and hasattr(base_env, "data"):
                if self._cam_renderer is None:
                    from orca_vla.cameras import VLACameraRenderer

                    self._cam_renderer = VLACameraRenderer(
                        base_env.model, base_env.data, width=self.render_width, height=self.render_height
                    )
                info["camera_top_rgb"] = self._cam_renderer.render("top")
                info["camera_side_rgb"] = self._cam_renderer.render("side")

        if self.real2sim_noise and obs.shape[0] >= 51:
            mean_finger_flex = float(np.mean(np.abs(obs[0:16])))
            fingers_occluding = mean_finger_flex > 0.35
            tags_top, tags_side = self._detect_visible_tags_top_and_side()
            noisy_pos, noisy_quat, noisy_qvel, noisy_n, noisy_align, tag_vis = self.cam_sensor.perturb_full_cube_state(
                obs[17:20],
                obs[20:24],
                obs[41:47],
                fingers_occluding=fingers_occluding,
                tags_top=tags_top,
                tags_side=tags_side,
            )
            obs[17:20] = noisy_pos
            obs[20:24] = noisy_quat
            obs[41:47] = noisy_qvel
            obs[47:50] = noisy_n
            obs[50] = noisy_align
            info["apriltag_visible"] = tag_vis
            info["tags_top"] = tags_top
            info["tags_side"] = tags_side
            info["observed_red_face_up_angle_rad"] = float(math.acos(np.clip(noisy_align, -1.0, 1.0)))

        if self.append_ultrasonic_obs:
            obs = np.concatenate([obs, np.array([us_d_norm, us_valid], dtype=np.float64)])
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.lock_wrist and abs(self.wrist_lock_deg) >= 1e-8:
            self._apply_locked_wrist_pose()
            base_env = self.env.unwrapped
            if hasattr(base_env, "_get_obs"):
                raw_obs = base_env._get_obs()
                cur = self.env
                if hasattr(cur, "_augment"):
                    obs = cur._augment(raw_obs)
                else:
                    obs = raw_obs
        self.us_sensor.reset(seed=seed)
        if obs.shape[0] >= 51:
            self.cam_sensor.reset(seed=seed, init_pos=obs[17:20], init_quat=obs[20:24])
        else:
            self.cam_sensor.reset(seed=seed)
        return self._process_obs(obs, info), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).copy()
        if self.lock_wrist and act.shape[0] > self._wrist_act_idx:
            act[self._wrist_act_idx] = 0.0
        obs, reward, terminated, truncated, info = self.env.step(act)
        return self._process_obs(obs, info), reward, terminated, truncated, info
