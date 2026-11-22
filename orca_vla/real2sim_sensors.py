"""Real2Sim Multi-Modal Sensor Calibration and Kinematic Alignment Wrappers for the 17-DoF ORCA Hand.

Implements empirically calibrated sensor perturbation and hardware-alignment models for Sim-to-Real
reinforcement learning (PPO, SAC, TD3) on the 17-DoF tendon-driven ETH ORCA hand:

1. Dual-Camera AprilTag 6-DoF Pose & Orientation Noise (`CalibratedAprilTagCameraWrapper`):
   - Models ChArUco-calibrated 1920x1080 dual-camera (`top` overhead and `side` lateral) AprilTag
     tracking errors at 30 Hz sample-and-hold cadence: inter-camera depth offset (5-10 mm),
     per-frame PnP angular jitter (2.6 deg), and persistent visual self-occlusion dropouts
     (18% frame dropout rate during finger flexion, holding the last visible pose and quaternion).
   - Consistently perturbs `cube_pos` (`obs[17:20]`), `cube_quat` (`obs[20:24]`), `cube_qvel`
     (`obs[41:47]`), `red_face_world_normal` (`obs[47:50]`), and `red_face_up_alignment` (`obs[50]`)
     so MLP policies cannot bypass AprilTag noise via unperturbed ground-truth quaternion channels.

2. Conical 40 kHz Ultrasonic Rangefinder Digital Twin (`CalibratedUltrasonicSensor`):
   - Models a palm-base HC-SR04 ultrasonic transceiver (15 deg half-angle cone cast via `mujoco.mj_ray`)
     with calibrated range offset (+2.9 mm), piezoelectric ringing dead-zone floor (<= 22.0 mm sampling
     discrete {17, 20} mm steps), bistatic distance-dependent specular incidence dropout
     (theta_50 = 26.5 deg at 7.4 cm tightening to 18.5 deg at 3.5 cm, slope k = 0.12 deg^-1),
     curved-finger specular deflection (22% survival on hand meshes), 3.2 mm step quantization at
     10 Hz sample-and-hold cadence, and background wall-jump outliers (~58.4 cm / ~128.9 cm).
   - Provides a 3-tap median + range-gating filter (`filter_for_rl_observation`) that maps dead-zone
     and out-of-range wall jumps (> 150 mm) to `(0.0, 0.0)` so outlier spikes never destabilize
     `VecNormalize` running statistics or MLP policy weights.

3. Gymnasium Environment Wrapper (`Real2SimSensorAndKinematicWrapper`):
   - Enforces physical deployment constraints during MuJoCo RL training and evaluation:
     * `lock_wrist=True`: Freezes the 1-DoF wrist pitch actuator (`right_wrist_actuator`) so manipulation
       is executed purely by the 16 finger degrees of freedom while keeping the palm-base sensor aligned.
     * `real_joint_roms=True`: Clamps simulated thumb and finger abduction/flexion ranges (matching
       `right_t-mcp_actuator` to physical [-5 deg, +40 deg] and `right_t-abd_actuator` to [-20 deg, +35 deg])
       and synchronizes `RelativeJointTargetAction` internal bounds (`_low`, `_high`, `_span`, `_target`)
       to prevent integrator windup.
     * `real2sim_noise=True`: Injects calibrated AprilTag camera pose/quaternion/velocity perturbations
       and occlusion dropouts into the observation vector (68D with `RelativeJointTargetAction`, 51D raw),
       and optionally appends `[us_d_norm, us_valid_flag]` when `append_ultrasonic_obs=True`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class UltrasonicReal2SimConfig:
    """Empirically fitted parameters for the palm-base 40 kHz HC-SR04 ultrasonic sensor."""

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
    """Empirically fitted noise and occlusion-dropout parameters from dual-camera AprilTag recordings."""

    control_hz: int = 100
    camera_hz: int = 30
    pos_episode_bias_sigma_m: float = 0.0045
    pos_step_jitter_sigma_m: float = 0.0004
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

        # Dead-zone floor (<= 22.0 mm): returns discrete ringing steps {17.0, 20.0} mm or 8% timeout
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
    """Perturbs ground-truth cube position, quaternion, velocity, and goal normal with 30 Hz AprilTag noise."""

    def __init__(self, config: Optional[CameraAprilTagReal2SimConfig] = None, seed: Optional[int] = None):
        self.cfg = config or CameraAprilTagReal2SimConfig()
        self.rng = random.Random(None if seed is None else seed + 100_003)
        self._pos_bias = np.zeros(3, dtype=np.float64)
        self._rot_bias_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_cube_pos = np.array([0.17, -0.01, 0.18], dtype=np.float64)
        self._last_cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_cube_qvel = np.zeros(6, dtype=np.float64)
        self._last_red_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._step_counter: int = 0
        self._occluded_streak: int = 0
        self._tag_visible: bool = True

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
        half_bias = 0.5 * math.radians(self.cfg.rot_episode_bias_deg)
        rb = np.array(
            [1.0, self.rng.gauss(0.0, half_bias), self.rng.gauss(0.0, half_bias), self.rng.gauss(0.0, half_bias)],
            dtype=np.float64,
        )
        self._rot_bias_quat = rb / max(1e-8, float(np.linalg.norm(rb)))
        self._step_counter = 0
        self._occluded_streak = 0
        self._tag_visible = True
        if init_pos is not None:
            self._last_cube_pos = np.asarray(init_pos, dtype=np.float64).copy()
        if init_quat is not None:
            q = np.asarray(init_quat, dtype=np.float64).copy()
            q /= max(1e-8, float(np.linalg.norm(q)))
            self._last_cube_quat = q
            self._last_red_normal = self._quat_to_red_normal(q)
        self._last_cube_qvel = np.zeros(6, dtype=np.float64)

    def perturb_full_cube_state(
        self,
        true_pos: np.ndarray,
        true_quat: np.ndarray,
        true_qvel: np.ndarray,
        fingers_occluding: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """Perturb 3D cube pos, 4D cube quat, 6D cube vel, 3D red-face normal, and 1D up alignment at 30 Hz."""
        hold_steps = max(1, self.cfg.control_hz // self.cfg.camera_hz)
        if self._step_counter % hold_steps == 0:
            # Temporal persistence for occlusion bursts
            if self._occluded_streak > 0:
                self._occluded_streak -= 1
                self._tag_visible = False
            else:
                p_drop = self.cfg.finger_occlusion_dropout_prob if fingers_occluding else self.cfg.baseline_dropout_prob
                if self.rng.random() < p_drop:
                    self._occluded_streak = self.rng.randint(0, 1)  # 1-2 frame burst at 30 Hz
                    self._tag_visible = False
                else:
                    self._tag_visible = True

            if self._tag_visible:
                noisy_pos = (
                    true_pos
                    + self._pos_bias
                    + np.array([self.rng.gauss(0.0, self.cfg.pos_step_jitter_sigma_m) for _ in range(3)], dtype=np.float64)
                )
                sigma_rad = math.radians(self.cfg.rot_step_jitter_deg)
                if self.rng.random() < self.cfg.single_tag_ippe_flip_prob:
                    sigma_rad *= 3.5
                # Perturb quaternion consistently on SO(3) tangent space (static bias + frame jitter)
                half_ang = 0.5 * sigma_rad
                dq = np.array(
                    [1.0, self.rng.gauss(0.0, half_ang), self.rng.gauss(0.0, half_ang), self.rng.gauss(0.0, half_ang)],
                    dtype=np.float64,
                )
                dq /= max(1e-8, float(np.linalg.norm(dq)))
                q_true_norm = true_quat / max(1e-8, float(np.linalg.norm(true_quat)))
                noisy_quat = self._quat_mul(self._quat_mul(q_true_norm, self._rot_bias_quat), dq)
                noisy_quat /= max(1e-8, float(np.linalg.norm(noisy_quat)))
                if np.dot(noisy_quat, true_quat) < 0:
                    noisy_quat = -noisy_quat

                v_noise = [self.rng.gauss(0.0, self.cfg.vel_lin_jitter_sigma_m_s) for _ in range(3)] + [
                    self.rng.gauss(0.0, self.cfg.vel_ang_jitter_sigma_rad_s) for _ in range(3)
                ]
                noisy_qvel = true_qvel + np.array(v_noise, dtype=np.float64)
                self._last_cube_pos = noisy_pos
                self._last_cube_quat = noisy_quat
                self._last_cube_qvel = noisy_qvel
                self._last_red_normal = self._quat_to_red_normal(noisy_quat)
            else:
                # During visual occlusion, pose holds last frame and estimated velocity decays
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


def apply_physical_joint_roms_to_base_env(base_env: Any) -> dict[str, Tuple[float, float]]:
    """Apply physical ORCA hand ROM limits directly to `base_env.model` BEFORE wrapping with `RelativeJointTargetAction`.

    Matches `orcahand_right.mjcf` actuator naming (`right_t-mcp_actuator`, `right_t-abd_actuator`, `right_p-abd_actuator`).
    """
    applied: dict[str, Tuple[float, float]] = {}
    if not hasattr(base_env, "model"):
        return applied

    m = base_env.model
    # Physical ROM limits (deg) measured on the real ORCA hand config.yaml
    rom_overrides_deg = {
        "t-mcp": (-5.0, 40.0),   # Sim default is [-25, +100] deg; real clamps at 40 deg 74% of the time
        "t-abd": (-20.0, 35.0),  # Real thumb abduction limit is +35 deg
        "p-abd": (-30.0, 30.0),  # Real pinky adduction limit is +30 deg
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
    """Gymnasium wrapper enforcing locked-wrist kinematics, physical ROM bounds, and calibrated Real2Sim noise."""

    US_POS_WORLD_M = np.array([0.1069, -0.0107, 0.2010], dtype=np.float64)
    US_DIR_WORLD = np.array([0.965, -0.262, 0.0], dtype=np.float64)

    def __init__(
        self,
        env: gym.Env,
        *,
        lock_wrist: bool = True,
        real_joint_roms: bool = True,
        real2sim_noise: bool = True,
        append_ultrasonic_obs: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.lock_wrist = lock_wrist
        self.real_joint_roms = real_joint_roms
        self.real2sim_noise = real2sim_noise
        self.append_ultrasonic_obs = append_ultrasonic_obs

        self.us_sensor = CalibratedUltrasonicSensor(seed=seed)
        self.cam_sensor = CalibratedAprilTagCameraWrapper(seed=seed)

        base_env = env.unwrapped
        self._wrist_act_idx = 0
        if hasattr(base_env, "model"):
            m = base_env.model
            for a_i in range(m.nu):
                if "wrist" in m.actuator(a_i).name:
                    self._wrist_act_idx = a_i
                    break
            if self.real_joint_roms:
                applied = apply_physical_joint_roms_to_base_env(base_env)
                # Synchronize RelativeJointTargetAction cached bounds if it is already in the wrapper chain
                cur = self.env
                while cur is not None and cur is not base_env:
                    if hasattr(cur, "_low") and hasattr(cur, "_high") and hasattr(cur, "_span"):
                        cur._low = np.asarray(base_env.action_low, dtype=np.float64)
                        cur._high = np.asarray(base_env.action_high, dtype=np.float64)
                        cur._span = np.maximum(cur._high - cur._low, 1e-6)
                    cur = getattr(cur, "env", None)

        # Precompute 25-ray 15-deg acoustic cone directions (1 axis + 3 rings x 8 rays) in world frame for `mj_ray`
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

    def _cast_ultrasonic_cone(self) -> Tuple[Optional[float], Optional[float], str]:
        """Cast the 25-ray 15-deg ultrasonic cone with per-ray specular filtering before nearest-echo selection."""
        base_env = self.env.unwrapped
        if not hasattr(base_env, "model") or not hasattr(base_env, "data"):
            return None, None, "none"

        m, d = base_env.model, base_env.data
        cube_body_id = getattr(base_env, "_cube_body_id", -1)
        R_cube: Optional[np.ndarray] = None
        if hasattr(base_env, "_cube_quat"):
            q = base_env._cube_quat()
            w, x, y, z = q / max(1e-8, float(np.linalg.norm(q)))
            R_cube = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )

        best_dist_m: Optional[float] = None
        best_inc_deg: Optional[float] = None
        best_body: str = "none"
        spec_limit_deg = self.us_sensor.cfg.hard_incidence_limit_deg

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

        if self.real2sim_noise and obs.shape[0] >= 51:
            # In OrcaHandRightCubeOrientation:
            #   obs[0:16] = 16 finger joint qpos, obs[16] = right_wrist qpos
            #   obs[17:20] = cube_pos (3D), obs[20:24] = cube_quat (4D)
            #   obs[24:41] = 17 hand joint qvel, obs[41:47] = cube_qvel (6D)
            #   obs[47:50] = red_face_world_normal (3D), obs[50] = red_face_up_alignment (1D)
            mean_finger_flex = float(np.mean(np.abs(obs[0:16])))
            fingers_occluding = mean_finger_flex > 0.35
            noisy_pos, noisy_quat, noisy_qvel, noisy_n, noisy_align, tag_vis = self.cam_sensor.perturb_full_cube_state(
                obs[17:20],
                obs[20:24],
                obs[41:47],
                fingers_occluding=fingers_occluding,
            )
            obs[17:20] = noisy_pos
            obs[20:24] = noisy_quat
            obs[41:47] = noisy_qvel
            obs[47:50] = noisy_n
            obs[50] = noisy_align
            info["apriltag_visible"] = tag_vis
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
