"""Unit tests for CalibratedUltrasonicSensor, CalibratedAprilTagCameraWrapper, Joe's real2sim-cameras scene, and Slurm script formatting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orca_vla.real2sim_sensors import (
    CalibratedAprilTagCameraWrapper,
    CalibratedUltrasonicSensor,
    Real2SimSensorAndKinematicWrapper,
    UltrasonicReal2SimConfig,
    core_to_sim_deg,
    sim_to_core_deg,
)


def test_ultrasonic_dead_zone_and_wall_jump_masking():
    for seed in range(50):
        sensor = CalibratedUltrasonicSensor(seed=seed)
        sensor.reset(seed=seed)
        raw_dz = sensor.simulate_raw_reading_mm(12.0, face_incidence_deg=0.0, hit_body="cube")
        assert raw_dz is None or raw_dz <= 22.0
        d_norm_dz, valid_dz = sensor.filter_for_rl_observation(raw_dz)
        assert d_norm_dz == 0.0 and valid_dz == 0.0

    sensor = CalibratedUltrasonicSensor(seed=123)
    sensor.reset(seed=123)

    d_norm_jump, valid_jump = sensor.filter_for_rl_observation(612.0)
    assert d_norm_jump == 0.0 and valid_jump == 0.0

    d_norm_ok, valid_ok = sensor.filter_for_rl_observation(54.4)
    assert 0.2 < d_norm_ok < 0.5 and valid_ok == 1.0


def test_ultrasonic_10hz_hold_cadence():
    cfg = UltrasonicReal2SimConfig(control_hz=100, ultrasonic_hz=10)
    sensor = CalibratedUltrasonicSensor(config=cfg, seed=7)
    sensor.reset(seed=7)

    readings = [sensor.step_100hz(50.0, face_incidence_deg=5.0, hit_body="cube")[0] for _ in range(10)]
    assert len(set(readings)) == 1


def test_camera_wrapper_perturbs_and_fuses_top_side_30hz():
    cam = CalibratedAprilTagCameraWrapper(seed=99)
    cam.reset(seed=99)

    p, q, v, n, a, vis = cam.perturb_full_cube_state(
        true_pos=np.array([0.17, -0.01, 0.18], dtype=np.float64),
        true_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        true_qvel=np.zeros(6, dtype=np.float64),
        fingers_occluding=False,
        tags_top=[0, 2],
        tags_side=[5],
    )
    assert len(p) == 3 and len(q) == 4 and len(v) == 6 and len(n) == 3 and isinstance(vis, bool)
    p2, q2, v2, n2, a2, vis2 = cam.perturb_full_cube_state(
        true_pos=np.array([0.18, 0.00, 0.18], dtype=np.float64),
        true_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        true_qvel=np.ones(6, dtype=np.float64),
        fingers_occluding=False,
        tags_top=[0, 2],
        tags_side=[5],
    )
    assert np.allclose(p, p2) and np.allclose(q, q2) and np.allclose(v, v2) and np.allclose(n, n2) and a == a2


def test_joint_map_and_real2sim_scene_wrapper():
    assert core_to_sim_deg("ring_abd", 0.0) == -10.0
    assert core_to_sim_deg("pinky_abd", 30.0) == 16.0
    assert core_to_sim_deg("index_abd", 15.0) == -15.0
    assert sim_to_core_deg("pinky_abd", 16.0) == 30.0

    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    sim_dir = repo_root.parent / "SEP730-Dexterous-Hand-Simulation" / "src"
    has_sim = importlib.util.find_spec("orca_sim") is not None
    if not has_sim:
        try:
            if (sim_dir / "orca_sim").exists():
                if str(sim_dir) not in sys.path:
                    sys.path.insert(0, str(sim_dir))
                has_sim = True
        except PermissionError:
            has_sim = False
    if has_sim:
        from orca_sim import OrcaHandRightCubeOrientation
        from orca_sim.wrappers import RelativeJointTargetAction

        env = OrcaHandRightCubeOrientation(
            version="v2",
            scene_file="scene_right_cube_orientation_real2sim.xml",
            initial_red_face="random",
            drop_height=0.13,
        )
        env = RelativeJointTargetAction(env, max_delta_frac=0.05)
        env = Real2SimSensorAndKinematicWrapper(
            env,
            lock_wrist=True,
            wrist_lock_deg=0.0,
            real_joint_roms=True,
            real2sim_noise=True,
            append_ultrasonic_obs=True,
            seed=42,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (70,)
        assert len(env._us_site_ids) == 25
        assert env._top_cam_id >= 0 and env._side_cam_id >= 0
        assert "observed_cube_ang_speed_rad_s" in info
        obs2, _, _, _, info2 = env.step(np.ones(17, dtype=np.float32))
        w_ctrl2 = float(env.unwrapped.data.ctrl[env._wrist_act_idx])
        assert abs(np.degrees(w_ctrl2) - 0.0) < 1e-4
        assert "tags_top" in info2 and "tags_side" in info2
        assert "observed_cube_ang_speed_rad_s" in info2
        env.close()


def test_ultrasonic_gating_skips_when_disabled():
    import gymnasium as gym
    from gymnasium import spaces

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(68,), dtype=np.float64)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(68, dtype=np.float64), {}

        def step(self, action):
            return np.zeros(68, dtype=np.float64), 0.0, False, False, {}

    w_off = Real2SimSensorAndKinematicWrapper(
        _DummyEnv(),
        lock_wrist=True,
        real_joint_roms=False,
        real2sim_noise=False,
        append_ultrasonic_obs=False,
        seed=0,
    )
    calls = {"us": 0}
    orig_cast = w_off._cast_ultrasonic_cone

    def _counted_cast():
        calls["us"] += 1
        return orig_cast()

    w_off._cast_ultrasonic_cone = _counted_cast
    w_off.reset(seed=0)
    for _ in range(25):
        w_off.step(np.zeros(17, dtype=np.float32))
    assert calls["us"] == 0, "Ultrasonic raycasting must be skipped when append_ultrasonic_obs=False"

    w_on = Real2SimSensorAndKinematicWrapper(
        _DummyEnv(),
        lock_wrist=True,
        real_joint_roms=False,
        real2sim_noise=False,
        append_ultrasonic_obs=True,
        seed=0,
    )
    calls_on = {"us": 0}
    orig_cast_on = w_on._cast_ultrasonic_cone

    def _counted_cast_on():
        calls_on["us"] += 1
        return orig_cast_on()

    w_on._cast_ultrasonic_cone = _counted_cast_on
    w_on.reset(seed=0)
    for _ in range(19):
        w_on.step(np.zeros(17, dtype=np.float32))
    # 20 total calls (reset + 19 steps) at 10 Hz (every 10 steps) -> exactly 2 raycast calls
    assert calls_on["us"] == 2, f"Expected 2 ultrasonic raycast calls at 10 Hz over 20 steps, got {calls_on['us']}"


def test_slurm_script_uses_unix_lf_line_endings():
    repo_root = Path(__file__).resolve().parent.parent
    slurm_sh = repo_root / "scripts" / "slurm_drac_rl_benchmark.sh"
    if slurm_sh.exists():
        raw_bytes = slurm_sh.read_bytes()
        assert b"\r\n" not in raw_bytes, "slurm_drac_rl_benchmark.sh must use LF line endings for sbatch on DRAC Nibi"
