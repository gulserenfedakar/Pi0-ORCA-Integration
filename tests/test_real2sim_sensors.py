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

    repo_root = Path(__file__).resolve().parent.parent
    sim_dir = repo_root.parent / "SEP730-Dexterous-Hand-Simulation" / "src"
    if (sim_dir / "orca_sim").exists():
        if str(sim_dir) not in sys.path:
            sys.path.insert(0, str(sim_dir))
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
            wrist_lock_deg=-9.6,
            real_joint_roms=True,
            real2sim_noise=True,
            append_ultrasonic_obs=True,
            seed=42,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (70,)
        assert len(env._us_site_ids) == 25
        assert env._top_cam_id >= 0 and env._side_cam_id >= 0
        # Verify wrist is locked at -9.6 deg (-0.16755 rad)
        w_qpos = float(env.unwrapped.data.qpos[env._wrist_qpos_adr])
        assert abs(np.degrees(w_qpos) - (-9.6)) < 0.05
        obs2, _, _, _, info2 = env.step(np.ones(17, dtype=np.float32))
        w_ctrl2 = float(env.unwrapped.data.ctrl[env._wrist_act_idx])
        w_qpos2 = float(env.unwrapped.data.qpos[env._wrist_qpos_adr])
        assert abs(np.degrees(w_ctrl2) - (-9.6)) < 1e-4
        assert abs(np.degrees(w_qpos2) - (-9.6)) < 0.5
        assert "tags_top" in info2 and "tags_side" in info2
        env.close()


def test_slurm_script_uses_unix_lf_line_endings():
    repo_root = Path(__file__).resolve().parent.parent
    slurm_sh = repo_root / "scripts" / "slurm_drac_rl_benchmark.sh"
    if slurm_sh.exists():
        raw_bytes = slurm_sh.read_bytes()
        assert b"\r\n" not in raw_bytes, "slurm_drac_rl_benchmark.sh must use LF line endings for sbatch on DRAC Nibi"
