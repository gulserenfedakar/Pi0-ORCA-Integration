"""Unit tests for CalibratedUltrasonicSensor, CalibratedAprilTagCameraWrapper, and Slurm script formatting."""

from __future__ import annotations

from pathlib import Path

from orca_vla.real2sim_sensors import (
    CalibratedAprilTagCameraWrapper,
    CalibratedUltrasonicSensor,
    UltrasonicReal2SimConfig,
)


def test_ultrasonic_dead_zone_and_wall_jump_masking():
    # Across 50 distinct seeds, dead-zone input (<= 22 mm) must NEVER leak past filter_for_rl_observation
    for seed in range(50):
        sensor = CalibratedUltrasonicSensor(seed=seed)
        sensor.reset(seed=seed)
        raw_dz = sensor.simulate_raw_reading_mm(12.0, face_incidence_deg=0.0, hit_body="cube")
        assert raw_dz is None or raw_dz <= 22.0
        d_norm_dz, valid_dz = sensor.filter_for_rl_observation(raw_dz)
        assert d_norm_dz == 0.0 and valid_dz == 0.0

    sensor = CalibratedUltrasonicSensor(seed=123)
    sensor.reset(seed=123)

    # High tilt angle / wall jump (>150 mm) must filter to (0.0, 0.0)
    d_norm_jump, valid_jump = sensor.filter_for_rl_observation(612.0)
    assert d_norm_jump == 0.0 and valid_jump == 0.0

    # Valid normal-incidence reading (54.4 mm) must pass with valid_flag = 1.0
    d_norm_ok, valid_ok = sensor.filter_for_rl_observation(54.4)
    assert 0.2 < d_norm_ok < 0.5 and valid_ok == 1.0


def test_ultrasonic_10hz_hold_cadence():
    cfg = UltrasonicReal2SimConfig(control_hz=100, ultrasonic_hz=10)
    sensor = CalibratedUltrasonicSensor(config=cfg, seed=7)
    sensor.reset(seed=7)

    readings = [sensor.step_100hz(50.0, face_incidence_deg=5.0, hit_body="cube")[0] for _ in range(10)]
    # All 10 steps within a 100 ms window must hold the identical 10 Hz sample
    assert len(set(readings)) == 1


def test_camera_wrapper_perturbs_full_cube_state_30hz():
    import numpy as np

    cam = CalibratedAprilTagCameraWrapper(seed=99)
    cam.reset(seed=99)

    p, q, v, n, a, vis = cam.perturb_full_cube_state(
        true_pos=np.array([0.17, -0.01, 0.18], dtype=np.float64),
        true_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        true_qvel=np.zeros(6, dtype=np.float64),
        fingers_occluding=False,
    )
    assert len(p) == 3 and len(q) == 4 and len(v) == 6 and len(n) == 3 and isinstance(vis, bool)
    # Next step within the 30 Hz window (3-step hold at 100 Hz) must return identical held observation
    p2, q2, v2, n2, a2, vis2 = cam.perturb_full_cube_state(
        true_pos=np.array([0.18, 0.00, 0.18], dtype=np.float64),
        true_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        true_qvel=np.ones(6, dtype=np.float64),
        fingers_occluding=False,
    )
    assert np.allclose(p, p2) and np.allclose(q, q2) and np.allclose(v, v2) and np.allclose(n, n2) and a == a2


def test_slurm_script_uses_unix_lf_line_endings():
    repo_root = Path(__file__).resolve().parent.parent
    slurm_sh = repo_root / "scripts" / "slurm_drac_rl_benchmark.sh"
    if slurm_sh.exists():
        raw_bytes = slurm_sh.read_bytes()
        assert b"\r\n" not in raw_bytes, "slurm_drac_rl_benchmark.sh must use LF line endings for sbatch on DRAC Nibi"
