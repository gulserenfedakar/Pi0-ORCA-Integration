"""Evaluate and compare trained RL policies (PPO, SAC, TD3) on the 17-DoF ORCA
in-hand cube reorientation benchmark under matched conditions (ideal or Real2Sim).

Usage:
    python scripts/evaluate_rl_reorient.py --runs-dir models --episodes 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure checkpoints pickled under NumPy 2.x (numpy._core, PCG64 class) load cleanly on NumPy 1.x
if not hasattr(np, "_core") and hasattr(np, "core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
try:
    import numpy.random._pickle as _np_pickle
    _orig_bg_ctor = _np_pickle.__bit_generator_ctor
    def _compat_bg_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return _orig_bg_ctor(bit_generator_name)
    _np_pickle.__bit_generator_ctor = _compat_bg_ctor
except Exception:
    pass
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for cand_sim in [
    os.environ.get("ORCA_SIM_ROOT"),
    *[str(p) for p in sorted(REPO_ROOT.parent.glob("*Dexterous-Hand-Simulation*"))],
    *[str(p) for p in sorted(Path.home().glob("*Dexterous-Hand-Simulation*"))],
]:
    if cand_sim and (Path(cand_sim) / "src" / "orca_sim").exists():
        sim_src = str((Path(cand_sim) / "src").resolve())
        if sim_src not in sys.path:
            sys.path.insert(0, sim_src)
        break

from orca_sim import OrcaHandRightCubeOrientation
from orca_sim.wrappers import RelativeJointTargetAction
from orca_vla.real2sim_sensors import (
    Real2SimSensorAndKinematicWrapper,
    apply_physical_joint_roms_to_base_env,
)

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}


def make_eval_env(
    init: str,
    action: str = "relative",
    max_delta_frac: float = 0.05,
    scene_file: str = "scene_right_cube_orientation.xml",
    drop_height: float = 0.05,
    lock_wrist: bool = False,
    wrist_lock_deg: float = 0.0,
    real_joint_roms: bool = False,
    real2sim_noise: bool = False,
    append_ultrasonic_obs: bool = False,
    render_camera_images: bool = False,
    seed: int = 20_000,
):
    def _thunk():
        env = OrcaHandRightCubeOrientation(
            version="v2",
            scene_file=scene_file,
            initial_red_face=init,
            drop_height=drop_height,
        )
        if real_joint_roms:
            apply_physical_joint_roms_to_base_env(env)
        if action == "relative":
            env = RelativeJointTargetAction(env, max_delta_frac=max_delta_frac)
        if lock_wrist or real_joint_roms or real2sim_noise or append_ultrasonic_obs or render_camera_images:
            env = Real2SimSensorAndKinematicWrapper(
                env,
                lock_wrist=lock_wrist,
                wrist_lock_deg=wrist_lock_deg,
                real_joint_roms=real_joint_roms,
                real2sim_noise=real2sim_noise,
                append_ultrasonic_obs=append_ultrasonic_obs,
                render_camera_images=render_camera_images,
                seed=seed,
            )
        return env

    return _thunk


def evaluate_run_dir(
    run_dir: Path,
    n_episodes: int = 200,
    override_scene: str | None = None,
    override_lock_wrist: bool | None = None,
    override_wrist_lock_deg: float | None = None,
    override_real_joint_roms: bool | None = None,
    override_real2sim_noise: bool | None = None,
    override_append_ultrasonic_obs: bool | None = None,
    render_camera_images: bool = False,
    run_label_suffix: str = "",
) -> dict | None:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        algo_name = cfg.get("algo", "ppo").lower()
        init_mode = cfg.get("init", "random")
        action_mode = cfg.get("action", "relative")
        max_delta_frac = float(cfg.get("max_delta_frac", 0.05))
        scene_file = str(cfg.get("scene_file", "scene_right_cube_orientation.xml"))
        drop_height = float(cfg.get("drop_height", 0.13 if "real2sim" in scene_file else 0.05))
        lock_wrist = bool(cfg.get("lock_wrist", False))
        wrist_lock_deg = float(cfg.get("wrist_lock_deg", 0.0))
        real_joint_roms = bool(cfg.get("real_joint_roms", False))
        real2sim_noise = bool(cfg.get("real2sim_noise", False))
        append_ultrasonic_obs = bool(cfg.get("append_ultrasonic_obs", False))
    else:
        name_lower = run_dir.name.lower()
        algo_name = "sac" if "sac" in name_lower else ("td3" if "td3" in name_lower else "ppo")
        init_mode = "random"
        action_mode = "relative"
        max_delta_frac = 0.05
        scene_file = "scene_right_cube_orientation.xml"
        drop_height = 0.05
        lock_wrist = "real2sim" in name_lower or "locked" in name_lower
        wrist_lock_deg = 0.0
        real_joint_roms = "real2sim" in name_lower or "locked" in name_lower
        real2sim_noise = "real2sim" in name_lower
        append_ultrasonic_obs = "70d" in name_lower

    if override_scene == "real2sim":
        scene_file = "scene_right_cube_orientation_real2sim.xml"
        drop_height = 0.13
    elif override_scene == "plain":
        scene_file = "scene_right_cube_orientation.xml"
        drop_height = 0.05
    if override_lock_wrist is not None:
        lock_wrist = override_lock_wrist
    if override_wrist_lock_deg is not None:
        wrist_lock_deg = override_wrist_lock_deg
    if override_real_joint_roms is not None:
        real_joint_roms = override_real_joint_roms
    if override_real2sim_noise is not None:
        real2sim_noise = override_real2sim_noise
    if override_append_ultrasonic_obs is not None:
        append_ultrasonic_obs = override_append_ultrasonic_obs

    model_path = run_dir / "best_model.zip"
    norm_path = run_dir / "best_vecnormalize.pkl"
    if not norm_path.exists():
        norm_path = run_dir / "vecnormalize.pkl"
        # Check if best_model.zip was saved at an earlier step than final_model.zip (before best_vecnormalize.pkl was introduced)
        eval_npz = run_dir / "evaluations.npz"
        ckpt_dir = run_dir / "checkpoints"
        if model_path.exists() and eval_npz.exists() and ckpt_dir.exists():
            try:
                ev = np.load(eval_npz)
                ts = ev["timesteps"]
                mean_rews = ev["results"].mean(axis=1)
                best_ts = int(ts[int(np.argmax(mean_rews))])
                last_ts = int(ts[-1])
                if best_ts < last_ts:
                    # Find closest checkpoint vecnormalize to best_ts
                    candidates = []
                    for pkl_p in ckpt_dir.glob("*_vecnormalize_*_steps.pkl"):
                        parts = pkl_p.stem.split("_")
                        for tok in parts:
                            if tok.isdigit():
                                candidates.append((abs(int(tok) - best_ts), pkl_p))
                                break
                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        norm_path = candidates[0][1]
            except Exception:
                pass
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        return None

    algo_cls = ALGO_CLASSES[algo_name]
    model = algo_cls.load(str(model_path), device="cpu")

    # Auto-detect 70D ultrasonic append from model observation space if not explicitly specified
    model_obs_shape = getattr(model.observation_space, "shape", None)
    if model_obs_shape == (70,) and not append_ultrasonic_obs:
        append_ultrasonic_obs = True
    elif model_obs_shape == (68,) and append_ultrasonic_obs:
        append_ultrasonic_obs = False

    venv = DummyVecEnv(
        [
            make_eval_env(
                init_mode,
                action_mode,
                max_delta_frac,
                scene_file=scene_file,
                drop_height=drop_height,
                lock_wrist=lock_wrist,
                wrist_lock_deg=wrist_lock_deg,
                real_joint_roms=real_joint_roms,
                real2sim_noise=real2sim_noise,
                append_ultrasonic_obs=append_ultrasonic_obs,
                render_camera_images=render_camera_images,
            )
        ]
    )
    if norm_path.exists():
        venv = VecNormalize.load(str(norm_path), venv)
        venv.training = False
        venv.norm_reward = False

    base_env = venv.venv.envs[0].unwrapped if hasattr(venv, "venv") else venv.envs[0].unwrapped

    successes = []
    goal_aligned = []
    drops = []
    hold_streaks = []
    min_angles = []
    final_angles = []
    infer_ms_list = []

    for ep in range(n_episodes):
        venv.seed(20_000 + ep)
        obs = venv.reset()
        max_streak = 0
        reached_goal = False
        min_ang = float(np.rad2deg(base_env._red_face_up_angle_rad()))

        for _ in range(base_env.max_episode_steps):
            t0 = time.perf_counter()
            act, _ = model.predict(obs, deterministic=True)
            infer_ms_list.append((time.perf_counter() - t0) * 1000.0)
            obs, _, dones, infos = venv.step(act)
            inf = infos[0]
            ang_deg = float(np.rad2deg(inf["red_face_up_angle_rad"]))
            min_ang = min(min_ang, ang_deg)
            max_streak = max(max_streak, int(inf["success_streak"]))
            if inf["goal_reached"]:
                reached_goal = True
            if dones[0]:
                final_angles.append(ang_deg)
                successes.append(bool(inf["is_success"]))
                drops.append(bool(inf["dropped"]))
                break
        else:
            final_angles.append(float(np.rad2deg(base_env._red_face_up_angle_rad())))
            successes.append(False)
            drops.append(bool(base_env._cube_dropped()))

        goal_aligned.append(reached_goal)
        hold_streaks.append(max_streak)
        min_angles.append(min_ang)

    # Also evaluate with terminal hold latch (using noisy observed angle/angvel when real2sim_noise is active)
    latched_successes = []
    for ep in range(n_episodes):
        venv.seed(20_000 + ep)
        obs = venv.reset()
        last_info: dict = {}
        for _ in range(base_env.max_episode_steps):
            if real2sim_noise and "observed_red_face_up_angle_rad" in last_info:
                ang_deg = float(np.rad2deg(last_info["observed_red_face_up_angle_rad"]))
                ang_spd = float(last_info.get("observed_cube_ang_speed_rad_s", np.linalg.norm(base_env._cube_qvel()[3:])))
            else:
                ang_deg = float(np.rad2deg(base_env._red_face_up_angle_rad()))
                ang_spd = float(np.linalg.norm(base_env._cube_qvel()[3:]))
            if ang_deg <= 14.5 and ang_spd <= 0.95 and not base_env._cube_dropped():
                act = np.zeros((1, 17), dtype=np.float32)
            else:
                act, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(act)
            last_info = infos[0]
            if dones[0]:
                latched_successes.append(bool(infos[0]["is_success"]))
                break
        else:
            latched_successes.append(False)

    aligned_min = [m for m, a in zip(min_angles, goal_aligned) if a]
    aligned_fin = [f for f, a in zip(final_angles, goal_aligned) if a]

    venv.close()
    return {
        "Run": run_dir.name + run_label_suffix,
        "Algorithm": algo_name.upper(),
        "Init": init_mode,
        "Action_Mode": action_mode,
        "Lock_Wrist": lock_wrist,
        "Wrist_Lock_Deg": wrist_lock_deg if lock_wrist else 0.0,
        "Real2Sim_Noise": real2sim_noise,
        "Obs_Dim": 70 if append_ultrasonic_obs else 68,
        "Episodes": n_episodes,
        "Policy_Latency_ms": round(float(np.mean(infer_ms_list)), 3),
        "Goal_Align_Rate_pct": round(100.0 * float(np.mean(goal_aligned)), 1),
        "Mean_Hold_Streak_steps": round(float(np.mean(hold_streaks)), 1),
        "Held_Success_Rate_pct": round(100.0 * float(np.mean(successes)), 1),
        "Latched_Success_Rate_pct": round(100.0 * float(np.mean(latched_successes)), 1),
        "Cube_Drop_Rate_pct": round(100.0 * float(np.mean(drops)), 1),
        "Min_Angle_Error_deg": round(float(np.mean(min_angles)), 2),
        "Aligned_Min_Angle_deg": round(float(np.mean(aligned_min)), 2) if aligned_min else 0.0,
        "Final_Angle_Error_deg": round(float(np.mean(final_angles)), 2),
        "Aligned_Final_Angle_deg": round(float(np.mean(aligned_fin)), 2) if aligned_fin else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "rl_reorient_benchmark.csv")
    parser.add_argument("--scene", choices=["auto", "real2sim", "plain"], default=None)
    parser.add_argument("--lock-wrist", action="store_true", default=None)
    parser.add_argument("--wrist-lock-deg", type=float, default=None)
    parser.add_argument("--real-joint-roms", action="store_true", default=None)
    parser.add_argument("--real2sim-noise", action="store_true", default=None)
    parser.add_argument("--append-ultrasonic-obs", action="store_true", default=None)
    parser.add_argument("--render-camera-images", action="store_true", default=False)
    parser.add_argument("--cross-eval", action="store_true", default=True, help="Also cross-evaluate ideal 68D policies under calibrated camera noise")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"No runs directory found at {args.runs_dir}")
        return

    rows = []
    for sub in sorted(args.runs_dir.iterdir()):
        if sub.is_dir():
            res = evaluate_run_dir(
                sub,
                n_episodes=args.episodes,
                override_scene=args.scene,
                override_lock_wrist=args.lock_wrist,
                override_wrist_lock_deg=args.wrist_lock_deg,
                override_real_joint_roms=args.real_joint_roms,
                override_real2sim_noise=args.real2sim_noise,
                override_append_ultrasonic_obs=args.append_ultrasonic_obs,
                render_camera_images=args.render_camera_images,
            )
            if res is not None:
                rows.append(res)
                print(
                    f"[{res['Algorithm']}] {res['Run']} (Obs={res['Obs_Dim']}D, Wrist={res['Wrist_Lock_Deg']}deg, Real2Sim={res['Real2Sim_Noise']}): "
                    f"Success={res['Held_Success_Rate_pct']}% | Latched={res['Latched_Success_Rate_pct']}% | "
                    f"Align={res['Goal_Align_Rate_pct']}% | HoldStreak={res['Mean_Hold_Streak_steps']}/25 | "
                    f"Drop={res['Cube_Drop_Rate_pct']}% | FinalAng={res['Final_Angle_Error_deg']}deg"
                )
                # Joe's requested cross-eval: ideal 68D policy evaluated under calibrated camera noise
                if args.cross_eval and not res["Real2Sim_Noise"] and res["Obs_Dim"] == 68:
                    cross = evaluate_run_dir(
                        sub,
                        n_episodes=args.episodes,
                        override_scene=args.scene or "real2sim",
                        override_lock_wrist=args.lock_wrist if args.lock_wrist is not None else res["Lock_Wrist"],
                        override_wrist_lock_deg=args.wrist_lock_deg,
                        override_real_joint_roms=args.real_joint_roms,
                        override_real2sim_noise=True,
                        override_append_ultrasonic_obs=False,
                        render_camera_images=args.render_camera_images,
                        run_label_suffix="_cross_eval_under_camera_noise",
                    )
                    if cross is not None:
                        rows.append(cross)
                        print(
                            f"[{cross['Algorithm']} CROSS-EVAL] {cross['Run']} (Real2Sim=True): "
                            f"Success={cross['Held_Success_Rate_pct']}% | Latched={cross['Latched_Success_Rate_pct']}% | "
                            f"Align={cross['Goal_Align_Rate_pct']}% | HoldStreak={cross['Mean_Hold_Streak_steps']}/25 | "
                            f"Drop={cross['Cube_Drop_Rate_pct']}% | FinalAng={cross['Final_Angle_Error_deg']}deg"
                        )

    if rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved RL comparison table to {args.out_csv}")


if __name__ == "__main__":
    main()
