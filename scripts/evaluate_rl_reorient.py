"""Evaluate and compare trained RL policies (PPO, SAC, TD3) on the 17-DoF ORCA
in-hand cube reorientation benchmark under matched conditions.

Usage:
    python scripts/evaluate_rl_reorient.py --runs-dir runs --episodes 20
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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
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

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}


def make_eval_env(init: str, action: str = "relative", max_delta_frac: float = 0.05):
    def _thunk():
        env = OrcaHandRightCubeOrientation(version="v2", initial_red_face=init)
        if action == "relative":
            env = RelativeJointTargetAction(env, max_delta_frac=max_delta_frac)
        return env

    return _thunk


def evaluate_run_dir(run_dir: Path, n_episodes: int = 20) -> dict | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    algo_name = cfg.get("algo", "ppo").lower()
    init_mode = cfg.get("init", "random")
    action_mode = cfg.get("action", "relative")
    max_delta_frac = float(cfg.get("max_delta_frac", 0.05))

    model_path = run_dir / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        return None

    algo_cls = ALGO_CLASSES[algo_name]
    model = algo_cls.load(str(model_path), device="cpu")

    venv = DummyVecEnv([make_eval_env(init_mode, action_mode, max_delta_frac)])
    norm_path = run_dir / "vecnormalize.pkl"
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

    venv.close()
    return {
        "Run": run_dir.name,
        "Algorithm": algo_name.upper(),
        "Init": init_mode,
        "Action_Mode": action_mode,
        "Episodes": n_episodes,
        "Policy_Latency_ms": round(float(np.mean(infer_ms_list)), 3),
        "Goal_Align_Rate_pct": round(100.0 * float(np.mean(goal_aligned)), 1),
        "Mean_Hold_Streak_steps": round(float(np.mean(hold_streaks)), 1),
        "Held_Success_Rate_pct": round(100.0 * float(np.mean(successes)), 1),
        "Cube_Drop_Rate_pct": round(100.0 * float(np.mean(drops)), 1),
        "Min_Angle_Error_deg": round(float(np.mean(min_angles)), 2),
        "Final_Angle_Error_deg": round(float(np.mean(final_angles)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "rl_algorithms_comparison.csv")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"No runs directory found at {args.runs_dir}")
        return

    rows = []
    for sub in sorted(args.runs_dir.iterdir()):
        if sub.is_dir():
            res = evaluate_run_dir(sub, n_episodes=args.episodes)
            if res is not None:
                rows.append(res)
                print(
                    f"[{res['Algorithm']}] {res['Run']}: Success={res['Held_Success_Rate_pct']}% | "
                    f"Align={res['Goal_Align_Rate_pct']}% | HoldStreak={res['Mean_Hold_Streak_steps']}/25 | "
                    f"Drop={res['Cube_Drop_Rate_pct']}% | FinalAng={res['Final_Angle_Error_deg']}deg"
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
