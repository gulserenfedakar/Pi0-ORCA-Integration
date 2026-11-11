import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import csv
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from scripts.run_cube_reorient import run_episode


def main() -> None:
    print("Loading policy 'pi0_libero' for 17-DoF In-Hand Cube Reorientation Benchmark...")
    config = _config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    print("Warming up XLA JIT compiler (1 step)...")
    _ = run_episode(
        policy,
        mode="synergy_reorient",
        camera="palm_front",
        init_red_face="down",
        seed=7,
        horizon=5,
        max_vla_steps=1,
    )

    experiments = [
        (
            "1. Standalone pi0 — Naive Direct Mapping (No Relative/Synergy)",
            "direct_17dof",
            "palm_front",
            "down",
            10,
            10,
            [7, 8],
            [0.0, 0.0],
        ),
        (
            "2. Standalone pi0 — Relative Control + 17-DoF Synergy (Front Cam)",
            "synergy_reorient",
            "palm_front",
            "random",
            10,
            10,
            [0, 6],
            [0.0, 0.0],
        ),
        (
            "3. Standalone pi0 — Relative Control + 17-DoF Synergy (Top Cam)",
            "synergy_reorient",
            "palm_top",
            "random",
            10,
            10,
            [0, 6],
            [0.0, 0.0],
        ),
        (
            "4. Hybrid pi0 + RL (PPO) — High-Level VLA + Low-Level RL (Nominal 180 deg Flip)",
            "pi0_rl_hybrid",
            "palm_front",
            "down",
            10,
            10,
            [7, 8],
            [0.0, 0.0],
        ),
        (
            "5. Hybrid pi0 + RL (PPO) — High-Level VLA + Low-Level RL (Randomized 5 Trials)",
            "pi0_rl_hybrid",
            "palm_front",
            "random",
            10,
            14,
            [0, 1, 4, 6, 7],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]

    summary_rows = []
    for label, mode, cam, init_mode, horizon, max_vla_steps, seeds, jitters in experiments:
        n_trials = len(seeds)
        print(f"\nRunning Experiment: {label} ({n_trials} trials)...")
        trial_results = []
        for t, (sd, jit) in enumerate(zip(seeds, jitters)):
            res = run_episode(
                policy,
                mode=mode,
                camera=cam,
                init_red_face=init_mode,
                seed=sd,
                horizon=horizon,
                max_vla_steps=max_vla_steps,
                xy_jitter=jit,
            )
            trial_results.append(res)
            print(
                f"  Trial {t+1}/{n_trials} (seed={sd}): Infer={res['infer_ms']:.1f}ms | E2E={res['e2e_ms']:.1f}ms | "
                f"InitAng={res['init_angle_deg']:.1f}deg -> MinAng={res['min_angle_deg']:.1f}deg -> FinalAng={res['final_angle_deg']:.1f}deg | "
                f"HoldStreak={res['max_hold_streak']}/25 | Dropped={res['dropped']} | Success={res['success']}"
            )

        row = {
            "Experiment": label,
            "Mode": mode,
            "Camera": cam,
            "Trials": n_trials,
            "Inference_Latency_ms": round(float(np.mean([r["infer_ms"] for r in trial_results])), 2),
            "E2E_Control_Latency_ms": round(float(np.mean([r["e2e_ms"] for r in trial_results])), 2),
            "Completion_Time_s": round(float(np.mean([r["completion_time_s"] for r in trial_results])), 2),
            "Goal_Align_Rate_pct": round(100.0 * float(np.mean([r["goal_reached"] for r in trial_results])), 1),
            "Mean_Hold_Streak_steps": round(float(np.mean([r["max_hold_streak"] for r in trial_results])), 1),
            "Task_Success_Rate_pct": round(100.0 * float(np.mean([r["success"] for r in trial_results])), 1),
            "Cube_Drop_Rate_pct": round(100.0 * float(np.mean([r["dropped"] for r in trial_results])), 1),
            "Min_Angle_Error_deg": round(float(np.mean([r["min_angle_deg"] for r in trial_results])), 2),
            "Final_Angle_Error_deg": round(float(np.mean([r["final_angle_deg"] for r in trial_results])), 2),
        }
        summary_rows.append(row)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "cube_reorient_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved cube reorientation benchmark CSV to {csv_path}")


if __name__ == "__main__":
    main()
