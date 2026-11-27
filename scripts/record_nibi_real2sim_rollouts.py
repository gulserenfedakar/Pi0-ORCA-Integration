"""Record side-by-side (top + side camera) rollout GIFs for the best Nibi Real2Sim PPO policies.

Usage (on DRAC Nibi):
    MUJOCO_GL=osmesa python scripts/record_nibi_real2sim_rollouts.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

from scripts.evaluate_rl_reorient import make_eval_env

TARGETS = [
    (
        "ppo_ideal_locked_wrist_68d_10000000_seed2",
        "Ideal Locked-Wrist 68D (84.0% Held)",
        "rollout_ideal_locked_wrist_68d_seed2.gif",
    ),
    (
        "ppo_real2sim_camera_68d_10000000_seed2",
        "Real2Sim Camera 68D (77.0% Held)",
        "rollout_real2sim_camera_68d_seed2.gif",
    ),
    (
        "ppo_real2sim_camera_ultrasonic_70d_10000000_seed0",
        "Real2Sim Cam+Ultrasonic 70D (60.5% Held)",
        "rollout_real2sim_camera_ultrasonic_70d_seed0.gif",
    ),
]


def _render_side_by_side(renderer: mujoco.Renderer, data: mujoco.MjData, label: str, step: int, ang_deg: float, streak: int, is_success: bool) -> Image.Image:
    renderer.update_scene(data, camera="top")
    top_rgb = renderer.render().copy()
    renderer.update_scene(data, camera="side")
    side_rgb = renderer.render().copy()

    combined = np.concatenate([top_rgb, side_rgb], axis=1)
    h, w, _ = combined.shape
    bar_h = 34
    canvas = Image.new("RGB", (w, h + bar_h), color=(15, 23, 42))
    canvas.paste(Image.fromarray(combined), (0, bar_h))

    draw = ImageDraw.Draw(canvas)
    status = "SUCCESS (HELD 25/25)" if is_success else f"Hold: {streak}/25"
    status_color = (74, 222, 128) if is_success else (248, 250, 252)
    draw.text((10, 9), f"{label}  |  Step {step:03d}  |  Err {ang_deg:5.1f} deg  |  {status}", fill=status_color)
    draw.text((8, bar_h + 6), "TOP CAM", fill=(255, 255, 255))
    draw.text((w // 2 + 8, bar_h + 6), "SIDE CAM", fill=(255, 255, 255))
    return canvas


def record_run(run_dir: Path, label: str, out_gif: Path) -> None:
    if not run_dir.exists():
        print(f"Skipping missing run_dir: {run_dir}")
        return
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model_path = run_dir / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    norm_path = run_dir / "best_vecnormalize.pkl"
    if not norm_path.exists():
        norm_path = run_dir / "vecnormalize.pkl"

    model = PPO.load(str(model_path), device="cpu")
    venv = DummyVecEnv(
        [
            make_eval_env(
                init=cfg.get("init", "random"),
                action=cfg.get("action", "relative"),
                max_delta_frac=float(cfg.get("max_delta_frac", 0.05)),
                scene_file=str(cfg.get("scene_file", "scene_right_cube_orientation_real2sim.xml")),
                drop_height=float(cfg.get("drop_height", 0.13)),
                lock_wrist=bool(cfg.get("lock_wrist", True)),
                wrist_lock_deg=float(cfg.get("wrist_lock_deg", 0.0)),
                real_joint_roms=bool(cfg.get("real_joint_roms", True)),
                real2sim_noise=bool(cfg.get("real2sim_noise", False)),
                append_ultrasonic_obs=bool(cfg.get("append_ultrasonic_obs", False)),
                seed=20_000,
            )
        ]
    )
    if norm_path.exists():
        venv = VecNormalize.load(str(norm_path), venv)
        venv.training = False
        venv.norm_reward = False

    base_env = venv.venv.envs[0].unwrapped if hasattr(venv, "venv") else venv.envs[0].unwrapped

    # Pick a deterministic seed with initial angle >= 90 deg that achieves held success
    chosen_seed = 20_000
    for ep in range(40):
        s = 20_000 + ep
        venv.seed(s)
        obs = venv.reset()
        init_ang = float(np.rad2deg(base_env._red_face_up_angle_rad()))
        if init_ang < 85.0:
            continue
        solved = False
        for _ in range(base_env.max_episode_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(act)
            if dones[0]:
                solved = bool(infos[0].get("is_success", False))
                break
        if solved:
            chosen_seed = s
            break

    renderer = mujoco.Renderer(base_env.model, height=240, width=320)
    venv.seed(chosen_seed)
    obs = venv.reset()
    frames: list[Image.Image] = []

    ang0 = float(np.rad2deg(base_env._red_face_up_angle_rad()))
    frames.append(_render_side_by_side(renderer, base_env.data, label, 0, ang0, 0, False))

    for step_i in range(1, base_env.max_episode_steps + 1):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        inf = infos[0]
        ang = float(np.rad2deg(inf.get("red_face_up_angle_rad", base_env._red_face_up_angle_rad())))
        streak = int(inf.get("success_streak", 0))
        succ = bool(inf.get("is_success", False))
        if step_i % 2 == 0 or dones[0]:
            frames.append(_render_side_by_side(renderer, base_env.data, label, step_i, ang, streak, succ))
        if dones[0]:
            for _ in range(12):
                frames.append(frames[-1])
            break

    renderer.close()
    venv.close()

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_gif),
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out_gif} ({len(frames)} frames, seed={chosen_seed})")


def main() -> None:
    out_dir = REPO_ROOT / "results" / "rollouts"
    for run_name, label, gif_name in TARGETS:
        record_run(REPO_ROOT / "runs" / run_name, label, out_dir / gif_name)


if __name__ == "__main__":
    main()
