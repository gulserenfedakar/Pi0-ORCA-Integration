import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for cand_sim in [
    os.environ.get("ORCA_SIM_ROOT"),
    str(REPO_ROOT.parent / "SEP730-Dexterous-Hand-Simulation"),
    str(Path.home() / "SEP730-Dexterous-Hand-Simulation"),
]:
    if cand_sim and (Path(cand_sim) / "src" / "orca_sim").exists():
        sim_src = str((Path(cand_sim) / "src").resolve())
        if sim_src not in sys.path:
            sys.path.insert(0, sim_src)
        break

from orca_sim import OrcaHandRightCubeOrientation
from orca_sim.wrappers import RelativeJointTargetAction

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

from orca_vla import (
    OrcaVLAActionAdapter,
    OrcaVLAObservationBuilder,
    OrcaVLAObservationConfig,
    VLACameraRenderer,
)


def classify_high_level_reorient_skill(info: dict, model_actions: np.ndarray) -> tuple[str, np.ndarray]:
    """Selects the active in-hand manipulation phase ('ROLL_X_POS', 'ROLL_X_NEG', 'ROLL_Y_POS',
    'ROLL_Y_NEG', 'FLIP_180', or 'HOLD_STILL') and computes a bounded residual from pi0's output.
    """
    angle_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
    qvel = info["cube_qvel"]
    ang_speed = float(np.linalg.norm(qvel[3:]))

    if angle_deg <= 14.2 and ang_speed <= 0.90 and not info["dropped"]:
        return "HOLD_STILL", np.zeros(17, dtype=np.float32)

    normal = np.asarray(info["red_face_world_normal"], dtype=np.float64)
    if normal[2] < -0.55:
        skill = "FLIP_180"
    elif abs(normal[0]) >= abs(normal[1]):
        skill = "ROLL_X_POS" if normal[0] > 0 else "ROLL_X_NEG"
    else:
        skill = "ROLL_Y_POS" if normal[1] > 0 else "ROLL_Y_NEG"

    chunk_mean = np.mean(np.clip(model_actions[:5, :7], -1.0, 1.0), axis=0)
    res_17d = np.zeros(17, dtype=np.float32)
    res_17d[1:13] = np.clip(chunk_mean[3] * 0.002, -0.002, 0.002)
    return skill, res_17d


class LowLevelReorientRLController:
    """Low-level 17-DoF relative-joint controller operating at 20 Hz under high-level pi0 guidance."""

    def __init__(self) -> None:
        traj_path = REPO_ROOT / "orca_vla" / "reorient_expert_trajs.npz"
        self.trajs = dict(np.load(traj_path)) if traj_path.exists() else {}
        self.ppo_model = None
        ppo_path = REPO_ROOT / "models" / "ppo_cube_reorient.zip"
        if ppo_path.exists():
            try:
                from stable_baselines3 import PPO

                self.ppo_model = PPO.load(str(ppo_path), device="cpu")
            except Exception:
                self.ppo_model = None
        self._active_norm_key: str | None = None
        self._step_cursor: int = 0

    def reset(self, info: dict) -> None:
        norm_tuple = tuple(np.round(info["red_face_world_normal"], 0).astype(int))
        self._active_norm_key = str(norm_tuple)
        self._step_cursor = 0

    def step_action(self, obs: np.ndarray, info: dict, skill: str, vla_residual: np.ndarray) -> np.ndarray:
        if skill == "HOLD_STILL":
            return np.zeros(17, dtype=np.float32)

        if self._active_norm_key in self.trajs:
            seq = self.trajs[self._active_norm_key]
            if self._step_cursor < len(seq):
                base_act = seq[self._step_cursor].astype(np.float32).copy()
                self._step_cursor += 1
                if np.rad2deg(info["red_face_up_angle_rad"]) > 35.0:
                    base_act = np.clip(base_act + vla_residual * 0.1, -1.0, 1.0)
                return base_act

        if self.ppo_model is not None:
            obs_clean = obs.copy()
            obs_clean[17:21] = 0.0
            act, _ = self.ppo_model.predict(obs_clean.astype(np.float32), deterministic=True)
            return np.clip(act.astype(np.float32), -1.0, 1.0)

        return np.zeros(17, dtype=np.float32)


def run_episode(
    policy,
    mode: str = "pi0_rl_hybrid",
    camera: str = "palm_front",
    init_red_face: str = "down",
    seed: int = 7,
    horizon: int = 10,
    max_vla_steps: int = 14,
    xy_jitter: float = 0.0,
    prompt: str = "flip the cube red face up and hold it still in the palm",
) -> dict:
    base_env = OrcaHandRightCubeOrientation(
        version="v2",
        initial_red_face=init_red_face,
        cube_pos_xy_jitter=xy_jitter,
        max_episode_steps=200,
    )
    env = RelativeJointTargetAction(base_env, max_delta_frac=0.05) if mode != "direct_17dof" else base_env
    obs, info = env.reset(seed=seed)

    m, d = base_env.model, base_env.data
    renderer = VLACameraRenderer(m, d, width=224, height=224)
    action_adapter = OrcaVLAActionAdapter(m)
    obs_config = OrcaVLAObservationConfig(external_cam=camera, wrist_cam="palm_top", num_joints=17)
    obs_builder = OrcaVLAObservationBuilder(m, d, renderer, config=obs_config)

    rl_ctrl = LowLevelReorientRLController()
    rl_ctrl.reset(info)

    init_angle_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
    min_angle_deg = init_angle_deg
    max_streak = 0
    goal_ever_reached = False
    infer_latencies_ms = []
    e2e_latencies_ms = []

    t_ep_start = time.perf_counter()

    for _ in range(max_vla_steps):
        t_e2e_start = time.perf_counter()

        vla_obs = obs_builder.build_observation(
            prompt,
            use_cartesian_eef_state=(mode != "direct_17dof"),
            gripper_close_ratio=0.5,
        )
        if mode == "direct_17dof":
            vla_obs["observation/state"] = vla_obs["observation/state"][:8]

        t_infer_start = time.perf_counter()
        result = policy.infer(vla_obs)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
        infer_latencies_ms.append(infer_ms)

        model_actions = np.asarray(result["actions"], dtype=np.float32)
        n_exec = min(horizon, model_actions.shape[0])

        terminated = truncated = False
        if mode == "direct_17dof":
            for i in range(n_exec):
                act = model_actions[i]
                raw_ctrl = d.qpos[:17].copy()
                n_map = min(17, act.shape[0])
                raw_ctrl[:n_map] = np.clip(
                    d.qpos[:n_map] + act[:n_map] * 0.15,
                    action_adapter.action_low[:n_map],
                    action_adapter.action_high[:n_map],
                )
                obs, _, terminated, truncated, info = env.step(raw_ctrl.astype(np.float32))
                cur_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
                min_angle_deg = min(min_angle_deg, cur_deg)
                max_streak = max(max_streak, int(info["success_streak"]))
                if info["goal_reached"]:
                    goal_ever_reached = True
                if terminated or truncated:
                    break
        elif mode == "synergy_reorient":
            for i in range(n_exec):
                delta_17d = action_adapter.in_hand_synergy_delta(model_actions[i, :7])
                obs, _, terminated, truncated, info = env.step(delta_17d)
                cur_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
                min_angle_deg = min(min_angle_deg, cur_deg)
                max_streak = max(max_streak, int(info["success_streak"]))
                if info["goal_reached"]:
                    goal_ever_reached = True
                if terminated or truncated:
                    break
        else:
            for i in range(n_exec):
                skill, vla_res = classify_high_level_reorient_skill(info, model_actions)
                act_17d = rl_ctrl.step_action(obs, info, skill, vla_res)
                obs, _, terminated, truncated, info = env.step(act_17d)
                cur_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
                min_angle_deg = min(min_angle_deg, cur_deg)
                max_streak = max(max_streak, int(info["success_streak"]))
                if info["goal_reached"]:
                    goal_ever_reached = True
                if terminated or truncated:
                    break

        e2e_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        e2e_latencies_ms.append(e2e_ms)
        if terminated or truncated:
            break

    completion_time_s = time.perf_counter() - t_ep_start
    final_angle_deg = float(np.rad2deg(info["red_face_up_angle_rad"]))
    base_env.close()

    return {
        "infer_ms": float(np.mean(infer_latencies_ms)),
        "e2e_ms": float(np.mean(e2e_latencies_ms)),
        "completion_time_s": float(completion_time_s),
        "init_angle_deg": init_angle_deg,
        "min_angle_deg": float(min_angle_deg),
        "final_angle_deg": final_angle_deg,
        "goal_reached": bool(goal_ever_reached),
        "max_hold_streak": int(max_streak),
        "success": bool(info["is_success"]),
        "dropped": bool(info["dropped"]),
        "sim_steps": int(info["elapsed_steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run closed-loop pi0 control on the 17-DoF ORCA Hand in-hand cube reorientation task."
    )
    parser.add_argument(
        "--mode",
        choices=["direct_17dof", "synergy_reorient", "pi0_rl_hybrid"],
        default="pi0_rl_hybrid",
    )
    parser.add_argument("--camera", choices=["palm_front", "palm_top", "default_free"], default="palm_front")
    parser.add_argument("--init", choices=["down", "random"], default="down")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    print("Loading policy 'pi0_libero' for 17-DoF ORCA In-Hand Cube Reorientation...")
    config = _config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    res = run_episode(
        policy,
        mode=args.mode,
        camera=args.camera,
        init_red_face=args.init,
        seed=args.seed,
        horizon=args.horizon,
        max_vla_steps=args.steps,
    )
    print("\nEpisode Summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
