"""Matched Reinforcement Learning benchmark trainer (PPO, SAC, TD3) on the 17-DoF ORCA
in-hand cube reorientation task (OrcaHandRightCubeOrientation).

Designed for multi-core CPU execution on local workstations and Digital Research
Alliance of Canada (DRAC) Slurm clusters under identical environment wrappers,
observation/reward normalization, held-success evaluation protocols, and optional
Real2Sim sensor perturbation and locked-wrist hardware constraints.

Examples:
    # Ideal baseline training (17-DoF unconstrained)
    python scripts/train_rl_reorient.py --algo ppo --steps 10_000_000 --n-envs 16

    # Sim-to-Real hardware-aligned training (locked wrist, physical thumb ROM, calibrated Real2Sim noise)
    python scripts/train_rl_reorient.py --algo ppo --steps 5_000_000 --n-envs 16 \
        --lock-wrist --real-joint-roms --real2sim-noise \
        --resume models/drac_ppo_10M/best_model.zip

    # 70D Ultrasonic-augmented fine-tuning (warm-starts 68D checkpoint via zero-padded input surgery)
    python scripts/train_rl_reorient.py --algo ppo --steps 5_000_000 --n-envs 16 \
        --lock-wrist --real-joint-roms --real2sim-noise --append-ultrasonic-obs \
        --resume models/drac_ppo_10M/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

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


def make_env(
    init: str,
    seed: int,
    action: str = "relative",
    max_delta_frac: float = 0.05,
    lock_wrist: bool = False,
    real_joint_roms: bool = False,
    real2sim_noise: bool = False,
    append_ultrasonic_obs: bool = False,
):
    def _thunk():
        from orca_sim import OrcaHandRightCubeOrientation
        from orca_sim.wrappers import RelativeJointTargetAction
        from orca_vla.real2sim_sensors import (
            Real2SimSensorAndKinematicWrapper,
            apply_physical_joint_roms_to_base_env,
        )

        env = OrcaHandRightCubeOrientation(version="v2", initial_red_face=init)
        # Apply physical joint ROM limits BEFORE RelativeJointTargetAction caches _low/_high/_span
        if real_joint_roms:
            apply_physical_joint_roms_to_base_env(env)
        if action == "relative":
            env = RelativeJointTargetAction(env, max_delta_frac=max_delta_frac)
        if lock_wrist or real_joint_roms or real2sim_noise or append_ultrasonic_obs:
            env = Real2SimSensorAndKinematicWrapper(
                env,
                lock_wrist=lock_wrist,
                real_joint_roms=real_joint_roms,
                real2sim_noise=real2sim_noise,
                append_ultrasonic_obs=append_ultrasonic_obs,
                seed=seed,
            )
        info_keys = ("is_success", "goal_reached", "success_streak", "dropped")
        env = Monitor(env, info_keywords=info_keys)
        env.reset(seed=seed)
        return env

    return _thunk


def _extend_linear_input_weights(new_module: torch.nn.Module, old_state_dict: dict[str, torch.Tensor]) -> None:
    """Copy pretrained weights into `new_module`, zero-padding extra input columns when obs_dim grows (e.g. 68 -> 70)."""
    new_sd = new_module.state_dict()
    for k, old_w in old_state_dict.items():
        if k not in new_sd:
            continue
        new_w = new_sd[k]
        if old_w.shape == new_w.shape:
            new_sd[k] = old_w.clone()
        elif old_w.ndim == 2 and new_w.ndim == 2 and old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
            padded = torch.zeros_like(new_w)
            padded[:, : old_w.shape[1]] = old_w
            new_sd[k] = padded
    new_module.load_state_dict(new_sd)


def _warm_up_offpolicy_replay_buffer(model: SAC | TD3, train_env: VecNormalize, n_steps: int = 10_000) -> None:
    """Pre-fill an empty off-policy replay buffer using the loaded actor before gradient updates begin."""
    obs = train_env.reset()
    n_envs = train_env.num_envs
    collected = 0
    while collected < n_steps:
        actions, _ = model.predict(obs, deterministic=False)
        scaled_actions = model.policy.scale_action(actions)
        next_obs, rewards, dones, infos = train_env.step(actions)
        model.replay_buffer.add(obs, next_obs, scaled_actions, rewards, dones, infos)
        obs = next_obs
        collected += n_envs
    print(f"Pre-warmed off-policy replay buffer with {collected:,} policy transitions.")


def build_model(
    algo: str,
    train_env: VecNormalize,
    action_mode: str,
    seed: int,
    device: str,
    resume_path: Path | None = None,
):
    if algo == "ppo":
        ppo_kwargs = dict(
            n_steps=512,
            batch_size=2048,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device=device,
            seed=seed,
            verbose=0,
        )
        fresh = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-0.5 if action_mode == "relative" else 0.0,
            ),
            **ppo_kwargs,
        )
        if resume_path is not None:
            old_model = PPO.load(str(resume_path), device=device)
            _extend_linear_input_weights(fresh.policy, old_model.policy.state_dict())
            print(f"Warm-started PPO policy weights from {resume_path}")
        return fresh

    if algo == "sac":
        sac_kwargs = dict(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            ent_coef="auto",
            device=device,
            seed=seed,
            verbose=0,
        )
        fresh = SAC(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=dict(net_arch=[256, 256]),
            **sac_kwargs,
        )
        if resume_path is not None:
            old_model = SAC.load(str(resume_path), device=device)
            _extend_linear_input_weights(fresh.policy, old_model.policy.state_dict())
            # Ensure entropy coefficient has enough floor to explore locked-wrist kinematics
            if hasattr(fresh, "log_ent_coef") and isinstance(fresh.log_ent_coef, torch.Tensor):
                with torch.no_grad():
                    fresh.log_ent_coef.copy_(torch.tensor([math.log(0.05)], device=fresh.device))
            _warm_up_offpolicy_replay_buffer(fresh, train_env, n_steps=10_000)
            print(f"Warm-started SAC policy weights and pre-filled replay buffer from {resume_path}")
        return fresh

    if algo == "td3":
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=0.15 * np.ones(n_actions, dtype=np.float32),
        )
        td3_kwargs = dict(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            action_noise=action_noise,
            device=device,
            seed=seed,
            verbose=0,
        )
        fresh = TD3(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=dict(net_arch=[256, 256]),
            **td3_kwargs,
        )
        if resume_path is not None:
            old_model = TD3.load(str(resume_path), device=device)
            _extend_linear_input_weights(fresh.policy, old_model.policy.state_dict())
            _warm_up_offpolicy_replay_buffer(fresh, train_env, n_steps=10_000)
            print(f"Warm-started TD3 policy weights and pre-filled replay buffer from {resume_path}")
        return fresh

    raise ValueError(f"Unsupported algorithm: {algo}")


def _load_or_extend_vecnormalize(train_env, resume_path: Path | None) -> VecNormalize:
    """Load `vecnormalize.pkl` from `resume_path.parent`, extending `obs_rms` if observation dim increased (68 -> 70)."""
    venv = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    if resume_path is None:
        return venv
    cand_norm = resume_path.parent / "vecnormalize.pkl"
    if not cand_norm.exists():
        return venv
    try:
        dummy = DummyVecEnv([lambda: gym.Env()])
        old_vn = VecNormalize.load(str(cand_norm), dummy)
        old_mean = old_vn.obs_rms.mean
        old_var = old_vn.obs_rms.var
        new_dim = venv.obs_rms.mean.shape[0]
        if old_mean.shape[0] == new_dim:
            venv.obs_rms.mean[:] = old_mean
            venv.obs_rms.var[:] = old_var
            venv.obs_rms.count = old_vn.obs_rms.count
            print(f"Loaded matching {new_dim}D VecNormalize statistics from {cand_norm}")
        elif old_mean.shape[0] < new_dim:
            venv.obs_rms.mean[: old_mean.shape[0]] = old_mean
            venv.obs_rms.var[: old_var.shape[0]] = old_var
            venv.obs_rms.count = old_vn.obs_rms.count
            print(f"Extended VecNormalize statistics ({old_mean.shape[0]}D -> {new_dim}D) from {cand_norm}")
    except Exception as exc:
        print(f"Warning: could not copy VecNormalize stats from {cand_norm}: {exc}")
    return venv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="sac")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total environment steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel simulation processes")
    parser.add_argument("--init", choices=["random", "down"], default="random")
    parser.add_argument("--eval-every", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--action", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--max-delta-frac", type=float, default=0.05)
    parser.add_argument("--lock-wrist", action="store_true", help="Freeze 1-DoF wrist pitch actuator during training")
    parser.add_argument("--real-joint-roms", action="store_true", help="Enforce physical hand thumb_mcp [-5, +40] deg bounds")
    parser.add_argument("--real2sim-noise", action="store_true", help="Enable calibrated dual-camera AprilTag & ultrasonic Real2Sim noise")
    parser.add_argument("--append-ultrasonic-obs", action="store_true", help="Append 2D [us_d_norm, us_valid] to observation space (68D -> 70D)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to existing model .zip checkpoint for warm-start fine-tuning")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_real2sim" if (args.real2sim_noise or args.lock_wrist) else ""
    out = args.out or (REPO_ROOT / "runs" / f"{args.algo}_reorient_{args.init}{suffix}_{timestamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "algo": args.algo,
                "steps": args.steps,
                "n_envs": args.n_envs,
                "init": args.init,
                "action": args.action,
                "max_delta_frac": args.max_delta_frac,
                "lock_wrist": args.lock_wrist,
                "real_joint_roms": args.real_joint_roms,
                "real2sim_noise": args.real2sim_noise,
                "append_ultrasonic_obs": args.append_ultrasonic_obs,
                "resume": str(args.resume) if args.resume else None,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    env_kwargs = dict(
        action=args.action,
        max_delta_frac=args.max_delta_frac,
        lock_wrist=args.lock_wrist,
        real_joint_roms=args.real_joint_roms,
        real2sim_noise=args.real2sim_noise,
        append_ultrasonic_obs=args.append_ultrasonic_obs,
    )
    if args.n_envs > 1:
        start_method = "forkserver" if os.name != "nt" else "spawn"
        train_env = SubprocVecEnv(
            [make_env(args.init, args.seed + i, **env_kwargs) for i in range(args.n_envs)],
            start_method=start_method,
        )
    else:
        train_env = DummyVecEnv([make_env(args.init, args.seed, **env_kwargs)])

    train_env = _load_or_extend_vecnormalize(train_env, args.resume)

    eval_env = DummyVecEnv([make_env(args.init, 10_000 + args.seed, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    class SyncNormEval(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            prev_best = self.best_mean_reward
            cont = super()._on_step()
            if self.best_mean_reward > prev_best:
                self.training_env.save(str(out / "vecnormalize.pkl"))
            return cont

    eval_cb = SyncNormEval(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(1, args.eval_every // args.n_envs),
        best_model_save_path=str(out),
        log_path=str(out),
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix=args.algo,
        save_vecnormalize=True,
    )

    model = build_model(args.algo, train_env, args.action, args.seed, args.device, resume_path=args.resume)
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    print(
        f"Training {args.algo.upper()} for {args.steps:,} steps on {args.n_envs} envs "
        f"(init={args.init}, action={args.action}, lock_wrist={args.lock_wrist}, "
        f"real_joint_roms={args.real_joint_roms}, real2sim_noise={args.real2sim_noise}, "
        f"append_ultrasonic_obs={args.append_ultrasonic_obs}) -> {out}"
    )
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=[eval_cb, ckpt_cb])
    model.save(out / "final_model")
    train_env.save(str(out / "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()
    print(f"Completed {args.algo.upper()} in {(time.time() - t0) / 60:.2f} min -> {out / 'progress.csv'}")


if __name__ == "__main__":
    main()
