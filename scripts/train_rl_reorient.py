"""Matched Reinforcement Learning benchmark trainer (PPO, SAC, TD3) on the 17-DoF ORCA
in-hand cube reorientation task (OrcaHandRightCubeOrientation).

Designed for multi-core CPU execution on local workstations and Digital Research
Alliance of Canada (DRAC) Slurm clusters under identical environment wrappers,
observation/reward normalization, and held-success evaluation protocols.

Examples:
    python scripts/train_rl_reorient.py --algo ppo --steps 10_000_000 --n-envs 16
    python scripts/train_rl_reorient.py --algo sac --steps 10_000_000 --n-envs 16
    python scripts/train_rl_reorient.py --algo td3 --steps 10_000_000 --n-envs 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def make_env(init: str, seed: int, action: str = "relative", max_delta_frac: float = 0.05):
    def _thunk():
        from orca_sim import OrcaHandRightCubeOrientation
        from orca_sim.wrappers import RelativeJointTargetAction

        env = OrcaHandRightCubeOrientation(version="v2", initial_red_face=init)
        if action == "relative":
            env = RelativeJointTargetAction(env, max_delta_frac=max_delta_frac)
        env = Monitor(env, info_keywords=("is_success", "goal_reached", "success_streak", "dropped"))
        env.reset(seed=seed)
        return env

    return _thunk


def build_model(algo: str, train_env: VecNormalize, action_mode: str, seed: int, device: str):
    if algo == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=512,
            batch_size=2048,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-0.5 if action_mode == "relative" else 0.0,
            ),
            device=device,
            seed=seed,
            verbose=0,
        )
    if algo == "sac":
        return SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            seed=seed,
            verbose=0,
        )
    if algo == "td3":
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=0.15 * np.ones(n_actions, dtype=np.float32),
        )
        return TD3(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            seed=seed,
            verbose=0,
        )
    raise ValueError(f"Unsupported algorithm: {algo}")


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
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (REPO_ROOT / "runs" / f"{args.algo}_reorient_{args.init}_{timestamp}")
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
                "seed": args.seed,
            },
            indent=2,
        )
    )

    env_kwargs = dict(action=args.action, max_delta_frac=args.max_delta_frac)
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = vec_cls([make_env(args.init, args.seed + i, **env_kwargs) for i in range(args.n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(args.init, 10_000 + args.seed, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    class SyncNormEval(EvalCallback):
        def _on_step(self) -> bool:
            self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

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

    model = build_model(args.algo, train_env, args.action, args.seed, args.device)
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    print(
        f"Training {args.algo.upper()} for {args.steps:,} steps on {args.n_envs} envs "
        f"(init={args.init}, action={args.action}) -> {out}"
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
