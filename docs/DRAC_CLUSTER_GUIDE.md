# DRAC (Digital Research Alliance of Canada) CPU Cluster Guide

This guide explains how to run the matched 10M-step Reinforcement Learning benchmark (`PPO`, `SAC`, `TD3`) on the 17-DoF `OrcaHandRightCubeOrientation` task using the Digital Research Alliance of Canada (DRAC) **Nibi** cluster (`nibi.alliancecan.ca`).

Because `OrcaHandRightCubeOrientation` uses low-dimensional proprioceptive and cube-pose observations (without camera rendering during pure RL training), it runs entirely on multi-core CPUs at `~2,409 FPS` for on-policy `PPO` and `~473–527 FPS` for off-policy `SAC` / `TD3` on 16 CPU cores and does not require GPU allocation.

---

## 1. One-Time Environment Setup on Nibi (`nibi.alliancecan.ca`)

Log in to the Nibi cluster via SSH:

```bash
ssh gfedakar@nibi.alliancecan.ca
```

Clone `Pi0-ORCA-Integration` inside your `$HOME` or `$PROJECT` directory (and ensure the `orca_sim` MuJoCo simulation package is installed or set via `ORCA_SIM_ROOT`):

```bash
cd ~
git clone https://github.com/gulserenfedakar/Pi0-ORCA-Integration.git
```

Create a Python 3.11 virtual environment using DRAC's prebuilt wheels (`--no-index` where available):

```bash
module load python/3.11 scipy-stack
virtualenv --no-download ~/orca_rl_venv
source ~/orca_rl_venv/bin/activate

pip install --upgrade pip
pip install numpy gymnasium mujoco torch stable-baselines3
pip install -e "$ORCA_SIM_ROOT"
pip install -e ~/Pi0-ORCA-Integration --no-deps
```

---

## 2. Submitting the Matched 10M-Step RL Benchmark (Slurm Job Array)

The script [`scripts/slurm_drac_rl_benchmark.sh`](../scripts/slurm_drac_rl_benchmark.sh) launches a 3-task Slurm job array (`--array=0-2`) that trains **`PPO`**, **`SAC`**, and **`TD3`** in parallel under identical conditions:
* **Environment:** `OrcaHandRightCubeOrientation(version="v2", initial_red_face="random")`
* **Action Wrapper:** `RelativeJointTargetAction(max_delta_frac=0.05)`
* **Normalization:** `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)`
* **Parallel Workers:** `16` (`SubprocVecEnv`)
* **Total Steps:** `10,000,000` steps per algorithm (`EvalCallback` every `200,000` steps over `20` episodes)

Submit the job array on Nibi (check your active Slurm account name via `sacctmgr show assoc user=$USER format=Account%20` if needed):

```bash
cd ~/Pi0-ORCA-Integration
sbatch --account=def-<pi_username> scripts/slurm_drac_rl_benchmark.sh
```

Monitor job progress in the Slurm queue:

```bash
sq
tail -f logs/orca_rl_*_1.out
```

Once the jobs complete, evaluate all trained checkpoints (`PPO`, `SAC`, `TD3`) across 20 held-out episodes to generate `results/rl_reorient_benchmark.csv`:

```bash
python scripts/evaluate_rl_reorient.py --runs-dir runs --episodes 20
```

---

## 3. Outputs Produced per Algorithm

Each run writes its artifacts to `runs/<algo>_reorient_random_10M/`:
* `progress.csv`: Training curves including `eval/success_rate`, `eval/mean_reward`, `eval/mean_ep_length`, and `time/fps`.
* `best_model.zip`: Checkpoint with the highest deterministic evaluation success rate.
* `final_model.zip`: Policy at 10,000,000 steps.
* `vecnormalize.pkl`: Running observation and reward normalization statistics.
* `checkpoints/`: Periodic snapshots saved every `500,000` steps.
