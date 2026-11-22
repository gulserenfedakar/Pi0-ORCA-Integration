#!/bin/bash
#SBATCH --job-name=orca_real2sim_rl
#SBATCH --output=logs/orca_rl_%A_%a.out
#SBATCH --error=logs/orca_rl_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

# 4-task matched Sim-to-Real ablation array:
#   Task 0: PPO (70D: Locked Wrist + Physical ROMs + Calibrated Camera & 25-Ray Ultrasonic Obs, warm-started from 10M PPO)
#   Task 1: SAC (70D: Locked Wrist + Physical ROMs + Calibrated Camera & 25-Ray Ultrasonic Obs, warm-started from 10M SAC)
#   Task 2: PPO (68D: Locked Wrist + Physical ROMs + Calibrated Camera Noise Only, plug-and-play with 68D replay tools)
#   Task 3: PPO (68D: Locked Wrist + Physical ROMs + Ideal Noise-Free Sensing Control Baseline)
ALGOS=("ppo" "sac" "ppo" "ppo")
MODES=("real2sim_camera_ultrasonic_70d" "real2sim_camera_ultrasonic_70d" "real2sim_camera_68d" "ideal_locked_wrist_68d")

ALGO="${ALGOS[$SLURM_ARRAY_TASK_ID]}"
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-10000000}"
N_ENVS="${SLURM_CPUS_PER_TASK:-16}"
INIT_MODE="${INIT_MODE:-random}"
SEED="${SEED:-0}"

mkdir -p logs runs

if command -v module &>/dev/null; then
    module load python/3.11 2>/dev/null || true
fi

if [ -f "$HOME/orca_rl_venv/bin/activate" ]; then
    source "$HOME/orca_rl_venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ORCA_SIM_ROOT="${ORCA_SIM_ROOT:-$HOME/ORCA-Dexterous-Hand-Simulation}"

EXTRA_FLAGS=(--lock-wrist --real-joint-roms)
if [[ "${MODE}" == real2sim_* ]]; then
    EXTRA_FLAGS+=(--real2sim-noise)
fi
if [[ "${MODE}" == *_70d ]]; then
    EXTRA_FLAGS+=(--append-ultrasonic-obs)
fi

# Only warm-start from unlocked-wrist 10M checkpoints if USE_RESUME=1 is explicitly set.
# By default, all 4 ablation tasks train from scratch under matched kinematic/sensor conditions.
if [[ "${USE_RESUME:-0}" == "1" ]]; then
    RESUME_CKPT="models/drac_${ALGO}_10M/best_model.zip"
    if [[ "${MODE}" == real2sim_* ]] && [ -f "${RESUME_CKPT}" ]; then
        EXTRA_FLAGS+=(--resume "${RESUME_CKPT}")
    fi
fi

echo "Starting RL job: algo=${ALGO}, mode=${MODE}, steps=${STEPS}, n_envs=${N_ENVS}, seed=${SEED}"
python scripts/train_rl_reorient.py \
    --algo "${ALGO}" \
    --steps "${STEPS}" \
    --n-envs "${N_ENVS}" \
    --init "${INIT_MODE}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}" \
    --out "runs/${ALGO}_${MODE}_${STEPS}_seed${SEED}"
