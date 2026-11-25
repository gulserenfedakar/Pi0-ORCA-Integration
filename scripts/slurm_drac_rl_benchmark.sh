#!/bin/bash
#SBATCH --job-name=orca_real2sim_rl
#SBATCH --output=logs/orca_rl_%A_%a.out
#SBATCH --error=logs/orca_rl_%A_%a.err
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

# 9-task matched 10M-step x 3-seed Sim-to-Real ablation array on Joe's real2sim bench scene
# (top + side calibrated cameras, 25 ultrasonic rangefinders us_0..us_24, wrist locked at 9.6 deg):
#   Tasks 0-2: PPO Ideal Locked-Wrist 68D (seeds 0, 1, 2)
#   Tasks 3-5: PPO Real2Sim Top+Side Camera 68D (seeds 0, 1, 2)
#   Tasks 6-8: PPO Real2Sim Top+Side Camera + 25-Ray Ultrasonic 70D (seeds 0, 1, 2)
ALGOS=("ppo" "ppo" "ppo" "ppo" "ppo" "ppo" "ppo" "ppo" "ppo")
MODES=(
  "ideal_locked_wrist_68d" "ideal_locked_wrist_68d" "ideal_locked_wrist_68d"
  "real2sim_camera_68d" "real2sim_camera_68d" "real2sim_camera_68d"
  "real2sim_camera_ultrasonic_70d" "real2sim_camera_ultrasonic_70d" "real2sim_camera_ultrasonic_70d"
)
SEEDS=(0 1 2 0 1 2 0 1 2)

ALGO="${ALGOS[$SLURM_ARRAY_TASK_ID]}"
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-10000000}"
N_ENVS="${SLURM_CPUS_PER_TASK:-16}"
INIT_MODE="${INIT_MODE:-random}"
WRIST_LOCK_DEG="${WRIST_LOCK_DEG:--9.6}"
SCENE_MODE="${SCENE_MODE:-real2sim}"

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
export OPENBLAS_NUM_THREADS=1
export ORCA_SIM_ROOT="${ORCA_SIM_ROOT:-$HOME/ORCA-Dexterous-Hand-Simulation}"

EXTRA_FLAGS=(--scene "${SCENE_MODE}" --lock-wrist --wrist-lock-deg "${WRIST_LOCK_DEG}" --real-joint-roms)
if [[ "${MODE}" == real2sim_* ]]; then
    EXTRA_FLAGS+=(--real2sim-noise)
fi
if [[ "${MODE}" == *_70d ]]; then
    EXTRA_FLAGS+=(--append-ultrasonic-obs)
fi

if [[ "${USE_RESUME:-0}" == "1" ]]; then
    RESUME_CKPT="models/drac_${ALGO}_10M/best_model.zip"
    if [[ "${MODE}" == real2sim_* ]] && [ -f "${RESUME_CKPT}" ]; then
        EXTRA_FLAGS+=(--resume "${RESUME_CKPT}")
    fi
fi

echo "Starting RL job: algo=${ALGO}, mode=${MODE}, steps=${STEPS}, n_envs=${N_ENVS}, seed=${SEED}, wrist=${WRIST_LOCK_DEG}deg, scene=${SCENE_MODE}"
python scripts/train_rl_reorient.py \
    --algo "${ALGO}" \
    --steps "${STEPS}" \
    --n-envs "${N_ENVS}" \
    --init "${INIT_MODE}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}" \
    --out "runs/${ALGO}_${MODE}_${STEPS}_seed${SEED}"
