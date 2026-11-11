#!/bin/bash
#SBATCH --job-name=orca_reorient_rl
#SBATCH --output=logs/orca_rl_%A_%a.out
#SBATCH --error=logs/orca_rl_%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

ALGOS=("ppo" "sac" "td3")
ALGO="${ALGOS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-10000000}"
N_ENVS="${SLURM_CPUS_PER_TASK:-16}"
INIT_MODE="${INIT_MODE:-random}"
SEED="${SEED:-0}"

mkdir -p logs runs

if command -v module &>/dev/null; then
    module load python/3.11 scipy-stack mujoco 2>/dev/null || true
fi

if [ -f "$HOME/orca_rl_venv/bin/activate" ]; then
    source "$HOME/orca_rl_venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ORCA_SIM_ROOT="${ORCA_SIM_ROOT:-$HOME/SEP730-Dexterous-Hand-Simulation}"

echo "Starting matched RL benchmark: algo=${ALGO}, steps=${STEPS}, n_envs=${N_ENVS}, init=${INIT_MODE}"
python scripts/train_rl_reorient.py \
    --algo "${ALGO}" \
    --steps "${STEPS}" \
    --n-envs "${N_ENVS}" \
    --init "${INIT_MODE}" \
    --seed "${SEED}" \
    --out "runs/${ALGO}_reorient_${INIT_MODE}_10M"
