#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-hdmi}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-hdmi}"
TASK="${TASK:-G1/hdmi/custom_object_011_no_chair}"
ALGO="${ALGO:-ppo_roa_train}"
EXP_NAME="${EXP_NAME:-custom_object_011_no_chair_server}"
HEADLESS="${HEADLESS:-true}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Common Miniconda install path on Linux servers.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[error] conda not found. Set up conda on PATH or source it before running this script."
  exit 1
fi

conda activate "$CONDA_ENV"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR"
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${EXP_NAME}_${TIMESTAMP}.log"

CMD=(
  "$PYTHON_BIN" scripts/train.py
  "algo=$ALGO"
  "task=$TASK"
  "headless=$HEADLESS"
  "wandb.mode=$WANDB_MODE"
  "wandb.project=$WANDB_PROJECT"
  "exp_name=$EXP_NAME"
)

if [[ -n "${NUM_ENVS:-}" ]]; then
  CMD+=("task.num_envs=$NUM_ENVS")
fi

if [[ -n "${TOTAL_FRAMES:-}" ]]; then
  CMD+=("total_frames=$TOTAL_FRAMES")
fi

if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  CMD+=("save_interval=$SAVE_INTERVAL")
fi

if [[ -n "${SEED:-}" ]]; then
  CMD+=("seed=$SEED")
fi

CMD+=("$@")

echo "[info] Repo root: $ROOT_DIR"
echo "[info] Conda env: $CONDA_ENV"
echo "[info] Python: $(command -v "$PYTHON_BIN")"
echo "[info] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[info] Task=$TASK"
echo "[info] Algo=$ALGO"
echo "[info] wandb.mode=$WANDB_MODE"
echo "[info] Log file: $LOG_FILE"
echo "[info] Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
