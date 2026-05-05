#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-hdmi}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-hdmi}"
HEADLESS="${HEADLESS:-true}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEACHER_DATASET="${TEACHER_DATASET:-locomotion}"
REF_DATASET="${REF_DATASET:-chair}"
TEACHER_ALGO="${TEACHER_ALGO:-ppo_roa_train}"
REF_ALGO="${REF_ALGO:-ppo_roa_finetune}"

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[error] conda not found. Set up conda on PATH or source it before running this script."
  exit 1
fi

conda activate "$CONDA_ENV"

if [[ $restore_nounset -eq 1 ]]; then
  set -u
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR"
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_custom_object_011_curriculum_${TIMESTAMP}.log"

CMD=(
  "$PYTHON_BIN" scripts/train_custom_object_011_curriculum.py
  "--teacher-dataset" "$TEACHER_DATASET"
  "--ref-dataset" "$REF_DATASET"
  "--teacher-algo" "$TEACHER_ALGO"
  "--ref-algo" "$REF_ALGO"
)

if [[ -n "${TEACHER_TOTAL_FRAMES:-}" ]]; then
  CMD+=("--teacher-total-frames" "$TEACHER_TOTAL_FRAMES")
fi

if [[ -n "${REF_TOTAL_FRAMES:-}" ]]; then
  CMD+=("--ref-total-frames" "$REF_TOTAL_FRAMES")
fi

if [[ -n "${TEACHER_NUM_ENVS:-}" ]]; then
  CMD+=("--teacher-override" "task.num_envs=$TEACHER_NUM_ENVS")
fi

if [[ -n "${REF_NUM_ENVS:-}" ]]; then
  CMD+=("--ref-override" "task.num_envs=$REF_NUM_ENVS")
fi

if [[ -n "$WANDB_MODE" ]]; then
  CMD+=("wandb.mode=$WANDB_MODE")
fi

if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=("wandb.project=$WANDB_PROJECT")
fi

if [[ -n "$HEADLESS" ]]; then
  CMD+=("headless=$HEADLESS")
fi

if [[ -n "${NUM_ENVS:-}" ]]; then
  CMD+=("task.num_envs=$NUM_ENVS")
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
echo "[info] WANDB_MODE=$WANDB_MODE"
echo "[info] WANDB_PROJECT=$WANDB_PROJECT"
echo "[info] HEADLESS=$HEADLESS"
echo "[info] TEACHER_DATASET=$TEACHER_DATASET"
echo "[info] REF_DATASET=$REF_DATASET"
echo "[info] TEACHER_ALGO=$TEACHER_ALGO"
echo "[info] REF_ALGO=$REF_ALGO"
echo "[info] Log file: $LOG_FILE"
echo "[info] Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
