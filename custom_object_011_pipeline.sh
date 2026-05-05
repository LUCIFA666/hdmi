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
LOG_FILE="$LOG_DIR/custom_object_011_pipeline_${TIMESTAMP}.log"

CMD=(
  "$PYTHON_BIN" scripts/custom_object_011_pipeline.py
  "$@"
)

has_override() {
  local prefix="$1"
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$prefix"* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "$WANDB_MODE" ]] && ! has_override "wandb.mode=" "$@"; then
  CMD+=("wandb.mode=$WANDB_MODE")
fi

if [[ -n "$WANDB_PROJECT" ]] && ! has_override "wandb.project=" "$@"; then
  CMD+=("wandb.project=$WANDB_PROJECT")
fi

if [[ -n "$HEADLESS" ]] && ! has_override "headless=" "$@"; then
  CMD+=("headless=$HEADLESS")
fi

echo "[info] Repo root: $ROOT_DIR"
echo "[info] Conda env: $CONDA_ENV"
echo "[info] Python: $(command -v "$PYTHON_BIN")"
echo "[info] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[info] WANDB_MODE=$WANDB_MODE"
echo "[info] WANDB_PROJECT=$WANDB_PROJECT"
echo "[info] HEADLESS=$HEADLESS"
echo "[info] Log file: $LOG_FILE"
echo "[info] Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
