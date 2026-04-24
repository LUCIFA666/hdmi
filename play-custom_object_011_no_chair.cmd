@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
if "%CUDA_VISIBLE_DEVICES%"=="" set "CUDA_VISIBLE_DEVICES=0"
if "%WANDB_MODE%"=="" set "WANDB_MODE=disabled"

set "TASK=G1/hdmi/custom_object_011_no_chair"
set "ALGO=ppo_roa_train"

if "%~1"=="" (
  echo [error] Missing checkpoint_path argument.
  echo [hint] Example:
  echo [hint]   play-custom_object_011_no_chair.cmd run:your_wandb_run_path
  exit /b 1
)

set "CHECKPOINT=%~1"

echo [info] Repo root: %ROOT%
echo [info] Python: %PYTHON_EXE%
echo [info] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [info] Task=%TASK%
echo [info] Checkpoint=%CHECKPOINT%

"%PYTHON_EXE%" scripts/play.py ^
  algo=%ALGO% ^
  task=%TASK% ^
  task.num_envs=1 ^
  checkpoint_path=%CHECKPOINT% ^
  wandb.mode=%WANDB_MODE%

exit /b %errorlevel%
