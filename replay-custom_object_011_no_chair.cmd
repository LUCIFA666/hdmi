@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
if "%CUDA_VISIBLE_DEVICES%"=="" set "CUDA_VISIBLE_DEVICES=0"
if "%WANDB_MODE%"=="" set "WANDB_MODE=disabled"

set "TASK=G1/hdmi/custom_object_011_no_chair"
set "ALGO=ppo_roa_train"

echo [info] Repo root: %ROOT%
echo [info] Python: %PYTHON_EXE%
echo [info] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [info] Task=%TASK%
echo [info] Replaying reference motion without chair object

"%PYTHON_EXE%" scripts/play.py ^
  algo=%ALGO% ^
  task=%TASK% ^
  wandb.mode=%WANDB_MODE% ^
  +task.command.replay_motion=true ^
  task.num_envs=1 ^
  %*

exit /b %errorlevel%
