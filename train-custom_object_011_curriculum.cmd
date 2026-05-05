@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

echo [info] Repo root: %ROOT%
echo [info] Python: %PYTHON_EXE%

"%PYTHON_EXE%" scripts\train_custom_object_011_curriculum.py %*

exit /b %errorlevel%
