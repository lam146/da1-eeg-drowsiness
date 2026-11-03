@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua co moi truong .venv. Xem docs/HUONG_DAN_UNG_DUNG_ML.md
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m n16_eeg.ml_gui
if errorlevel 1 (
  echo Ung dung gap loi. Hay gui noi dung loi de kiem tra.
  pause
)
endlocal
