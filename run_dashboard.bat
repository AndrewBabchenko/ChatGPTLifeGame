@echo off
REM Run Training Dashboard App with virtual environment
echo Starting Training Dashboard...
.venv_rocm\Scripts\python.exe scripts\dashboard_app.py
pause
