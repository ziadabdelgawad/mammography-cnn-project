@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m src.train --model regularized --epochs 35
pause
