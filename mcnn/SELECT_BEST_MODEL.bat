@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m src.select_best
pause
