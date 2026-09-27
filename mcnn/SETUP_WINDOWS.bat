@echo off
cd /d "%~dp0"
echo ================================================================
echo Mammography CNN Project - Windows Setup
echo ================================================================
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.12 was not found.
  echo Install Python 3.12 and make sure the py launcher is available.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto :fail
)
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
echo Installing project requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.
echo SETUP COMPLETE.
echo Next: STEP1_PREPARE_DATASET.bat
pause
exit /b 0
:fail
echo.
echo SETUP FAILED. Review the error above.
pause
exit /b 1
