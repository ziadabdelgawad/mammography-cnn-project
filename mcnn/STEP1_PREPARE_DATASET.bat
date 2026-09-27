@echo off
cd /d "%~dp0"
echo ================================================================
echo STEP 1 - REGISTER AND VERIFY REVISED CBIS-DDSM
echo ================================================================
echo Expected source folder: train, val and test
echo Each split must contain: benign and malign
echo The images will NOT be copied into this project.
echo.
set /p SRC=Paste the FULL path to the INNER cbis_ddsm_rev folder: 
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m src.prepare_cbis_ddsm "%SRC%" --overwrite
pause
