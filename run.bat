@echo off
cd /d "%~dp0"

echo ===================================================
echo              LAUNCHING DONUTS
echo ===================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please create the .venv folder first.
    pause
    exit /b
)

echo [1/2] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/2] Starting DONUTS dashboard...
python dashboard\dashboard.py

echo ===================================================
echo DONUTS has stopped.
echo ===================================================

pause