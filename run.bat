@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo [1/3] Creating virtual environment...
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)

echo [2/3] Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [3/3] Starting Tuna Hotspot PoC...
.venv\Scripts\python.exe -m streamlit run app.py
endlocal
