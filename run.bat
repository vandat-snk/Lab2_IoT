@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Chua co moi truong Python. Hay chay setup.ps1 theo README.md.
  pause
  exit /b 1
)
echo Mo http://localhost:8000 trong trinh duyet. Bam Ctrl+C de dung backend.
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
pause
