@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   tiny-model dashboard
echo ============================================
echo.
echo Starting dashboard server...
echo Browser will open http://127.0.0.1:5000
echo Close this window to stop the dashboard.
echo.
start "" /b cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5000"
python dashboard.py
echo.
echo Dashboard stopped.
pause
