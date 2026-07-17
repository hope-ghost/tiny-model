@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Install dependencies
echo ============================================
echo.
echo [1/2] Installing PyTorch (CUDA 12.8)...
pip install torch --index-url https://download.pytorch.org/whl/cu128
echo.
echo [2/2] Installing other dependencies...
pip install "tokenizers>=0.15.0" "numpy>=1.24.0" "tqdm>=4.66.0" flask
echo.
echo Done.
pause
