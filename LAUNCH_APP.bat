@echo off
echo ========================================
echo   Advanced Data Science Platform
echo   Quick Launch Script
echo ========================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
portable_python\python.exe -m streamlit run app.py

echo.
echo Press any key to exit...
pause >nul
