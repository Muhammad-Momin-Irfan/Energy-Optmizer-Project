@echo off
echo ==========================================
echo 🚀 LAUNCHING SMART HOTEL DASHBOARD
echo ==========================================

echo.
echo Activating Virtual Environment...
call .\venv\Scripts\activate.bat

echo.
echo Starting Streamlit...
python -m streamlit run app\main.py

pause