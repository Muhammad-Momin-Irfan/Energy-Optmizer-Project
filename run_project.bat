@echo off
echo ==========================================
echo 🚀 STARTING SMART HOTEL PIPELINE
echo ==========================================

echo.
echo [1/5] Activating Virtual Environment...
call .\venv\Scripts\activate.bat

echo.
echo [2/5] Running Preprocessing (Step 2)...
python scripts\step2_preprocessing.py

echo.
echo [3/5] Generating EDA Graphs (Step 3)...
python scripts\step3_eda.py

echo.
echo [4/5] Training AI Models (Step 4)...
python scripts\step4_modeling.py

echo.
echo [5/5] Testing Logic (Steps 5 & 6)...
python scripts\step5_6_prediction.py

echo.
echo 🎉 Pipeline Complete! Launching the Web App...
python -m streamlit run app\main.py

pause