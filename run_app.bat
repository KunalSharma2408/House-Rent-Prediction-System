@echo off
echo ========================================================
echo High-Level House Price Predictor
echo Made by: Kunal Sharma, IITG, BSc AI and DS
echo ========================================================
echo.

echo [1/3] Installing necessary dependencies...
pip install -q pandas scikit-learn streamlit joblib
echo Dependencies installed successfully.
echo.

if not exist "model.pkl" (
    echo [2/3] Model not found. Training the machine learning model...
    python train_model.py
    echo.
) else (
    echo [2/3] Model already exists. Skipping training.
    echo.
)

echo [3/3] Starting the Streamlit application...
streamlit run app.py
pause
