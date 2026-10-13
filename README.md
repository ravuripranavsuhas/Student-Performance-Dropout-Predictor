# Student Dropout Prediction System

A beginner-friendly Streamlit machine-learning project for VS Code.

## Features
- Uses the UCI Predict Students' Dropout and Academic Success dataset.
- Predicts the actual `Dropout` target against `Graduate` and `Enrolled`.
- Trains Logistic Regression, Decision Tree, and Random Forest.
- Dashboard with outcome counts and model metrics.
- Individual prediction with estimated probability and configurable Low/Medium/High bands.
- CSV batch prediction and downloadable template.
- Local SQLite prediction history.
- Downloadable PDF reports.
- Confusion matrix, classification report, and feature-importance chart.

## Folder structure
```text
student_dropout_predictor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── student_input_template.csv
```

`student_history.db` is created automatically when the app starts.

## Run on Windows in VS Code
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Optional but recommended) create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, you can use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Run the app:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Internet requirement
The app downloads the public UCI dataset on first launch. Internet access is required. Streamlit caches the downloaded dataset and trained models during a session.

## Important notes
- The app uses the dataset's actual `Target` values. It treats `Dropout` as positive class 1 and groups `Graduate` and `Enrolled` as class 0.
- The probability and risk bands are estimates. Low/Medium/High thresholds are demonstration bands, not validated decision thresholds.
- Individual prediction form values use numeric-coded dataset fields. Read the feature help text and use the dataset's coding scheme.
- For bulk prediction, upload a CSV with the dataset's feature column names. Download the template from the app.
- Prediction history is stored in `student_history.db` in the project folder. Keep this database private.
- This is an educational prototype, not a substitute for human review or a validated institutional early-warning system.
