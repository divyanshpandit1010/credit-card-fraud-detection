# Credit Card Fraud Detection

ML model to detect fraudulent credit card transactions, deployed as a live REST API.

## Results
- **F1-Score (Fraud class):** 0.87
- **ROC-AUC Score:** 0.97
- **Dataset:** 284,807 transactions, 0.17% fraud rate

## Tech Stack
- Python
- Scikit-learn
- SMOTE (imbalanced-learn)
- FastAPI
- Render (deployment)

## Key Decisions
- Used **SMOTE** to handle severe class imbalance (99.83% normal)
- Chose **F1-score** over accuracy as main metric
- **Random Forest** for robustness and interpretability

## Live API
https://fraud-detection-api.onrender.com/docs

## How to Run Locally
pip install -r requirements.txt
uvicorn app:app --reload
