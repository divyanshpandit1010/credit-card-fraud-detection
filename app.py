"""
Credit Card Fraud Detection API
Author: Divyansh Pandit
GitHub: github.com/divyanshpandit1010
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detector")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float

@app.get("/")
def home():
    return FileResponse("./dashboard.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("./dashboard.html")

@app.post("/predict")
def predict(t: Transaction):
    amount_scaled = scaler.transform([[t.Amount]])[0][0]
    time_scaled = scaler.transform([[t.Time]])[0][0]
    
    features = np.array([[
        t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7,
        t.V8, t.V9, t.V10, t.V11, t.V12, t.V13, t.V14,
        t.V15, t.V16, t.V17, t.V18, t.V19, t.V20, t.V21,
        t.V22, t.V23, t.V24, t.V25, t.V26, t.V27, t.V28,
        amount_scaled, time_scaled
    ]])
    
    prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

# Lower threshold to 0.3 instead of default 0.5
is_fraud = bool(probability >= 0.3)

return {
    "is_fraud": is_fraud,
    "fraud_probability": round(float(probability), 4),
    "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
}
