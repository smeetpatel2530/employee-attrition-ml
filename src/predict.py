"""
Employee Attrition Prediction - Inference Modules
"""
import joblib
import pandas as pd

_model = _scaler = _features = None


def _load():
    global _model, _scaler, _features
    if _model is None:
        _model    = joblib.load("models/attrition_model.pkl")
        _scaler   = joblib.load("models/scaler.pkl")
        _features = joblib.load("models/feature_names.pkl")


def predict_attrition(input_dict: dict) -> dict:
    _load()
    df = pd.DataFrame([input_dict])
    for col in _features:
        if col not in df.columns:
            df[col] = 0
    df = df[_features]
    X_scaled = _scaler.transform(df)
    prob  = _model.predict_proba(X_scaled)[0][1]
    label = "Yes" if prob >= 0.5 else "No"
    risk  = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    return {
        "attrition_probability": round(float(prob) * 100, 2),
        "attrition_label": label,
        "risk_level": risk
    }
