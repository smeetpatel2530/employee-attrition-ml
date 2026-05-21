# Employee Attrition Prediction System
> **Author:** Smeet Patel | M.Tech CSE, Delhi Technological University | 2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange?logo=mlflow)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An end-to-end machine learning system that predicts employee attrition risk using the IBM HR Analytics dataset. Built with a production-grade pipeline: data preprocessing → SMOTE balancing → model comparison → MLflow tracking → Flask REST API → deployment on Render.

---

## Live Demo
🚀 **[Try it live →](https://employee-attrition-smeet.onrender.com)**

---

## Problem Statement
Employee attrition costs companies 33–200% of an employee's annual salary. This system uses ML to identify at-risk employees early, enabling proactive retention.

**Dataset:** IBM HR Analytics — 1,470 employees, 35 features.

---

## Architecture

```
Data Ingestion (CSV)
        │
        ▼
Preprocessing Pipeline
  ├── Label encoding
  ├── SMOTE (handle 16% class imbalance)
  └── StandardScaler
        │
        ▼
Model Comparison (MLflow tracked)
  ├── Random Forest   (CV F1: 0.84)
  ├── Gradient Boost  (CV F1: 0.86)
  └── XGBoost ← Best (CV F1: 0.89)
        │
        ▼
Flask REST API → Render Deployment
```

---

## Results

| Model | CV F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.84 | 0.91 | 0.83 | 0.85 |
| Gradient Boosting | 0.86 | 0.93 | 0.85 | 0.87 |
| **XGBoost (Best)** | **0.89** | **0.95** | **0.88** | **0.90** |

**Key EDA Findings:**
- Overtime employees are 3x more likely to leave
- Monthly income < $3,000 is the strongest predictor
- First 3 years at company = peak attrition risk
- Low job satisfaction (1–2/4) doubles attrition probability

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | pandas, NumPy, scikit-learn |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Models | XGBoost, Random Forest, Gradient Boosting |
| Experiment Tracking | MLflow |
| API | Flask, Gunicorn |
| Deployment | Render |

---

## Setup

```bash
git clone https://github.com/smeet-patel/employee-attrition-ml.git
cd employee-attrition-ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset from Kaggle → place at data/HR-Employee-Attrition.csv
python src/train.py     # train model
python app.py           # start Flask at http://localhost:5000
```

---

## API

**POST** `/predict`
```json
{ "Age": 32, "MonthlyIncome": 3000, "OverTime": 1, "JobSatisfaction": 1,
  "YearsAtCompany": 2, "WorkLifeBalance": 1, "DistanceFromHome": 25,
  "NumCompaniesWorked": 5, "TotalWorkingYears": 6, "YearsInCurrentRole": 1,
  "TrainingTimesLastYear": 1, "JobRole": 8 }
```
**Response:**
```json
{ "success": true, "result": { "attrition_probability": 78.4, "attrition_label": "Yes", "risk_level": "High" } }
```

---

## Project Structure

```
employee-attrition-ml/
├── src/
│   ├── train.py       # Full training pipeline
│   └── predict.py     # Inference module
├── notebooks/
│   └── 01_EDA.ipynb   # Exploratory data analysis
├── templates/
│   └── index.html     # Flask web UI
├── data/              # Dataset (download from Kaggle)
├── models/            # Saved artifacts (auto-generated)
├── app.py             # Flask entry point
├── Procfile           # Render/Heroku deploy config
└── requirements.txt
```

---

## Future Work
- [ ] SHAP values for prediction explainability
- [ ] Streamlit monitoring dashboard
- [ ] Docker + GitHub Actions CI/CD
- [ ] Multi-class risk tiering (Low / Medium / High)

---

## License
MIT © 2026 Smeet Patel
