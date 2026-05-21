"""
Employee Attrition Prediction - Training Pipeline
Author: Smeet Patel | M.Tech CSE, DTU
"""
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SEED = 42


def load_data(path: str = "data/HR-Employee-Attrition.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"],
            errors="ignore", inplace=True)
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[INFO] After SMOTE: {np.bincount(y_res)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    return X_scaled, y_res, scaler, encoders, X.columns.tolist()


def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=SEED, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            use_label_encoder=False, eval_metric="logloss", random_state=SEED),
    }
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        results[name] = {"model": model, "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std()}
        print(f"[{name}] CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
    print(f"\n[INFO] Best model: {best_name}")
    return best_name, results[best_name]["model"], results


def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
    print("\n" + "="*50)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(classification_report(y_test, y_pred))
    return metrics


def main():
    os.makedirs("models", exist_ok=True)
    df = load_data()
    X, y, scaler, encoders, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    best_name, best_model, _ = train_models(X_train, y_train)
    best_model.fit(X_train, y_train)
    evaluate(best_model, X_test, y_test, best_name)
    joblib.dump(best_model, "models/attrition_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("\n[INFO] Artifacts saved to models/")


if __name__ == "__main__":
    main()
