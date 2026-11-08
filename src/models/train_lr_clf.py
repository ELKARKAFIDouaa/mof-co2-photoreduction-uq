import sys, pickle
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.config_classification import (
    TARGET_CLF, CLASS_NAMES, CLF_MODELS_DIR, CLF_PKL_DIR, CLF_FIG_DIR,
    CLF_RESULTS_DIR, LR_CLF_PARAMS
)
from src.evaluation_classification import evaluate_clf, aggregate_cv_clf

MODEL_NAME = "LogisticRegression"

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Classification : {TARGET_CLF}")
print("=" * 60)

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()
print(f"Dataset: {df_model.shape[0]} rows ({df.shape[0]-df_model.shape[0]} NaN removed)")

# LR est sensible à l'échelle → StandardScaler
scaler   = StandardScaler()
le       = LabelEncoder()

X_raw = df_model[FEATURES].values
y_raw = df_model[TARGET_CLF].values

X = scaler.fit_transform(X_raw)
y = le.fit_transform(y_raw)      # encode classes en 0,1,2,...
classes_encoded = le.classes_
print(f"Classes : {classes_encoded} → {list(range(len(classes_encoded)))}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD
# =============================================================================
kf            = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics  = []
fold_preds    = np.zeros(len(y), dtype=int)
fold_probas   = np.zeros((len(y), len(classes_encoded)))
fold_bins     = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    m = LogisticRegression(**LR_CLF_PARAMS)
    m.fit(X_train, y_train)

    fold_preds[val_idx]     = m.predict(X_val)
    fold_probas[val_idx, :] = m.predict_proba(X_val)

    metrics, bins = evaluate_clf(
        y_val, fold_preds[val_idx], fold_probas[val_idx],
        class_names=list(classes_encoded),
        model_name=f"{MODEL_NAME}_fold{fold_idx}", verbose=True
    )
    fold_metrics.append(metrics)
    fold_bins.append(bins)

# =============================================================================
# 3. MÉTRIQUES OOF GLOBALES
# =============================================================================
print("\n--- Global OOF metrics ---")
oof_metrics, oof_bins = evaluate_clf(
    y, fold_preds, fold_probas,
    class_names=list(classes_encoded),
    model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_clf(fold_metrics)

# =============================================================================
# 4. ENTRAÎNEMENT FINAL
# =============================================================================
final_model = LogisticRegression(**LR_CLF_PARAMS)
final_model.fit(X, y)

# Sauvegarde modèle + scaler + encodeur
model_path = CLF_MODELS_DIR / "lr_clf.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model, "scaler": scaler, "label_encoder": le}, f)
print(f"Model saved: {model_path}")

# Sauvegarde OOF pour eval et compare
oof_path = CLF_PKL_DIR / "oof_lr.pkl"
with open(oof_path, "wb") as f:
    pickle.dump({
        "y_true"    : y,
        "y_pred"    : fold_preds,
        "y_proba"   : fold_probas,
        "oof_bins"  : oof_bins,
        "oof_metrics": oof_metrics,
        "cv_summary": cv_summary,
        "classes"   : classes_encoded,
    }, f)
print(f"OOF saved: {oof_path}")

# =============================================================================
# 5. MLFLOW
# =============================================================================
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"{MODEL_NAME}_clf") as run:
    mlflow.log_param("model_type", MODEL_NAME)
    for k, v in LR_CLF_PARAMS.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("oof_F1_macro", oof_metrics["F1_macro"])
    mlflow.log_metric("oof_AUC",      oof_metrics["AUC_OvR"])
    mlflow.log_metric("oof_LogLoss",  oof_metrics["LogLoss"])
    mlflow.log_metric("oof_ECE",      oof_metrics["ECE"])
    mlflow.log_metric("oof_Accuracy", oof_metrics["Accuracy"])
    for k, v in cv_summary.items():
        mlflow.log_metric(f"cv_{k}", v)
    mlflow.set_tag("pipeline_stage", "classification")
    mlflow.set_tag("model_type",     MODEL_NAME)
    mlflow.set_tag("target",         TARGET_CLF)
    mlflow.sklearn.log_model(final_model, f"model_{MODEL_NAME}")
    mlflow.log_artifact(str(model_path))
    print(f"MLflow run: {run.info.run_id}")

print(f"\nDone. OOF F1={oof_metrics['F1_macro']} | AUC={oof_metrics['AUC_OvR']}")