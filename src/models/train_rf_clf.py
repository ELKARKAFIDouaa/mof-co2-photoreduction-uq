import sys, pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.config_classification import (
    TARGET_CLF, CLF_MODELS_DIR, CLF_PKL_DIR, RF_CLF_PARAMS
)
from src.evaluation_classification import evaluate_clf, aggregate_cv_clf

MODEL_NAME = "RandomForest"

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Classification : {TARGET_CLF}")
print("=" * 60)

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()
print(f"Dataset: {df_model.shape[0]} rows ({df.shape[0]-df_model.shape[0]} NaN removed)")

# RF n'a pas besoin de scaling — LabelEncoder suffit
le    = LabelEncoder()
X     = df_model[FEATURES].values
y     = le.fit_transform(df_model[TARGET_CLF].values)
classes_encoded = le.classes_
print(f"Classes : {list(classes_encoded)}")

# =============================================================================
# 2. CROSS-VALIDATION
# =============================================================================
kf           = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds   = np.zeros(len(y), dtype=int)
fold_probas  = np.zeros((len(y), len(classes_encoded)))
fold_bins    = []
importances  = np.zeros(len(FEATURES))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    m = RandomForestClassifier(**RF_CLF_PARAMS)
    m.fit(X_train, y_train)

    fold_preds[val_idx]     = m.predict(X_val)
    fold_probas[val_idx, :] = m.predict_proba(X_val)
    importances            += m.feature_importances_

    metrics, bins = evaluate_clf(
        y_val, fold_preds[val_idx], fold_probas[val_idx],
        class_names=list(classes_encoded),
        model_name=f"{MODEL_NAME}_fold{fold_idx}", verbose=True
    )
    fold_metrics.append(metrics)
    fold_bins.append(bins)

importances /= CV_FOLDS
importance_dict = dict(zip(FEATURES, importances))

# =============================================================================
# 3. OOF GLOBALES
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
final_model = RandomForestClassifier(**RF_CLF_PARAMS)
final_model.fit(X, y)

model_path = CLF_MODELS_DIR / "rf_clf.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model, "label_encoder": le}, f)
print(f"Model saved: {model_path}")

oof_path = CLF_PKL_DIR / "oof_rf.pkl"
with open(oof_path, "wb") as f:
    pickle.dump({
        "y_true"        : y,
        "y_pred"        : fold_preds,
        "y_proba"       : fold_probas,
        "oof_bins"      : oof_bins,
        "oof_metrics"   : oof_metrics,
        "cv_summary"    : cv_summary,
        "classes"       : classes_encoded,
        "importances"   : importance_dict,
    }, f)
print(f"OOF saved: {oof_path}")

# =============================================================================
# 5. MLFLOW
# =============================================================================
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"{MODEL_NAME}_clf") as run:
    mlflow.log_param("model_type", MODEL_NAME)
    for k, v in RF_CLF_PARAMS.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("oof_F1_macro", oof_metrics["F1_macro"])
    mlflow.log_metric("oof_AUC",      oof_metrics["AUC_OvR"])
    mlflow.log_metric("oof_LogLoss",  oof_metrics["LogLoss"])
    mlflow.log_metric("oof_ECE",      oof_metrics["ECE"])
    mlflow.log_metric("oof_Accuracy", oof_metrics["Accuracy"])
    for k, v in cv_summary.items():
        mlflow.log_metric(f"cv_{k}", v)
    # Importance top 10
    for rank, (feat, score) in enumerate(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        clean = feat.replace(".", "_")[:80]
        mlflow.log_metric(f"imp_rank{rank:02d}_{clean}", round(float(score), 6))
    mlflow.set_tag("pipeline_stage", "classification")
    mlflow.set_tag("model_type",     MODEL_NAME)
    mlflow.set_tag("target",         TARGET_CLF)
    mlflow.sklearn.log_model(final_model, f"model_{MODEL_NAME}")
    mlflow.log_artifact(str(model_path))
    print(f"MLflow run: {run.info.run_id}")

print(f"\nDone. OOF F1={oof_metrics['F1_macro']} | AUC={oof_metrics['AUC_OvR']}")