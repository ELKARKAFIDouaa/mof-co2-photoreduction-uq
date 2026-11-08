import sys, pickle
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.config_classification import (
    TARGET_CLF, CLF_MODELS_DIR, CLF_PKL_DIR, SVM_CLF_PARAMS
)
from src.evaluation_classification import evaluate_clf, aggregate_cv_clf

MODEL_NAME = "SVM_RBF"

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()

# SVM sensible à l'échelle → StandardScaler
scaler = StandardScaler()
le     = LabelEncoder()
X      = scaler.fit_transform(df_model[FEATURES].values)
y      = le.fit_transform(df_model[TARGET_CLF].values)
classes_encoded = le.classes_
print(f"Classes : {list(classes_encoded)}")

kf           = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds   = np.zeros(len(y), dtype=int)
fold_probas  = np.zeros((len(y), len(classes_encoded)))
fold_bins    = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    m = SVC(**SVM_CLF_PARAMS)
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

oof_metrics, oof_bins = evaluate_clf(
    y, fold_preds, fold_probas,
    class_names=list(classes_encoded),
    model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_clf(fold_metrics)

final_model = SVC(**SVM_CLF_PARAMS)
final_model.fit(X, y)

model_path = CLF_MODELS_DIR / "svm_clf.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model, "scaler": scaler, "label_encoder": le}, f)

oof_path = CLF_PKL_DIR / "oof_svm.pkl"
with open(oof_path, "wb") as f:
    pickle.dump({
        "y_true": y, "y_pred": fold_preds, "y_proba": fold_probas,
        "oof_bins": oof_bins, "oof_metrics": oof_metrics,
        "cv_summary": cv_summary, "classes": classes_encoded,
    }, f)

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"{MODEL_NAME}_clf") as run:
    mlflow.log_param("model_type", MODEL_NAME)
    for k, v in SVM_CLF_PARAMS.items():
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
    mlflow.log_artifact(str(model_path))
    print(f"MLflow run: {run.info.run_id}")

print(f"\nDone. OOF F1={oof_metrics['F1_macro']} | AUC={oof_metrics['AUC_OvR']}")