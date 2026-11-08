import sys, pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.config_classification import (
    TARGET_CLF, CLF_MODELS_DIR, CLF_PKL_DIR,
    XGB_CLF_PARAMS, XGB_CLF_EARLY_STOPPING
)
from src.evaluation_classification import (
    evaluate_clf, aggregate_cv_clf, compute_sample_weights_clf
)

MODEL_NAME = "XGBoost"

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Classification : {TARGET_CLF}")
print("=" * 60)

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()
le       = LabelEncoder()
X        = df_model[FEATURES].values
y        = le.fit_transform(df_model[TARGET_CLF].values)
classes_encoded = le.classes_
n_classes       = len(classes_encoded)
print(f"Classes : {list(classes_encoded)}")

xgb_init = {k: v for k, v in XGB_CLF_PARAMS.items()}
# num_class requis pour multiclasse
xgb_init["num_class"]      = n_classes
xgb_init["objective"]      = "multi:softprob"

# =============================================================================
# 2. CROSS-VALIDATION avec early stopping + sample_weight
# =============================================================================
kf            = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics  = []
fold_preds    = np.zeros(len(y), dtype=int)
fold_probas   = np.zeros((len(y), n_classes))
fold_bins     = []
importances   = np.zeros(len(FEATURES))
best_iters    = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_full, X_val = X[train_idx], X[val_idx]
    y_train_full, y_val = y[train_idx], y[val_idx]

    # sous-ensemble interne pour early stopping
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED
    )
    sw_tr = compute_sample_weights_clf(y_tr)

    m = XGBClassifier(**xgb_init)
    m.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_es, y_es)],
        early_stopping_rounds=XGB_CLF_EARLY_STOPPING,
        verbose=False
    )
    best_iters.append(m.best_iteration)

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
    print(f"    best_iter={m.best_iteration}")

importances    /= CV_FOLDS
importance_dict = dict(zip(FEATURES, importances))
avg_best_iter   = int(np.mean(best_iters))

# =============================================================================
# 3. OOF GLOBALES
# =============================================================================
oof_metrics, oof_bins = evaluate_clf(
    y, fold_preds, fold_probas,
    class_names=list(classes_encoded),
    model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_clf(fold_metrics)

# =============================================================================
# 4. ENTRAÎNEMENT FINAL
# =============================================================================
final_params = {**xgb_init, "n_estimators": avg_best_iter}
final_model  = XGBClassifier(**final_params)
sw_all       = compute_sample_weights_clf(y)
final_model.fit(X, y, sample_weight=sw_all, verbose=False)

model_path = CLF_MODELS_DIR / "xgb_clf.pkl"
final_model.save_model(str(model_path).replace(".pkl", ".json"))
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model, "label_encoder": le,
                 "avg_best_iter": avg_best_iter}, f)
print(f"Model saved: {model_path}")

oof_path = CLF_PKL_DIR / "oof_xgb.pkl"
with open(oof_path, "wb") as f:
    pickle.dump({
        "y_true": y, "y_pred": fold_preds, "y_proba": fold_probas,
        "oof_bins": oof_bins, "oof_metrics": oof_metrics,
        "cv_summary": cv_summary, "classes": classes_encoded,
        "importances": importance_dict, "best_iters": best_iters,
    }, f)

# =============================================================================
# 5. MLFLOW
# =============================================================================
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"{MODEL_NAME}_clf") as run:
    mlflow.log_param("model_type",    MODEL_NAME)
    mlflow.log_param("avg_best_iter", avg_best_iter)
    for k, v in XGB_CLF_PARAMS.items():
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