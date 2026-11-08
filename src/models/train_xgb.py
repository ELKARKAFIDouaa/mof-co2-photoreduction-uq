import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, TARGET_TAUX,
    CV_FOLDS, RANDOM_SEED, XGB_PARAMS,
    SAVED_MODELS_DIR, RESULTS_DIR, RESULTS_FIG_DIR
)
from src.evaluation import (
    evaluate_model, aggregate_cv_results,
    plot_predictions, plot_calibration_curve
)
from src.mlflow_logger import (
    setup_mlflow, log_cv_results, log_model_params,
    log_feature_importance, log_artifacts_and_model, log_tags
)

MODEL_NAME = "XGBoost"

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Training on {TARGET_TAUX}")
print("=" * 60)

df = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
print(f"Rows used: {df_model.shape[0]} "
      f"({df.shape[0] - df_model.shape[0]} NaN removed)")

missing_feats = [f for f in FEATURES if f not in df_model.columns]
if missing_feats:
    raise ValueError(f"Missing features: {missing_feats}")

X = df_model[FEATURES].values
y = df_model[TARGET_TAUX].values

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD WITH EARLY STOPPING
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold CV with early stopping ---")

# Separate early_stopping_rounds from XGB_PARAMS
# In XGBoost >= 2.0, early_stopping_rounds must be passed to the constructor
early_stopping_rounds = XGB_PARAMS.pop("early_stopping_rounds", 50)
xgb_init_params = XGB_PARAMS.copy()

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds = np.zeros_like(y, dtype=float)
importances = np.zeros(len(FEATURES))
best_iterations = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  Fold {fold_idx}/{CV_FOLDS}")

    X_train_full, X_val = X[train_idx], X[val_idx]
    y_train_full, y_val = y[train_idx], y[val_idx]

    # --- Internal subset for early stopping ---
    # Take 20% of training fold as internal eval set.
    # Decision: do NOT touch X_val (external validation fold).
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=RANDOM_SEED
    )

    # ✅ FIX: early_stopping_rounds passed to constructor (XGBoost >= 2.0)
    model_fold = XGBRegressor(**xgb_init_params, early_stopping_rounds=early_stopping_rounds)
    model_fold.fit(
        X_tr, y_tr,
        eval_set=[(X_es, y_es)],
        verbose=False
    )

    best_iter = model_fold.best_iteration
    best_iterations.append(best_iter)
    print(f"    Best iteration: {best_iter}")

    y_pred_val = model_fold.predict(X_val)
    fold_preds[val_idx] = y_pred_val

    metrics, _ = evaluate_model(
        y_val, y_pred_val,
        model_name=f"{MODEL_NAME}_fold{fold_idx}",
        verbose=True
    )
    fold_metrics.append(metrics)
    importances += model_fold.feature_importances_

importances /= CV_FOLDS
importance_dict = dict(zip(FEATURES, importances))

# =============================================================================
# 3. OOF METRICS
# =============================================================================
print("\n--- Global OOF metrics ---")
oof_metrics, oof_cal_detail = evaluate_model(
    y, fold_preds, model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_results(fold_metrics)

avg_best_iter = int(np.mean(best_iterations))
print(f"\nAverage best iteration across folds: {avg_best_iter}")
print("\nCV summary:")
for k, v in cv_summary.items():
    print(f"  {k:30s} : {v}")

# =============================================================================
# 4. FINAL TRAINING (with n_estimators = average best_iter)
# =============================================================================
print(f"\n--- Final training (n_estimators={avg_best_iter}) ---")

# Final model uses fixed n_estimators — no early stopping needed
final_params = {**xgb_init_params, "n_estimators": avg_best_iter}
final_model = XGBRegressor(**final_params)
final_model.fit(X, y, verbose=False)

model_path = SAVED_MODELS_DIR / "xgb_taux_log.json"
final_model.save_model(str(model_path))
print(f"Model saved: {model_path}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
fig_pred_path = RESULTS_FIG_DIR / "xgb_pred_vs_real.png"
fig_cal_path = RESULTS_FIG_DIR / "xgb_calibration.png"

plot_predictions(y, fold_preds, model_name=f"{MODEL_NAME} (OOF)", save_path=fig_pred_path)
plot_calibration_curve(oof_cal_detail, model_name=f"{MODEL_NAME} (OOF)", save_path=fig_cal_path)

# =============================================================================
# 6. MLFLOW LOGGING
# =============================================================================
print("\n--- MLflow Logging ---")
setup_mlflow()

with mlflow.start_run(run_name=f"{MODEL_NAME}_taux_log") as run:
    print(f"Run ID: {run.info.run_id}")

    # Restore original XGB_PARAMS for logging (re-add early_stopping_rounds)
    xgb_params_for_logging = {**xgb_init_params, "early_stopping_rounds": early_stopping_rounds}
    log_model_params(xgb_params_for_logging, MODEL_NAME)
    mlflow.log_param("best_iteration_avg", avg_best_iter)

    for metric_name, metric_val in oof_metrics.items():
        if isinstance(metric_val, (int, float)):
            mlflow.log_metric(f"oof_{metric_name}", metric_val)

    log_cv_results(cv_summary, fold_metrics, prefix="cv_")
    log_feature_importance(importance_dict)
    log_tags(MODEL_NAME, TARGET_TAUX, CV_FOLDS, FEATURES)
    log_artifacts_and_model(
        final_model, MODEL_NAME,
        artifacts_paths=[fig_pred_path, fig_cal_path, model_path]
    )

print(f"\n✓ {MODEL_NAME} completed.")
print(f"  OOF MAE: {oof_metrics['MAE']} | R²: {oof_metrics['R2']}")