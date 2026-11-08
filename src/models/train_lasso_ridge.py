import numpy as np
import pandas as pd
import pickle
import mlflow
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, TARGET_TAUX,
    CV_FOLDS, RANDOM_SEED,
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

MODEL_NAME = "LassoRidge"

# Lasso and Ridge parameters
LASSO_PARAMS = {
    "alpha": 0.01,
    "max_iter": 10000,
    "random_state": RANDOM_SEED
}

RIDGE_PARAMS = {
    "alpha": 1.0,
    "max_iter": 10000,
    "random_state": RANDOM_SEED
}

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Training on {TARGET_TAUX}")
print("=" * 60)

df = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
print(f"Dataset loaded: {df_model.shape[0]} rows, {df_model.shape[1]} columns")

# Linear models require feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_raw = df_model[FEATURES].values
y_raw = df_model[TARGET_TAUX].values.reshape(-1, 1)

X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw).flatten()

print(f"Features standardized: mean≈0, std≈1")
print(f"X : {X.shape} | y : {y.shape}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD FOR LASSO
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation for LASSO ---")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
lasso_fold_metrics = []
lasso_fold_preds = np.zeros_like(y, dtype=float)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  LASSO Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_fold = Lasso(**LASSO_PARAMS)
    model_fold.fit(X_train, y_train)

    y_pred_val = model_fold.predict(X_val)
    lasso_fold_preds[val_idx] = y_pred_val

    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val_orig = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

    metrics, _ = evaluate_model(
        y_val_orig, y_pred_val_orig,
        model_name=f"LASSO_fold{fold_idx}",
        verbose=True
    )
    lasso_fold_metrics.append(metrics)

# =============================================================================
# 3. CROSS-VALIDATION 5-FOLD FOR RIDGE
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation for RIDGE ---")

ridge_fold_metrics = []
ridge_fold_preds = np.zeros_like(y, dtype=float)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  RIDGE Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_fold = Ridge(**RIDGE_PARAMS)
    model_fold.fit(X_train, y_train)

    y_pred_val = model_fold.predict(X_val)
    ridge_fold_preds[val_idx] = y_pred_val

    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val_orig = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

    metrics, _ = evaluate_model(
        y_val_orig, y_pred_val_orig,
        model_name=f"RIDGE_fold{fold_idx}",
        verbose=True
    )
    ridge_fold_metrics.append(metrics)

# =============================================================================
# 4. GLOBAL OOF METRICS
# =============================================================================
print("\n--- Global OOF metrics ---")

lasso_preds_orig = scaler_y.inverse_transform(lasso_fold_preds.reshape(-1, 1)).flatten()
ridge_preds_orig = scaler_y.inverse_transform(ridge_fold_preds.reshape(-1, 1)).flatten()
y_orig = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

print("\n--- LASSO OOF ---")
lasso_oof_metrics, lasso_cal_detail = evaluate_model(
    y_orig, lasso_preds_orig,
    model_name="LASSO_OOF", verbose=True
)

print("\n--- RIDGE OOF ---")
ridge_oof_metrics, ridge_cal_detail = evaluate_model(
    y_orig, ridge_preds_orig,
    model_name="RIDGE_OOF", verbose=True
)

lasso_cv_summary = aggregate_cv_results(lasso_fold_metrics)
ridge_cv_summary = aggregate_cv_results(ridge_fold_metrics)

# =============================================================================
# 5. FINAL TRAINING ON ALL DATA
# =============================================================================
print("\n--- Final training on 100% of data ---")

final_lasso = Lasso(**LASSO_PARAMS)
final_lasso.fit(X, y)

final_ridge = Ridge(**RIDGE_PARAMS)
final_ridge.fit(X, y)

# Save models and scalers
lasso_path = SAVED_MODELS_DIR / "lasso_taux_log.pkl"
ridge_path = SAVED_MODELS_DIR / "ridge_taux_log.pkl"

with open(lasso_path, "wb") as f:
    pickle.dump({'model': final_lasso, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
with open(ridge_path, "wb") as f:
    pickle.dump({'model': final_ridge, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

print(f"LASSO model saved: {lasso_path}")
print(f"RIDGE model saved: {ridge_path}")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
fig_lasso_pred = RESULTS_FIG_DIR / "lasso_pred_vs_real.png"
fig_lasso_cal = RESULTS_FIG_DIR / "lasso_calibration.png"
fig_ridge_pred = RESULTS_FIG_DIR / "ridge_pred_vs_real.png"
fig_ridge_cal = RESULTS_FIG_DIR / "ridge_calibration.png"

plot_predictions(y_orig, lasso_preds_orig, model_name="LASSO (OOF)", save_path=fig_lasso_pred)
plot_calibration_curve(lasso_cal_detail, model_name="LASSO (OOF)", save_path=fig_lasso_cal)
plot_predictions(y_orig, ridge_preds_orig, model_name="RIDGE (OOF)", save_path=fig_ridge_pred)
plot_calibration_curve(ridge_cal_detail, model_name="RIDGE (OOF)", save_path=fig_ridge_cal)

# =============================================================================
# 7. MLFLOW LOGGING
# =============================================================================
print("\n--- MLflow Logging ---")
setup_mlflow()

# Log LASSO
with mlflow.start_run(run_name="LASSO_taux_log") as run:
    print(f"LASSO Run ID: {run.info.run_id}")
    log_model_params(LASSO_PARAMS, "LASSO")
    for metric_name, metric_val in lasso_oof_metrics.items():
        if isinstance(metric_val, (int, float)):
            mlflow.log_metric(f"oof_{metric_name}", metric_val)
    log_cv_results(lasso_cv_summary, lasso_fold_metrics, prefix="cv_")
    log_tags("LASSO", TARGET_TAUX, CV_FOLDS, FEATURES)
    log_artifacts_and_model(final_lasso, "LASSO", artifacts_paths=[fig_lasso_pred, fig_lasso_cal, lasso_path])

# Log RIDGE
with mlflow.start_run(run_name="RIDGE_taux_log") as run:
    print(f"RIDGE Run ID: {run.info.run_id}")
    log_model_params(RIDGE_PARAMS, "RIDGE")
    for metric_name, metric_val in ridge_oof_metrics.items():
        if isinstance(metric_val, (int, float)):
            mlflow.log_metric(f"oof_{metric_name}", metric_val)
    log_cv_results(ridge_cv_summary, ridge_fold_metrics, prefix="cv_")
    log_tags("RIDGE", TARGET_TAUX, CV_FOLDS, FEATURES)
    log_artifacts_and_model(final_ridge, "RIDGE", artifacts_paths=[fig_ridge_pred, fig_ridge_cal, ridge_path])

print(f"\n✓ {MODEL_NAME} completed.")
print(f"  LASSO OOF MAE: {lasso_oof_metrics['MAE']} | R²: {lasso_oof_metrics['R2']}")
print(f"  RIDGE OOF MAE: {ridge_oof_metrics['MAE']} | R²: {ridge_oof_metrics['R2']}")