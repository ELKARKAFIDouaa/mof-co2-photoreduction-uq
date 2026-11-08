import numpy as np
import pandas as pd
import pickle
import mlflow
from pathlib import Path
from sklearn.svm import SVR
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

MODEL_NAME = "SVM_RBF"

# SVM parameters
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "epsilon": 0.1,
    "cache_size": 500
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

# SVM requires feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_raw = df_model[FEATURES].values
y_raw = df_model[TARGET_TAUX].values.reshape(-1, 1)

X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw).flatten()

print(f"Features standardized: mean≈0, std≈1")
print(f"X : {X.shape} | y : {y.shape}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation ---")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds = np.zeros_like(y, dtype=float)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Training ---
    model_fold = SVR(**SVM_PARAMS)
    model_fold.fit(X_train, y_train)

    # --- Prediction on validation ---
    y_pred_val = model_fold.predict(X_val)
    fold_preds[val_idx] = y_pred_val

    # Convert back to original scale for metrics
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val_orig = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

    # --- Fold metrics ---
    metrics, _ = evaluate_model(
        y_val_orig, y_pred_val_orig,
        model_name=f"{MODEL_NAME}_fold{fold_idx}",
        verbose=True
    )
    fold_metrics.append(metrics)

# =============================================================================
# 3. GLOBAL OOF METRICS (Out-Of-Fold)
# =============================================================================
print("\n--- Global OOF metrics (all out-of-fold predictions) ---")

fold_preds_orig = scaler_y.inverse_transform(fold_preds.reshape(-1, 1)).flatten()
y_orig = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

oof_metrics, oof_cal_detail = evaluate_model(
    y_orig, fold_preds_orig,
    model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_results(fold_metrics)

print("\nCV summary (mean ± std per fold):")
for k, v in cv_summary.items():
    print(f"  {k:30s} : {v}")

# =============================================================================
# 4. FINAL TRAINING ON ALL DATA
# =============================================================================
print("\n--- Final training on 100% of data ---")
final_model = SVR(**SVM_PARAMS)
final_model.fit(X, y)

# Save model and scalers
model_path = SAVED_MODELS_DIR / "svm_rbf_taux_log.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        'model': final_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, f)
print(f"Model saved: {model_path}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
fig_pred_path = RESULTS_FIG_DIR / "svm_rbf_pred_vs_real.png"
fig_cal_path = RESULTS_FIG_DIR / "svm_rbf_calibration.png"

plot_predictions(
    y_orig, fold_preds_orig,
    model_name=f"{MODEL_NAME} (OOF)",
    save_path=fig_pred_path
)
plot_calibration_curve(
    oof_cal_detail,
    model_name=f"{MODEL_NAME} (OOF)",
    save_path=fig_cal_path
)

# =============================================================================
# 6. MLFLOW LOGGING
# =============================================================================
print("\n--- MLflow Logging ---")
setup_mlflow()

with mlflow.start_run(run_name=f"{MODEL_NAME}_taux_log") as run:
    print(f"Run ID: {run.info.run_id}")

    # Hyperparameters
    log_model_params(SVM_PARAMS, MODEL_NAME)

    # OOF metrics
    for metric_name, metric_val in oof_metrics.items():
        if isinstance(metric_val, (int, float)):
            mlflow.log_metric(f"oof_{metric_name}", metric_val)

    # CV metrics
    log_cv_results(cv_summary, fold_metrics, prefix="cv_")

    # Tags
    log_tags(MODEL_NAME, TARGET_TAUX, CV_FOLDS, FEATURES)

    # Artifacts
    log_artifacts_and_model(
        final_model, MODEL_NAME,
        artifacts_paths=[fig_pred_path, fig_cal_path, model_path]
    )

print(f"\n✓ {MODEL_NAME} completed.")
print(f"  OOF MAE: {oof_metrics['MAE']} | R²: {oof_metrics['R2']}")
