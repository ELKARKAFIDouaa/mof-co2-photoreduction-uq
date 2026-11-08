import joblib
import numpy as np
import pandas as pd

import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
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

MODEL_NAME = "GPR_Matern"

# GPR specific parameters
GPR_PARAMS_MATERN = {
    "alpha"               : 0.1,    # bruit obs réaliste (données expérimentales)
    "normalize_y"         : True,   # laisser GPR gérer la normalisation y
    "n_restarts_optimizer": 10,
    "random_state"        : RANDOM_SEED
}

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Training on {TARGET_TAUX}")
print("=" * 60)
print("\n📊 GPR with Matérn kernel (ν=3/2)")
print("   - Kernel: smooth but realistic for chemical data")
print("   - Provides uncertainty estimates for each prediction\n")

df = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
print(f"Dataset loaded: {df_model.shape[0]} rows used "
      f"({df.shape[0] - df_model.shape[0]} NaN removed)")

missing_feats = [f for f in FEATURES if f not in df_model.columns]
if missing_feats:
    raise ValueError(f"Missing features in CSV: {missing_feats}")

# IMPORTANT: GPR is sensitive to feature scaling
scaler_X = StandardScaler()


X_raw = df_model[FEATURES].values
y_raw = df_model[TARGET_TAUX].values.reshape(-1, 1)

X = scaler_X.fit_transform(X_raw)
y = y_raw.flatten() 

print(f"Features standardized: mean≈0, std≈1")
print(f"Target standardized: mean≈0, std≈1")
print(f"X: {X.shape} | y: {y.shape}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation with Matérn kernel ---")

#  bornes WhiteKernel plus contraintes
kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * \
         Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + \
         WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
# borne haute à 1.0, pas 10.0

print(f"Kernel structure:\n{kernel}\n")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds = np.zeros_like(y, dtype=float)
fold_uncertainties = []  # Store uncertainty estimates per fold

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Initialize and train GPR ---
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        **GPR_PARAMS_MATERN
    )
    gpr.fit(X_train, y_train)

    # --- Predict on validation (with uncertainty) ---
    y_pred_val, y_std_val = gpr.predict(X_val, return_std=True)
    fold_preds[val_idx] = y_pred_val
    fold_uncertainties.append(y_std_val)

    # Convert back to original scale for metrics
    y_val_orig  = y_val ##car on a éviter le scaling de y
    y_pred_orig = y_pred_val 

    # --- Fold metrics (on original scale) ---
    metrics, _ = evaluate_model(
        y_val_orig,y_pred_orig,
        model_name=f"{MODEL_NAME}_fold{fold_idx}",
        verbose=True
    )
    fold_metrics.append(metrics)

    # Print uncertainty info
    print(f"    Avg prediction uncertainty (std): {y_std_val.mean():.4f} (original scale)")

# =============================================================================
# 3. GLOBAL OOF METRICS (Out-Of-Fold)
# =============================================================================
print("\n--- Global OOF metrics (all out-of-fold predictions) ---")

# Convert OOF predictions back to original scale
y_orig      = y_raw.flatten()
fold_preds_orig = fold_preds

oof_metrics, oof_cal_detail = evaluate_model(
    y_orig, fold_preds_orig, 
    model_name=f"{MODEL_NAME}_OOF", 
    verbose=True
)
cv_summary = aggregate_cv_results(fold_metrics)

print("\nCV summary (mean ± std per fold):")
for k, v in cv_summary.items():
    print(f"  {k:30s} : {v}")

# =============================================================================
# 4. FINAL TRAINING ON ALL DATA
# =============================================================================
print("\n--- Final training on 100% of data ---")

final_gpr = GaussianProcessRegressor(
    kernel=kernel,
    **GPR_PARAMS_MATERN
)
final_gpr.fit(X, y)

# Save model and scalers
model_path = SAVED_MODELS_DIR / "gpr_matern_taux_log.pkl"
joblib.dump({
    'model': final_gpr,
    'scaler_X': scaler_X
}, model_path, compress=3)
obj = joblib.load(model_path)
print("Reload OK")
print(f"Model saved: {model_path}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
fig_pred_path = RESULTS_FIG_DIR / "gpr_matern_pred_vs_real.png"
fig_cal_path = RESULTS_FIG_DIR / "gpr_matern_calibration.png"

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

# Additional plot: prediction uncertainty
fig_uncertainty, ax = plt.subplots(figsize=(8, 5))
all_uncertainties = np.concatenate(fold_uncertainties)
ax.hist(all_uncertainties , bins=30, alpha=0.7, color='#4C72B0')
ax.set_xlabel("Prediction uncertainty (std) in original scale")
ax.set_ylabel("Frequency")
ax.set_title(f"{MODEL_NAME} — Distribution of prediction uncertainties")
uncertainty_path = RESULTS_FIG_DIR / "gpr_matern_uncertainty.png"
plt.savefig(uncertainty_path, dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# 6. MLFLOW LOGGING
# =============================================================================
print("\n--- MLflow Logging ---")
setup_mlflow()

with mlflow.start_run(run_name=f"{MODEL_NAME}_taux_log") as run:
    print(f"Run ID: {run.info.run_id}")

    # Log parameters
    mlflow.log_param("model_type", MODEL_NAME)
    mlflow.log_param("kernel_type", "Matern_nu1.5")
    mlflow.log_param("kernel_string", str(kernel))
    for param, value in GPR_PARAMS_MATERN.items():
        mlflow.log_param(f"gpr_{param}", value)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("random_seed", RANDOM_SEED)

    # Log metrics
    for metric_name, metric_val in oof_metrics.items():
        if isinstance(metric_val, (int, float)):
            mlflow.log_metric(f"oof_{metric_name}", metric_val)

    # Log CV summary
    log_cv_results(cv_summary, fold_metrics, prefix="cv_")

    # Log uncertainty info
    mlflow.log_metric("avg_prediction_uncertainty", all_uncertainties.mean())
    mlflow.log_metric("std_prediction_uncertainty", all_uncertainties.std())

    # Tags
    log_tags(MODEL_NAME, TARGET_TAUX, CV_FOLDS, FEATURES)
    mlflow.set_tag("kernel", "Matern_nu1.5")

    # Artifacts
    log_artifacts_and_model(
        final_gpr, MODEL_NAME,
        artifacts_paths=[fig_pred_path, fig_cal_path, uncertainty_path, model_path]
    )

print(f"\n✓ {MODEL_NAME} completed.")
print(f"  OOF MAE: {oof_metrics['MAE']} | R²: {oof_metrics['R2']}")