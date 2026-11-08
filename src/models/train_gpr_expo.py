import numpy as np
import pandas as pd
import joblib
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel, WhiteKernel, RBF
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

MODEL_NAME = "GPR_Exponential"

# GPR specific parameters
GPR_PARAMS_EXP = {
    "alpha": 1e-6,           # Noise level in y (added to diagonal)
    "normalize_y": True,      # Normalize target values
    "n_restarts_optimizer": 10,  # Number of restarts for kernel optimization
    "random_state": RANDOM_SEED
}

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("=" * 60)
print(f"  {MODEL_NAME} — Training on {TARGET_TAUX}")
print("=" * 60)
print("\n📊 GPR with Exponential kernel (RBF - Radial Basis Function)")
print("   - Kernel: very smooth, infinite differentiability")
print("   - Assumes underlying function is extremely smooth")
print("   - Best for continuous, slowly varying phenomena\n")

df = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
print(f"Dataset loaded: {df_model.shape[0]} rows used "
      f"({df.shape[0] - df_model.shape[0]} NaN removed)")

missing_feats = [f for f in FEATURES if f not in df_model.columns]
if missing_feats:
    raise ValueError(f"Missing features in CSV: {missing_feats}")

# GPR is sensitive to feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_raw = df_model[FEATURES].values
y_raw = df_model[TARGET_TAUX].values.reshape(-1, 1)

X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw).flatten()

print(f"Features standardized: mean≈0, std≈1")
print(f"Target standardized: mean≈0, std≈1")
print(f"X: {X.shape} | y: {y.shape}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation with Exponential (RBF) kernel ---")

# Define Exponential/RBF kernel
# Structure: ConstantKernel * RBF(length_scale) + WhiteKernel(noise)
# RBF = exp(-r²/(2ℓ²)) which is the exponential squared kernel
base_kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
              RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# Add white noise kernel for observation noise
kernel = base_kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))

print(f"Kernel structure (Exponential/RBF):\n{kernel}\n")
print("Note: RBF kernel = exp(-||x-x'||²/(2ℓ²)) → infinitely smooth")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds = np.zeros_like(y, dtype=float)
fold_uncertainties = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Initialize and train GPR ---
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        **GPR_PARAMS_EXP
    )
    gpr.fit(X_train, y_train)

    # --- Predict on validation (with uncertainty) ---
    y_pred_val, y_std_val = gpr.predict(X_val, return_std=True)
    fold_preds[val_idx] = y_pred_val
    fold_uncertainties.append(y_std_val)

    # Convert back to original scale for metrics
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val_orig = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

    # --- Fold metrics (on original scale) ---
    metrics, _ = evaluate_model(
        y_val_orig, y_pred_val_orig,
        model_name=f"{MODEL_NAME}_fold{fold_idx}",
        verbose=True
    )
    fold_metrics.append(metrics)

    # Print uncertainty info
    print(f"    Avg prediction uncertainty (std): {y_std_val.mean():.4f} "
          f"({y_std_val.mean() * scaler_y.scale_[0]:.4f} in original scale)")
    
    # Print optimized kernel parameters
    print(f"    Optimized kernel: {gpr.kernel_}")

# =============================================================================
# 3. GLOBAL OOF METRICS (Out-Of-Fold)
# =============================================================================
print("\n--- Global OOF metrics (all out-of-fold predictions) ---")

# Convert OOF predictions back to original scale
fold_preds_orig = scaler_y.inverse_transform(fold_preds.reshape(-1, 1)).flatten()
y_orig = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

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
    **GPR_PARAMS_EXP
)
final_gpr.fit(X, y)

# Save model and scalers
model_path = SAVED_MODELS_DIR / "gpr_exponential_taux_log.pkl"
joblib.dump({
    'model': final_gpr,
    'scaler_X': scaler_X
}, model_path, compress=3)
print(f"Model saved: {model_path}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
fig_pred_path = RESULTS_FIG_DIR / "gpr_exponential_pred_vs_real.png"
fig_cal_path = RESULTS_FIG_DIR / "gpr_exponential_calibration.png"

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
ax.hist(all_uncertainties * scaler_y.scale_[0], bins=30, alpha=0.7, color='#DD8452')
ax.set_xlabel("Prediction uncertainty (std) in original scale")
ax.set_ylabel("Frequency")
ax.set_title(f"{MODEL_NAME} — Distribution of prediction uncertainties")
uncertainty_path = RESULTS_FIG_DIR / "gpr_exponential_uncertainty.png"
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
    mlflow.log_param("kernel_type", "Exponential_RBF")
    mlflow.log_param("kernel_string", str(kernel))
    for param, value in GPR_PARAMS_EXP.items():
        mlflow.log_param(f"gpr_{param}", value)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("random_seed", RANDOM_SEED)

    # Log final kernel parameters
    if hasattr(final_gpr, 'kernel_'):
        mlflow.log_param("final_kernel", str(final_gpr.kernel_))

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
    mlflow.set_tag("kernel", "Exponential_RBF")

    # Artifacts
    log_artifacts_and_model(
        final_gpr, MODEL_NAME,
        artifacts_paths=[fig_pred_path, fig_cal_path, uncertainty_path, model_path]
    )

print(f"\n✓ {MODEL_NAME} completed.")
print(f"  OOF MAE: {oof_metrics['MAE']} | R²: {oof_metrics['R2']}")