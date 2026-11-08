import numpy as np
import pandas as pd
import pickle
import mlflow
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

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

MODEL_NAME = "DecisionTree"

# Decision Tree parameters
DT_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
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

missing_feats = [f for f in FEATURES if f not in df_model.columns]
if missing_feats:
    raise ValueError(f"Missing features in CSV: {missing_feats}")

X = df_model[FEATURES].values
y = df_model[TARGET_TAUX].values
print(f"X : {X.shape} | y : {y.shape}")

# =============================================================================
# 2. CROSS-VALIDATION 5-FOLD
# =============================================================================
print(f"\n--- {CV_FOLDS}-Fold Cross-Validation ---")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
fold_preds = np.zeros_like(y, dtype=float)
importances = np.zeros(len(FEATURES))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\n  Fold {fold_idx}/{CV_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_fold = DecisionTreeRegressor(**DT_PARAMS)
    model_fold.fit(X_train, y_train)

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
# 3. GLOBAL OOF METRICS
# =============================================================================
print("\n--- Global OOF metrics ---")
oof_metrics, oof_cal_detail = evaluate_model(
    y, fold_preds, model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_results(fold_metrics)

print("\nCV summary (mean ± std per fold):")
for k, v in cv_summary.items():
    print(f"  {k:30s} : {v}")

# =============================================================================
# 4. FINAL TRAINING ON ALL DATA
# =============================================================================
print("\n--- Final training on 100% of data ---")
final_model = DecisionTreeRegressor(**DT_PARAMS)
final_model.fit(X, y)

model_path = SAVED_MODELS_DIR / "dt_taux_log.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Model saved: {model_path}")

# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
fig_pred_path = RESULTS_FIG_DIR / "dt_pred_vs_real.png"
fig_cal_path = RESULTS_FIG_DIR / "dt_calibration.png"

plot_predictions(y, fold_preds, model_name=f"{MODEL_NAME} (OOF)", save_path=fig_pred_path)
plot_calibration_curve(oof_cal_detail, model_name=f"{MODEL_NAME} (OOF)", save_path=fig_cal_path)

# =============================================================================
# 6. MLFLOW LOGGING
# =============================================================================
print("\n--- MLflow Logging ---")
setup_mlflow()

with mlflow.start_run(run_name=f"{MODEL_NAME}_taux_log") as run:
    print(f"Run ID: {run.info.run_id}")

    log_model_params(DT_PARAMS, MODEL_NAME)

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