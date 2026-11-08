import sys, pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)
from src.config_classification import (
    TARGET_CLF, CLF_MODELS_DIR, CLF_PKL_DIR,
    LGBM_CLF_PARAMS, LGBM_CLF_EARLY_STOPPING
)
from src.evaluation_classification import evaluate_clf, aggregate_cv_clf

MODEL_NAME = "LightGBM"

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()
le       = LabelEncoder()
X        = df_model[FEATURES].values
y        = le.fit_transform(df_model[TARGET_CLF].values)
classes_encoded = le.classes_
print(f"Classes : {list(classes_encoded)}")

kf            = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics  = []
fold_preds    = np.zeros(len(y), dtype=int)
fold_probas   = np.zeros((len(y), len(classes_encoded)))
fold_bins     = []
importances_split = np.zeros(len(FEATURES))
importances_gain  = np.zeros(len(FEATURES))
best_iters        = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_full, X_val = X[train_idx], X[val_idx]
    y_train_full, y_val = y[train_idx], y[val_idx]

    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED
    )

    evals_result = {}
    m = LGBMClassifier(**LGBM_CLF_PARAMS)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_es, y_es)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(LGBM_CLF_EARLY_STOPPING, verbose=False),
            log_evaluation(period=-1),
            record_evaluation(evals_result),
        ]
    )
    best_iters.append(m.best_iteration_)
    importances_split += m.feature_importances_
    importances_gain  += m.booster_.feature_importance(importance_type="gain")

    fold_preds[val_idx]     = m.predict(X_val)
    fold_probas[val_idx, :] = m.predict_proba(X_val)

    metrics, bins = evaluate_clf(
        y_val, fold_preds[val_idx], fold_probas[val_idx],
        class_names=list(classes_encoded),
        model_name=f"{MODEL_NAME}_fold{fold_idx}", verbose=True
    )
    fold_metrics.append(metrics)
    fold_bins.append(bins)
    print(f"    best_iter={m.best_iteration_}")

importances_gain_norm = importances_gain / importances_gain.sum()
importance_dict       = dict(zip(FEATURES, importances_gain_norm))
avg_best_iter         = int(np.mean(best_iters))

oof_metrics, oof_bins = evaluate_clf(
    y, fold_preds, fold_probas,
    class_names=list(classes_encoded),
    model_name=f"{MODEL_NAME}_OOF", verbose=True
)
cv_summary = aggregate_cv_clf(fold_metrics)

final_params = {**LGBM_CLF_PARAMS, "n_estimators": avg_best_iter}
final_model  = LGBMClassifier(**final_params)
final_model.fit(X, y)

model_path = CLF_MODELS_DIR / "lgbm_clf.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model, "label_encoder": le}, f)

oof_path = CLF_PKL_DIR / "oof_lgbm.pkl"
with open(oof_path, "wb") as f:
    pickle.dump({
        "y_true": y, "y_pred": fold_preds, "y_proba": fold_probas,
        "oof_bins": oof_bins, "oof_metrics": oof_metrics,
        "cv_summary": cv_summary, "classes": classes_encoded,
        "importances": importance_dict, "best_iters": best_iters,
    }, f)

mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"{MODEL_NAME}_clf") as run:
    mlflow.log_param("model_type",    MODEL_NAME)
    mlflow.log_param("avg_best_iter", avg_best_iter)
    for k, v in LGBM_CLF_PARAMS.items():
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