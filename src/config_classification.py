import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT, PROCESSED_DIR, MODELS_DIR,
    FEATURES, CV_FOLDS, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)

# ---------------------------------------------------------------------------
# Cible classification
# ---------------------------------------------------------------------------
TARGET_CLF = "produit_classe"

# ---------------------------------------------------------------------------
# Dossiers spécifiques classification
# ---------------------------------------------------------------------------
CLF_MODELS_DIR  = MODELS_DIR / "saved_models" / "classification"
CLF_RESULTS_DIR = MODELS_DIR / "results"      / "classification"
CLF_FIG_DIR     = CLF_RESULTS_DIR / "figures"
CLF_PKL_DIR     = CLF_RESULTS_DIR / "pkl"

for _d in [CLF_MODELS_DIR, CLF_RESULTS_DIR, CLF_FIG_DIR, CLF_PKL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes attendues (adapter si encodage différent)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["CO", "CH4", "CH3OH", "HCOOH", "autre"]

# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------

LR_CLF_PARAMS = {
    "C"           : 1.0,
    "max_iter"    : 1000,
    "solver"      : "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

RF_CLF_PARAMS = {
    "n_estimators"    : 700,
    "max_features"    : "sqrt",
    "min_samples_leaf": 3,
    "max_depth"       : None,
    "class_weight"    : "balanced_subsample",
    "n_jobs"          : -1,
    "random_state"    : RANDOM_SEED,
}

XGB_CLF_PARAMS = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "max_depth"        : 4,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "eval_metric"      : "mlogloss",
    "use_label_encoder": False,
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,
}
XGB_CLF_EARLY_STOPPING = 50

LGBM_CLF_PARAMS = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 25,
    "min_child_samples": 10,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "class_weight"     : "balanced",
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,
    "verbose"          : -1,
}
LGBM_CLF_EARLY_STOPPING = 50

SVM_CLF_PARAMS = {
    "kernel"                 : "rbf",
    "C"                      : 1.0,
    "gamma"                  : "scale",
    "probability"            : True,
    "decision_function_shape": "ovr",
    "class_weight"           : "balanced",
    "random_state"           : RANDOM_SEED,
}

# ---------------------------------------------------------------------------
# Mapping fichiers pkl
# ---------------------------------------------------------------------------
CLF_MODEL_FILES = {
    "LogisticRegression": CLF_MODELS_DIR / "lr_clf.pkl",
    "RandomForest"      : CLF_MODELS_DIR / "rf_clf.pkl",
    "XGBoost"           : CLF_MODELS_DIR / "xgb_clf.pkl",
    "LightGBM"          : CLF_MODELS_DIR / "lgbm_clf.pkl",
    "SVM_RBF"           : CLF_MODELS_DIR / "svm_clf.pkl",
}

CLF_OOF_FILES = {
    "LogisticRegression": CLF_PKL_DIR / "oof_lr.pkl",
    "RandomForest"      : CLF_PKL_DIR / "oof_rf.pkl",
    "XGBoost"           : CLF_PKL_DIR / "oof_xgb.pkl",
    "LightGBM"          : CLF_PKL_DIR / "oof_lgbm.pkl",
    "SVM_RBF"           : CLF_PKL_DIR / "oof_svm.pkl",
}