# =============================================================================
# compare_models.py — Figures de comparaison inter-modèles
# =============================================================================
# Charge les modèles depuis SAVED_MODELS_DIR (svm_rbf_taux_log.pkl, etc.)
# Recalcule les prédictions OOF → compute_ece → calibration curves
# Produit :
#   1. Calibration curves — tous les modèles superposés
#   2. Learning curves   — MAE vs n_samples
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, TARGET_TAUX,
    CV_FOLDS, RANDOM_SEED, RESULTS_FIG_DIR,
    RF_PARAMS, XGB_PARAMS, LGBM_PARAMS,
    SVM_PARAMS, LASSO_PARAMS, RIDGE_PARAMS, DT_PARAMS,
    SAVED_MODELS_DIR,
)
from src.evaluation import compute_ece

# =============================================================================
# 1. CHARGEMENT DONNÉES
# =============================================================================
df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
X        = df_model[FEATURES].values
y        = df_model[TARGET_TAUX].values
print(f"Données : {X.shape[0]} observations, {X.shape[1]} features")


PKL_FILES = {
    "GPR Matern"    : "gpr_matern_taux_log.pkl",
    "GPR Exponential": "gpr_exponential_taux_log.pkl",
    "XGBoost"       : "xgboost_taux_log.pkl",
    "LightGBM"      : "lightgbm_taux_log.pkl",
    "RandomForest"  : "random_forest_taux_log.pkl",
    "SVM RBF"       : "svm_rbf_taux_log.pkl",
    "Decision Tree" : "decision_tree_taux_log.pkl",
    "Lasso"         : "lasso_taux_log.pkl",
    "Ridge"         : "ridge_taux_log.pkl",
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf",
]

# =============================================================================
# 3. CHARGEMENT DES PKL
# =============================================================================
def load_pkl(pkl_path: Path) -> dict:
    
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    return {"model": obj, "scaler_X": None, "scaler_y": None}


available, missing = {}, []
for name, fname in PKL_FILES.items():
    path = SAVED_MODELS_DIR / fname
    if path.exists():
        available[name] = path
    else:
        missing.append(f"{name} → {fname}")

if missing:
    print(f"\n⚠️  PKL introuvables (ignorés) :")
    for m in missing:
        print(f"   {m}")

print(f"\n✓ Modèles trouvés : {list(available.keys())}\n")

# =============================================================================
# 4. FONCTION OOF — recalcule prédictions cross-val depuis le modèle pkl
# =============================================================================
def compute_oof_cal_detail(pkl_data: dict, X: np.ndarray, y: np.ndarray,
                            cv_folds: int, random_seed: int):
    
    model    = pkl_data["model"]
    has_scaler = (
        pkl_data.get("scaler_X") is not None
        and pkl_data.get("scaler_y") is not None
    )

    kf  = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr        = y[train_idx]

        if has_scaler:
            # Refit scaler sur le fold train uniquement (pas de leakage)
            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(y_tr.reshape(-1, 1))
            X_tr_s  = sx.transform(X_tr)
            X_val_s = sx.transform(X_val)
            y_tr_s  = sy.transform(y_tr.reshape(-1, 1)).flatten()

            model.fit(X_tr_s, y_tr_s)
            y_pred_s     = model.predict(X_val_s)
            oof[val_idx] = sy.inverse_transform(
                y_pred_s.reshape(-1, 1)
            ).flatten()
        else:
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict(X_val)

    _, cal_detail = compute_ece(y, oof)
    return cal_detail


# =============================================================================
# 5. CALCUL OOF POUR TOUS LES MODÈLES DISPONIBLES
# =============================================================================
print("--- Calcul OOF calibration ---")
all_cal_details = {}

for name, path in available.items():
    print(f"  {name}...", end=" ", flush=True)
    try:
        pkl_data = load_pkl(path)
        all_cal_details[name] = compute_oof_cal_detail(
            pkl_data, X, y, CV_FOLDS, RANDOM_SEED
        )
        print("✓")
    except Exception as e:
        print(f"✗  {e}")

# =============================================================================
# 6. FIGURE A — Calibration curves superposées
# =============================================================================
if all_cal_details:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect calibration", zorder=5)

    for (name, cal), color in zip(all_cal_details.items(), COLORS):
        exp = [v["expected_coverage"] for v in cal.values()]
        obs = [v["observed_coverage"] for v in cal.values()]
        ax.plot(exp, obs, "o-", color=color, lw=1.5,
                markersize=4, label=name, alpha=0.85)

    ax.set_xlabel("Expected coverage (α)", fontsize=12)
    ax.set_ylabel("Observed coverage",     fontsize=12)
    ax.set_title("Calibration Curves — All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    fig_cal_path = RESULTS_FIG_DIR / "all_calibration_curves.png"
    plt.savefig(fig_cal_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\n✓ Calibration curves sauvées : {fig_cal_path}")

# =============================================================================
# 7. FIGURE B — Learning curves (MAE vs n_samples)
# =============================================================================

print("\n--- Learning curves (peut prendre 2-3 min) ---")

xgb_params_lc  = {k: v for k, v in XGB_PARAMS.items()
                  if k not in ["early_stopping_rounds", "eval_metric"]}
lgbm_params_lc = {**LGBM_PARAMS, "verbose": -1}
kernel_lc      = ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=0.1)

# GPR nécessite un scaling → Pipeline pour que learning_curve l'applique proprement
gpr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr",    GaussianProcessRegressor(
                   kernel=kernel_lc, alpha=0.1,
                   normalize_y=True, n_restarts_optimizer=3,
                   random_state=RANDOM_SEED)),
])

MODELS_LC = {
    "GPR Matern"   : gpr_pipeline,
    "XGBoost"      : XGBRegressor(**xgb_params_lc, verbosity=0),
    "LightGBM"     : LGBMRegressor(**lgbm_params_lc),
    "RandomForest" : RandomForestRegressor(**RF_PARAMS),
    "SVM RBF"      : SVR(**SVM_PARAMS),
    "Decision Tree": DecisionTreeRegressor(**DT_PARAMS),
    "Lasso"        : Lasso(**LASSO_PARAMS),
    "Ridge"        : Ridge(**{k: v for k, v in RIDGE_PARAMS.items()
                              if k != "random_state"}),
}

fig, ax = plt.subplots(figsize=(11, 6))

for (name, model), color in zip(MODELS_LC.items(), COLORS):
    print(f"  {name}...", end=" ", flush=True)
    try:
        train_sizes, _, test_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.15, 1.0, 8),
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        test_mae = -test_scores.mean(axis=1)
        test_std =  test_scores.std(axis=1)
        ax.plot(train_sizes, test_mae, "o-", color=color, lw=2, label=name)
        ax.fill_between(train_sizes,
                        test_mae - test_std,
                        test_mae + test_std,
                        alpha=0.10, color=color)
        print("✓")
    except Exception as e:
        print(f"✗  {e}")

ax.set_xlabel("Number of training samples", fontsize=12)
ax.set_ylabel("MAE (5-fold CV)",            fontsize=12)
ax.set_title("Learning Curves — All Models",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()

fig_lc_path = RESULTS_FIG_DIR / "learning_curves_all.png"
plt.savefig(fig_lc_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Learning curves sauvées : {fig_lc_path}")