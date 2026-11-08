from pathlib import Path
 

PROJECT_ROOT = Path("D:/Mes projets/PFE/fastapi-ml")
 

# Données
DATA_DIR          = PROJECT_ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed" / "pipeline_outputs"
FIGURES_DIR       = DATA_DIR / "figures"
 
# Fichiers d'entrée pour la modélisation
FEATURE_SELECTION_CSV  = PROCESSED_DIR / "feature_selection.csv"
SELECTED_FEATURES_JSON = PROCESSED_DIR / "selected_features.json"
 

# Modèles
MODELS_DIR        = PROJECT_ROOT / "models"
SCALERS_DIR       = MODELS_DIR / "scalers"
SAVED_MODELS_DIR  = MODELS_DIR / "saved_models"
RESULTS_DIR       = MODELS_DIR / "results"
RESULTS_FIG_DIR   = RESULTS_DIR / "figures"
 
for _dir in [SAVED_MODELS_DIR, RESULTS_DIR, RESULTS_FIG_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# MLflow
MLFLOW_TRACKING_URI    = str(PROJECT_ROOT / "mlflow_runs")
MLFLOW_EXPERIMENT_NAME = "MOF_photocatalysis_pipeline"
 

# Colonnes cibles et features

TARGET_TAUX     = "taux_log"
TARGET_SEL      = "performance.selectivite_pct"
TARGET_Produit = "produit_classe"
 
# 12 features sélectionnées (ordonnées par ρ_taux décroissant)
FEATURES = [
    "cocatalyseur.type_enc",                          # ρ_taux=+0.58
    "mof.metal_enc",                                  # ρ_taux=+0.56
    "mof.ligand_enc",                                 # ρ_taux=+0.55
    "cocatalyseur.charge_pct",                        # ρ_taux=+0.43
    "proprietes_electroniques.band_gap_eV",           # ρ_taux=+0.28
    "conditions_reaction.solvant_enc",                # ρ_taux=+0.28
    "source_lumiere_enc",                             # ρ_taux=+0.10
    "conditions_reaction.concentration_catalyseur_g_L",# ρ_taux=+0.09
    "proprietes_structurales.BET_m2_g",               # ρ_sel=+0.18
    "BET_x_duree",                                    # ρ_sel=+0.17
    "conditions_reaction.duree_irradiation_h",        # ρ_taux=+0.07
    "delta_LUMO_CH4",                                 # feature quantique
]
 

# Configuration Cross-Validation

CV_FOLDS    = 5
RANDOM_SEED = 42
 

# Hyperparamètres des modèles

 
RF_PARAMS = {
    "n_estimators"    : 700,
    "max_features"    : "sqrt",
    "min_samples_leaf": 3,
    "max_depth"       : None,
    "n_jobs"          : -1,
    "random_state"    : RANDOM_SEED,
}

XGB_PARAMS = {
    "n_estimators"        : 800,
    "learning_rate"       : 0.05,
    "max_depth"           : 5,
    "reg_alpha"           : 0.1,
    "reg_lambda"          : 1.0,
    "subsample"           : 0.8,
    "colsample_bytree"    : 0.8,
    "early_stopping_rounds": 50,
    "eval_metric"         : "mae",
    "random_state"        : RANDOM_SEED,
    "n_jobs"              : -1,
}

LGBM_PARAMS = {
    "n_estimators"     : 800,
    "learning_rate"    : 0.05,
    "num_leaves"       : 31,
    "min_child_samples": 10,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

# =============================================================================
# HYPERPARAMÈTRES DES NOUVEAUX MODÈLES
# =============================================================================

# SVM avec noyau RBF
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "epsilon": 0.1,
    "cache_size": 500
}

# Lasso (régression L1)
LASSO_PARAMS = {
    "alpha": 0.01,
    "max_iter": 10000,
    "random_state": RANDOM_SEED
}

# Ridge (régression L2)
RIDGE_PARAMS = {
    "alpha": 1.0,
    "max_iter": 10000,
    "random_state": RANDOM_SEED
}

# Decision Tree
DT_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED
}

# Gradient Boosting Decision Tree (sklearn)
GBDT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_SEED
}

# GPR Matérn
GPR_MATERN_PARAMS = {
    "alpha": 1e-6,
    "normalize_y": True,
    "n_restarts_optimizer": 10,
    "random_state": RANDOM_SEED
}

# GPR Exponential (RBF)
GPR_EXPONENTIAL_PARAMS = {
    "alpha": 1e-6,
    "normalize_y": True,
    "n_restarts_optimizer": 10,
    "random_state": RANDOM_SEED
}