# shap_xgboost.py — Figures SHAP pour le rapport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, TARGET_TAUX,
    SAVED_MODELS_DIR, RESULTS_FIG_DIR,
)

# =============================================================================
# 1. CHARGEMENT DONNÉES + MODÈLE
# =============================================================================
df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_TAUX]).copy()
X_raw    = df_model[FEATURES].values          # valeurs encodées/scalées
X_df     = df_model[FEATURES].copy()          # DataFrame pour SHAP (garde les noms)


final_xgb = XGBRegressor()
final_xgb.load_model(SAVED_MODELS_DIR / "xgb_taux_log.json")
print(f"Model  : {type(final_xgb).__name__}")


# 2. LABELS LISIBLES — remplace les noms de colonnes encodées


FEATURE_LABELS = {
    "proprietes_electroniques.band_gap_eV"          : "Band Gap (eV)",
    "mof.ligand_enc"                                 : "Ligand (enc.)",
    "cocatalyseur.charge_pct"                        : "Cocatalyst charge (%)",
    "BET_x_duree"                                    : "BET × Irradiation time",
    "mof.metal_enc"                                  : "Metal (enc.)",
    "cocatalyseur.type_enc"                          : "Cocatalyst type (enc.)",
    "conditions_reaction.duree_irradiation_h"        : "Irradiation duration (h)",
    "conditions_reaction.concentration_catalyseur_g_L": "Catalyst conc. (g/L)",
    "proprietes_structurales.BET_m2_g"               : "BET surface (m²/g)",
    "delta_LUMO_CH4"                                 : "ΔLUMO CH₄ (eV)",
    "conditions_reaction.solvant_enc"                : "Solvent (enc.)",
    "source_lumiere_enc"                             : "Light source (enc.)",
}

# Noms courts pour l'affichage (dans l'ordre de FEATURES)
feature_labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
print(f"Features : {len(feature_labels)} labels définis")

# 3. CALCUL SHAP — TreeExplainer (exact, rapide pour XGBoost)

print("\nCalcul SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_raw)    # shape : (n_obs, n_features)
print(f"SHAP values shape : {shap_values.shape}")
print(f"SHAP mean |val| par feature (top 5) :")
mean_abs = np.abs(shap_values).mean(axis=0)
top5_idx = np.argsort(mean_abs)[::-1][:5]
for i in top5_idx:
    print(f"  {feature_labels[i]:35s} : {mean_abs[i]:.4f}")


#  Beeswarm plot (summary)

# Lecture : chaque point = une observation
#   - Position X  = valeur SHAP (impact sur la prédiction taux_log)
#   - Couleur     = valeur de la feature (rouge = haute, bleu = basse)
#   - Features    = triées par importance décroissante (|SHAP| moyen)
# Interprétation malgré l'encodage :
#   - "Band Gap rouge → SHAP négatif" = band gap élevé → taux de production prédit plus bas
#   - "BET×durée rouge → SHAP positif" = interaction BET/temps élevée → production prédite plus haute
#   - Pour les enc. : rouge = modalité encodée haute (ex: ligand fréquent/performant)

print("\nSHAP beeswarm...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_raw,
    feature_names=feature_labels,
    show=False,
    max_display=len(FEATURES),      # affiche toutes les features
    alpha=0.7,
    plot_size=None,                  # on contrôle via plt.figure
)
plt.title(
    "SHAP Beeswarm Plot (XGBoost)\n"
    "Impact of each feature on predicted log(production rate)",
    fontsize=11, fontweight="bold", pad=12
)
plt.xlabel("SHAP value (impact on model output: log production rate)", fontsize=10)
plt.tight_layout()

fig_a_path = RESULTS_FIG_DIR / "shap_beeswarm_xgb.png"
plt.savefig(fig_a_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Sauvée : {fig_a_path}")


# Bar plot (importance globale, mean |SHAP|)

# Lecture : barre = moyenne de |SHAP| sur toutes les observations
#   - Mesure l'importance MOYENNE, pas le signe de l'effet
#   - Comparable à l'importance RF mais interprétable en unités de la cible
#   - La valeur X = impact moyen en unités de log(taux_production)

print("SHAP bar (importance globale)...")
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values,
    X_raw,
    feature_names=feature_labels,
    plot_type="bar",
    show=False,
    max_display=len(FEATURES),
    plot_size=None,
)
plt.title(
    "SHAP Feature Importance (XGBoost)\n"
    "Mean |SHAP value| across all observations",
    fontsize=11, fontweight="bold", pad=12
)
plt.xlabel("Mean |SHAP value| (average impact on log production rate)", fontsize=10)
plt.tight_layout()

fig_b_path = RESULTS_FIG_DIR / "shap_bar_xgb.png"
plt.savefig(fig_b_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Sauvée : {fig_b_path}")


# 6. BONUS — Table de synthèse SHAP (utile pour le rapport)

shap_summary = pd.DataFrame({
    "Feature"          : feature_labels,
    "Feature_raw"      : FEATURES,
    "Mean_abs_SHAP"    : mean_abs,
    "Encoding"         : [
        "numeric",        # band_gap_eV
        "target_enc",     # ligand
        "numeric",        # cocatalyst charge
        "interaction",    # BET x duree
        "target_enc",     # metal
        "target_enc",     # cocatalyst type
        "numeric",        # duree irradiation
        "numeric",        # concentration
        "numeric",        # BET
        "numeric",        # delta LUMO CH4
        "onehot_enc",     # solvant
        "ordinal_enc",    # source lumiere
    ][:len(FEATURES)],
}).sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)

print("\nTop features par importance SHAP :")
print(shap_summary[["Feature", "Mean_abs_SHAP", "Encoding"]].to_string(index=False))

table_path = RESULTS_FIG_DIR / "shap_importance_table.csv"
shap_summary.to_csv(table_path, index=False)
print(f"\n✓ Table sauvée : {table_path}")