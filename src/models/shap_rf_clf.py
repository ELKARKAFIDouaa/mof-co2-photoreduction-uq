# =============================================================================
# shap_rf_clf.py — Analyse SHAP avec TreeExplainer sur RandomForest
# =============================================================================
# Pourquoi RF et pas LightGBM pour SHAP ?
#   - RF a été retenu si meilleur F1 ; sinon changer MODEL_FOR_SHAP
#   - TreeExplainer supporte nativement RF et LightGBM
#   - Les valeurs SHAP multiclasses donnent une importance par classe
#     → permet de voir quelles features déterminent CO vs CH4 vs CH3OH
#
# Produit :
#   1. SHAP beeswarm (summary plot) — importance + direction
#   2. SHAP bar plot (mean |SHAP|) — importance globale
#   3. SHAP heatmap par classe — quelle feature détermine quel produit
#   4. pkl des valeurs SHAP pour réutilisation
# =============================================================================

import sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    FEATURE_SELECTION_CSV, FEATURES, RANDOM_SEED
)
from src.config_classification import (
    TARGET_CLF, CLF_PKL_DIR, CLF_FIG_DIR,
    CLF_MODELS_DIR, RF_CLF_PARAMS
)

# Changer ici si LightGBM est meilleur modèle
MODEL_FOR_SHAP = "RandomForest"
MODEL_PKL_KEY  = "RandomForest"

# =============================================================================
# 1. CHARGER MODÈLE FINAL + DONNÉES
# =============================================================================
print("=" * 60)
print(f"  SHAP Analysis — {MODEL_FOR_SHAP}")
print("=" * 60)

model_path = CLF_MODELS_DIR / "rf_clf.pkl"
# Si LightGBM : model_path = CLF_MODELS_DIR / "lgbm_clf.pkl"

with open(model_path, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
le    = saved["label_encoder"]
classes = le.classes_
print(f"Model loaded. Classes: {list(classes)}")

df       = pd.read_csv(FEATURE_SELECTION_CSV)
df_model = df.dropna(subset=[TARGET_CLF]).copy()
X        = df_model[FEATURES].values
y        = le.transform(df_model[TARGET_CLF].values)

# Créer un DataFrame avec noms de features propres pour les plots
feature_names_clean = [f.replace(".", "_").replace("conditions_reaction_", "")
                        .replace("proprietes_", "").replace("cocatalyseur_", "cata_")
                        .replace("mof_", "") for f in FEATURES]
X_df = pd.DataFrame(X, columns=feature_names_clean)

# =============================================================================
# 2. CALCUL SHAP AVEC TreeExplainer
# =============================================================================
print("\nComputing SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)
# shap_values : list[n_classes] de arrays (n_samples, n_features)
# ou array (n_samples, n_features, n_classes) selon la version SHAP

# Normalisation de la forme selon la version SHAP
if isinstance(shap_values, list):
    # list[n_classes] → stack en (n_samples, n_features, n_classes)
    shap_array = np.stack(shap_values, axis=2)
else:
    shap_array = shap_values

print(f"SHAP values shape: {shap_array.shape}")

# Sauvegarder pour réutilisation
shap_pkl_path = CLF_PKL_DIR / "shap_rf_clf.pkl"
with open(shap_pkl_path, "wb") as f:
    pickle.dump({
        "shap_values"  : shap_array,
        "X_df"         : X_df,
        "classes"      : classes,
        "feature_names": feature_names_clean,
        "explainer"    : explainer,
    }, f)
print(f"SHAP values saved: {shap_pkl_path}")

# =============================================================================
# 3. FIGURE A : SHAP Summary Beeswarm — importance globale toutes classes
# =============================================================================
# Pour multiclasse : moyenne des |SHAP| sur toutes les classes
shap_mean_abs = np.abs(shap_array).mean(axis=2)   # (n_samples, n_features)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_mean_abs,
    X_df,
    feature_names=feature_names_clean,
    plot_type="dot",
    show=False,
    max_display=12
)
plt.title(f"SHAP Summary — {MODEL_FOR_SHAP} (all classes averaged)",
          fontsize=12, fontweight="bold")
plt.tight_layout()
fig_bee_path = CLF_FIG_DIR / "shap_rf_clf_beeswarm.png"
plt.savefig(fig_bee_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Beeswarm saved: {fig_bee_path}")

# =============================================================================
# 4. FIGURE B : SHAP Bar Plot — mean |SHAP| par feature
# =============================================================================
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_mean_abs,
    X_df,
    feature_names=feature_names_clean,
    plot_type="bar",
    show=False,
    max_display=12
)
plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {MODEL_FOR_SHAP}",
          fontsize=12, fontweight="bold")
plt.tight_layout()
fig_bar_path = CLF_FIG_DIR / "shap_rf_clf_bar.png"
plt.savefig(fig_bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Bar plot saved: {fig_bar_path}")

# =============================================================================
# 5. FIGURE C : SHAP par classe — heatmap mean |SHAP| (feature × classe)
# =============================================================================
# Pour chaque classe : mean |SHAP| par feature
# → montre quelles features discriminent chaque produit
n_classes   = len(classes)
mean_shap_per_class = np.abs(shap_array).mean(axis=0)  # (n_features, n_classes)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(mean_shap_per_class, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(n_classes))
ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(feature_names_clean)))
ax.set_yticklabels(feature_names_clean, fontsize=9)
ax.set_xlabel("Predicted product class", fontsize=11)
ax.set_ylabel("Feature",                 fontsize=11)
ax.set_title(f"Mean |SHAP| per Class — {MODEL_FOR_SHAP}\n"
             f"(Which features drive each product prediction)",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Mean |SHAP value|")

# Annoter les valeurs
for i in range(len(feature_names_clean)):
    for j in range(n_classes):
        val = mean_shap_per_class[i, j]
        ax.text(j, i, f"{val:.3f}",
                ha="center", va="center", fontsize=8,
                color="white" if val > mean_shap_per_class.max() * 0.6 else "black")

plt.tight_layout()
fig_hm_path = CLF_FIG_DIR / "shap_rf_clf_per_class_heatmap.png"
plt.savefig(fig_hm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Per-class heatmap saved: {fig_hm_path}")

# =============================================================================
# 6. RAPPORT TEXTE
# =============================================================================
report_lines = [
    "=" * 60,
    f"SHAP ANALYSIS — {MODEL_FOR_SHAP}",
    "=" * 60,
    "",
    "Top 5 features par importance SHAP globale (mean |SHAP|):",
]
global_importance = mean_shap_per_class.mean(axis=1)
sorted_idx = np.argsort(global_importance)[::-1]
for rank, idx in enumerate(sorted_idx[:5], 1):
    report_lines.append(
        f"  {rank}. {feature_names_clean[idx]:40s} : {global_importance[idx]:.4f}"
    )

report_lines += ["", "Top feature par classe :"]
for j, cls in enumerate(classes):
    top_idx = np.argmax(mean_shap_per_class[:, j])
    report_lines.append(
        f"  {cls:10s} → {feature_names_clean[top_idx]} "
        f"(mean |SHAP| = {mean_shap_per_class[top_idx, j]:.4f})"
    )

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = CLF_PKL_DIR.parent / "shap_clf_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nSHAP report saved: {report_path}")
print(f"\nDone. SHAP analysis complete for {MODEL_FOR_SHAP}.")