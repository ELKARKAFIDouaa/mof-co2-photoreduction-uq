import sys, pickle
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, FEATURES
)
from src.config_classification import (
    CLF_PKL_DIR, CLF_FIG_DIR, CLF_RESULTS_DIR, CLF_OOF_FILES, CLASS_NAMES
)
from src.evaluation_classification import (
    plot_confusion_matrix,
    plot_all_reliability_diagrams,
    plot_decision_matrix_clf
)

# =============================================================================
# 1. CHARGER TOUS LES OOF
# =============================================================================
print("=" * 60)
print("  EVALUATION CONSOLIDEE — Classification")
print("=" * 60)

MODEL_KEYS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "SVM_RBF"]

oof_data = {}
for name in MODEL_KEYS:
    path = CLF_OOF_FILES[name]
    if not path.exists():
        print(f"  WARNING: OOF not found for {name} → {path}")
        print(f"  Run train_{name.lower()}_clf.py first.")
        continue
    with open(path, "rb") as f:
        oof_data[name] = pickle.load(f)
    print(f"  Loaded OOF: {name}")

available_models = list(oof_data.keys())
if not available_models:
    raise RuntimeError("No OOF files found. Run train_*_clf.py first.")

# Récupérer les classes depuis le premier modèle disponible
classes = oof_data[available_models[0]]["classes"]
print(f"\nClasses: {list(classes)}")

# =============================================================================
# 2. CONSTRUIRE LA TABLE DE MÉTRIQUES
# =============================================================================
all_metrics = []
all_bins    = []

for name in available_models:
    d = oof_data[name]
    m = d["oof_metrics"].copy()
    m["model"] = name
    all_metrics.append(m)
    all_bins.append(d["oof_bins"])

df_metrics = pd.DataFrame(all_metrics).set_index("model")
print("\n--- OOF Metrics Summary ---")
print(df_metrics[["F1_macro","AUC_OvR","LogLoss","ECE","Accuracy"]].to_string())

# Identifier le meilleur modèle (F1 macro)
best_model_name = df_metrics["F1_macro"].idxmax()
print(f"\nBest model (F1 macro): {best_model_name}")

# =============================================================================
# 3. FIGURE 1 : Decision matrix heatmap
# =============================================================================
fig_dm_path = CLF_FIG_DIR / "clf_decision_matrix.png"
plot_decision_matrix_clf(all_metrics, save_path=fig_dm_path)

# =============================================================================
# 4. FIGURE 2 : Reliability diagrams — tous les modèles, 1 seule figure
# =============================================================================
fig_rd_path = CLF_FIG_DIR / "clf_reliability_diagrams_all.png"
plot_all_reliability_diagrams(all_bins, available_models, save_path=fig_rd_path)

# =============================================================================
# 5. FIGURE 3 : Confusion matrix du meilleur modèle
# =============================================================================
best_data   = oof_data[best_model_name]
fig_cm_path = CLF_FIG_DIR / f"clf_confusion_matrix_{best_model_name}.png"
plot_confusion_matrix(
    y_true      = best_data["y_true"],
    y_pred      = best_data["y_pred"],
    class_names = list(classes),
    model_name  = f"{best_model_name} (OOF)",
    save_path   = fig_cm_path,
)
# =============================================================================
# 5. FIGURE 3 : Confusion matrix — meilleur modèle + SVM
# =============================================================================
# Meilleur modèle (F1)
best_data   = oof_data[best_model_name]
fig_cm_path = CLF_FIG_DIR / f"clf_confusion_matrix_{best_model_name}.png"
plot_confusion_matrix(
    y_true      = best_data["y_true"],
    y_pred      = best_data["y_pred"],
    class_names = list(classes),
    model_name  = f"{best_model_name} (OOF)",
    save_path   = fig_cm_path,
)

# SVM — pour comparer 
fig_cm_svm_path = CLF_FIG_DIR / "clf_confusion_matrix_SVM_RBF.png"
if "SVM_RBF" in oof_data:
    svm_data = oof_data["SVM_RBF"]
    plot_confusion_matrix(
        y_true      = svm_data["y_true"],
        y_pred      = svm_data["y_pred"],
        class_names = list(classes),
        model_name  = "SVM_RBF (OOF)",
        save_path   = fig_cm_svm_path,
    )
else:
    print("  WARNING: SVM_RBF OOF non disponible — confusion matrix ignorée")
    fig_cm_svm_path = None
# =============================================================================
# 6. RAPPORT TEXTE
# =============================================================================
report_lines = [
    "=" * 60,
    "CLASSIFICATION EVALUATION REPORT",
    "=" * 60,
    "",
    f"Models evaluated : {available_models}",
    f"Target           : produit_classe",
    f"Classes          : {list(classes)}",
    f"CV               : 5-Fold OOF",
    "",
    "--- OOF METRICS ---",
]
for name in available_models:
    m = oof_data[name]["oof_metrics"]
    report_lines.append(
        f"  {name:22s} F1={m['F1_macro']:.4f}  AUC={m['AUC_OvR']:.4f}"
        f"  LogLoss={m['LogLoss']:.4f}  ECE={m['ECE']:.4f}  Acc={m['Accuracy']:.4f}"
    )
report_lines += [
    "",
    f"Best model (F1)  : {best_model_name}",
    "",
    "--- CV STABILITY (best model) ---",
]
cv = oof_data[best_model_name].get("cv_summary", {})
for k, v in cv.items():
    report_lines.append(f"  {k:30s} : {v}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = CLF_RESULTS_DIR / "clf_evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved: {report_path}")

# =============================================================================
# 7. LOG MLFLOW CONSOLIDÉ
# =============================================================================
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name="Classification_Evaluation") as run:
    mlflow.set_tag("pipeline_stage", "classification_evaluation")

    for name in available_models:
        m = oof_data[name]["oof_metrics"]
        pfx = name.lower()
        mlflow.log_metric(f"{pfx}_F1",       m["F1_macro"])
        mlflow.log_metric(f"{pfx}_AUC",      m["AUC_OvR"])
        mlflow.log_metric(f"{pfx}_LogLoss",  m["LogLoss"])
        mlflow.log_metric(f"{pfx}_ECE",      m["ECE"])
        mlflow.log_metric(f"{pfx}_Accuracy", m["Accuracy"])

    mlflow.set_tag("best_model", best_model_name)

    for path in [fig_dm_path, fig_rd_path, fig_cm_path, report_path]:
        if Path(path).exists():
            mlflow.log_artifact(str(path))

    print(f"\nMLflow run: {run.info.run_id}")

print(f"\nDone. Best model: {best_model_name}")
print(f"  F1={df_metrics.loc[best_model_name,'F1_macro']} "
      f"| AUC={df_metrics.loc[best_model_name,'AUC_OvR']}")