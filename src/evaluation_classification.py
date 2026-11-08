import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, roc_auc_score, log_loss, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------------------------
# ECE classification (reliability diagram)
# ---------------------------------------------------------------------------

def compute_ece_clf(y_true, y_proba, n_bins=10):
    """
    ECE pour classification multiclasse.
    Découpe les probabilités max (confiance) en bins et mesure
    l'écart entre confiance moyenne et accuracy réelle du bin.
    ECE = Σ (|bin|/N) × |accuracy(bin) − confidence(bin)|
    """
    y_pred      = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    correctness = (y_pred == y_true).astype(int)

    bins_data = []
    ece       = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correctness[mask].mean()
        conf = confidences[mask].mean()
        frac = mask.sum() / len(y_true)
        ece += frac * abs(acc - conf)
        bins_data.append({
            "bin_center": (lo + hi) / 2,
            "accuracy"  : float(acc),
            "confidence": float(conf),
            "n_samples" : int(mask.sum()),
        })

    return float(ece), bins_data


# ---------------------------------------------------------------------------
# Évaluation principale
# ---------------------------------------------------------------------------

def evaluate_clf(y_true, y_pred, y_proba, class_names, model_name="model", verbose=True):
    """
    Calcule F1 macro, AUC-ROC OvR, Log-Loss, ECE, Accuracy.
    Retourne metrics dict + bins_data pour reliability diagram.
    """
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    ll  = log_loss(y_true, y_proba)
    ece, bins = compute_ece_clf(y_true, y_proba)

    n_classes = y_proba.shape[1]
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    metrics = {
        "model"   : model_name,
        "F1_macro": round(f1,  4),
        "AUC_OvR" : round(auc, 4),
        "LogLoss" : round(ll,  4),
        "ECE"     : round(ece, 4),
        "Accuracy": round(acc, 4),
    }

    if verbose:
        print(f"\n{'='*52}")
        print(f"  Classification : {model_name}")
        print(f"{'='*52}")
        print(f"  F1-score macro : {f1:.4f}   <- prioritaire")
        print(f"  AUC-ROC (OvR)  : {auc:.4f}   <- standard biblio")
        print(f"  Log-Loss       : {ll:.4f}   <- calibration proba")
        print(f"  ECE            : {ece:.4f}   <- fiabilite intervalles")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"{'='*52}")
        print(classification_report(y_true, y_pred,
                                    target_names=class_names,
                                    zero_division=0))
    return metrics, bins


def aggregate_cv_clf(fold_metrics_list):
    """Moyenne +/- std sur les 5 folds."""
    df = pd.DataFrame(fold_metrics_list)
    numeric = df.select_dtypes(include=np.number).columns
    summary = {}
    for col in numeric:
        summary[f"{col}_mean"] = round(df[col].mean(), 4)
        summary[f"{col}_std"]  = round(df[col].std(),  4)
    return summary


def compute_sample_weights_clf(y):
    """Poids par échantillon pour XGBoost (équivalent class_weight='balanced')."""
    return compute_sample_weight("balanced", y)


# ---------------------------------------------------------------------------
# Figure 1 : Confusion matrix normalisée (meilleur modèle uniquement)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    """Côte à côte : counts bruts + normalisée par ligne (recall)."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} - Confusion Matrix (OOF)",
                 fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalized (recall per class)"],
        ["d", ".2f"]
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title(title,        fontsize=11)
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 : Reliability diagrams — tous les modèles, 1 seule figure
# ---------------------------------------------------------------------------

def plot_all_reliability_diagrams(all_bins_data, model_names, save_path=None):
    """
    Reliability diagrams superposés — équivalent des calibration curves
    de la régression.
    Axe X = confiance moyenne du bin | Axe Y = accuracy réelle du bin
    """
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],"r--", lw=2, label="Perfect calibration", zorder=5)

    for bins_data, name, color in zip(all_bins_data, model_names, colors):
        conf = [b["confidence"] for b in bins_data]
        acc  = [b["accuracy"]   for b in bins_data]
        ax.plot(conf, acc, "o-", color=color, lw=1.8,
                markersize=5, label=name, alpha=0.88)

    ax.set_xlabel("Mean confidence (max predicted proba)", fontsize=12)
    ax.set_ylabel("Fraction of correct predictions",       fontsize=12)
    ax.set_title("Reliability Diagrams - All Classification Models",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Reliability diagrams saved: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 : Decision matrix heatmap — tous les modèles
# ---------------------------------------------------------------------------

def plot_decision_matrix_clf(all_metrics, save_path=None):
    """Heatmap RdYlGn normalisée sur F1/AUC/ECE/LogLoss/Accuracy."""
    metric_cols = ["F1_macro", "AUC_OvR", "LogLoss", "ECE", "Accuracy"]
    df = pd.DataFrame(all_metrics).set_index("model")

    metrics_max = ["F1_macro", "AUC_OvR", "Accuracy"]
    metrics_min = ["LogLoss",  "ECE"]

    norm = df[metric_cols].copy().astype(float)
    for col in metrics_max:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5
    for col in metrics_min:
        rng = norm[col].max() - norm[col].min()
        norm[col] = 1 - (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5

    col_labels = ["F1 macro↑", "AUC↑", "LogLoss↓", "ECE↓", "Accuracy↑"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index, fontsize=10)

    raw = df[metric_cols].values
    for i in range(len(norm.index)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{raw[i,j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < norm.values[i,j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Normalized score (green = best)")
    ax.set_title("Decision Matrix - Classification Models (produit_classe)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Decision matrix saved: {save_path}")
    plt.show()
    return fig