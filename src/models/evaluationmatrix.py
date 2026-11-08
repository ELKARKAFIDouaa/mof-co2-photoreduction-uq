import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "Model":    ["GPR Matern","XGBoost","LightGBM","RandomForest",
                 "SVM RBF","Decision Tree","Lasso","Ridge"],
    "MAE↓":     [0.876, 0.867, 0.953, 0.929, 0.937, 1.046, 1.104, 1.124],
    "R²↑":      [0.715, 0.709, 0.676, 0.678, 0.665, 0.603, 0.603, 0.592],
    "RMSE↓":    [1.23,  1.21,  1.31,  1.28,  1.30,  1.45,  1.48,  1.51],
    "ECE↓":     [0.083, 0.083, 0.064, 0.083, 0.091, 0.069, 0.036, 0.032],
    "LogLik↑":  [-1.626,-1.636,-1.690,-1.687,-1.694,-1.792,-1.791,-1.805],
}
df = pd.DataFrame(data).set_index("Model")

# Normaliser 0→1 (meilleur = 1)
metrics_min = ["MAE↓", "RMSE↓", "ECE↓"]
metrics_max = ["R²↑", "LogLik↑"]

norm = df.copy()
for col in metrics_min:
    norm[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
for col in metrics_max:
    norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(norm.columns)))
ax.set_xticklabels(norm.columns, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(norm.index)))
ax.set_yticklabels(norm.index, fontsize=10)

# Valeurs réelles dans chaque cellule
for i in range(len(norm.index)):
    for j in range(len(norm.columns)):
        ax.text(j, i, f"{df.values[i,j]:.3f}",
                ha="center", va="center", fontsize=9,
                color="black" if 0.3 < norm.values[i,j] < 0.8 else "white")

plt.colorbar(im, ax=ax, label="Normalized score (green = best)")
ax.set_title("Decision Matrix — 8 Surrogate Models for taux_log Prediction",
             fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("D:\\Mes projets\\PFE\\Project\\models\\results\\figures\\decision_matrix.png",
            dpi=200, bbox_inches="tight")
plt.show()