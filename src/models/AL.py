import numpy as np
import pandas as pd
import math
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. SEED GLOBALE — REPRODUCTIBILITÉ TOTALE
# ============================================================
SEED = 42
np.random.seed(SEED)

FIGURES_PATH = r"D:\Mes projets\PFE\Project\models\results\figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

N_RUNS = 5  # Nombre de runs pour moyenner les résultats

# ============================================================
# 1. CHARGEMENT ET NETTOYAGE
# ============================================================

def load_and_clean_mof_dataset():
    df = pd.read_csv(r"D:\Mes projets\PFE\Project\data\processed\pipeline_outputs\feature_selection.csv")
    print(f"Shape initial: {df.shape}")

    target_col = 'taux_log'
    df = df.dropna(subset=[target_col])

    feature_cols = [col for col in df.columns
                    if col != target_col and np.issubdtype(df[col].dtype, np.number)]

    X = df[feature_cols].values
    y = df[target_col].values

    if np.isnan(X).sum() > 0:
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    print(f"Dataset final: {X.shape[0]} échantillons, {X.shape[1]} features")
    return X, y, feature_cols


# ============================================================
# 2. FONCTIONS D'ACQUISITION
# ============================================================

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def norm_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def acquisition_ucb(mu, sigma, kappa=2.0):
    return mu + kappa * sigma

def acquisition_ei(mu, sigma, y_best):
    if sigma < 1e-10:
        return 0.0
    improvement = y_best - mu
    z = improvement / sigma
    return improvement * norm_cdf(z) + sigma * norm_pdf(z)

def acquisition_pi(mu, sigma, y_best, xi=0.01):
    if sigma < 1e-10:
        return 0.0
    z = (y_best - mu - xi) / sigma
    return norm_cdf(z)


# ============================================================
# 3. BOUCLE ACTIVE LEARNING — FOCALISÉE SUR R²
# ============================================================

def active_learning_loop(X_pool, y_pool, X_init, y_init, X_val, y_val,
                         acquisition_func, n_iterations=50, acq_params=None, run_seed=42):

    np.random.seed(run_seed)

    D_X = list(X_init.copy())
    D_y = list(y_init.copy())
    r2_history = []

    if acq_params is None:
        acq_params = {}

    X_pool = X_pool.copy()
    y_pool = y_pool.copy()

    for t in range(n_iterations):
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4, normalize_y=True,
            n_restarts_optimizer=2, random_state=run_seed
        )
        gpr.fit(np.array(D_X), np.array(D_y))

        scores = []
        for x in X_pool:
            mu, sigma = gpr.predict([x], return_std=True)
            sigma = max(sigma[0], 1e-6)
            mu = mu[0]

            if acquisition_func == 'ucb':
                score = acquisition_ucb(mu, sigma, acq_params.get('kappa', 2.0))
            elif acquisition_func == 'ei':
                score = acquisition_ei(mu, sigma, min(D_y))
            elif acquisition_func == 'pi':
                score = acquisition_pi(mu, sigma, min(D_y), acq_params.get('xi', 0.01))
            else:
                score = np.random.randn()
            scores.append(score)

        best_idx = np.argmax(scores)
        D_X.append(X_pool[best_idx])
        D_y.append(y_pool[best_idx])
        X_pool = np.delete(X_pool, best_idx, axis=0)
        y_pool = np.delete(y_pool, best_idx, axis=0)

        gpr_eval = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-4, random_state=run_seed
        )
        gpr_eval.fit(np.array(D_X), np.array(D_y))
        y_pred = gpr_eval.predict(X_val)
        r2_history.append(r2_score(y_val, y_pred))

    return r2_history


# ============================================================
# 4. BASELINE ALÉATOIRE
# ============================================================

def random_baseline(X_pool, y_pool, X_init, y_init, X_val, y_val,
                    n_iterations=50, n_reps=5):
    all_r2 = []
    for rep in range(n_reps):
        np.random.seed(SEED + rep)
        n_pool = len(X_pool)
        indices = np.random.choice(n_pool, size=min(n_iterations, n_pool), replace=False)
        r2_history = []

        for k in range(1, n_iterations + 1):
            idx = indices[:min(k, len(indices))]
            sub_X = np.vstack([X_init] + [X_pool[i] for i in idx])
            sub_y = np.hstack([y_init] + [y_pool[i] for i in idx])

            gpr = GaussianProcessRegressor(
                kernel=Matern(nu=2.5), alpha=1e-4, random_state=SEED
            )
            gpr.fit(sub_X, sub_y)
            y_pred = gpr.predict(X_val)
            r2_history.append(r2_score(y_val, y_pred))

        all_r2.append(r2_history)

    return np.mean(all_r2, axis=0), np.std(all_r2, axis=0)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X, y, feature_names = load_and_clean_mof_dataset()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split unique et reproductible
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    n_init = 10
    np.random.seed(SEED)
    init_idx = np.random.choice(len(X_train), size=n_init, replace=False)
    X_init = X_train[init_idx]
    y_init = y_train[init_idx]

    mask = np.ones(len(X_train), dtype=bool)
    mask[init_idx] = False
    X_pool = X_train[mask]
    y_pool = y_train[mask]

    print(f"\nInitialisation : {n_init} points | Pool : {len(X_pool)} | Val : {len(X_val)}")

    strategies = {
        'UCB (κ=2.0)':   ('ucb', {'kappa': 2.0}),
        'Expected Improvement': ('ei', {}),
        'Prob. of Improvement (ξ=0.01)': ('pi', {'xi': 0.01}),
    }

    # Moyenner sur N_RUNS pour chaque stratégie
    results_r2_mean = {}
    results_r2_std  = {}

    print(f"\n{'='*60}")
    print(f"RUNS (N={N_RUNS}) PAR STRATÉGIE — FOCUS R²")
    print(f"{'='*60}")

    for name, (func, params) in strategies.items():
        print(f"\n{name}:")
        all_runs = []
        for run in range(N_RUNS):
            r2_hist = active_learning_loop(
                X_pool, y_pool, X_init, y_init, X_val, y_val,
                func, n_iterations=50, acq_params=params,
                run_seed=SEED + run
            )
            all_runs.append(r2_hist)
            print(f"  Run {run+1}/{N_RUNS} — R² final: {r2_hist[-1]:.4f}")

        results_r2_mean[name] = np.mean(all_runs, axis=0)
        results_r2_std[name]  = np.std(all_runs,  axis=0)

    print(f"\nBaseline aléatoire ({N_RUNS} runs):")
    rand_mean_r2, rand_std_r2 = random_baseline(
        X_pool, y_pool, X_init, y_init, X_val, y_val,
        n_iterations=50, n_reps=N_RUNS
    )

    # ============================================================
    # VISUALISATION — R² UNIQUEMENT
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = range(1, 51)
    colors = {'UCB (κ=2.0)': '#1f77b4',
               'Expected Improvement': '#ff7f0e',
               'Prob. of Improvement (ξ=0.01)': '#2ca02c'}

    # Courbes d'évolution R²
    for name, mean_r2 in results_r2_mean.items():
        std_r2 = results_r2_std[name]
        ax1.plot(iterations, mean_r2, label=name, linewidth=2, color=colors[name])
        ax1.fill_between(iterations,
                         mean_r2 - std_r2,
                         mean_r2 + std_r2,
                         alpha=0.15, color=colors[name])

    ax1.plot(iterations, rand_mean_r2, 'k--', label='Random baseline', linewidth=2)
    ax1.fill_between(iterations,
                     rand_mean_r2 - rand_std_r2,
                     rand_mean_r2 + rand_std_r2,
                     color='gray', alpha=0.15)

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('R² (5-fold, moyenne sur runs)', fontsize=11)
    ax1.set_title('R² Evolution — Active Learning', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Barres finales R²
    final_r2 = {name: results_r2_mean[name][-1] for name in results_r2_mean}
    final_r2['Random'] = rand_mean_r2[-1]

    bar_colors = [colors.get(n, '#7f7f7f') for n in final_r2.keys()]
    bars = ax2.bar(final_r2.keys(), final_r2.values(), color=bar_colors, width=0.5)

    for bar, val in zip(bars, final_r2.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Final R² (after 50 iterations)', fontsize=11)
    ax2.set_title('Final R² Comparison', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    ax2.set_ylim(0, max(final_r2.values()) * 1.15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Active Learning — R² Focus (seed={SEED}, {N_RUNS} runs averaged)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    figure_path = os.path.join(FIGURES_PATH, 'AL_r2_focus.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {figure_path}")
    plt.show()

    # ============================================================
    # RÉSULTATS FINAUX
    # ============================================================

    print(f"\n{'='*60}")
    print("RÉSULTATS FINAUX — R²")
    print(f"{'='*60}")
    print(f"{'Stratégie':<40} {'R² final':>10} {'± std':>8} {'Gain vs Random':>16}")
    print("-" * 78)

    for name, mean_r2 in results_r2_mean.items():
        r2_f = mean_r2[-1]
        std_f = results_r2_std[name][-1]
        gain  = (r2_f - rand_mean_r2[-1]) / abs(rand_mean_r2[-1]) * 100
        print(f"{name:<40} {r2_f:>10.4f} {std_f:>8.4f} {gain:>+15.2f}%")

    print(f"{'Random baseline':<40} {rand_mean_r2[-1]:>10.4f} {rand_std_r2[-1]:>8.4f} {'---':>16}")

    best = max(results_r2_mean.items(), key=lambda x: x[1][-1])
    gain_best = (best[1][-1] - rand_mean_r2[-1]) / abs(rand_mean_r2[-1]) * 100

    print(f"\n{'='*60}")
    print(f"✅ Meilleure stratégie (R²) : {best[0]}")
    print(f"   R² final moyen           : {best[1][-1]:.4f}")
    print(f"   Gain vs Random           : {gain_best:+.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()