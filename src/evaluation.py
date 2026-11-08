import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def compute_mae(y_true, y_pred):
    """Mean Absolute Error — unit: same as log_rate."""
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true, y_pred):
    """R² — between -∞ and 1. >0.7 = good for experimental chemistry data."""
    return r2_score(y_true, y_pred)


def compute_rmse(y_true, y_pred):
    """RMSE — more sensitive to outliers than MAE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ---------------------------------------------------------------------------
# Gaussian log-likelihood
# ---------------------------------------------------------------------------

def compute_log_likelihood(y_true, y_pred, sigma=None):
    """
    Log-likelihood under Gaussian assumption: N(y_pred, sigma²).

    If sigma is not provided, it is estimated as the RMSE of residuals
    (empirical sigma on validation fold).

    Higher values (less negative) indicate better calibration.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    sigma  : float or None — assumed standard deviation of errors

    Returns
    -------
    float : average log-likelihood per observation
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    if sigma is None:
        sigma = np.std(residuals)          # empirical estimation
        sigma = max(sigma, 1e-6)           # avoid division by zero

    n = len(y_true)
    ll = -0.5 * np.log(2 * np.pi * sigma**2) - (residuals**2) / (2 * sigma**2)
    return float(ll.mean())


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(y_true, y_pred, sigma=None, n_bins=10):
    """
    Expected Calibration Error for regression.

    Principle: build Gaussian confidence intervals at different confidence
    levels α (10%, 20%, ..., 90%) and verify that the actual proportion
    of observations within the interval matches α.

    ECE = average of |observed_coverage - α| over all levels.

    Interpretation:
        ECE ≈ 0.00 → perfectly calibrated
        ECE ≈ 0.10 → 10% average deviation between confidence and reality

    Parameters
    ----------
    y_true  : array-like
    y_pred  : array-like
    sigma   : float or None — standard deviation of errors (estimated if None)
    n_bins  : int — number of confidence levels tested

    Returns
    --------
    float : ECE
    dict  : details per level (useful for plotting calibration curve)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    if sigma is None:
        sigma = np.std(residuals)
        sigma = max(sigma, 1e-6)

    from scipy import stats

    alphas = np.linspace(0.1, 0.9, n_bins)
    calibration_detail = {}
    ece_terms = []

    for alpha in alphas:
        # Symmetric confidence interval at level alpha
        z = stats.norm.ppf((1 + alpha) / 2)
        lower = y_pred - z * sigma
        upper = y_pred + z * sigma

        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        ece_terms.append(abs(coverage - alpha))
        calibration_detail[round(alpha, 2)] = {
            "expected_coverage": round(alpha, 3),
            "observed_coverage": round(float(coverage), 3),
            "error"            : round(abs(coverage - alpha), 3),
        }

    ece = float(np.mean(ece_terms))
    return ece, calibration_detail


# ---------------------------------------------------------------------------
# Main function: all metrics at once
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, sigma=None, model_name="model", verbose=True):
    """
    Computes MAE, R², RMSE, Log-likelihood and ECE.

    Returns a dict with all metrics.
    """
    mae  = compute_mae(y_true, y_pred)
    r2   = compute_r2(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    ll   = compute_log_likelihood(y_true, y_pred, sigma)
    ece, cal_detail = compute_ece(y_true, y_pred, sigma)

    metrics = {
        "model"          : model_name,
        "MAE"            : round(mae,  4),
        "R2"             : round(r2,   4),
        "RMSE"           : round(rmse, 4),
        "LogLikelihood"  : round(ll,   4),
        "ECE"            : round(ece,  4),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Evaluation: {model_name}")
        print(f"{'='*50}")
        print(f"  MAE            : {mae:.4f}   ← priority")
        print(f"  R²             : {r2:.4f}   ← explained variance")
        print(f"  RMSE           : {rmse:.4f}")
        print(f"  Log-Likelihood : {ll:.4f}   ← probabilistic calibration")
        print(f"  ECE            : {ece:.4f}   ← interval reliability")
        print(f"{'='*50}\n")

    return metrics, cal_detail


# ---------------------------------------------------------------------------
# Aggregate CV results (mean ± std across folds)
# ---------------------------------------------------------------------------

def aggregate_cv_results(fold_metrics_list):
    """
    Takes a list of dicts (one per fold) and returns mean ± std.

    Parameters
    ----------
    fold_metrics_list : list[dict]

    Returns
    -------
    dict with keys 'mean' and 'std' for each metric
    """
    df = pd.DataFrame(fold_metrics_list)
    numeric_cols = df.select_dtypes(include=np.number).columns
    summary = {}
    for col in numeric_cols:
        summary[f"{col}_mean"] = round(df[col].mean(), 4)
        summary[f"{col}_std"]  = round(df[col].std(),  4)
    return summary


# ---------------------------------------------------------------------------
# Visualization: predictions vs actual + residuals
# ---------------------------------------------------------------------------

def plot_predictions(y_true, y_pred, model_name="model", save_path=None):
    """
    Two side-by-side plots:
      - Left: y_pred vs y_true (ideal line)
      - Right: residuals vs y_pred (homoscedasticity)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Prediction Diagnostics", fontsize=13, fontweight="bold")

    # --- Left: pred vs actual ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Ideal y=x")
    ax.set_xlabel("Actual log_rate")
    ax.set_ylabel("Predicted log_rate")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    # --- Right: residuals ---
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3, color="#DD8452")
    ax.axhline(0, color="r", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted log_rate")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals vs Predictions")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    plt.show()
    return fig


def plot_calibration_curve(cal_detail, model_name="model", save_path=None):
    """
    Calibration curve: observed vs expected coverage.
    A curve close to the diagonal = good ECE.
    """
    expected = [v["expected_coverage"] for v in cal_detail.values()]
    observed = [v["observed_coverage"] for v in cal_detail.values()]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
    ax.plot(expected, observed, "o-", color="#4C72B0", lw=2, label=model_name)
    ax.fill_between(expected, expected, observed, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Expected coverage (α)")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Calibration curve — {model_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    plt.show()
    return fig