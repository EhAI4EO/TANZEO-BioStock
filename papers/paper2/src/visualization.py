"""Aggregate results and produce the comparison figures used in the paper.

Covers: feature-selection sweep aggregation, per-model scatter/KDE density
plots against the 1:1 line, radar charts of test/train metrics across
models, and a Taylor diagram summarizing correlation/variance/CRMSE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# --------------------------------------------------------------------------
# Feature-selection sweep aggregation
# --------------------------------------------------------------------------

def aggregate_feature_selection_results(results_dir: str | Path) -> pd.DataFrame:
    """Combine `results_<method>.csv` sweep files into one tidy comparison table."""
    results_dir = Path(results_dir)
    frames = []
    for path in sorted(results_dir.glob("results_*.csv")):
        method = path.stem.replace("results_", "")
        df = pd.read_csv(path)
        df["method"] = method
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    if "k_features" in combined.columns:
        combined["n_components"] = combined.get("n_components", pd.Series(dtype=float)).fillna(
            combined["k_features"]
        )

    grouped = (
        combined.groupby(["method", "n_components"])
        .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
             RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"))
        .reset_index()
    )
    grouped["RMSE_mean"] = np.sqrt(grouped["RMSE_mean"])
    grouped["RMSE_std"] = np.sqrt(grouped["RMSE_std"])
    return grouped


# --------------------------------------------------------------------------
# Scatter / KDE density plot (predicted vs. actual canopy height)
# --------------------------------------------------------------------------

def plot_scatter_density(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
    axis_limit: float = 90,
) -> plt.Figure:
    """Scatter of predicted vs. true canopy height, colored by KDE density."""
    metrics = {
        "R²": f"{np_r2(y_true, y_pred):.3f}",
        "RMSE": f"{np_rmse(y_true, y_pred):.3f}",
        "MAE": f"{np_mae(y_true, y_pred):.3f}",
        "MAPE": f"{np_mape(y_true, y_pred) * 100:.2f}%",
    }
    stats_text = "\n".join(f"{k:5s}: {v}" for k, v in metrics.items())

    values = np.vstack([y_true, y_pred])
    kernel = stats.gaussian_kde(values)(values)

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.scatterplot(x=y_true, y=y_pred, c=kernel, cmap="Spectral_r", alpha=0.9, ax=ax)
    sns.kdeplot(
        x=y_true, y=y_pred, levels=7, fill=False, alpha=1, cut=7,
        bw_adjust=0.8, thresh=0.05, ax=ax, linewidths=1,
    )
    ax.plot([0, axis_limit], [0, axis_limit], "--k", linewidth=1)

    ax.set_xlabel("True FCH (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted FCH (m)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.tick_params(axis="both", which="major", labelsize=12)

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment="top", bbox=props)

    kde_line = mlines.Line2D([], [], color="black", linewidth=1, label="KDE Contours")
    one2one_line = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1, label="1:1 Line")
    legend = ax.legend(handles=[kde_line, one2one_line], fontsize=12, loc="lower right",
                        frameon=True, fancybox=True, framealpha=0.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    return fig


def np_r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


def np_rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))


def np_mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def np_mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)


# --------------------------------------------------------------------------
# Radar chart comparing normalized metrics across models
# --------------------------------------------------------------------------

def plot_radar_metrics(
    model_names: list[str],
    values: np.ndarray,  # shape (n_models, 4): [RMSE, MAE, MAPE(%), R2(%)]
    axis_limits: np.ndarray,  # shape (4, 2): [min, max] per metric
    save_path: str | Path | None = None,
) -> plt.Figure:
    metrics = ["RMSE (m)", "MAE (m)", "MAPE (%)", "R\u00b2 (%)"]

    values_norm = (values - axis_limits[:, 0]) / (axis_limits[:, 1] - axis_limits[:, 0])
    values_norm = np.clip(values_norm, 0, 1)
    values_norm = np.concatenate([values_norm, values_norm[:, [0]]], axis=1)

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, model in enumerate(model_names):
        ax.plot(angles, values_norm[i], label=model, linewidth=2)
        ax.fill(angles, values_norm[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.tick_params(axis="x", pad=30)
    ax.set_yticklabels([])

    legend = ax.legend(title="Models", fontsize=14, title_fontsize=16,
                        loc="upper right", bbox_to_anchor=(1.1, 0.2),
                        frameon=True, fancybox=True, framealpha=0.9)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Taylor diagram
# --------------------------------------------------------------------------

def _taylor_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """correlation, std(pred)/std(true), centered RMSE (normalized by std(true))."""
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    std_true = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)
    std_norm = std_pred / std_true
    crmse = np.sqrt(std_pred**2 + std_true**2 - 2 * std_pred * std_true * corr) / std_true
    return corr, std_norm, crmse


def plot_taylor_diagram(
    y_true: np.ndarray,
    model_preds: list[np.ndarray],
    model_names: list[str],
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, tuple[float, float, float]]]:
    """First-quadrant Taylor diagram with a zoomed inset for high-correlation models."""
    ref_std = np.std(y_true)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.plot([0], [ref_std], "ko", label="Reference")

    results = {}
    for name, y_pred in zip(model_names, model_preds):
        corr, std_norm, crmse = _taylor_stats(y_true, y_pred)
        theta = np.arccos(corr)
        ax.plot(theta, std_norm, "o", label=name)
        results[name] = (corr, std_norm, crmse)

    ax.set_title("Taylor Diagram (First Quadrant)", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.35], polar=True)
    ax_inset.set_theta_zero_location("N")
    ax_inset.set_theta_direction(-1)
    ax_inset.set_thetamin(0)
    ax_inset.set_thetamax(25)
    ax_inset.plot([0], [ref_std], "ko")
    for name, (corr, std_norm, _) in results.items():
        ax_inset.plot(np.arccos(corr), std_norm, "o")

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    return fig, results
