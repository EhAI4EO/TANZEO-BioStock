# -*- coding: utf-8 -*-
"""
evaluation.py

Model evaluation and interpretation plots: Random Forest feature
importances, sensor-group contribution radar chart, and SHAP-based
interpretation (optional; requires the ``shap`` package).

All plotting functions accept an ``output_path``; if given, the figure is
saved (publication-resolution) in addition to being returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

logger = logging.getLogger(__name__)

# Sensor-group boundaries in the ORIGINAL 20-band stack index space,
# matching the original "multi-sensor contribution" analysis.
SENSOR_GROUP_RANGES = {
    "Sentinel-2 bands": (0, 9),
    "Spectral indices": (10, 16),
    "Sentinel-1": (17, 18),
    "DEM": (19, 19),
}


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    title: str = "Random Forest Feature Importance",
    output_path: Optional[Path] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Bar plot of Random Forest feature importances, sorted descending."""
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.bar(range(len(importances)), importances[order], color="lightgreen", align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=90)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved feature-importance plot to %s", output_path)
    return fig


def sensor_group_contributions(
    selected_original_indices: Sequence[int], importances: Sequence[float]
) -> dict[str, float]:
    """Aggregate RF feature importances by sensor group.

    Parameters
    ----------
    selected_original_indices : sequence of int
        Original (pre-SelectKBest) 0-based band indices of the surviving
        features, e.g. from ``selector.get_support(indices=True)``.
    importances : sequence of float
        RF importances, in the same order as ``selected_original_indices``.

    Returns
    -------
    dict of group name -> normalized contribution (sums to ~1.0).
    """
    sums = {g: 0.0 for g in SENSOR_GROUP_RANGES}
    for orig_idx, imp in zip(selected_original_indices, importances):
        for group, (lo, hi) in SENSOR_GROUP_RANGES.items():
            if lo <= orig_idx <= hi:
                sums[group] += float(imp)
                break
    total = sum(sums.values())
    return {g: (v / total if total > 0 else 0.0) for g, v in sums.items()}


def plot_sensor_group_radar(
    group_contributions: dict[str, float],
    output_path: Optional[Path] = None,
    font_family: str = "DejaVu Serif",
    dpi: int = 300,
) -> plt.Figure:
    """Radar chart of sensor-group contributions to RF feature importance.

    Parameters
    ----------
    group_contributions : dict
        Output of :func:`sensor_group_contributions`.
    output_path : Path, optional
    font_family : str
        Defaults to a widely-available serif font. The original script
        used "Times New Roman", which may be unavailable outside Windows;
        override this argument if that font is installed.
    dpi : int
        NOTE: the original script used dpi=2000, which produces very large
        files; 300 is used as a more portable, publication-adequate
        default. Override if higher resolution is required.
    """
    rcParams["font.family"] = font_family

    groups = list(group_contributions.keys())
    vals = np.array([group_contributions[g] for g in groups])

    n = len(groups)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    vals_closed = np.concatenate([vals, [vals[0]]])

    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, vals_closed, linewidth=2)
    ax.fill(angles_closed, vals_closed, alpha=0.20)
    ax.set_ylim(0, max(0.35, vals.max() * 1.25))
    ax.set_thetagrids(np.degrees(angles), [])

    for angle, label in zip(angles, groups):
        ax.text(angle, ax.get_ylim()[1] * 1.08, label, ha="center", va="center", fontsize=13)
    for angle, v in zip(angles, vals):
        ax.text(angle, v + 0.02, f"{v:.2f}", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved sensor-group radar chart to %s", output_path)
    return fig


def plot_shap_summary(
    model,
    X_eval: np.ndarray,
    feature_names: Sequence[str],
    output_path: Optional[Path] = None,
    max_display: int = 20,
    dpi: int = 300,
):
    """SHAP beeswarm summary plot for a tree-based regressor.

    Requires the optional ``shap`` dependency (see requirements.txt).

    Raises
    ------
    ImportError
        If ``shap`` is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package is required for SHAP-based interpretation "
            "plots. Install it with `pip install shap`."
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    shap.summary_plot(
        shap_values, X_eval, feature_names=list(feature_names),
        show=False, max_display=min(max_display, len(feature_names)),
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved SHAP summary plot to %s", output_path)
    return fig, shap_values
