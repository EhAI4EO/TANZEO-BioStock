# -*- coding: utf-8 -*-
"""
visualization.py

Exploratory and quicklook plots: target-variable boxplots (with IQR
outlier detection), Sentinel-2 false-color composites, and spectral-index
mosaics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_target_boxplot(
    values: pd.Series,
    title: str,
    ylabel: str,
    color: str = "skyblue",
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Vertical boxplot of a target variable with the mean marked.

    Parameters
    ----------
    values : pd.Series
        Cleaned (outlier-removed) target values.
    title, ylabel : str
    color : str
    output_path : Path, optional
    dpi : int
        NOTE: the original scripts used dpi up to 1000; 300 is used here
        as a portable default adequate for most journal figure requirements.
        Override for camera-ready submission if needed.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 8), dpi=dpi)
    sns.boxplot(y=values, color=color, width=0.4, linewidth=1.5, fliersize=5, ax=ax)

    mean_val = values.mean()
    ax.scatter(0, mean_val, color="red", s=100, marker="D", label=f"Mean: {mean_val:.2f}")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved boxplot to %s", output_path)
    return fig


def plot_s2_false_color(
    s2_norm: np.ndarray,
    nir_band: int = 6,
    red_band: int = 2,
    green_band: int = 1,
    brightness_factor: float = 2.0,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Sentinel-2 (NIR, Red, Green) false-color quicklook composite.

    Parameters
    ----------
    s2_norm : np.ndarray
        Normalized (0-1) Sentinel-2 stack, shape (H, W, 10).
    nir_band, red_band, green_band : int
        Zero-based band indices within the 10-band stack
        (defaults: B8=6, B4=2, B3=1).
    brightness_factor : float
        Linear brightening factor applied before clipping to [0, 1].
    """
    nir = np.clip(s2_norm[:, :, nir_band] * brightness_factor, 0, 1)
    red = np.clip(s2_norm[:, :, red_band] * brightness_factor, 0, 1)
    green = np.clip(s2_norm[:, :, green_band] * brightness_factor, 0, 1)
    rgb = np.stack([nir, red, green], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.imshow(rgb)
    ax.set_title("Sentinel-2 False Color Composite (NIR, Red, Green)")
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved S2 false-color composite to %s", output_path)
    return fig


def plot_spectral_indices_grid(
    indices: dict[str, np.ndarray],
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """2x4 grid of spectral-index maps with individual colorbars.

    Parameters
    ----------
    indices : dict
        Output of ``raster_utils.compute_spectral_indices``.
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 12), dpi=dpi)
    axes = axes.flatten()

    for i, (name, data) in enumerate(indices.items()):
        im = axes[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(name, fontsize=14)
        axes[i].axis("off")
        cbar = fig.colorbar(im, ax=axes[i], shrink=0.7)
        cbar.set_label("Index Value", rotation=270, labelpad=15)

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved spectral-index grid to %s", output_path)
    return fig


def plot_predicted_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Optional[Path] = None, dpi: int = 300
) -> plt.Figure:
    """Scatter plot of predicted vs. true values with a 1:1 reference line."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    ax.scatter(y_true, y_pred, alpha=0.6, color="green")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved predicted-vs-true plot to %s", output_path)
    return fig
