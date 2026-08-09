# -*- coding: utf-8 -*-
"""
model_utils.py

Configuration loading, feature selection, Random Forest training /
hyperparameter tuning, model serialization, and regression metrics.

Scientific behaviour preserved from the original research scripts:
    - Rows containing any zero-valued feature are dropped before model
      fitting (``filter_zero_rows``), matching the original
      ``mask_train = ~(X_train == 0).any(axis=1)`` filtering.
    - Feature selection via ``SelectKBest(f_regression, k=...)``.
    - Two-stage Random Forest fitting: a baseline
      ``RandomForestRegressor(n_estimators=100, random_state=42)`` followed
      by ``GridSearchCV`` hyperparameter tuning over the same parameter
      grid used in the original scripts. The GridSearchCV-tuned model is
      the one used for wall-to-wall prediction in all three workflows.
    - Metrics: RMSE, MAE, R^2, bias, nRMSE (%), MAPE (%), matching the
      original evaluation blocks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

# Hyperparameter grid used for GridSearchCV-based Random Forest tuning in
# all three workflows (FCH, AGB, AGC), matching the original scripts.
DEFAULT_RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(config_path: Path) -> dict:
    """Load and lightly validate a YAML configuration file.

    Parameters
    ----------
    config_path : Path

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist. The error message points the
        user to ``config/config.example.yaml``.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Copy config/config.example.yaml to config/config.yaml and "
            f"edit the paths/parameters for your environment."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# --------------------------------------------------------------------------- #
# Sample filtering / feature selection
# --------------------------------------------------------------------------- #
def filter_zero_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples that contain any exactly-zero feature value.

    Zero feature values typically indicate a patch that fell (partially)
    outside valid raster coverage. Matches the original scripts'
    ``mask = ~(X == 0).any(axis=1)`` filtering.
    """
    mask = ~(X == 0).any(axis=1)
    n_before = len(X)
    logger.info("Zero-row filtering: %d -> %d samples.", n_before, mask.sum())
    return X[mask], y[mask]


def select_k_best_features(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Fit ``SelectKBest(f_regression, k=k)`` on the training set.

    Returns
    -------
    (X_train_selected, X_test_selected, fitted_selector)
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector


def select_best_k(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    k_min: int = 3, k_max: Optional[int] = None, random_state: int = 42,
) -> tuple[int, SelectKBest, RandomForestRegressor, dict]:
    """Sweep k over ``[k_min, k_max]`` and keep the model with the best test R^2.

    Reproduces the original "best-K" search loop.

    Returns
    -------
    (best_k, best_selector, best_model, best_metrics)
    """
    if k_max is None:
        k_max = X_train.shape[1]

    best_k, best_score = None, -np.inf
    best_selector, best_model, best_metrics = None, None, {}

    for k in range(k_min, k_max + 1):
        selector = SelectKBest(score_func=f_regression, k=k)
        X_tr = selector.fit_transform(X_train, y_train)
        X_te = selector.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        r2 = r2_score(y_test, y_pred)
        if r2 > best_score:
            best_k, best_score = k, r2
            best_selector, best_model = selector, model
            best_metrics = compute_regression_metrics(y_test, y_pred)

    logger.info("Best K feature-selection sweep: k=%d, R2=%.4f.", best_k, best_score)
    return best_k, best_selector, best_model, best_metrics


# --------------------------------------------------------------------------- #
# Model training
# --------------------------------------------------------------------------- #
def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, random_state: int = 42
) -> tuple[RandomForestRegressor, float]:
    """Train a baseline Random Forest regressor.

    Returns
    -------
    (model, training_time_seconds)
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Trained baseline RandomForestRegressor in %.2fs.", elapsed)
    return model, elapsed


def tune_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    param_grid: Optional[dict] = None, cv: int = 5, random_state: int = 42, n_jobs: int = -1,
) -> tuple[RandomForestRegressor, dict, float]:
    """Tune a Random Forest via GridSearchCV.

    This is the final model used for wall-to-wall prediction in all three
    workflows (FCH, AGB, AGC), matching the original scripts.

    Returns
    -------
    (best_estimator, best_params, tuning_time_seconds)
    """
    grid = param_grid or DEFAULT_RF_PARAM_GRID
    search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid=grid, cv=cv, n_jobs=n_jobs, verbose=1, scoring="neg_mean_squared_error",
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(
        "GridSearchCV RF tuning complete in %.2fs. Best params: %s", elapsed, search.best_params_
    )
    return search.best_estimator_, search.best_params_, elapsed


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    bias: float
    nrmse_percent: float
    mape_percent: float

    def as_dict(self) -> dict:
        return {
            "rmse": self.rmse, "mae": self.mae, "r2": self.r2, "bias": self.bias,
            "nrmse_percent": self.nrmse_percent, "mape_percent": self.mape_percent,
        }

    def log(self, label: str = "Model", unit: str = "") -> None:
        u = f" {unit}" if unit else ""
        logger.info(
            "%s -- RMSE: %.2f%s | MAE: %.2f%s | Bias: %.2f%s | R2: %.4f | nRMSE: %.2f%% | MAPE: %.2f%%",
            label, self.rmse, u, self.mae, u, self.bias, u, self.r2, self.nrmse_percent, self.mape_percent,
        )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute RMSE, MAE, R^2, bias, nRMSE(%), and MAPE(%).

    MAPE is undefined (division by zero) if any ``y_true`` value is zero;
    in that case it is reported as ``np.nan`` rather than raising.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mean_y = np.mean(y_true)
    nrmse = float((rmse / mean_y) * 100) if mean_y != 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_terms = np.abs((y_true - y_pred) / y_true)
    mape = float(np.nanmean(np.where(y_true != 0, mape_terms, np.nan)) * 100)

    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2, bias=bias, nrmse_percent=nrmse, mape_percent=mape)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def save_model(model: Any, path: Path, overwrite: bool = False) -> None:
    """Save a fitted model with joblib.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists and overwrite=False: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> Any:
    """Load a joblib-serialized model.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run the corresponding training "
            f"stage first (see README.md, 'How to Run')."
        )
    return joblib.load(path)


def check_feature_order(feature_names: list[str], expected_names: list[str]) -> None:
    """Raise a clear error if a model's expected feature order/names differ.

    Used before wall-to-wall prediction to catch predictor-ordering
    mismatches between training and inference.
    """
    if list(feature_names) != list(expected_names):
        raise ValueError(
            "Feature name/order mismatch between training and prediction.\n"
            f"  Expected: {expected_names}\n"
            f"  Got:      {feature_names}\n"
            "This usually means the predictor stack was rebuilt with a "
            "different band order, or the wrong SelectKBest selector was applied."
        )
