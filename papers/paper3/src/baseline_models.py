# -*- coding: utf-8 -*-
"""
baseline_models.py

OPTIONAL comparison baselines against the production Random Forest model:
Support Vector Regression, CatBoost, and a 1D-CNN.

These reproduce the "several regression algorithms were compared" step
mentioned in the paper's abstract (Random Forest was ultimately selected
for the operational pipeline). They are intentionally **not** run by the
default `run_fch_mapping.py` / `run_agb_mapping.py` / `run_agc_mapping.py`
entry points; invoke them explicitly (e.g. via `--evaluate-baselines`) if
you want to reproduce the model-comparison step.

Status of each baseline, verified against the original scripts:
    - SVR:      complete, runs on CPU with scikit-learn (no extra deps).
    - CatBoost: complete, but requires the optional `catboost` package.
    - 1D-CNN:   complete, but requires the optional `tensorflow` package.
      This was only present in the FCH script's comparison; it was not
      re-run for AGB/AGC in the original materials, so it is offered here
      as a general-purpose utility rather than a claim that it was used
      for all three targets.

These baselines are provided as reproducible, documented utilities for
model comparison and are independent of the operational FCH/AGB/AGC
pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from sklearn.svm import SVR

from .model_utils import RegressionMetrics, compute_regression_metrics

logger = logging.getLogger(__name__)


def evaluate_svr(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1,
) -> tuple[SVR, RegressionMetrics, float]:
    """Train and evaluate a Support Vector Regressor baseline."""
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    metrics.log(label="SVR baseline")
    return model, metrics, elapsed


def evaluate_catboost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    iterations: int = 300, learning_rate: float = 0.1, depth: int = 6, random_state: int = 42,
):
    """Train and evaluate a CatBoost regressor baseline.

    Raises
    ------
    ImportError
        If the optional `catboost` package is not installed.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is an optional dependency for baseline comparison. "
            "Install it with `pip install catboost`."
        ) from exc

    model = CatBoostRegressor(
        verbose=0, iterations=iterations, learning_rate=learning_rate, depth=depth, random_state=random_state,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    metrics.log(label="CatBoost baseline")
    return model, metrics, elapsed


def evaluate_1d_cnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 100, batch_size: int = 32, learning_rate: float = 5e-4, validation_split: float = 0.2,
):
    """Train and evaluate a small 1D-CNN regression baseline.

    Reproduces the architecture used in the original FCH comparison
    (Conv1D(64) -> Dropout -> Conv1D(128) -> BatchNorm -> Conv1D(256) ->
    Flatten -> Dense(128) -> Dropout -> Dense(1)).

    Raises
    ------
    ImportError
        If the optional `tensorflow` package is not installed.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is an optional dependency for the 1D-CNN baseline. "
            "Install it with `pip install tensorflow`."
        ) from exc

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same", input_shape=(X_train_cnn.shape[1], 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    start = time.time()
    model.fit(X_train_cnn, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    elapsed = time.time() - start

    y_pred = model.predict(X_test_cnn, verbose=0).flatten()
    metrics = compute_regression_metrics(y_test, y_pred)
    metrics.log(label="1D-CNN baseline")
    return model, metrics, elapsed


def run_all_baselines(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    include_catboost: bool = True, include_cnn: bool = True,
) -> dict:
    """Run SVR (always), and optionally CatBoost / 1D-CNN, skipping missing deps.

    Returns
    -------
    dict mapping baseline name -> {"metrics": ..., "train_time_s": ...},
    with an ``"error"`` key instead if a dependency was unavailable.
    """
    results = {}

    _, metrics, elapsed = evaluate_svr(X_train, y_train, X_test, y_test)
    results["svr"] = {"metrics": metrics.as_dict(), "train_time_s": elapsed}

    if include_catboost:
        try:
            _, metrics, elapsed = evaluate_catboost(X_train, y_train, X_test, y_test)
            results["catboost"] = {"metrics": metrics.as_dict(), "train_time_s": elapsed}
        except ImportError as exc:
            logger.warning("Skipping CatBoost baseline: %s", exc)
            results["catboost"] = {"error": str(exc)}

    if include_cnn:
        try:
            _, metrics, elapsed = evaluate_1d_cnn(X_train, y_train, X_test, y_test)
            results["cnn_1d"] = {"metrics": metrics.as_dict(), "train_time_s": elapsed}
        except ImportError as exc:
            logger.warning("Skipping 1D-CNN baseline: %s", exc)
            results["cnn_1d"] = {"error": str(exc)}

    return results
