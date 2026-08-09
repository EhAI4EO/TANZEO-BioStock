"""Train and cross-validate the four benchmark regressors used in the paper.

Each model is fit on an RFE-reduced feature set (RFE base estimator matches
the model family), hyperparameter-tuned with `GridSearchCV` (skipped for
OLS, which has no hyperparameters to tune), and evaluated with 5-fold CV.
Per-fold train/test metrics and predictions are saved to CSV, and the
best-fit estimator is pickled with `joblib`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from tqdm import tqdm

warnings.filterwarnings("ignore")


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict | None  # None => no grid search (e.g. OLS)
    rfe_base_estimator: Any
    n_features_to_select: int = 10


MODEL_SPECS: dict[str, ModelSpec] = {
    "RF": ModelSpec(
        name="RF",
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid={
            "n_estimators": [50, 100, 200, 300, 500, 600, 700, 800, 900, 1000],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        rfe_base_estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    ),
    "SVR": ModelSpec(
        name="SVR",
        estimator=SVR(),
        param_grid={
            "C": [10, 100, 1000],
            "epsilon": [0.01, 0.1, 0.5, 1],
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": [3, 4, 5],
        },
        rfe_base_estimator=SVR(kernel="linear"),
    ),
    "GBR": ModelSpec(
        name="GBR",
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid={
            "n_estimators": [100, 200, 300, 500, 600, 700, 800, 900, 1000],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "max_features": ["sqrt", "log2", None],
        },
        rfe_base_estimator=GradientBoostingRegressor(random_state=42),
    ),
    "OLS": ModelSpec(
        name="OLS",
        estimator=LinearRegression(),
        param_grid=None,
        rfe_base_estimator=LinearRegression(),
    ),
}


def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def train_evaluate_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """RFE-reduce, (optionally) grid-search, then 5-fold CV a model.

    Returns (best_estimator, per_fold_metrics_df, per_fold_predictions_df).
    """
    selector = RFE(
        estimator=spec.rfe_base_estimator,
        n_features_to_select=spec.n_features_to_select,
        step=1,
    )
    X_reduced = selector.fit_transform(X, y)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if spec.param_grid is not None:
        print(f"[{spec.name}] running grid search for best hyperparameters...")
        grid_search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(X_reduced, y)
        print(f"[{spec.name}] best params: {grid_search.best_params_}")
        print(f"[{spec.name}] best CV R2: {grid_search.best_score_:.3f}")
        best_model = grid_search.best_estimator_
    else:
        best_model = spec.estimator

    all_results, all_predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(cv.split(X_reduced), total=cv.get_n_splits(), desc=f"{spec.name} CV folds"), start=1
    ):
        best_model.fit(X_reduced[train_idx], y[train_idx])
        y_train_pred = best_model.predict(X_reduced[train_idx])
        y_test_pred = best_model.predict(X_reduced[test_idx])

        train_metrics = _compute_metrics(y[train_idx], y_train_pred)
        test_metrics = _compute_metrics(y[test_idx], y_test_pred)

        all_results.append({
            "fold": fold,
            **{f"Train_{k}": v for k, v in train_metrics.items()},
            **{f"Test_{k}": v for k, v in test_metrics.items()},
        })
        all_predictions.append(pd.DataFrame({
            "fold": fold,
            "y_actual": np.ravel(y[test_idx]),
            "y_predicted": np.ravel(y_test_pred),
        }))

    df_results = pd.DataFrame(all_results)
    df_predictions = pd.concat(all_predictions, ignore_index=True)

    print(f"[{spec.name}] mean metrics across folds:")
    print(df_results.mean(numeric_only=True))

    return best_model, df_results, df_predictions


def run_all_models(
    X: np.ndarray, y: np.ndarray, output_dir: str | Path, model_names: list[str] | None = None
) -> dict[str, dict]:
    """Train/evaluate every model in `model_names` (default: all) and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = model_names or list(MODEL_SPECS.keys())
    outcomes = {}
    for name in model_names:
        spec = MODEL_SPECS[name]
        best_model, df_results, df_predictions = train_evaluate_model(spec, X, y)

        joblib.dump(best_model, output_dir / f"best_{name.lower()}_model.pkl")
        df_results.to_csv(output_dir / f"{name}_metrics.csv", index=False)
        df_predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)

        outcomes[name] = {
            "model": best_model,
            "metrics": df_results,
            "predictions": df_predictions,
        }
    return outcomes
