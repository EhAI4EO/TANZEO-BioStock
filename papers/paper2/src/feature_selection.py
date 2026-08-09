"""Compare feature selection / dimensionality reduction methods.

Each method is swept across an increasing number of retained
features/components, scored with a 5-fold cross-validated Random Forest
baseline. This consolidates what were six near-identical experiment cells
in the original notebook (PCA, NMF, SelectKBest x2, RFE, SFS) into one
parameterized sweep, run once per method.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

RF_DEFAULTS = dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)


def _fold_metrics(rf, X_train, y_train, X_test, y_test) -> tuple[float, float, np.ndarray]:
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    return r2, rmse, rf.feature_importances_


def run_selection_sweep(
    method_name: str,
    transform_fn: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, dict]],
    X: np.ndarray,
    y: np.ndarray,
    param_range: range,
    n_splits: int = 5,
    random_state: int = 42,
    rf_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Sweep `transform_fn` over `param_range`, scoring each with CV Random Forest.

    `transform_fn(X, y, param)` must return `(X_transformed, extra_info_dict)`,
    where `extra_info_dict` is merged into every result row (e.g. explained
    variance, selected feature indices).
    """
    rf_kwargs = rf_kwargs or RF_DEFAULTS
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_results = []
    for param in tqdm(param_range, desc=f"{method_name} sweep"):
        X_t, extra_info = transform_fn(X, y, param)

        fold_r2, fold_rmse = [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_t), start=1):
            rf = RandomForestRegressor(**rf_kwargs)
            r2, rmse, importances = _fold_metrics(
                rf, X_t[train_idx], y[train_idx], X_t[test_idx], y[test_idx]
            )
            fold_r2.append(r2)
            fold_rmse.append(rmse)
            all_results.append({
                "n_components": param,
                "fold": fold,
                "R2": r2,
                "RMSE": rmse,
                "feature_importances": importances,
                "ranked_features": np.argsort(importances)[::-1],
                **extra_info,
            })

        tqdm.write(
            f"[{method_name} | {param}] Mean R2={np.mean(fold_r2):.3f}, "
            f"Mean RMSE={np.mean(fold_rmse):.3f}"
        )

    return pd.DataFrame(all_results)


# --- transform_fn implementations, one per method -------------------------

def _pca_transform(X, y, n_components):
    pca = PCA(n_components=n_components)
    X_t = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    return X_t, {
        "explained_variance_ratio": explained,
        "cumulative_variance": explained.cumsum(),
    }


def _nmf_transform(X, y, n_components):
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=500)
    X_t = nmf.fit_transform(X)
    return X_t, {"reconstruction_error": nmf.reconstruction_err_}


def _select_k_best_f_transform(X, y, k):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_t = selector.fit_transform(X, y)
    return X_t, {
        "selected_features": selector.get_support(indices=True),
        "f_regression_scores": selector.scores_,
    }


def _select_k_best_mi_transform(X, y, k):
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    X_t = selector.fit_transform(X, y)
    return X_t, {
        "selected_features": selector.get_support(indices=True),
        "mutual_info_scores": selector.scores_,
    }


def _rfe_transform(X, y, k):
    base_estimator = RandomForestRegressor(**RF_DEFAULTS)
    selector = RFE(estimator=base_estimator, n_features_to_select=k, step=1)
    X_t = selector.fit_transform(X, y)
    return X_t, {
        "selected_features": selector.get_support(indices=True),
        "ranking": selector.ranking_,
    }


SWEEP_METHODS: dict[str, Callable] = {
    "PCA": _pca_transform,
    "NMF": _nmf_transform,
    "SelectKBest_f_regression": _select_k_best_f_transform,
    "SelectKBest_mutual_info": _select_k_best_mi_transform,
    "RFE": _rfe_transform,
}


def run_all_sweeps(X: np.ndarray, y: np.ndarray, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Run every registered sweep method and write one CSV per method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = X.shape[1]
    results = {}
    for name, transform_fn in SWEEP_METHODS.items():
        df = run_selection_sweep(name, transform_fn, X, y, range(1, n_features + 1))
        df.to_csv(output_dir / f"results_{name}.csv", index=False)
        results[name] = df
    return results


def sequential_feature_selection(
    X: np.ndarray, y: np.ndarray, n_features_to_select: int = 5
) -> np.ndarray:
    """One-shot forward sequential feature selection, returns selected column indices."""
    sfs = SequentialFeatureSelector(
        RandomForestRegressor(),
        n_features_to_select=n_features_to_select,
        direction="forward",
    )
    sfs.fit_transform(X, y)
    return np.where(sfs.get_support())[0]


def variance_threshold_reduce(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    from sklearn.feature_selection import VarianceThreshold

    return VarianceThreshold(threshold=threshold).fit_transform(X)


def linear_regression_rfe(X: np.ndarray, y: np.ndarray, n_features_to_select: int = 5) -> np.ndarray:
    """RFE with an OLS base estimator, returns selected column indices."""
    rfe = RFE(LinearRegression(), n_features_to_select=n_features_to_select)
    rfe.fit_transform(X, y)
    return np.where(rfe.support_)[0]
