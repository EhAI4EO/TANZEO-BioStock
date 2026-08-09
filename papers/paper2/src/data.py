"""Load the GEE-sampled point dataset and prepare it for scikit-learn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import Config


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_xy(dataset: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split into predictor/target arrays using the band lists from config."""
    predictors = config.get("bands", "predictors")
    target = config.get("bands", "target")

    x_data = dataset[predictors]
    y_data = dataset[[target]]
    return x_data.values, y_data.values.ravel(), predictors


def scale_features(x_array: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(x_array)
    return x_norm, scaler
