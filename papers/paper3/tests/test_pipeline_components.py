# -*- coding: utf-8 -*-
"""
tests/test_pipeline_components.py

Lightweight validation tests for critical reusable functions, per the
project's reproducibility requirements. These do NOT require the original remote-sensing datasets -- they
exercise geometry, feature selection, filtering, and configuration logic
with synthetic data. They are validation checks for the packaged
functions, not a reproduction of the manuscript's reported FCH/AGB/AGC
accuracy figures.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_preparation as dp
from src import model_utils as mu
from src import raster_utils as ru


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_rotated_buffer_area_preserved():
    from shapely.geometry import Point

    pt = Point(0, 0)
    poly = dp.create_rotated_buffer(pt, width=100, height=17, angle_deg=15)
    assert pytest.approx(poly.area, rel=1e-6) == 100 * 17


def test_aligned_rectangle_bounds():
    from shapely.geometry import Point

    pt = Point(10, 20)
    poly = dp.create_aligned_rectangle(pt, width=225, height=900)
    minx, miny, maxx, maxy = poly.bounds
    assert pytest.approx(maxx - minx) == 225
    assert pytest.approx(maxy - miny) == 900
    assert pytest.approx((minx + maxx) / 2) == 10
    assert pytest.approx((miny + maxy) / 2) == 20


def test_icesat2_track_angle_descending_and_ascending():
    assert dp.icesat2_track_angle(418, default_angle=90, offset_deg=5.5) == 84.5
    assert dp.icesat2_track_angle(730, default_angle=90, offset_deg=5.5) == 95.5
    assert dp.icesat2_track_angle(999, default_angle=90, offset_deg=5.5) == 90


# --------------------------------------------------------------------------- #
# IQR outlier filtering
# --------------------------------------------------------------------------- #
def test_iqr_bounds_reasonable():
    import pandas as pd

    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    lower, upper = dp.iqr_bounds(values)
    assert lower < values.median() < upper
    assert 100 > upper  # the extreme outlier should fall outside the fence


# --------------------------------------------------------------------------- #
# Patch reduction (center_mode)
# --------------------------------------------------------------------------- #
def test_reduce_patch_full_mean():
    data = np.ones((2, 5, 5)) * 3.0
    data[0, 0, 0] = 30.0  # perturb one corner pixel
    result = ru._reduce_patch(data, center_mode=0)
    expected_band0 = (24 * 3.0 + 30.0) / 25
    assert result[0] == pytest.approx(expected_band0)
    assert result[1] == pytest.approx(3.0)


def test_reduce_patch_center_pixel():
    data = np.zeros((1, 5, 5))
    data[0, 2, 2] = 42.0
    result = ru._reduce_patch(data, center_mode=1)
    assert result[0] == pytest.approx(42.0)


def test_reduce_patch_5x5_window():
    data = np.ones((1, 9, 9))
    data[0, 4, 4] = 5.0  # center pixel of the 5x5 window
    result = ru._reduce_patch(data, center_mode=25)
    # 5x5 window centered at (4,4): 24 pixels of value 1 + 1 pixel of value 5
    expected = (24 * 1.0 + 5.0) / 25
    assert result[0] == pytest.approx(expected)


def test_reduce_patch_invalid_center_mode_raises():
    data = np.ones((1, 5, 5))
    with pytest.raises(ValueError):
        ru._reduce_patch(data, center_mode=8)  # not an odd perfect square


# --------------------------------------------------------------------------- #
# Feature filtering / selection
# --------------------------------------------------------------------------- #
def test_filter_zero_rows_drops_any_zero_feature():
    X = np.array([[1.0, 2.0], [0.0, 5.0], [3.0, 4.0]])
    y = np.array([10.0, 20.0, 30.0])
    X_f, y_f = mu.filter_zero_rows(X, y)
    assert X_f.shape[0] == 2
    assert list(y_f) == [10.0, 30.0]


def test_select_k_best_features_shape():
    rng = np.random.default_rng(42)
    X_train = rng.random((50, 10))
    y_train = X_train[:, 0] * 5 + rng.random(50) * 0.01
    X_test = rng.random((10, 10))

    X_train_sel, X_test_sel, selector = mu.select_k_best_features(X_train, y_train, X_test, k=3)
    assert X_train_sel.shape == (50, 3)
    assert X_test_sel.shape == (10, 3)
    assert selector.get_support().sum() == 3


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_regression_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    metrics = mu.compute_regression_metrics(y_true, y_pred)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.bias == pytest.approx(0.0)


def test_feature_order_check_raises_on_mismatch():
    with pytest.raises(ValueError):
        mu.check_feature_order(["a", "b"], ["a", "c"])
    mu.check_feature_order(["a", "b"], ["a", "b"])  # should not raise


# --------------------------------------------------------------------------- #
# Configuration loading
# --------------------------------------------------------------------------- #
def test_load_config_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        mu.load_config(missing)


def test_load_config_reads_example(tmp_path):
    example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
    config = mu.load_config(example)
    assert "paths" in config
    assert "fch" in config and "agb" in config and "agc" in config
    assert config["crs_epsg"] == 21037
