# -*- coding: utf-8 -*-
"""
agc_mapping.py

Above-Ground Carbon (AGC) mapping workflow: Excel field-data conversion,
boundary clipping, axis-aligned buffer construction, patch feature
extraction (with FCH as an auxiliary 21st predictor band), SelectKBest
feature selection, Random Forest training + GridSearchCV tuning, and
wall-to-wall AGC prediction.

**Depends on FCH:** this stage requires the FCH raster produced by
``fch_mapping.predict()`` (``paths.fch_output_map`` in config.yaml) as an
auxiliary predictor band. Run the FCH stage first.

Default configuration (see config/config.example.yaml, section `agc`):
    - Field-plot buffer: 225 m x 900 m, axis-aligned (same geometry family
      as AGB, since both targets are typically measured on the same field
      plots).
    - Patch extraction bounding box: 22 x 90 pixels, reduced via a
      centered 5x5-pixel window mean (center_mode=25).
    - Predictors: the 20-band AGC-area stack + the FCH raster (21 total).
    - Feature selection: SelectKBest sweep over k, keeping the best test R^2.
    - Model: RandomForestRegressor, tuned via GridSearchCV.

All of the values above are configurable via `config.yaml` -- see
`config/config.example.yaml`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from . import data_preparation as dp
from . import model_utils as mu
from . import raster_utils as ru

logger = logging.getLogger(__name__)


def prepare_field_data(config: dict) -> gpd.GeoDataFrame:
    """Convert the AGC Excel field table to a GeoDataFrame and clip it.

    Reproduces the original two-step AGC preprocessing: Excel(X, Y) ->
    GeoPackage, then spatial clip against a study-area boundary shapefile.
    """
    agc_cfg = config["agc"]
    paths = config["paths"]

    gdf = dp.excel_xy_to_geodataframe(
        Path(paths["agc_field_excel"]), crs_epsg=config["crs_epsg"]
    )
    if paths.get("agc_boundary_shapefile"):
        gdf = dp.clip_by_boundary(gdf, Path(paths["agc_boundary_shapefile"]))
    return gdf


def prepare_training_geometries(config: dict, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove outliers and build field-plot buffers for the AGC points."""
    agc_cfg = config["agc"]
    gdf_clean = dp.remove_outliers(gdf, target_column=agc_cfg["target_column"])
    gdf_buffered = dp.build_field_plot_buffers(
        gdf_clean, width=agc_cfg["plot_width_m"], height=agc_cfg["plot_height_m"], crs_epsg=config["crs_epsg"]
    )
    return gdf_buffered


def extract_training_samples(config: dict, gdf_buffered: gpd.GeoDataFrame):
    """Split into train/test and extract 21-band (20-band stack + FCH) features."""
    agc_cfg = config["agc"]
    paths = config["paths"]

    train_gdf, test_gdf = dp.train_test_split_gdf(
        gdf_buffered, test_size=agc_cfg["test_size"], random_state=config["random_seed"]
    )

    common_kwargs = dict(
        stack_path=Path(paths["agc_predictor_stack_20band"]),
        auxiliary_path=Path(paths["fch_output_map"]),
        target_column=agc_cfg["target_column"],
        patch_width=agc_cfg["patch_width_px"],
        patch_height=agc_cfg["patch_height_px"],
        pixel_size=config["pixel_size_m"],
        selected_bands=list(range(1, 22)),
        center_mode=agc_cfg["center_mode"],
    )
    X_train, y_train = ru.extract_patch_features_with_auxiliary_band(gdf_subset=train_gdf, **common_kwargs)
    X_test, y_test = ru.extract_patch_features_with_auxiliary_band(gdf_subset=test_gdf, **common_kwargs)
    return X_train, y_train, X_test, y_test


def train(config: dict) -> dict:
    """Run the full AGC training stage.

    Requires ``paths.fch_output_map`` and ``paths.agc_predictor_stack_20band``
    to already exist on disk.

    Returns
    -------
    dict with keys: model, selector, best_k, metrics, selected_feature_names,
    tune_time_s.
    """
    paths = config["paths"]

    logger.info("AGC stage 1/4: loading and clipping field data.")
    gdf = prepare_field_data(config)

    logger.info("AGC stage 2/4: preparing field-plot buffers.")
    gdf_buffered = prepare_training_geometries(config, gdf)

    logger.info("AGC stage 3/4: extracting training/test samples (21-band, FCH-augmented).")
    X_train, y_train, X_test, y_test = extract_training_samples(config, gdf_buffered)

    logger.info("AGC stage 4/4: SelectKBest sweep + Random Forest tuning.")
    best_k, selector, _, sweep_metrics = mu.select_best_k(
        X_train, y_train, X_test, y_test, k_min=3, random_state=config["random_seed"]
    )
    selected_names = [ru.BAND_NAMES_21[i] for i in selector.get_support(indices=True)]

    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    best_model, best_params, tune_time = mu.tune_random_forest(
        X_train_sel, y_train, random_state=config["random_seed"]
    )
    y_pred = best_model.predict(X_test_sel)
    metrics = mu.compute_regression_metrics(y_test, y_pred)
    metrics.log(label="AGC Random Forest (tuned)", unit="Mg/ha")

    mu.save_model(best_model, Path(paths["agc_model"]), overwrite=config.get("overwrite_outputs", False))
    mu.save_model(selector, Path(paths["agc_selector"]), overwrite=config.get("overwrite_outputs", False))
    np.save(Path(paths["agc_selected_band_indices"]), selector.get_support(indices=True))

    return {
        "model": best_model, "selector": selector, "best_k": best_k,
        "metrics": metrics.as_dict(), "sweep_metrics": sweep_metrics,
        "best_params": best_params, "selected_feature_names": selected_names,
        "tune_time_s": tune_time,
    }


def predict(config: dict) -> Path:
    """Generate the wall-to-wall AGC map using the tuned model.

    Requires ``train()`` to have been run previously, and the FCH map
    (``paths.fch_output_map``) to exist.

    Returns
    -------
    Path to the output AGC GeoTIFF.
    """
    agc_cfg = config["agc"]
    paths = config["paths"]

    selected_idx = np.load(Path(paths["agc_selected_band_indices"]))
    output_path = Path(paths["agc_output_map"])

    ru.predict_wall_to_wall(
        model_path=Path(paths["agc_model"]),
        output_path=output_path,
        primary_raster_path=Path(paths["agc_predictor_stack_20band"]),
        auxiliary_raster_path=Path(paths["fch_output_map"]),
        patch_size=(agc_cfg["patch_height_px"], agc_cfg["patch_width_px"]),
        center_mode=agc_cfg["center_mode"],
        min_valid_ratio=agc_cfg.get("min_valid_ratio", 0.5),
        selected_band_indices=list(selected_idx),
        overwrite=config.get("overwrite_outputs", False),
    )
    return output_path
