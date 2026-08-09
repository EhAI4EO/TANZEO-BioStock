# -*- coding: utf-8 -*-
"""
agb_mapping.py

Above-Ground Biomass (AGB) mapping workflow: field-plot cleaning,
axis-aligned buffer construction, patch feature extraction (with FCH as an
auxiliary 21st predictor band), SelectKBest-swept feature selection,
Random Forest training + GridSearchCV tuning, and wall-to-wall AGB
prediction.

**Depends on FCH:** this stage requires the FCH raster produced by
``fch_mapping.predict()`` (``paths.fch_output_map`` in config.yaml) as an
auxiliary predictor band. Run the FCH stage first.

Default configuration (see config/config.example.yaml, section `agb`):
    - Field-plot buffer: 225 m x 900 m, axis-aligned.
    - Patch extraction bounding box: 22 x 90 pixels, reduced via a
      centered 5x5-pixel window mean (center_mode=25).
    - Predictors: the 20-band FCH stack + the FCH raster itself (21 total).
    - Feature selection: SelectKBest sweep over k, keeping the best test R^2.
    - Model: RandomForestRegressor, tuned via GridSearchCV on the
      SelectKBest-reduced feature set.

All of the values above are configurable via `config.yaml` -- see
`config/config.example.yaml`. Alternative plot geometries are also
available there as optional settings.
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


def prepare_training_geometries(config: dict) -> gpd.GeoDataFrame:
    """Load AGB field points, remove outliers, and build field-plot buffers."""
    agb_cfg = config["agb"]
    paths = config["paths"]

    gdf = gpd.read_file(Path(paths["agb_field_points"]))
    gdf_clean = dp.remove_outliers(gdf, target_column=agb_cfg["target_column"])

    gdf_buffered = dp.build_field_plot_buffers(
        gdf_clean,
        width=agb_cfg["plot_width_m"],
        height=agb_cfg["plot_height_m"],
        crs_epsg=config["crs_epsg"],
    )
    return gdf_buffered


def extract_training_samples(config: dict, gdf_buffered: gpd.GeoDataFrame):
    """Split into train/test and extract 21-band (20-band stack + FCH) features."""
    agb_cfg = config["agb"]
    paths = config["paths"]

    train_gdf, test_gdf = dp.train_test_split_gdf(
        gdf_buffered, test_size=agb_cfg["test_size"], random_state=config["random_seed"]
    )

    common_kwargs = dict(
        stack_path=Path(paths["agb_predictor_stack_20band"]),
        auxiliary_path=Path(paths["fch_output_map"]),
        target_column=agb_cfg["target_column"],
        patch_width=agb_cfg["patch_width_px"],
        patch_height=agb_cfg["patch_height_px"],
        pixel_size=config["pixel_size_m"],
        selected_bands=list(range(1, 22)),
        center_mode=agb_cfg["center_mode"],
    )
    X_train, y_train = ru.extract_patch_features_with_auxiliary_band(gdf_subset=train_gdf, **common_kwargs)
    X_test, y_test = ru.extract_patch_features_with_auxiliary_band(gdf_subset=test_gdf, **common_kwargs)
    return X_train, y_train, X_test, y_test


def train(config: dict) -> dict:
    """Run the full AGB training stage.

    Requires ``paths.fch_output_map`` and ``paths.agb_predictor_stack_20band``
    to already exist on disk (build the 20-band stack the same way as the
    FCH stage's ``fch_mapping.build_predictor_stack``, using AGB-area
    Sentinel-1/2/DEM inputs from config).

    Returns
    -------
    dict with keys: model, selector, best_k, metrics, selected_feature_names,
    tune_time_s.
    """
    agb_cfg = config["agb"]
    paths = config["paths"]

    logger.info("AGB stage 1/3: preparing field-plot buffers.")
    gdf_buffered = prepare_training_geometries(config)

    logger.info("AGB stage 2/3: extracting training/test samples (21-band, FCH-augmented).")
    X_train, y_train, X_test, y_test = extract_training_samples(config, gdf_buffered)

    logger.info("AGB stage 3/3: SelectKBest sweep + Random Forest tuning.")
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
    metrics.log(label="AGB Random Forest (tuned)", unit="Mg/ha")

    mu.save_model(best_model, Path(paths["agb_model"]), overwrite=config.get("overwrite_outputs", False))
    mu.save_model(selector, Path(paths["agb_selector"]), overwrite=config.get("overwrite_outputs", False))
    np.save(Path(paths["agb_selected_band_indices"]), selector.get_support(indices=True))

    return {
        "model": best_model, "selector": selector, "best_k": best_k,
        "metrics": metrics.as_dict(), "sweep_metrics": sweep_metrics,
        "best_params": best_params, "selected_feature_names": selected_names,
        "tune_time_s": tune_time,
    }


def predict(config: dict) -> Path:
    """Generate the wall-to-wall AGB map using the tuned model.

    Requires ``train()`` to have been run previously, and the FCH map
    (``paths.fch_output_map``) to exist.

    Returns
    -------
    Path to the output AGB GeoTIFF.
    """
    agb_cfg = config["agb"]
    paths = config["paths"]

    selected_idx = np.load(Path(paths["agb_selected_band_indices"]))
    output_path = Path(paths["agb_output_map"])

    ru.predict_wall_to_wall(
        model_path=Path(paths["agb_model"]),
        output_path=output_path,
        primary_raster_path=Path(paths["agb_predictor_stack_20band"]),
        auxiliary_raster_path=Path(paths["fch_output_map"]),
        patch_size=(agb_cfg["patch_height_px"], agb_cfg["patch_width_px"]),
        center_mode=agb_cfg["center_mode"],
        min_valid_ratio=agb_cfg.get("min_valid_ratio", 0.5),
        selected_band_indices=list(selected_idx),
        overwrite=config.get("overwrite_outputs", False),
    )
    return output_path
