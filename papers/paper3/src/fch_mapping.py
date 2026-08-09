# -*- coding: utf-8 -*-
"""
fch_mapping.py

Forest Canopy Height (FCH) mapping workflow: ICESat-2 ATL08 point cleaning,
rotated-footprint buffer construction, 20-band predictor stack assembly,
SelectKBest feature selection, Random Forest training + GridSearchCV
tuning, and wall-to-wall 10 m FCH prediction.

Default configuration (see config/config.example.yaml, section `fch`):
    - Footprint buffer: 100 m x 17 m, rotated per ATL08 ground-track ID.
    - Patch extraction window: 11 x 3 pixels, full-patch mean (center_mode=0).
    - Feature selection: SelectKBest(f_regression, k=16) of 20 bands.
    - Model: RandomForestRegressor, tuned via GridSearchCV.

All of the values above are configurable via `config.yaml` -- see
`config/config.example.yaml`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np

from . import data_preparation as dp
from . import model_utils as mu
from . import raster_utils as ru

logger = logging.getLogger(__name__)


def prepare_training_geometries(config: dict) -> gpd.GeoDataFrame:
    """Load ATL08 points, remove outliers, and build rotated footprint buffers.

    Parameters
    ----------
    config : dict
        Full loaded configuration (see config/config.example.yaml).

    Returns
    -------
    GeoDataFrame with an active ``rotated_buffer`` geometry and the target
    column ``h_canopy``.
    """
    fch_cfg = config["fch"]
    paths = config["paths"]

    gdf = gpd.read_file(Path(paths["icesat2_points"]))
    gdf_clean = dp.remove_outliers(gdf, target_column=fch_cfg["target_column"])

    gdf_buffered = dp.build_icesat2_buffers(
        gdf_clean,
        width=fch_cfg["footprint_width_m"],
        height=fch_cfg["footprint_height_m"],
        crs_epsg=config["crs_epsg"],
        descending_track_ids=tuple(fch_cfg["descending_track_ids"]),
        ascending_track_ids=tuple(fch_cfg["ascending_track_ids"]),
        offset_deg=fch_cfg["orientation_offset_deg"],
    )
    return gdf_buffered


def build_predictor_stack(config: dict, selected_indices: Optional[list[int]] = None):
    """Assemble the 20-band (or SelectKBest-reduced) FCH predictor stack.

    Parameters
    ----------
    config : dict
    selected_indices : list of int, optional
        Zero-based band indices to keep (e.g. from a fitted SelectKBest
        selector). If omitted, all 20 bands are returned.

    Returns
    -------
    (stack, profile)
    """
    paths = config["paths"]
    return ru.stack_predictor_bands(
        s2_path=Path(paths["sentinel2"]),
        s1_path=Path(paths["sentinel1"]),
        dem_path=Path(paths["dem"]),
        selected_indices=selected_indices,
    )


def extract_training_samples(config: dict, gdf_buffered: gpd.GeoDataFrame, stacked_raster_path: Path):
    """Split into train/test and extract 20-band patch features + labels."""
    fch_cfg = config["fch"]
    train_gdf, test_gdf = dp.train_test_split_gdf(
        gdf_buffered, test_size=fch_cfg["test_size"], random_state=config["random_seed"]
    )

    common_kwargs = dict(
        raster_path=stacked_raster_path,
        target_column=fch_cfg["target_column"],
        patch_width=fch_cfg["patch_width_px"],
        patch_height=fch_cfg["patch_height_px"],
        pixel_size=config["pixel_size_m"],
        selected_bands=list(range(1, 21)),
        center_mode=fch_cfg["center_mode"],
        keep_out_of_bounds=False,  # matches the original FCH extraction logic
    )
    X_train, y_train = ru.extract_patch_features(gdf_subset=train_gdf, **common_kwargs)
    X_test, y_test = ru.extract_patch_features(gdf_subset=test_gdf, **common_kwargs)

    X_train, y_train = mu.filter_zero_rows(X_train, y_train)
    X_test, y_test = mu.filter_zero_rows(X_test, y_test)
    return X_train, y_train, X_test, y_test


def train(config: dict) -> dict:
    """Run the full FCH training stage: data prep -> features -> RF tuning.

    Returns
    -------
    dict with keys: model, selector, metrics, selected_band_names,
    train_time_s, tune_time_s.
    """
    fch_cfg = config["fch"]
    paths = config["paths"]

    logger.info("FCH stage 1/4: preparing ICESat-2 buffers.")
    gdf_buffered = prepare_training_geometries(config)

    logger.info("FCH stage 2/4: building the 20-band predictor stack.")
    stack, profile = build_predictor_stack(config)
    stack_path = Path(paths["fch_predictor_stack_20band"])
    ru.write_raster(stack_path, stack, profile, overwrite=config.get("overwrite_outputs", False))

    logger.info("FCH stage 3/4: extracting training/test samples.")
    X_train, y_train, X_test, y_test = extract_training_samples(config, gdf_buffered, stack_path)

    logger.info("FCH stage 4/4: feature selection + Random Forest training.")
    X_train_sel, X_test_sel, selector = mu.select_k_best_features(
        X_train, y_train, X_test, k=fch_cfg["select_k_best"]
    )
    selected_idx = selector.get_support(indices=True)
    selected_names = [ru.BAND_NAMES_20[i] for i in selected_idx]

    _, baseline_time = mu.train_random_forest(X_train_sel, y_train, random_state=config["random_seed"])
    best_model, best_params, tune_time = mu.tune_random_forest(
        X_train_sel, y_train, random_state=config["random_seed"]
    )

    y_pred = best_model.predict(X_test_sel)
    metrics = mu.compute_regression_metrics(y_test, y_pred)
    metrics.log(label="FCH Random Forest (tuned)", unit="m")

    mu.save_model(
        best_model, Path(paths["fch_model"]), overwrite=config.get("overwrite_outputs", False)
    )
    mu.save_model(
        selector, Path(paths["fch_selector"]), overwrite=config.get("overwrite_outputs", False)
    )
    np.save(Path(paths["fch_selected_band_indices"]), selected_idx)

    return {
        "model": best_model, "selector": selector, "metrics": metrics.as_dict(),
        "best_params": best_params, "selected_band_names": selected_names,
        "baseline_train_time_s": baseline_time, "tune_time_s": tune_time,
    }


def predict(config: dict) -> Path:
    """Generate the wall-to-wall 10 m FCH map using the tuned model.

    Requires ``train()`` to have been run previously (the model, selector,
    and predictor stack must exist on disk).

    Returns
    -------
    Path to the output FCH GeoTIFF.
    """
    fch_cfg = config["fch"]
    paths = config["paths"]

    selected_idx = np.load(Path(paths["fch_selected_band_indices"]))
    stack, profile = build_predictor_stack(config, selected_indices=list(selected_idx))
    stack_path = Path(paths["fch_prediction_stack"])
    ru.write_raster(stack_path, stack, profile, overwrite=config.get("overwrite_outputs", False))

    output_path = Path(paths["fch_output_map"])
    ru.predict_wall_to_wall(
        model_path=Path(paths["fch_model"]),
        output_path=output_path,
        primary_raster_path=stack_path,
        auxiliary_raster_path=None,
        patch_size=(fch_cfg["patch_height_px"], fch_cfg["patch_width_px"]),
        center_mode=fch_cfg["center_mode"],
        min_valid_ratio=fch_cfg.get("min_valid_ratio", 0.5),
        selected_band_indices=None,  # stack already reduced to selected bands
        overwrite=config.get("overwrite_outputs", False),
    )
    return output_path
