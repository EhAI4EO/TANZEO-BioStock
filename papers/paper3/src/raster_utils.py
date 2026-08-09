# -*- coding: utf-8 -*-
"""
raster_utils.py

Raster I/O, band normalization, spectral-index computation, multi-sensor
band stacking, patch-based feature extraction from vector footprints, and
block-wise wall-to-wall prediction.

Scientific behaviour preserved from the original research scripts:
    - Per-band min-max normalization to [0, 1] (``normalize_all_bands``).
    - Spectral indices: NDVI, NDMI, NDPI, EVI, NBR, NBR2, NIRv, computed
      with zero-safe division.
    - Patch feature extraction supports a ``center_mode`` parameter that
      selects either the full-patch mean (0), a single center pixel (1),
      or the mean of an odd k x k window centered on the patch
      (any odd perfect square, e.g. 9, 25 for 3x3 / 5x5).
    - Wall-to-wall prediction is block-wise (row-by-row) rather than
      whole-array in memory, matching the original design intent, and
      skips patches whose valid-pixel fraction falls below
      ``min_valid_ratio``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Normalization & spectral indices
# --------------------------------------------------------------------------- #
def normalize_all_bands(image: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Array of shape (H, W) or (H, W, B).

    Returns
    -------
    np.ndarray (float32), same shape as ``image``.
    """
    image = image.astype(np.float32)
    min_vals = image.min(axis=(0, 1), keepdims=True) if image.ndim == 3 else image.min()
    max_vals = image.max(axis=(0, 1), keepdims=True) if image.ndim == 3 else image.max()
    range_vals = max_vals - min_vals
    if np.isscalar(range_vals):
        range_vals = 1e-6 if range_vals == 0 else range_vals
    else:
        range_vals = np.where(range_vals == 0, 1e-6, range_vals)
    return (image - min_vals) / range_vals


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.where(denominator == 0, 0, numerator / denominator)


def compute_spectral_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute NDVI, NDMI, NDPI, EVI, NBR, NBR2, NIRv from normalized bands.

    Parameters
    ----------
    bands : dict
        Must contain normalized (0-1) Sentinel-2 bands under the keys
        ``"B2"``, ``"B4"``, ``"B8"``, ``"B11"``, ``"B12"``.

    Returns
    -------
    dict of index name -> np.ndarray, in a fixed, documented order
    matching the original 20-band stack layout:
    NDVI, NDMI, NDPI, EVI, NBR, NBR2, NIRv.
    """
    b2, b4, b8, b11, b12 = bands["B2"], bands["B4"], bands["B8"], bands["B11"], bands["B12"]

    ndvi = _safe_divide(b8 - b4, b8 + b4)
    ndmi = _safe_divide(b8 - b11, b8 + b11)
    ndpi = _safe_divide(b8 - (0.74 * b4 + 0.26 * b11), b8 + (0.74 * b4 + 0.26 * b11))
    evi = 2.5 * _safe_divide(b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1)
    nbr = _safe_divide(b8 - b12, b8 + b12)
    nbr2 = _safe_divide(b11 - b12, b11 + b12)
    nirv = b8 * ndvi

    return {
        "NDVI": ndvi, "NDMI": ndmi, "NDPI": ndpi, "EVI": evi,
        "NBR": nbr, "NBR2": nbr2, "NIRv": nirv,
    }


# Fixed, documented band order for the 20-band FCH predictor stack.
# Indices 0-9: Sentinel-2 (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
# Indices 10-16: spectral indices (NDVI,NDMI,NDPI,EVI,NBR,NBR2,NIRv)
# Indices 17-18: Sentinel-1 (VV, VH)
# Index 19: DEM
BAND_NAMES_20 = [
    "B2 (Blue)", "B3 (Green)", "B4 (Red)", "B5 (Red Edge 1)", "B6 (Red Edge 2)",
    "B7 (Red Edge 3)", "B8 (NIR1)", "B8A (NIR2)", "B11 (SWIR1)", "B12 (SWIR2)",
    "NDVI", "NDMI", "NDPI", "EVI", "NBR", "NBR2", "NIRv",
    "S1 VV", "S1 VH", "DEM",
]
# AGB / AGC use the 20-band stack plus FCH as a 21st predictor.
BAND_NAMES_21 = BAND_NAMES_20 + ["FCH"]


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a raster into a (H, W, B) array plus its rasterio profile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
        profile = src.profile
    return np.moveaxis(arr, 0, -1), profile


def write_raster(path: Path, array: np.ndarray, profile: dict, overwrite: bool = False) -> None:
    """Write a (H, W) or (H, W, B) array to disk with the given profile.

    Raises
    ------
    FileExistsError
        If ``path`` already exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists and overwrite=False: {path}. "
            f"Set overwrite=True (or --overwrite) to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    bands = array.shape[-1]

    out_profile = profile.copy()
    out_profile.update(count=bands)

    with rasterio.open(path, "w", **out_profile) as dst:
        for i in range(bands):
            dst.write(array[:, :, i], i + 1)
    logger.info("Wrote raster with %d band(s) to %s", bands, path)


def stack_predictor_bands(
    s2_path: Path,
    s1_path: Path,
    dem_path: Path,
    selected_indices: Optional[Sequence[int]] = None,
    brightness_factor: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Build the 20-band FCH predictor stack from S2, S1, and DEM rasters.

    Parameters
    ----------
    s2_path : Path
        10-band Sentinel-2 reflectance composite (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12).
    s1_path : Path
        2-band Sentinel-1 composite (VV, VH).
    dem_path : Path
        1-band DEM raster.
    selected_indices : sequence of int, optional
        If given, subsets the 20-band stack to these zero-based band
        indices (e.g. the output of a fitted ``SelectKBest`` selector).
    brightness_factor : float
        Unused in the numerical stack (kept only for parity with the
        original RGB-quicklook code, which is in ``visualization.py``).

    Returns
    -------
    (stack, profile) where ``stack`` has shape (H, W, len(selected_indices)
    or 20) and ``profile`` is a rasterio profile derived from ``s2_path``
    with ``count`` set accordingly and ``dtype='float32'``.
    """
    s2, _ = read_raster(s2_path)
    s1, _ = read_raster(s1_path)
    dem, _ = read_raster(dem_path)

    s2_norm = normalize_all_bands(s2)
    s1 = np.nan_to_num(s1, nan=0.0)
    s1_norm = normalize_all_bands(s1)
    dem_norm = normalize_all_bands(np.squeeze(dem))

    band_map = {
        "B2": s2_norm[:, :, 0], "B4": s2_norm[:, :, 2],
        "B8": s2_norm[:, :, 6], "B11": s2_norm[:, :, 8], "B12": s2_norm[:, :, 9],
    }
    indices = compute_spectral_indices(band_map)
    si_stack = np.stack(
        [indices["NDVI"], indices["NDMI"], indices["NDPI"], indices["EVI"],
         indices["NBR"], indices["NBR2"], indices["NIRv"]], axis=-1
    )
    si_norm = normalize_all_bands(si_stack)

    h, w = s2_norm.shape[:2]
    full_stack = np.zeros((h, w, 20), dtype=np.float32)
    full_stack[:, :, 0:10] = s2_norm
    full_stack[:, :, 10:17] = si_norm
    full_stack[:, :, 17:19] = s1_norm[:, :, :2]
    full_stack[:, :, 19] = dem_norm

    if selected_indices is not None:
        full_stack = full_stack[:, :, list(selected_indices)]

    with rasterio.open(s2_path) as src:
        profile = src.profile.copy()
    profile.update(count=full_stack.shape[-1], dtype="float32")

    return full_stack, profile


# --------------------------------------------------------------------------- #
# Patch-based feature extraction (training samples)
# --------------------------------------------------------------------------- #
def _reduce_patch(selected_data: np.ndarray, center_mode: int) -> np.ndarray:
    """Reduce a (bands, H, W) patch to a (bands,) feature vector.

    Parameters
    ----------
    selected_data : np.ndarray
        Shape (bands, H, W).
    center_mode : int
        0 = mean over the full patch; 1 = center pixel only;
        any odd perfect square k*k (9, 25, ...) = mean over the centered
        k x k window.
    """
    h, w = selected_data.shape[1:]
    center_y, center_x = h // 2, w // 2

    if center_mode == 0:
        return selected_data.mean(axis=(1, 2))
    if center_mode == 1:
        return selected_data[:, center_y, center_x]

    k = int(round(center_mode ** 0.5))
    if k * k != center_mode or k % 2 == 0:
        raise ValueError(
            f"Unsupported center_mode={center_mode}. Use 0 (full-patch mean), "
            f"1 (center pixel), or an odd perfect square (9, 25, ...)."
        )
    offset = k // 2
    window = selected_data[
        :, center_y - offset: center_y + offset + 1, center_x - offset: center_x + offset + 1
    ]
    return window.mean(axis=(1, 2))


def extract_patch_features(
    gdf_subset: gpd.GeoDataFrame,
    raster_path: Path,
    target_column: str,
    patch_width: int,
    patch_height: int,
    pixel_size: float = 10.0,
    selected_bands: Optional[Sequence[int]] = None,
    center_mode: int = 0,
    keep_out_of_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract predictor features for each footprint geometry in ``gdf_subset``.

    For each geometry, a bounding box of size (``patch_width`` x
    ``patch_height`` pixels) centered on the geometry centroid is clipped
    from ``raster_path``, and reduced to a feature vector via
    ``center_mode`` (see :func:`_reduce_patch`).

    This consolidates the several near-duplicate
    ``extract_mean_features_with_partial_zeros`` variants found across the
    original FCH / AGB / AGC scripts into one parameterized implementation.

    Parameters
    ----------
    gdf_subset : GeoDataFrame
        Footprint geometries (rotated or axis-aligned buffers) with a
        ``target_column`` field.
    raster_path : Path
        Predictor raster (band-stacked GeoTIFF).
    target_column : str
        Name of the target field (``"h_canopy"``, ``"AGB"``, or ``"AGC"``).
    patch_width, patch_height : int
        Bounding-box size in pixels used to clip around each geometry
        centroid before applying ``center_mode`` reduction.
    pixel_size : float
        Raster pixel size in meters (used to convert pixel counts to map
        units for the bounding box).
    selected_bands : sequence of int, optional
        1-based band indices to extract (default: all bands).
    center_mode : int
        See :func:`_reduce_patch`.
    keep_out_of_bounds : bool
        If True (default, matching the final/active AGB & AGC extraction
        logic), points whose patch extends beyond the raster bounds are
        still processed via a clipped ``rasterio.mask``. If False
        (matching the original FCH extraction logic), such points are
        skipped entirely.

    Returns
    -------
    (features, labels) : (np.ndarray of shape (n, n_bands), np.ndarray of shape (n,))
    """
    features, labels = [], []
    skipped_oob = 0
    skipped_empty = 0
    skipped_error = 0

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        nodata_value = src.nodata
        total_bands = src.count

        if selected_bands is None:
            selected_bands = list(range(1, total_bands + 1))

        if gdf_subset.crs != raster_crs:
            gdf_subset = gdf_subset.to_crs(raster_crs)

        half_w_m = (patch_width // 2) * pixel_size
        half_h_m = (patch_height // 2) * pixel_size
        geom_col = gdf_subset.geometry.name

        for idx, point in gdf_subset.iterrows():
            x_c, y_c = point[geom_col].centroid.x, point[geom_col].centroid.y

            out_of_bounds = (
                x_c - half_w_m < raster_bounds.left or x_c + half_w_m > raster_bounds.right
                or y_c - half_h_m < raster_bounds.bottom or y_c + half_h_m > raster_bounds.top
            )
            if out_of_bounds and not keep_out_of_bounds:
                skipped_oob += 1
                continue

            bbox = box(x_c - half_w_m, y_c - half_h_m, x_c + half_w_m, y_c + half_h_m)
            bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=gdf_subset.crs)

            try:
                out_image, _ = rio_mask(src, bbox_gdf.geometry, crop=True, all_touched=True)
                if nodata_value is not None:
                    out_image = np.ma.masked_equal(out_image, nodata_value)
                if isinstance(out_image, np.ma.MaskedArray) and np.all(out_image.mask):
                    skipped_empty += 1
                    continue

                selected_data = out_image[[b - 1 for b in selected_bands], :, :]
                band_vals = _reduce_patch(np.asarray(selected_data), center_mode=center_mode)

                features.append(band_vals)
                labels.append(point[target_column])
            except Exception as exc:  # noqa: BLE001 - log & continue, matches original robustness
                logger.warning("Error extracting features at index %s: %s", idx, exc)
                skipped_error += 1
                continue

    logger.info(
        "Extracted %d feature vectors from %d input geometries "
        "(skipped: %d out-of-bounds, %d empty, %d errors).",
        len(features), len(gdf_subset), skipped_oob, skipped_empty, skipped_error,
    )
    return np.array(features), np.array(labels)


def extract_patch_features_with_auxiliary_band(
    gdf_subset: gpd.GeoDataFrame,
    stack_path: Path,
    auxiliary_path: Path,
    target_column: str,
    patch_width: int,
    patch_height: int,
    pixel_size: float = 10.0,
    selected_bands: Optional[Sequence[int]] = None,
    center_mode: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`extract_patch_features`, but appends a normalized
    auxiliary raster band (FCH) as the final predictor.

    This reproduces the AGB / AGC "stacked input" extraction logic, where
    the previously mapped FCH raster is min-max normalized (by its
    valid-data maximum) and concatenated as the 21st predictor band before
    patch reduction.

    Parameters
    ----------
    gdf_subset : GeoDataFrame
    stack_path : Path
        20-band predictor stack (same layout as FCH mapping).
    auxiliary_path : Path
        Single-band FCH raster (must share CRS, transform, and dimensions
        with ``stack_path``).
    target_column : str
        ``"AGB"`` or ``"AGC"``.
    patch_width, patch_height : int
    pixel_size : float
    selected_bands : sequence of int, optional
        1-based indices into the 21-band combined stack.
    center_mode : int

    Returns
    -------
    (features, labels)

    Raises
    ------
    ValueError
        If ``stack_path`` and ``auxiliary_path`` are not spatially aligned.
    """
    features, labels = [], []
    skipped = {"invalid_canopy": 0, "empty": 0, "error": 0}

    with rasterio.open(auxiliary_path) as canopy_src:
        canopy_data = canopy_src.read(1)
        if canopy_src.nodata is not None:
            canopy_data = np.ma.masked_equal(canopy_data, canopy_src.nodata)
        canopy_max = canopy_data.max()
        if canopy_max <= 0 or np.isnan(canopy_max):
            raise ValueError(f"Invalid maximum value in auxiliary raster: {auxiliary_path}")

    with rasterio.open(stack_path) as src_stack, rasterio.open(auxiliary_path) as src_aux:
        if src_stack.crs != src_aux.crs:
            raise ValueError("CRS mismatch between predictor stack and auxiliary raster.")
        if src_stack.transform != src_aux.transform:
            raise ValueError("Transform mismatch between predictor stack and auxiliary raster.")
        if (src_stack.width, src_stack.height) != (src_aux.width, src_aux.height):
            raise ValueError("Dimension mismatch between predictor stack and auxiliary raster.")

        raster_crs = src_stack.crs
        total_bands = src_stack.count + 1
        if selected_bands is None:
            selected_bands = list(range(1, total_bands + 1))

        if gdf_subset.crs != raster_crs:
            gdf_subset = gdf_subset.to_crs(raster_crs)

        half_w_m = (patch_width // 2) * pixel_size
        half_h_m = (patch_height // 2) * pixel_size
        geom_col = gdf_subset.geometry.name

        for idx, point in gdf_subset.iterrows():
            x_c, y_c = point[geom_col].centroid.x, point[geom_col].centroid.y
            bbox = box(x_c - half_w_m, y_c - half_h_m, x_c + half_w_m, y_c + half_h_m)
            bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs=gdf_subset.crs)

            try:
                img_stack, _ = rio_mask(src_stack, bbox_gdf.geometry, crop=True, all_touched=True)
                img_aux, _ = rio_mask(src_aux, bbox_gdf.geometry, crop=True, all_touched=True)

                if src_stack.nodata is not None:
                    img_stack = np.ma.masked_equal(img_stack, src_stack.nodata)
                if src_aux.nodata is not None:
                    img_aux = np.ma.masked_equal(img_aux, src_aux.nodata)

                img_aux = img_aux / canopy_max
                if img_aux.ndim == 2:
                    img_aux = img_aux[np.newaxis, :, :]

                combined = np.ma.masked_invalid(np.vstack([img_stack, img_aux]))
                combined = np.ma.masked_greater(np.abs(combined), 1e4)
                if isinstance(combined, np.ma.MaskedArray) and np.all(combined.mask):
                    skipped["empty"] += 1
                    continue

                selected_data = combined[[b - 1 for b in selected_bands], :, :]
                band_vals = _reduce_patch(np.asarray(selected_data), center_mode=center_mode)

                # Sanity checks on the auxiliary (FCH) channel, which is the
                # last selected band whenever selected_bands includes band 21.
                if selected_bands[-1] == total_bands:
                    aux_val = band_vals[-1]
                    if not np.isfinite(aux_val) or abs(aux_val) > 1e3:
                        skipped["invalid_canopy"] += 1
                        continue
                    if np.isclose(aux_val, 1.0, atol=1e-4):
                        skipped["invalid_canopy"] += 1
                        continue
                    band_vals[-1] = np.clip(aux_val, 0, 1)

                features.append(band_vals)
                labels.append(point[target_column])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error extracting features (with auxiliary band) at index %s: %s", idx, exc)
                skipped["error"] += 1
                continue

    logger.info(
        "Extracted %d feature vectors (with auxiliary band) from %d geometries "
        "(skipped: %s).", len(features), len(gdf_subset), skipped,
    )
    return np.array(features), np.array(labels)


# --------------------------------------------------------------------------- #
# Wall-to-wall (block-wise) prediction
# --------------------------------------------------------------------------- #
def predict_wall_to_wall(
    model_path: Path,
    output_path: Path,
    primary_raster_path: Path,
    auxiliary_raster_path: Optional[Path] = None,
    patch_size: tuple[int, int] = (11, 3),
    center_mode: int = 0,
    min_valid_ratio: float = 0.5,
    selected_band_indices: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> None:
    """Apply a trained regressor to a full raster, block-wise by row.

    Consolidates the several ``predict_large_image_with_rf`` variants
    found in the original scripts (plain patch-mean, NaN-aware masking,
    dual-image stacking, and center-window restriction) into a single
    production implementation covering all of their capabilities:
    NaN-aware masking, optional auxiliary-raster stacking, and a
    ``center_mode``-restricted patch reduction with a minimum valid-pixel
    fraction per patch.

    Parameters
    ----------
    model_path : Path
        Path to a joblib-serialized regressor.
    output_path : Path
        Output single-band GeoTIFF path.
    primary_raster_path : Path
        Predictor stack raster (e.g. the 20-band FCH stack, or the 20-band
        AGB/AGC stack when ``auxiliary_raster_path`` supplies the FCH band).
    auxiliary_raster_path : Path, optional
        Single-band raster (e.g. FCH) to concatenate as the final
        predictor band. Must share CRS, transform, and dimensions with
        ``primary_raster_path``.
    patch_size : (int, int)
        (height, width) of the window used for edge padding; must be >=
        the window implied by ``center_mode``.
    center_mode : int
        0 = mean over the full ``patch_size`` window; 1 = center pixel;
        odd perfect square k*k = mean over a centered k x k window.
    min_valid_ratio : float
        Minimum fraction of valid (non-NaN) pixels within the reduction
        window required to produce a prediction for a given pixel.
    selected_band_indices : sequence of int, optional
        Zero-based indices into the (stacked) feature vector to pass to
        the model, e.g. the output of a fitted ``SelectKBest`` selector.
        Required if the model was trained on a feature-selected subset.
    overwrite : bool
        Whether to overwrite an existing output file.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite=False: {output_path}")

    model = joblib.load(model_path)

    with rasterio.open(primary_raster_path) as src_primary:
        profile = src_primary.profile.copy()
        h, w = src_primary.height, src_primary.width
        img = src_primary.read().astype(np.float32)  # (C, H, W)
        valid_mask = np.array([src_primary.read_masks(i + 1).astype(bool) for i in range(src_primary.count)])
        transform = src_primary.transform
        crs = src_primary.crs

    if auxiliary_raster_path is not None:
        with rasterio.open(auxiliary_raster_path) as src_aux:
            if src_aux.crs != crs:
                raise ValueError("CRS mismatch between primary and auxiliary prediction rasters.")
            if src_aux.transform != transform:
                raise ValueError("Transform mismatch between primary and auxiliary prediction rasters.")
            if (src_aux.width, src_aux.height) != (w, h):
                raise ValueError("Dimension mismatch between primary and auxiliary prediction rasters.")
            aux = src_aux.read().astype(np.float32)  # (1, H, W)
            aux_mask = np.array([src_aux.read_masks(1).astype(bool)])
        combined_invalid = ~(np.all(valid_mask, axis=0) & np.all(aux_mask, axis=0))
        stacked = np.concatenate([img, aux], axis=0)
    else:
        combined_invalid = ~np.all(valid_mask, axis=0)
        stacked = img

    stacked = np.moveaxis(stacked, 0, -1)  # (H, W, C)
    stacked[combined_invalid] = np.nan

    pad_h, pad_w = patch_size[0] // 2, patch_size[1] // 2
    padded = np.pad(stacked, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant", constant_values=np.nan)

    if center_mode in (0, 1):
        c_half_h, c_half_w = pad_h, pad_w
    else:
        k = int(round(center_mode ** 0.5))
        if k * k != center_mode or k % 2 == 0:
            raise ValueError(f"Unsupported center_mode={center_mode}.")
        c_half_h = c_half_w = k // 2

    prediction_map = np.full((h, w), np.nan, dtype=np.float32)

    for row in tqdm(range(h), desc=f"Predicting rows -> {output_path.name}"):
        row_features, valid_cols = [], []
        for col in range(w):
            patch = padded[row: row + patch_size[0], col: col + patch_size[1], :]
            cy, cx = patch_size[0] // 2, patch_size[1] // 2

            if center_mode in (0,):
                window = patch
            elif center_mode == 1:
                window = patch[cy: cy + 1, cx: cx + 1, :]
            else:
                window = patch[cy - c_half_h: cy + c_half_h + 1, cx - c_half_w: cx + c_half_w + 1, :]

            valid_frac = np.sum(~np.isnan(window[:, :, 0])) / (window.shape[0] * window.shape[1])
            if valid_frac < min_valid_ratio:
                continue

            mean_patch = np.nanmean(window, axis=(0, 1))
            if selected_band_indices is not None:
                mean_patch = mean_patch[list(selected_band_indices)]

            row_features.append(mean_patch)
            valid_cols.append(col)

        if row_features:
            preds = model.predict(np.array(row_features))
            prediction_map[row, valid_cols] = preds.astype(np.float32)

    profile.update(height=h, width=w, transform=transform, dtype=rasterio.float32, count=1, nodata=np.nan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction_map, 1)

    logger.info("Wall-to-wall prediction written to %s (center_mode=%d).", output_path, center_mode)
