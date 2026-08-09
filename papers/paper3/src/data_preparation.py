# -*- coding: utf-8 -*-
"""
data_preparation.py

Loading and cleaning of field / reference datasets (ICESat-2 ATL08 canopy
height points, AGB field plots, AGC field plots) and construction of the
sampling-footprint buffer geometries used for raster feature extraction.

Scientific behaviour preserved from the original research scripts:
    - IQR-based outlier removal (1.5 * IQR fences) on the target variable.
    - ICESat-2 (FCH) footprints are approximated with *rotated* rectangular
      buffers, oriented according to the ATL08 ground-track ID, to reflect
      the true along-track footprint geometry of each beam.
    - AGB / AGC field plots use *axis-aligned* rectangular buffers.

Buffer dimensions for AGB / AGC (default: 225 m x 900 m) and for ICESat-2
footprints (default: 100 m x 17 m) are configurable via `config.yaml` --
see `config/config.example.yaml` for the current defaults and adjustable
parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Outlier filtering
# --------------------------------------------------------------------------- #
def iqr_bounds(values: pd.Series, whisker: float = 1.5) -> tuple[float, float]:
    """Compute lower/upper IQR fences for outlier detection.

    Parameters
    ----------
    values : pd.Series
        Numeric values (NaNs are ignored).
    whisker : float, default 1.5
        Standard Tukey whisker multiplier, matching the original scripts.

    Returns
    -------
    (lower_bound, upper_bound)
    """
    clean = values.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return q1 - whisker * iqr, q3 + whisker * iqr


def remove_outliers(
    gdf: gpd.GeoDataFrame, target_column: str, whisker: float = 1.5
) -> gpd.GeoDataFrame:
    """Filter out rows whose ``target_column`` falls outside the IQR fences.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input point layer.
    target_column : str
        Name of the numeric target field (e.g. ``"h_canopy"``, ``"AGB"``,
        ``"AGC"``).
    whisker : float, default 1.5
        IQR whisker multiplier.

    Returns
    -------
    GeoDataFrame
        Filtered copy of ``gdf``.

    Raises
    ------
    KeyError
        If ``target_column`` is not present in ``gdf``.
    """
    if target_column not in gdf.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in input layer. "
            f"Available columns: {list(gdf.columns)}"
        )

    values = gdf[target_column]
    lower, upper = iqr_bounds(values, whisker=whisker)
    n_before = len(gdf)
    mask = (values >= lower) & (values <= upper)
    cleaned = gdf[mask].copy()

    logger.info(
        "Outlier filtering on '%s': %d -> %d points kept "
        "(bounds=[%.3f, %.3f], whisker=%.2f)",
        target_column, n_before, len(cleaned), lower, upper, whisker,
    )
    return cleaned


# --------------------------------------------------------------------------- #
# Buffer geometry construction
# --------------------------------------------------------------------------- #
def create_rotated_buffer(point: Point, width: float, height: float, angle_deg: float) -> Polygon:
    """Create a rectangular buffer around ``point``, rotated by ``angle_deg``.

    Used to approximate the along-track footprint of an ICESat-2 ATL08
    segment, whose orientation depends on the satellite ground track.

    Parameters
    ----------
    point : shapely.geometry.Point
        Center of the buffer, in a projected (metric) CRS.
    width, height : float
        Buffer dimensions in map units (meters), before rotation.
    angle_deg : float
        Rotation angle in degrees, applied about the buffer center.

    Returns
    -------
    shapely.geometry.Polygon
    """
    half_w, half_h = width / 2, height / 2
    rectangle = Polygon(
        [(-half_w, -half_h), (-half_w, half_h), (half_w, half_h), (half_w, -half_h)]
    )
    rotated = rotate(rectangle, angle_deg, origin=(0, 0), use_radians=False)
    return translate(rotated, xoff=point.x, yoff=point.y)


def create_aligned_rectangle(point: Point, width: float, height: float) -> Polygon:
    """Create an axis-aligned rectangular buffer around ``point``.

    Used for AGB / AGC field-plot footprints.

    Parameters
    ----------
    point : shapely.geometry.Point
    width, height : float
        Buffer dimensions in map units (meters).

    Returns
    -------
    shapely.geometry.Polygon
    """
    half_w, half_h = width / 2, height / 2
    return Polygon(
        [
            (point.x - half_w, point.y - half_h),
            (point.x - half_w, point.y + half_h),
            (point.x + half_w, point.y + half_h),
            (point.x + half_w, point.y - half_h),
        ]
    )


def icesat2_track_angle(track_id: Optional[int], default_angle: float = 90.0, offset_deg: float = 5.5) -> float:
    """Return the ATL08 footprint rotation angle for a given ground-track ID.

    Reproduces the original mapping:
        - track_id in {418, 860} -> descending orbit -> (default - offset)
        - track_id == 730         -> ascending orbit  -> (default + offset)
        - anything else           -> default orientation

    These specific track IDs are dataset-specific (derived from the West
    Usambara ATL08 acquisitions used in this study) and are exposed via
    ``config.yaml`` rather than hard-coded, so they can be adapted to a
    different study area / acquisition set.
    """
    if track_id in (418, 860):
        return default_angle - offset_deg
    if track_id == 730:
        return default_angle + offset_deg
    return default_angle


def build_icesat2_buffers(
    gdf: gpd.GeoDataFrame,
    width: float,
    height: float,
    crs_epsg: int,
    descending_track_ids: tuple[int, ...] = (418, 860),
    ascending_track_ids: tuple[int, ...] = (730,),
    default_angle: float = 90.0,
    offset_deg: float = 5.5,
    track_id_column: str = "track_id",
) -> gpd.GeoDataFrame:
    """Replace point geometries with rotated ATL08 footprint buffers.

    The original point geometry is preserved as WKT in a ``point_wkt``
    column, and the active geometry column becomes ``rotated_buffer``.

    Parameters
    ----------
    gdf : GeoDataFrame
        Cleaned ICESat-2 ATL08 points (must contain a point geometry and,
        optionally, ``track_id_column``).
    width, height : float
        Footprint dimensions in meters (default 100 x 17, matching the
        approximate ATL08 along-track segment footprint used in this study).
    crs_epsg : int
        Target projected EPSG code (e.g. 21037 for Arc 1960 / UTM 37S).
    descending_track_ids, ascending_track_ids : tuple of int
        Ground-track IDs assigned a negative / positive angular offset.
    default_angle, offset_deg : float
        Base orientation and offset (degrees) applied per track.
    track_id_column : str
        Name of the column holding the ATL08 ground-track ID.

    Returns
    -------
    GeoDataFrame with an active ``rotated_buffer`` geometry column.
    """
    gdf = gdf.copy()
    if gdf.crs is None or not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=crs_epsg)

    def _angle(track_id):
        if track_id in descending_track_ids:
            return default_angle - offset_deg
        if track_id in ascending_track_ids:
            return default_angle + offset_deg
        return default_angle

    buffer_geoms = []
    for _, row in gdf.iterrows():
        pt = row.geometry
        track_id = row.get(track_id_column)
        angle = _angle(track_id)
        buffer_geoms.append(create_rotated_buffer(pt, width=width, height=height, angle_deg=angle))

    gdf["rotated_buffer"] = buffer_geoms
    gdf["point_wkt"] = gdf.geometry.to_wkt()
    gdf = gdf.set_geometry("rotated_buffer")
    gdf = gdf.drop(columns=["geometry"], errors="ignore")

    geometry_columns = gdf.select_dtypes(include="geometry").columns.tolist()
    drop_cols = [c for c in geometry_columns if c != "rotated_buffer"]
    gdf = gdf.drop(columns=drop_cols)
    gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)

    logger.info("Built %d rotated ICESat-2 footprint buffers (%.0f x %.0f m).", len(gdf), width, height)
    return gdf


def build_field_plot_buffers(
    gdf: gpd.GeoDataFrame, width: float, height: float, crs_epsg: int
) -> gpd.GeoDataFrame:
    """Replace point geometries with axis-aligned field-plot buffers.

    Used for AGB and AGC field inventory plots.

    Parameters
    ----------
    gdf : GeoDataFrame
        Cleaned field-plot points.
    width, height : float
        Plot dimensions in meters.
    crs_epsg : int
        Target projected EPSG code.

    Returns
    -------
    GeoDataFrame with an active ``rect_buffer`` geometry column.
    """
    gdf = gdf.copy()
    if gdf.crs is None or not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=crs_epsg)

    buffer_geoms = [create_aligned_rectangle(row.geometry, width, height) for _, row in gdf.iterrows()]
    gdf["rect_buffer"] = buffer_geoms
    gdf["point_wkt"] = gdf.geometry.to_wkt()
    gdf = gdf.set_geometry("rect_buffer")
    gdf = gdf.drop(columns=["geometry"], errors="ignore")

    geometry_columns = gdf.select_dtypes(include="geometry").columns.tolist()
    drop_cols = [c for c in geometry_columns if c != "rect_buffer"]
    gdf = gdf.drop(columns=drop_cols)
    gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)

    logger.info("Built %d axis-aligned field-plot buffers (%.0f x %.0f m).", len(gdf), width, height)
    return gdf


# --------------------------------------------------------------------------- #
# Tabular -> vector conversion (AGC field data)
# --------------------------------------------------------------------------- #
def excel_xy_to_geodataframe(
    excel_path: Path, x_column: str = "X", y_column: str = "Y", crs_epsg: int = 21037
) -> gpd.GeoDataFrame:
    """Convert an Excel table with X/Y coordinate columns into a GeoDataFrame.

    Parameters
    ----------
    excel_path : Path
        Path to the field-inventory Excel file (e.g. AGC field measurements).
    x_column, y_column : str
        Names of the coordinate columns (whitespace in column names is
        stripped before lookup, matching the original script).
    crs_epsg : int
        EPSG code of the coordinates in the Excel file.

    Returns
    -------
    GeoDataFrame

    Raises
    ------
    FileNotFoundError
        If ``excel_path`` does not exist.
    ValueError
        If the X/Y columns are missing.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Field data Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(
            f"'{x_column}' / '{y_column}' columns not found in {excel_path}. "
            f"Available columns: {list(df.columns)}"
        )

    geometry = [Point(xy) for xy in zip(df[x_column], df[y_column])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{crs_epsg}")
    logger.info("Converted %d rows from %s to point geometries (EPSG:%d).", len(gdf), excel_path.name, crs_epsg)
    return gdf


def clip_by_boundary(gdf: gpd.GeoDataFrame, boundary_path: Path) -> gpd.GeoDataFrame:
    """Clip ``gdf`` to a study-area / plot-extent boundary shapefile.

    Parameters
    ----------
    gdf : GeoDataFrame
    boundary_path : Path
        Path to a boundary vector file (e.g. a .shp study-extent mask).

    Returns
    -------
    GeoDataFrame
    """
    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary/clip file not found: {boundary_path}")

    clipper = gpd.read_file(boundary_path)
    if gdf.crs != clipper.crs:
        clipper = clipper.to_crs(gdf.crs)

    clipped = gpd.clip(gdf, clipper)
    logger.info("Clipped %d -> %d features using %s.", len(gdf), len(clipped), boundary_path.name)
    return clipped


def train_test_split_gdf(gdf: gpd.GeoDataFrame, test_size: float, random_state: int):
    """Thin wrapper around ``sklearn.model_selection.train_test_split``.

    Kept as a separate function (rather than calling sklearn directly in
    each mapping module) so the random seed and split ratio are always
    sourced from configuration.
    """
    from sklearn.model_selection import train_test_split

    train_gdf, test_gdf = train_test_split(gdf, test_size=test_size, random_state=random_state)
    logger.info("Train/test split: %d / %d (test_size=%.2f, seed=%d).", len(train_gdf), len(test_gdf), test_size, random_state)
    return train_gdf, test_gdf
