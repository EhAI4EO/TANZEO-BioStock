"""Google Earth Engine data preparation pipeline.

Builds a monthly Sentinel-1 + Sentinel-2 + SRTM feature stack, matches it to
GEDI L2A (RH98) canopy-height footprints, samples the stack into a point
dataset, and exports both an image composite and a training/test table to
Google Drive.

This mirrors the exploratory TANZEO-GEDI Colab notebook, reorganized into
importable, parameterized functions and driven entirely by `config.yaml`
instead of hardcoded paths/credentials.
"""

from __future__ import annotations

import ee

from .config import Config

# Sentinel-2 bands + derived indices kept after preprocessing
S2_SELECTED_BANDS = [
    "VV", "VH", "NDVI", "MNDWI", "NDMI", "B2", "B3", "B4", "B8", "B11",
    "QA60", "MSK_CLASSI_OPAQUE", "MSK_CLASSI_CIRRUS",
]


def initialize_ee(config: Config) -> None:
    """Authenticate and initialize the Earth Engine session for this project."""
    ee.Authenticate()
    ee.Initialize(project=config.get("gee", "project"))


def load_assets(config: Config) -> dict[str, ee.FeatureCollection]:
    """Load the in-situ / boundary FeatureCollections referenced in config."""
    root = config.get("gee", "asset_root")
    return {
        "forest_boundaries": ee.FeatureCollection(
            f"{root}/{config.get('gee', 'forest_boundaries_asset')}"
        ),
        "inventory_plots": ee.FeatureCollection(
            f"{root}/{config.get('gee', 'inventory_plots_asset')}"
        ),
        "agb": ee.FeatureCollection(f"{root}/{config.get('gee', 'agb_asset')}"),
    }


def make_rectangle_builder(config: Config):
    """Return a per-feature function that builds a plot-sized rectangle buffer.

    Buffer half-width/half-height depend on the region name, matching the
    field-plot geometry used for two forest reserves (smaller plots) versus
    the rest of the study area.
    """
    aoi_cfg = config["aoi"]
    region_property = aoi_cfg["region_property"]
    small_regex = aoi_cfg["small_buffer_regions_regex"]
    small_w, small_h = aoi_cfg["small_buffer_half_width_m"], aoi_cfg["small_buffer_half_height_m"]
    default_w, default_h = aoi_cfg["default_half_width_m"], aoi_cfg["default_half_height_m"]

    def create_rectangle(feature: ee.Feature) -> ee.Feature:
        region = feature.getString(region_property)

        half_width = ee.Algorithms.If(region.match(small_regex), small_w, default_w)
        half_height = ee.Algorithms.If(region.match(small_regex), small_h, default_h)

        point = feature.geometry().transform("EPSG:3857", 1)
        coords = point.coordinates()
        x = ee.Number(coords.get(0))
        y = ee.Number(coords.get(1))

        rect = ee.Geometry.Rectangle(
            [x.subtract(half_width), y.subtract(half_height),
             x.add(half_width), y.add(half_height)],
            proj="EPSG:3857",
            geodesic=False,
        )
        return ee.Feature(rect).copyProperties(feature)

    return create_rectangle


def build_plot_rectangles(agb: ee.FeatureCollection, config: Config) -> ee.FeatureCollection:
    """Convert AGB plot points into rectangular sampling footprints."""
    return agb.map(make_rectangle_builder(config))


def add_s2_indices(image: ee.Image) -> ee.Image:
    """Add NDVI, MNDWI, NDMI, NDRE, EVI to a (reflectance-scaled) S2 image.

    Images with no bands (e.g. empty monthly composites where no cloud-free
    scene existed) are passed through unmodified.
    """

    def add_all_indices(img: ee.Image) -> ee.Image:
        img = img.divide(10000)
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
        ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
        evi = img.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ).rename("EVI")
        return img.addBands([ndvi, mndwi, ndmi, ndre, evi])

    is_empty = ee.Algorithms.If(
        ee.Algorithms.IsEqual(image.get("empty"), None),
        ee.Algorithms.IsEqual(image.bandNames().length(), 0),
        image.get("empty"),
    )
    return ee.Algorithms.If(is_empty, image, add_all_indices(image))


def monthly_sentinel2_composites(aoi: ee.Geometry, config: Config) -> ee.ImageCollection:
    """Monthly cloud-filtered median Sentinel-2 composites with spectral indices."""
    time_cfg = config["time"]
    year, start_month, end_month = time_cfg["year"], time_cfg["start_month"], time_cfg["end_month"]
    cloud_max = config.get("sentinel2", "cloudy_pixel_percentage_max", default=5)

    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    def monthly_composite(month: int) -> ee.Image:
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")
        monthly_images = (
            collection.filterDate(start_date, end_date)
            .filterBounds(aoi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        )
        return (
            monthly_images.median()
            .clip(aoi)
            .set("month", month)
            .set("system:time_start", start_date.millis())
        )

    months = list(range(start_month, end_month + 1))
    composites = ee.ImageCollection([monthly_composite(m) for m in months])
    return composites.map(add_s2_indices)


def monthly_sentinel1_composites(aoi: ee.Geometry, config: Config) -> ee.ImageCollection:
    """Monthly median Sentinel-1 VV/VH composites."""
    time_cfg = config["time"]
    year, start_month, end_month = time_cfg["year"], time_cfg["start_month"], time_cfg["end_month"]

    s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD")

    def monthly_composite(month: int) -> ee.Image:
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")
        monthly_images = (
            s1_collection.filterDate(start_date, end_date)
            .filterBounds(aoi)
            .select(["VV", "VH"])
        )
        return (
            monthly_images.median()
            .clip(aoi)
            .set("month", month)
            .set("system:time_start", start_date.millis())
        )

    months = list(range(start_month, end_month + 1))
    return ee.ImageCollection([monthly_composite(m) for m in months])


def combine_s1_s2_elevation(
    s1_composites: ee.ImageCollection,
    s2_composites: ee.ImageCollection,
    srtm: ee.Image,
    n_months: int = 12,
) -> ee.ImageCollection:
    """Stack matching S1 + S2 + elevation bands for each month, index-aligned."""
    s1_list = s1_composites.toList(n_months)
    s2_list = s2_composites.toList(n_months)

    def combine_monthly(i):
        s1_img = ee.Image(s1_list.get(i))
        s2_img = ee.Image(s2_list.get(i))
        combined = s1_img.addBands(s2_img).set("month", ee.Number(i).add(1))
        return combined.addBands(srtm.rename("elevation"))

    return ee.ImageCollection.fromImages(
        ee.List.sequence(0, n_months - 1).map(combine_monthly)
    )


def gedi_canopy_height(aoi: ee.Geometry, config: Config) -> ee.ImageCollection:
    """GEDI L2A RH98 canopy height, monthly, quality-filtered (quality_flag == 1)."""
    time_cfg = config["time"]
    gedi = (
        ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        .filterDate(time_cfg["gedi_start_date"], time_cfg["gedi_end_date"])
        .filterBounds(aoi)
    )

    def calculate_height_metrics(image: ee.Image) -> ee.Image:
        rh98 = image.select("rh98")
        quality = image.select("quality_flag")
        masked = rh98.updateMask(quality.eq(1))
        return masked.rename("canopy_height")

    return gedi.map(calculate_height_metrics)


def match_and_merge_collections(
    height_metrics: ee.ImageCollection, combined_collection: ee.ImageCollection
) -> ee.ImageCollection:
    """Attach the GEDI canopy-height band matching each S1/S2 composite's month/year."""
    list_heights = height_metrics.toList(height_metrics.size())

    def match_and_merge(i):
        img_a = ee.Image(list_heights.get(i))
        date_a = img_a.date()
        matched_b = (
            combined_collection.filter(
                ee.Filter.calendarRange(date_a.get("month"), date_a.get("month"), "month")
            )
            .filter(ee.Filter.calendarRange(date_a.get("year"), date_a.get("year"), "year"))
            .sort("system:time_start")
            .first()
        )
        merged = img_a.addBands(ee.Image(matched_b))
        return merged.set("system:time_start", img_a.get("system:time_start"))

    n = height_metrics.size()
    return ee.ImageCollection(
        ee.List.sequence(0, n.subtract(1)).map(match_and_merge)
    )


def remove_images_with_few_bands(
    collection: ee.ImageCollection, min_band_count: int
) -> ee.ImageCollection:
    """Drop composites that didn't accumulate enough bands (e.g. all-cloud months)."""

    def add_band_count(img: ee.Image) -> ee.Image:
        return img.set("band_count", img.bandNames().size())

    with_counts = collection.map(add_band_count)
    return with_counts.filter(ee.Filter.gte("band_count", min_band_count))


def sample_image_collection(
    dataset_col: ee.ImageCollection, aoi: ee.Geometry, config: Config
) -> ee.FeatureCollection:
    """Sample every image in the collection into geolocated point features."""
    sampling_cfg = config["sampling"]
    scale = sampling_cfg["scale_m"]
    num_pixels = sampling_cfg["num_pixels"]

    def sample_image(img: ee.Image) -> ee.FeatureCollection:
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        samples = img.sample(region=aoi, scale=scale, numPixels=num_pixels, geometries=True)

        def add_coords(f: ee.Feature) -> ee.Feature:
            coords = f.geometry().coordinates()
            return f.set({
                "image_id": img.id(),
                "date": date,
                "longitude": coords.get(0),
                "latitude": coords.get(1),
            })

        return samples.map(add_coords)

    return dataset_col.map(sample_image).flatten()


def train_test_split_fc(
    samples: ee.FeatureCollection, config: Config
) -> tuple[ee.FeatureCollection, ee.FeatureCollection]:
    """Random train/test split of a FeatureCollection via a random column."""
    sampling_cfg = config["sampling"]
    seed = sampling_cfg["random_seed"]
    train_fraction = sampling_cfg["train_fraction"]

    with_rand = samples.randomColumn("rand", seed=seed)
    train = with_rand.filter(ee.Filter.lt("rand", train_fraction))
    test = with_rand.filter(ee.Filter.gte("rand", train_fraction))
    return train, test


def export_image_to_drive(image: ee.Image, region: ee.Geometry, config: Config) -> ee.batch.Task:
    export_cfg = config["export"]
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=export_cfg["image_export_description"],
        folder=export_cfg["drive_folder"],
        region=region,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    return task


def export_table_to_drive(samples: ee.FeatureCollection, config: Config) -> ee.batch.Task:
    export_cfg = config["export"]
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=export_cfg["table_export_description"],
        folder=export_cfg["drive_folder"],
        fileFormat="CSV",
    )
    task.start()
    return task


def run_pipeline(config: Config) -> None:
    """End-to-end: build the feature stack, sample it, and export to Drive."""
    initialize_ee(config)
    assets = load_assets(config)

    aoi = assets["forest_boundaries"]
    aoi_geometry = aoi.geometry()

    srtm = ee.Image("CGIAR/SRTM90_V4").clip(aoi)

    s2_composites = monthly_sentinel2_composites(aoi, config)
    s1_composites = monthly_sentinel1_composites(aoi, config)
    combined = combine_s1_s2_elevation(s1_composites, s2_composites, srtm)
    combined = combined.select(S2_SELECTED_BANDS + ["elevation"])

    height_metrics = gedi_canopy_height(aoi, config)
    merged = match_and_merge_collections(height_metrics, combined)

    min_band_count = config.get("sampling", "min_band_count", default=12)
    filtered = remove_images_with_few_bands(merged, min_band_count)
    dataset_col = filtered.select(config.get("bands", "final"))

    date_range = config.get("export", "image_export_date_range")
    selected_image = dataset_col.filterDate(*date_range).median().toFloat()
    export_image_to_drive(selected_image, aoi_geometry, config)

    samples = sample_image_collection(dataset_col, aoi_geometry, config)
    export_table_to_drive(samples, config)

    print(
        "GEE export tasks started. Check the Earth Engine Tasks tab or the "
        f"'{config.get('export', 'drive_folder')}' folder in Google Drive."
    )
