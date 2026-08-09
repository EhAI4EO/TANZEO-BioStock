# Phase 3

## Reference
- **Title:** Multi-source Remote Sensing of Tropical Montane Forest Structure for Degradation Risk and Restoration Prioritization
- **Authors:** S. Ehsan Khankeshizadeh; Soheil Zaghian; Sadegh Jamali; Ernest William Mauya; Torbern Tagesson; Ali Mohammadzadeh
- **Status:** Published
- **Journal:** Science of Remote Sensing
- **Year:** 2026
- **Article Number:** 100481
- **DOI:** [https://doi.org/10.1016/j.srs.2026.100481](https://doi.org/10.1016/j.srs.2026.100481)

---

## Methodology Workflow
<p align="center">
  <img src="Figure/methodology_flowchart.jpg" alt="Methodology workflow of Phase 3" width="900">
</p>

---

## Abstract
This study develops a reproducible multi-sensor machine learning-based framework for mapping forest structure, biomass-related patterns, carbon stocks indicators, and ecological vulnerability in tropical forests of the West Usambara (WUSA) Mountains, Tanzania. Sparse but accurate ICESat-2 LiDAR canopy-height observations were integrated with Sentinel-1, Sentinel-2, and topographic data within a machine learning-based framework, in which several regression algorithms were compared and the optimized Random Forest approach was selected to generate a wall-to-wall 10 m resolution forest canopy height (FCH) map that represents broad spatial heterogeneity across the study area. Building on this, the resulting continuous FCH layer was incorporated as a structural predictor in RF models for estimating above-ground biomass (AGB) and above-ground carbon (AGC), which were calibrated and evaluated using available field inventory data collected across multiple forest reserves in the WUSA region. Results indicate that incorporating FCH provides modest relative improvements in AGB and AGC estimation compared to models based solely on optical and radar predictors, although absolute predictive accuracy remains moderate. Furthermore, feature-importance analyses showed that Sentinel-2 spectral bands dominated model performance (≈45–60%), while FCH provided consistent complementary structural information. The integrated FCH, AGB, and AGC layers were further used in a geospatial vulnerability assessment to identify spatial patterns of degradation risk and restoration potential, providing spatially explicit references to support conservation planning and management. To further strengthen the analysis, Monte Carlo-based uncertainty and sensitivity assessments were implemented to quantify uncertainty in AGB and AGC predictions and to evaluate the propagation of these uncertainties into the derived vulnerability maps. Results show that the vulnerability estimates were most sensitive to FCH and slope weights, whereas AGC exhibited comparatively low influence. The source code for this work is publicly available at https://github.com/EhAI4EO/TANZEO-BioStock.

---

## Data & Outputs
The source code and associated project resources are available in this repository.

---

## Outputs
- 10 m forest canopy height map
- Above-ground biomass map
- Above-ground carbon map
- Forest degradation-risk assessment
- Restoration-priority map
- Uncertainty and sensitivity analysis outputs

> **Repository scope note:** this code release currently covers the **FCH, AGB, and AGC mapping stages only** (Random Forest training and wall-to-wall prediction). The Monte Carlo uncertainty/sensitivity assessment and the degradation-risk / restoration-priority vulnerability workflow described in the abstract above are part of the published study but **are not yet included in this repository**; no code for them has been added or reconstructed. This release will be extended if/when that code is finalized.

---

## Repository Structure

```text
papers/paper3/
├── README.md                     # this file
├── requirements.txt               # pip dependencies
├── environment.yml                # Conda environment (recommended for the geospatial stack)
├── config/
│   └── config.example.yaml        # copy to config.yaml and edit for your environment
├── src/
│   ├── __init__.py
│   ├── data_preparation.py        # point loading, outlier filtering, buffer geometry construction
│   ├── raster_utils.py            # band stacking, spectral indices, patch extraction, wall-to-wall prediction
│   ├── model_utils.py             # config loading, feature selection, RF training/tuning, metrics, I/O
│   ├── fch_mapping.py             # FCH training + prediction orchestration
│   ├── agb_mapping.py             # AGB training + prediction orchestration (depends on FCH output)
│   ├── agc_mapping.py             # AGC training + prediction orchestration (depends on FCH output)
│   ├── baseline_models.py         # OPTIONAL: SVR / CatBoost / 1D-CNN comparison baselines
│   ├── evaluation.py              # feature importance, sensor-group radar chart, SHAP plots
│   └── visualization.py           # EDA boxplots, S2 false-color composite, spectral-index grid
├── scripts/
│   ├── run_fch_mapping.py
│   ├── run_agb_mapping.py
│   ├── run_agc_mapping.py
│   └── run_full_workflow.py       # FCH -> AGB -> AGC in dependency order
├── tests/
│   └── test_pipeline_components.py  # unit tests for geometry, patch reduction, filtering, metrics, config loading
├── Figure/
│   └── methodology_flowchart.jpg
├── outputs/
│   └── README.md                  # expected output layout (directory is otherwise git-ignored)
└── .gitignore
```

---

## Workflow Overview

1. **Data preparation** -- load ICESat-2 ATL08 points (FCH) or field-inventory points (AGB/AGC), remove outliers via IQR fences, and build sampling-footprint buffers (rotated for ICESat-2, axis-aligned for field plots).
2. **Predictor stacking** -- build a 20-band raster stack per study extent: 10 Sentinel-2 bands, 7 spectral indices (NDVI, NDMI, NDPI, EVI, NBR, NBR2, NIRv), 2 Sentinel-1 bands (VV, VH), 1 DEM band.
3. **FCH modelling** -- extract patch-mean features (11x3 px) at each ICESat-2 footprint, select the best 16 of 20 bands (`SelectKBest`), train and tune a Random Forest (`GridSearchCV`), and produce a wall-to-wall 10 m FCH map.
4. **AGB / AGC modelling** -- extract patch features (22x90 px bounding box, 5x5-pixel center-window mean) at each field plot from the 20-band stack **plus the FCH map as a 21st predictor**, sweep `SelectKBest` over k to find the best feature subset, tune a Random Forest, and produce wall-to-wall AGB / AGC maps.
5. **Evaluation** -- RMSE, MAE, R², bias, nRMSE(%), MAPE(%) on a held-out test split; RF feature importances and a sensor-group contribution radar chart; optional SHAP interpretation.

**Stage dependency:** AGB and AGC both require the FCH map produced in step 3 as an input. Run FCH mapping before AGB/AGC mapping (`run_full_workflow.py` enforces this order automatically).

---

## Requirements

- Python 3.11 (3.9-3.12 are also expected to work; not individually verified)
- See `requirements.txt` / `environment.yml` for the full dependency list
- Core: NumPy, Pandas, scikit-learn, Rasterio, GeoPandas, Shapely, PyYAML, joblib, tqdm, Matplotlib, Seaborn
- Optional (baseline model comparison / SHAP only): `catboost`, `tensorflow`, `shap`

---

## Installation

### Option A -- Conda (recommended for the geospatial stack)

```bash
conda env create -f environment.yml
conda activate tanzeo-paper3
```

### Option B -- pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install` fails to build GDAL/Rasterio/Fiona on your platform, use Option A instead.

---

## Configuration

Copy the example configuration and edit it for your local paths:

```bash
cp config/config.example.yaml config/config.yaml
```

Then edit `config/config.yaml`:
- Set all paths under `paths:` to your local input data and desired output locations.
- Review `crs_epsg` (default: 21037 -- Arc 1960 / UTM zone 37S) if working in a different projection.
- The example configuration provides the default parameters used by the packaged workflow. Review and adjust the `fch:`, `agb:`, and `agc:` sections (sampling geometry, patch dimensions, feature selection, and model settings) for your own datasets before execution.

`config/config.yaml` is git-ignored and must never be committed with personal machine paths or credentials.

---

## Input Data

**No raw remote-sensing or field-inventory data are distributed in this repository.** You must supply:

| Data | Used by | Notes |
|------|---------|-------|
| ICESat-2 ATL08 points (`.gpkg`), with `h_canopy` and `track_id` fields | FCH | |
| Sentinel-2 composite (10 bands: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12) | FCH, AGB, AGC | one raster per study extent |
| Sentinel-1 composite (VV, VH) | FCH, AGB, AGC | |
| DEM | FCH, AGB, AGC | |
| AGB field plots (`.gpkg`, `AGB` field) | AGB | |
| AGC field measurements (`.xlsx`, `X`/`Y`/`AGC` columns) + boundary shapefile | AGC | |

All predictor rasters must share the same CRS, resolution (10 m by default), and pixel alignment; the AGB/AGC auxiliary-band extraction (`raster_utils.extract_patch_features_with_auxiliary_band`) and wall-to-wall prediction (`raster_utils.predict_wall_to_wall`) explicitly validate CRS, transform, and dimension agreement between the predictor stack and the FCH raster, and raise a clear error on mismatch.

---

## How to Run

### Run FCH Mapping

```bash
python scripts/run_fch_mapping.py --config config/config.yaml
```

Optional flags: `--stage {train,predict,all}` (default `all`), `--evaluate-baselines` (also runs the optional SVR/CatBoost/1D-CNN comparison from `src/baseline_models.py`).

### Run AGB Mapping

Requires the FCH stage to have been run first (`paths.fch_output_map` must exist).

```bash
python scripts/run_agb_mapping.py --config config/config.yaml
```

### Run AGC Mapping

Requires the FCH stage to have been run first.

```bash
python scripts/run_agc_mapping.py --config config/config.yaml
```

### Run the Full Workflow

Runs FCH -> AGB -> AGC in the correct order:

```bash
python scripts/run_full_workflow.py --config config/config.yaml
```

Use `--skip-fch`, `--skip-agb`, `--skip-agc` to skip individual stages (e.g. if you already have a fresh FCH map on disk).

---

## Output Files

See `outputs/README.md` for the full expected directory layout. In summary, each stage writes:
- a trained model (`*_random_forest.joblib`) and fitted feature selector (`*_selectkbest.joblib`),
- the selected-band index array (`*_selected_band_indices.npy`),
- and a wall-to-wall prediction GeoTIFF (`FCH_Map_10m.tif`, `AGB_Map_10m.tif`, `AGC_Map_10m.tif`).

None of these are written if `overwrite_outputs: false` (the default) and the target file already exists -- the pipeline raises `FileExistsError` rather than silently overwriting prior results.

---

## Reproducibility

- Random seed (`random_seed: 42` by default) is applied consistently to train/test splitting and Random Forest fitting.
- Predictor band order is fixed and documented in `src/raster_utils.BAND_NAMES_20` / `BAND_NAMES_21`; `model_utils.check_feature_order` is available to validate a model's expected feature order before prediction.
- Feature selection (`SelectKBest`) selectors are serialized alongside their models so the exact selected-band subset used at training time is reused at prediction time.
- CRS / transform / dimension checks run automatically before FCH-auxiliary-band feature extraction and before wall-to-wall prediction.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Configuration file not found` | `config/config.yaml` missing | `cp config/config.example.yaml config/config.yaml` and edit paths |
| `FileNotFoundError` on a raster/vector path | A `paths:` entry in `config.yaml` is wrong or the file wasn't supplied | Check the "Input Data" table above |
| `ValueError: CRS mismatch...` / `Transform mismatch...` / `Dimension mismatch...` | The FCH raster and the AGB/AGC predictor stack are not spatially aligned | Regenerate one of the rasters so both share CRS, transform, and pixel grid |
| `FileExistsError: Output already exists and overwrite=False` | You re-ran a stage without clearing previous outputs | Delete the specific output file, or set `overwrite_outputs: true` in `config.yaml` |
| `ImportError` when using `--evaluate-baselines` | `catboost` / `tensorflow` / `shap` not installed | `pip install catboost tensorflow shap` (all optional) |
| AGB/AGC prediction fails immediately | FCH stage was not run first | `python scripts/run_fch_mapping.py --config config/config.yaml` |

---

## Citation

If you use this code, please cite:

```bibtex
@article{khankeshizadeh2026wusa,
  title   = {Multi-source Remote Sensing of Tropical Montane Forest Structure for Degradation Risk and Restoration Prioritization},
  author  = {Khankeshizadeh, S. Ehsan and Zaghian, Soheil and Jamali, Sadegh and Mauya, Ernest William and Tagesson, Torbern and Mohammadzadeh, Ali},
  journal = {Science of Remote Sensing},
  year    = {2026},
  volume  = {},
  pages   = {100481},
  doi     = {10.1016/j.srs.2026.100481}
}
```
