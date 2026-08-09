# outputs/

This directory is populated when you run the FCH / AGB / AGC workflows. It
is git-ignored (see `.gitignore`) except for this file, so that the
directory structure ships with the repository without committing large
generated rasters.

Expected subfolders after a full run (see the root `README.md`, section
"Output Files", for details on each file):

```text
outputs/
├── fch/
│   ├── Predictor_Stack_20band.tif
│   ├── Predictor_Stack_Selected.tif
│   ├── fch_random_forest.joblib
│   ├── fch_selectkbest.joblib
│   ├── fch_selected_band_indices.npy
│   └── FCH_Map_10m.tif
├── agb/
│   ├── agb_random_forest.joblib
│   ├── agb_selectkbest.joblib
│   ├── agb_selected_band_indices.npy
│   └── AGB_Map_10m.tif
└── agc/
    ├── agc_random_forest.joblib
    ├── agc_selectkbest.joblib
    ├── agc_selected_band_indices.npy
    └── AGC_Map_10m.tif
```
