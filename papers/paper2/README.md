

# Phase 2

### Reference
- **Title:** Forest Canopy Height Mapping in Tanzanian Tropical Rainforests Using Multimodal Remote Sensing Data and Machine Learning
- **Authors:** Soheil Zaghian; Seyed Ehsan Khankeshizadeh; Sadegh Jamali; Torbern Tagesson; Ernest William Mauya; Ali Mohammadzadeh; Filbert Francis
- **Status:** Accepted
- **Journal/Conference:** 2026 ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences (ISPRS Annals) 
- **DOI:** [ISPRS Conference 2026](https://doi.org/10.5194/isprs-annals-XI-3-2026-835-2026)

---

## Abstract
Forest canopy height (FCH) is a critical biophysical parameter that characterizes forest structure and provides fundamental information for estimating above-ground biomass and carbon stocks. The Global Ecosystem Dynamics Investigation (GEDI) Level 2A (L2A) product offers accurate canopy height observations; however, its point-based nature constrains spatial continuity in FCH mapping. This study integrates the multimodal remote sensing datasets for continuous FCH mapping in Tanzania’s West Usambara (WUSA) forest, recognized globally for its rich biodiversity and ecological significance. Hence, remote sensing data, including Sentinel-1 polarizations (VV and VH), Sentinel-2 spectral bands and vegetation indices, and the SRTM digital elevation model (DEM), were integrated and matched with GEDI canopy height data used as reference for FCH modelling. The optimal feature set was derived by evaluating the performance of several feature selection and extraction methods, including Principal Component Analysis (PCA), Minimum Noise Fraction (MNF), Recursive Feature Elimination (RFE), Sequential Feature Selection (SFS), and the Selected K-Best approach using F-value and mutual information scoring functions. The feature set derived from RFE, comprising ten features from all data sources, demonstrated the highest accuracy and reliability in FCH modelling. Subsequently, four machine learning algorithms, including Random Forest (RF), Gradient Boosting Regressor (GBR), Support Vector Regressor (SVR), and Ordinary Least Squares (OLS), were evaluated for FCH modelling. Accordingly, RF achieved higher R² than GBR, SVR, and OLS, with differences of 0.9%, 8.7%, and 16.4%, respectively. Therefore, the RF model, as the most reliable model, was employed for FCH mapping across the WUSA forest.---

## Data & Outputs
<p align="center">
  <img src="Figure/Flowchart_new_2.png" width="40%">
  <img src="Figure/image.png" width="25%">

</p>

<p align="center">
</p>
The datasets used in this study are publicly available on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.17914046


---

## How to Run
1. Create the Python environment
2. Install dependencies
3. Run training and prediction scripts

*(Detailed instructions will be provided once the code is finalized.)*

---

## Outputs
- Forest canopy height maps
- Validation metrics and figures

