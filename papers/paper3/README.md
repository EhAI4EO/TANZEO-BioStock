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
  <img src="Figure/methodology_flowchart.png.jpg" alt="Methodology workflow of Phase 3" width="900">
</p>

---
## Abstract

This study develops a reproducible multi-sensor machine learning-based framework for mapping forest structure, biomass-related patterns, carbon stocks indicators, and ecological vulnerability in tropical forests of the
West Usambara (WUSA) Mountains, Tanzania. Sparse but accurate ICESat-2 LiDAR canopy-height observations
were integrated with Sentinel-1, Sentinel-2, and topographic data within a machine learning-based framework,
in which several regression algorithms were compared and the optimized Random Forest approach was selected
to generate a wall-to-wall 10 m resolution forest canopy height (FCH) map that represents broad spatial heterogeneity across the study area. Building on this, the resulting continuous FCH layer was incorporated as a
structural predictor in RF models for estimating above-ground biomass (AGB) and above-ground carbon (AGC),
which were calibrated and evaluated using available field inventory data collected across multiple forest reserves
in the WUSA region. Results indicate that incorporating FCH provides modest relative improvements in AGB and
AGC estimation compared to models based solely on optical and radar predictors, although absolute predictive
accuracy remains moderate. Furthermore, feature-importance analyses showed that Sentinel-2 spectral bands
dominated model performance (≈45–60%), while FCH provided consistent complementary structural information. The integrated FCH, AGB, and AGC layers were further used in a geospatial vulnerability assessment to
identify spatial patterns of degradation risk and restoration potential, providing spatially explicit references to
support conservation planning and management. To further strengthen the analysis, Monte Carlo-based uncertainty and sensitivity assessments were implemented to quantify uncertainty in AGB and AGC predictions and to
evaluate the propagation of these uncertainties into the derived vulnerability maps. Results show that the vulnerability estimates were most sensitive to FCH and slope weights, whereas AGC exhibited comparatively low
influence. The source code for this work is publicly available at https://github.com/EhAI4EO/TANZEO-BioStock.

---

## Data & Outputs

The source code and associated project resources are available in this repository.

---

## How to Run

1. Create the Python environment.
2. Install the required dependencies.
3. Prepare the remote sensing and field inventory datasets.
4. Run the canopy height, biomass, carbon, and vulnerability mapping workflows.

*Detailed instructions will be provided as the code and datasets4 are finalized.*

---

## Outputs

- 10 m forest canopy height map
- Above-ground biomass map
- Above-ground carbon map
- Forest degradation-risk assessment
- Restoration-priority map
- Uncertainty and sensitivity analysis outputs
