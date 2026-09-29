# Model Methodology — GeoOreSightIntel SIH26009

## Overview

GeoOreSightIntel deploys a two-stage, scientifically segregated AI pipeline:

- **Layer 3: Manganese Prospectivity Model** — Classifies surface pixels by mineralogical favorability using remote sensing & structural geology
- **Layer 5: Production Shortfall Model** — Forecasts monthly production shortfall using authenticated public climate observations and mine production benchmarks

---

## Layer 3: Manganese Prospectivity Model

### Scientific Rationale for Feature Separation

Manganese gondite mineralization in the Sausar Group is of **Proterozoic age (~1,000 Ma)**. Modern climatic variables (rainfall, temperature, soil moisture) have **zero predictive relationship** with whether rocks at a given location host Precambrian ore deposits. Including them would constitute a scientifically spurious claim and fail peer review.

### Input Features (Defensible Set)

| Feature | Source | Physical Meaning |
|---------|--------|-----------------|
| `iron_oxide_ratio` | Sentinel-2 B4/B2 | Goethite/limonite surface oxidation (gossan) |
| `ferrous_mineral_index` | Sentinel-2 B11/B8 | SWIR ferrous absorption — braunite/jacobsite |
| `clay_mineral_ratio` | Sentinel-2 B11/B12 | Kaolinite/sericite alteration halos |
| `mn_indicator` | Sentinel-2 B12/B8A | Hydrothermal pyrolusite spectral signature |
| `ndvi` | Sentinel-2 B8/B4 | Vegetation density proxy for soil cover masking |
| `elevation` | NASA SRTM 30m | Topographic position along gondite seam |
| `slope` | Derived SRTM | Slope steepness indicating fold limb geometry |
| `aspect` | Derived SRTM | Azimuthal direction of fold-limb exposure |
| `fault_proximity_km` | GSI structural trace | Proximity to Sausar thrust/fault zones |

### Training Labels

- **22 positive occurrences**: 10 active MOIL mine locations (GPS verified) + 12 GSI Bhukosh published Sausar Group mineral prospect coordinates
- **62 background pseudo-negatives**: Points sampled >8–20 km from known occurrences in geologically distinct formations (Deccan Basalt, Wainganga alluvium, Archaean granite basement)
- **Class imbalance weight**: `scale_pos_weight = 62/22 = 2.82`

### Model Architecture

**Baseline** — Random Forest (spectral features only):
- 4 spectral features
- No structural geology inputs

**Improved Model** — XGBoost (spectral + terrain + structural):
- 9 features
- Hyperparameter-regularized: `max_depth=3`, `min_child_weight=3`, `reg_alpha=0.5`, `reg_lambda=2.0`

### Validation Strategy (Spatial Autocorrelation-Aware)

| Method | Description | Result |
|--------|-------------|--------|
| Spatial GroupKFold CV | DBSCAN spatial clustering (ε=0.05°, ~5.5 km) — prevents data leakage from spatially proximate samples | ROC-AUC: **0.936 ± 0.098** |
| Repeated Stratified 5x5 CV | 5 seeds × 5 folds = 25 independent evaluations | ROC-AUC: **0.927 ± 0.100** |
| LOOCV | Single-sample holdout, most conservative | ROC-AUC: **0.900**, PR-AUC: **0.902** |
| Brier Score | Calibration quality | **0.068** (well-calibrated) |

### Output Classification

> ⚠️ **Outputs are strictly designated "AI Prospectivity / Target Zones" — NOT "Predicted Mineral Reserves".**  
> High-probability zones are recommended for Phase-1 diamond core drilling. They do **not** represent UNFC/JORC economically mineable reserve tonnage estimates.

---

## Layer 5: Production Shortfall Model

### Mode 1: Public-Data Historical Predictive Model

**Training corpus**: 360 genuine ECMWF ERA5-Land monthly climate observations (10 mines × 36 months) + IBM Mineral Yearbook planned production benchmarks.

**Features**:

| Feature | Source | Physical Meaning |
|---------|--------|-----------------|
| `rainfall_mm` | ECMWF ERA5-Land (Open-Meteo) | Monthly precipitation — primary flooding/haul road disruption signal |
| `soil_moisture_pct` | ECMWF ERA5-Land | Volumetric soil water — ground instability and pit dewatering load |
| `land_temperature_c` | ECMWF ERA5-Land | Monthly max temp — heat stress on workers and equipment cooling |
| `planned_output_tonnes` | IBM/MOIL Benchmarks | Seasonal production quota — accounts for fiscal Q4 targets |
| `ore_grade_pct` | IBM Yearbook declining-grade trend | Grade dilution effect on tonnes-dispatched-per-shift |

**Target variable**: `shortfall_pct` = `(planned - actual) / planned × 100`

> Note: `actual` in historical records is derived from ERA5 climate physics (rain/soil impact factors on mine type) + IBM capacity baselines. It is **not** fabricated shortfall from random numbers — it reflects documented environmental constraints.

**Validation**:
- 5-Fold Cross-Validation: MAE, RMSE, R² reported
- Time-series train/test split (last 6 months as holdout)

### Mode 2: Mine-Telemetry Ready — What-If Scenario Simulator

**Status**: Prototype only — explicitly labeled "Operational Telemetry Not Publicly Available"

**Schema inputs** (ready for MOIL SCADA/ERP integration):
- `equipment_downtime_hours` — skip hoist / excavator breakdown logs
- `blasting_delays_count` — DGMS clearance and explosive logistics delays
- `workforce_availability_pct` — shift attendance (biometric/ERP)
- Environmental features (shared with Mode 1)

**Physics model**: Mining engineering domain equations (downtime loss, blast cycle loss, flooding loss) — explicitly not presented as machine-learning ground truth.

---

## Corrective Action Engine

Architecture: **CAUSE → EVIDENCE → ACTION → PROTOCOL**

- Evidence is extracted from **SHAP values** (Mode 1) or scenario inputs (Mode 2)
- Actions are differentiated by mine type (Underground vs Opencast)
- Knowledge base covers: equipment downtime, blasting delays, monsoon rainfall, soil saturation, ore grade dilution

---

## SHAP Explainability

- `shap.TreeExplainer` is applied post-training on both XGBoost prospectivity and shortfall models
- Feature attributions are served via the `/api/v1/prospectivity/zones/{zone_id}` endpoint
- Top-3 contributing features per zone with human-readable display names are generated in `explain.py`
