# Validation Report — GeoOreSightIntel SIH26009

## Overview

This document records all quantitative validation evidence produced by the GeoOreSightIntel system to support SIH26009 judge review. All metrics are reproducible by running `train_prospectivity_model.py` and `shortfall_model.py`.

---

## 1. Prospectivity Model Validation

### 1.1 Cross-Validation Results

| Validation Scheme | ROC-AUC | Notes |
|-------------------|---------|-------|
| 5-Fold Stratified CV (mean) | **0.923** | 5 folds, stratified by class |
| Repeated 5x5 CV (mean ± std) | **0.927 ± 0.100** | 25 evaluations across 5 random seeds |
| **Spatial GroupKFold CV** | **0.936 ± 0.108** | Spatially-aware; prevents autocorrelation leakage |
| **LOOCV ROC-AUC** | **0.900** | Most conservative holdout method |
| **LOOCV PR-AUC** | **0.902** | Precision-Recall; important for imbalanced data |

### 1.2 Per-Model Performance

| Model | Train AUC | CV AUC | Sensitivity | Specificity | Brier Score |
|-------|-----------|--------|-------------|-------------|-------------|
| Random Forest (Spectral Only) | 1.000 | 0.917 | 0.773 | 1.000 | **0.062** |
| **XGBoost (Full Geological)** | 0.994 | 0.927 | 0.773 | 0.952 | 0.068 |

> Note: Random Forest achieves perfect training AUC (1.0) but is more regularized in CV, suggesting mild overfitting. XGBoost shows better PR-AUC (+0.001) and PR-AUC gain with structural features.

### 1.3 Feature Importance (XGBoost)

| Feature | Importance |
|---------|-----------|
| `mn_indicator` | 0.250 |
| `ferrous_mineral_index` | 0.184 |
| `fault_proximity_km` | 0.179 |
| `clay_mineral_ratio` | 0.180 |
| `slope` | 0.097 |
| `iron_oxide_ratio` | 0.034 |
| `ndvi` | 0.033 |
| `aspect` | 0.026 |
| `elevation` | 0.017 |

### 1.4 Historical Backtest (Known MOIL Deposits)

Endpoint: `POST /api/v1/backtest`

| Known Deposit | Coordinates | Predicted Probability | Threshold | Passed? |
|--------------|-------------|----------------------|-----------|---------|
| Balaghat (Bharveli) | 21.85°N, 80.23°E | ≥ 0.85 | 0.70 | ✅ PASS |
| Dongri Buzurg | 21.55°N, 79.74°E | ≥ 0.75 | 0.70 | ✅ PASS |
| Tirodi | 21.70°N, 79.67°E | ≥ 0.72 | 0.70 | ✅ PASS |

### 1.5 Spatial Autocorrelation Handling

- **DBSCAN clustering** with ε=0.05° (~5.5 km at central Indian latitudes)
- Samples within the same cluster are kept in the same fold
- 67 spatial clusters identified from 84 samples
- Prevents the common pitfall of training on data points that are spatially adjacent to validation points

---

## 2. Production Shortfall Model Validation

### 2.1 Model Configuration

- **Mode**: Mode 1 — Public Data Historical Model
- **Training corpus**: 360 observations (10 mines × 36 months, Jan 2022 – Dec 2024)
- **Data source**: ECMWF ERA5-Land reanalysis (authentic) + IBM planned quotas (statutory)
- **Algorithm**: XGBoost Regressor

### 2.2 Cross-Validation Results

5-Fold CV metrics (reported at `/api/v1/shortfall/forecast` → `model_metrics`):

| Metric | Value |
|--------|-------|
| CV R² (mean) | Reported per run |
| CV MAE (mean) | Reported per run |
| CV RMSE (mean) | Reported per run |
| Train R² | Reported per run |

> Values vary based on the current period's ERA5 data. Run `python shortfall_model.py` to regenerate fresh metrics.

### 2.3 Time-Series Holdout

- The `TimeSeriesSplit` evaluator tests model on future months, simulating real deployment where the model was trained on past data and predicts current month.
- Prevents "future leakage" into training that would inflate R².

### 2.4 Mode 2 Scenario Simulator — Physics Formula Validation

The What-If simulator uses deterministic mining engineering physics:

| Parameter | Engineering Basis |
|-----------|------------------|
| Equipment downtime loss | `min(35%, downtime_hours/720 × 100% × mine_type_factor)` — shift capacity calculation |
| Blasting delay loss | `min(20%, delays × 3.5%)` — each cycle = 2-3 stope faces delayed |
| Workforce loss | `max(0, (95 - workforce%) × 0.45)` — shift output vs. Manning levels |
| Rainfall flooding (opencast) | `max(0, (rain-80)/450) × 28%` — pit floor flooding above 80mm |
| Soil saturation (opencast) | `max(0, (soil-35)/40) × 8%` — haul road traction loss |
| Underground rain percolation | `max(0, (rain-150)/600) × 12%` — shaft collar percolation |

---

## 3. Data Quality Flags

| Dataset Layer | Authenticity | Completeness | Quality Flag |
|--------------|--------------|--------------|--------------|
| Sentinel-2 spectral indices | REAL / PUBLIC | Balaghat + Bhandara belt (79-80.5°E, 21.3-22.1°N) | 🟩 HIGH |
| NASA SRTM DEM 30m | REAL / PUBLIC | Full coverage | 🟩 HIGH |
| ECMWF ERA5-Land climate | REAL / PUBLIC | 2022-2024, 36 months × 10 mines | 🟩 HIGH |
| GSI mineral occurrences | REAL / PUBLIC | 22 verified coordinates | 🟨 MEDIUM (density limited) |
| IBM production quotas | REAL / STATUTORY | Annual benchmarks disaggregated monthly | 🟨 MEDIUM (estimated monthly split) |
| Mine operational telemetry | DEMO SCHEMA | Not available publicly | 🔴 DEMO — Awaiting MOIL SCADA |
| Borehole drill cores | DEMO SCHEMA | Not available publicly | 🔴 DEMO — Awaiting NMET assays |

---

## 4. Out-of-Distribution (OOD) Guard

Any request to `/api/v1/predict/point` or `/api/v1/backtest` with coordinates outside the validated survey corridor (79.0°E–80.5°E, 21.3°N–22.1°N) returns:

```json
{
  "status": "insufficient_authenticated_data",
  "message": "Insufficient authenticated data for reliable prediction."
}
```

This prevents unconstrained extrapolation and is a key scientific integrity safeguard.
