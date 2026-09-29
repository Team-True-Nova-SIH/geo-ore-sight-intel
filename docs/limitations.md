# System Limitations — GeoOreSightIntel SIH26009

> **Purpose**: This document transparently discloses all known scientific, data, and technical limitations of the GeoOreSightIntel platform. Disclosing limitations proactively is required for scientific integrity and is expected by Ministry of Steel evaluation panels.

---

## 1. Prospectivity Model Limitations

### 1.1 Output is NOT Mineral Reserves

**Limitation**: The system produces "AI Prospectivity Scores" — it does NOT calculate mineral reserve tonnage.

- UNFC-2019 / JORC Code require **borehole-verified, geologically modelled** mineral estimates
- Our output is a relative favorability score derived from surface remote sensing
- This maps to UNFC **G4 (Reconnaissance)** or **G3 (Prognostic)** category — not G1/G2 proven/probable reserves
- To convert prospectivity to a legally reportable reserve estimate, MOIL would need to conduct Phase-1 diamond core drilling, laboratory assay, and a qualified person (QP) technical report

### 1.2 Spatial Coverage is Limited

**Limitation**: The Sentinel-2 spectral data, SRTM terrain, and GSI training labels are authenticated only for:
- Balaghat District, Madhya Pradesh
- Nagpur-Bhandara District, Maharashtra
- Coverage extent: **79.0°E–80.5°E, 21.3°N–22.1°N**

Predictions requested outside this corridor are blocked with a data-gap error. The system is **not** calibrated for Odisha, Karnataka, Goa, or Rajasthan manganese belts.

### 1.3 Training Data Density

**Limitation**: Only **84 training samples** (22 positive occurrences + 62 background points).

- Small sample sizes increase variance in cross-validation metrics
- High CV standard deviations (±0.09–0.10 AUC) indicate sensitivity to sample splits
- A larger curated label set from GSI Bhukosh would substantially improve model stability

### 1.4 Subsurface is Not Directly Modelled

**Limitation**: All spectral and topographic features are derived from **surface remote sensing**. Subsurface ore geometry (depth, dip, thickness) requires physical borehole data.

- The `geology_adapter.py` subsurface module demonstrates the integration schema
- It currently uses **reference published GSI stratigraphic columns** as illustrations, not actual drill core assay data
- This is explicitly labeled in all API responses as "DEMO / INTEGRATION READY"

---

## 2. Production Shortfall Model Limitations

### 2.1 No Actual Mine Production Records

**Limitation**: MOIL Ltd's actual monthly production data (tonnes dispatched, shift reports, winding engine logs) is **proprietary operational data not in the public domain**.

- The Mode 1 historical model uses **simulated "actual" production** derived from ERA5 climate physics applied to IBM planned quotas
- This is more defensible than pure random noise, but is not the same as mining the MOIL SAP/ERP system
- R² and MAE metrics reflect the model fitting this physics-derived target — they cannot guarantee the same accuracy on real MOIL ERP data

### 2.2 Equipment Telemetry is Not Available

**Limitation**: Mine equipment downtime, shift hoist logs, excavator breakdown records, and blasting clearance logs are proprietary and not publicly accessible.

- Mode 2 (What-If Simulator) demonstrates the API schema and engineering equations for when this data becomes available
- It is explicitly labeled: "PROTOTYPE SCENARIO — OPERATIONAL TELEMETRY NOT PUBLICLY AVAILABLE"
- Any shortfall figure derived from Mode 2 is **scenario planning output**, not historical ground truth

### 2.3 Ore Grade is Approximated

**Limitation**: Per-mine monthly ore grade (% Mn) is estimated from IBM Mineral Yearbook annual average trends and a declining grade trajectory, not from shift-by-shift dispatch weighbridge records.

- Grade is modelled as: `base_grade - (time_offset × 0.35%) + sinusoidal_noise`
- This follows the documented IBM trend of ~0.3-0.35% annual Mn% grade decline in Sausar Group deposits
- Actual dispatch grade may differ significantly by stope and shift

---

## 3. Technical and Infrastructure Limitations

### 3.1 Google Earth Engine Dependency

**Limitation**: The full data pipeline (`data_pipeline.py`) requires an authenticated Google Earth Engine account.

- In the absence of GEE credentials, the system uses a **deterministic fallback feature generator** based on known geophysical patterns
- The fallback still produces scientifically plausible synthetic features for the Balaghat Belt, but is not derived from actual satellite image analysis
- The prospectivity `model.pkl` shipped with the repository was trained on GEE-extracted features

### 3.2 Shortfall Model Requires ERA5 Connectivity

**Limitation**: If the Open-Meteo API is unavailable, `fetch_real_weather.py` cannot pull fresh ERA5 climate data.

- The system caches `historical_public_data.csv` which contains 36 months of pre-fetched ERA5 records
- If this file is present, the API will function correctly in offline mode
- New deployment to a new environment requires internet access on first boot to fetch climate records

### 3.3 SHAP Explainability is Post-Hoc

**Limitation**: SHAP values explain model predictions statistically, not physically.

- A high SHAP value for `ferrous_mineral_index` at a given pixel means this feature contributed most to raising the predicted probability — it does not constitute field-verified geological evidence
- SHAP explanations should be treated as hypothesis-generation guidance for field geologists, not as proof of ore presence

---

## 4. Scope Not Covered

| Out of Scope | Reason |
|-------------|--------|
| Actual mine tonnage reserve estimation | Requires physical drilling, assay, QP reporting |
| Underground ventilation / geomechanical analysis | Requires proprietary mine plans |
| Workforce scheduling optimization | Requires MOIL HR/biometric data |
| Real-time SCADA integration | Requires MOIL IT infrastructure access |
| Other Indian manganese belts (Odisha, Karnataka) | Outside the authenticated satellite corridor |
| Mineral price forecasting | Economic model scope not addressed |

---

## 5. Scientific Integrity Statement

All output labeled "REAL / PUBLIC" in `/api/v1/provenance` has been independently verifiable against the cited public sources (ERA5, Sentinel-2, SRTM, GSI). All output labeled "DEMO" or "PROTOTYPE SCENARIO" is clearly segregated and is never presented as historical ground truth. The system architecture is designed to fail safely and transparently when data is insufficient, rather than producing confident but fabricated predictions.
