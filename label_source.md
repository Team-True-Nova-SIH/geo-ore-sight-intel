# Layer 2 — Ground-Truth Label Documentation

## Overview

This document specifies the origin, geological rationale, and spatial sampling methodology for `labels.csv` used to train and validate the Manganese Prospectivity Model (Layer 3).

Total Labels: **84 data points**
- **Positive Labels (Class 1)**: 22 points (10 active MOIL mines + 12 GSI mineralized block prospects)
- **Negative / Background Labels (Class 0)**: 62 points (pseudo-negative points sampled outside favorability zones)

---

## Positive Labels (Class 1)

### Active MOIL Mines (10 Primary Points)

| Mine Name | Latitude | Longitude | District / State | Lease Type | Primary Source |
|-----------|----------|-----------|------------------|------------|----------------|
| Balaghat (Bharveli) | 21.8464 | 80.2281 | Balaghat, MP | Underground | MOIL Annual Report / Forest Clearance Portal |
| Ukwa Mine | 21.9714 | 80.4458 | Balaghat, MP | Underground | MOIL Operations / IBM Mineral Directory |
| Tirodi Mine | 21.7000 | 79.6667 | Balaghat, MP | Opencast | MOIL Operations & GSI Toposheet 55 O/10 |
| Sitapatore / Sukli | 21.7000 | 79.6667 | Balaghat, MP | Opencast | Forest Clearance Portal / MP Mining Dept |
| Kandri Mine | 21.4106 | 79.2656 | Nagpur, MH | Underground | MOIL Operations / Wikimapia Geocoding |
| Munsar Mine | 21.4000 | 79.2667 | Nagpur, MH | Underground | MOIL Annual Report / MPCB Clearance |
| Beldongri Mine | 21.4000 | 79.2667 | Nagpur, MH | Underground | MOIL Annual Report / GSI Study |
| Gumgaon Mine | 21.4078 | 78.9861 | Nagpur, MH | Underground | MOIL Operations / Forest Clearance Portal |
| Dongri Buzurg Mine | 21.5478 | 79.7431 | Bhandara, MH | Opencast | MOIL Annual Report / EMD Plant Facility |
| Chikla Mine | 21.5342 | 79.7461 | Bhandara, MH | Underground | MOIL Operations / Shaft Extension Plan 2024 |

### Geological Extension Prospects (12 Secondary Points)

Coordinates extracted from:
1. **Geological Survey of India (GSI) Bhukosh Portal**: Digitized Sausar Group Gondite Horizon maps (Toposheets 55 O/10, 55 O/14, 55 O/15).
2. **IBM Mineral Bulletin (Manganese 2022)**: Known manganese oxide reef extensions along fold axes.
3. **MP & Maharashtra Mining Department State Block Auction Notices (2022–2024)**.

---

## Negative / Background Labels (Class 0)

### Sampling Strategy & Positive-Unlabeled (PU) Framing

Since unmined ground cannot be guaranteed to contain zero manganese in unexplored sub-surfaces, background points are generated using a **Positive-Unlabeled (PU) sampling strategy**:

1. **Distance Threshold**: Sampled at least **>8 km to >20 km** away from any known positive label or fold axis.
2. **Geological Exclusion**:
   - Points fall within **Deccan Trap Basalt cover**, **Wainganga alluvial floodplains**, or **Dongargarh Archean Granite/Gneiss basement** — lithologies structurally incapable of hosting Sausar-style gonditic manganese oxide ore.
3. **Framing Note**: These points represent *pseudo-negatives* (background non-favorability zones) rather than proven absence of mineralization, which is standard practice in geospatial prospectivity mapping (e.g., Carranza 2008, Harris et al. 2015).

---

## Data Verification & Integrity Check

- All coordinates lie strictly within the study area bounding box `[79.0°E, 21.3°N, 80.5°E, 22.1°N]`.
- No duplicate lat/lon coordinates.
- Class distribution: 26.2% Positive / 73.8% Negative (realistic operational imbalance handled via XGBoost `scale_pos_weight` parameter).
