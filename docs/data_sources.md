# GeoOreSightIntel — Authoritative Data Sources & Provenance Registry
**Smart India Hackathon 2026 (SIH26009) — Ministry of Steel / MOIL Limited**

---

## 1. Executive Summary & Data Integrity Policy

To ensure **absolute scientific defensibility and judge trust**, the GeoOreSightIntel platform strictly differentiates between:
1. **AUTHENTIC / PUBLIC DATA**: Live Earth Observation feeds, reanalysis climate models, official geological memoirs, and statutory corporate filings.
2. **ENTERPRISE INTEGRATION DEMONSTRATION**: High-fidelity data structures formatted for proprietary mine SCADA, ERP, and diamond-core drilling feeds that are legally restricted to MOIL Ltd.

> [!IMPORTANT]
> **Scientific Integrity Directives:**
> - **Reserves vs Prospectivity**: Output is strictly designated as **"AI Prospectivity / Target Zones"**. Mineral reserve estimation under UNFC-111 / JORC standards requires dense sub-surface diamond core drilling and 3D variography block modeling. AI satellite inference provides exploration prioritization, NOT mineable reserve tonnage.
> - **No Fake Telemetry**: Equipment breakdown and shift attendance data are not fabricated as historical ground truth. The system provides an authenticated public-data model (Mode 1) and a dedicated What-If Scenario Simulator (Mode 2).
> - **Feature Disentanglement**: Operational weather variables (rainfall, soil moisture, land temperature) are **strictly excluded** from Precambrian manganese prospectivity models to avoid scientifically spurious correlations.

---

## 2. Comprehensive Data Source Inventory

### Dataset 1: Copernicus Sentinel-2 Level-2A (Surface Reflectance)
- **Source**: European Space Agency (ESA) / Copernicus Open Access Hub / Google Earth Engine
- **Identifier**: `COPERNICUS/S2_SR_HARMONIZED`
- **URL / Reference**: [Copernicus Sentinel-2 User Handbook](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- **Spatial Coverage**: Balaghat & Nagpur-Bhandara Manganese Corridor, Central India `[79.0°E, 21.3°N, 80.5°E, 22.1°N]`
- **Temporal Coverage**: 2024-01-01 to 2024-06-30 (Dry/pre-monsoon cloud-free composite)
- **Spatial / Spectral Resolution**: 10m (B2, B3, B4, B8), 20m (B11, B12 SWIR)
- **License / Accessibility**: Open Access (CC-BY 4.0 / Copernicus Data Policy)
- **Preprocessing Method**: 
  - Median composite filtered for `<20%` cloud pixel cover using Scene Classification Layer (SCL).
  - Derived diagnostic spectroscopic indices:
    - **Iron Oxide Ratio**: $(B4 - B2) / (B4 + B2)$ (detects surface gossan & hematite/goethite capping).
    - **Ferrous Mineral Index**: $B11 / B8$ (SWIR1 to NIR ratio diagnostic of $Fe^{2+}/Mn^{2+}$ silicates in gondite).
    - **Clay Mineral Ratio**: $B11 / B12$ (SWIR1 to SWIR2 ratio isolating hydrothermal alteration envelopes).
    - **Normalized Difference Vegetation Index (NDVI)**: $(B8 - B4) / (B8 + B4)$ (measures canopy cover masking bedrock).
    - **Manganese Composite Indicator**: $(B4 / B2) \times (B11 / B8)$.
- **Date Accessed / Verified**: 2026-09-28

---

### Dataset 2: NASA Shuttle Radar Topography Mission (SRTM) DEM
- **Source**: NASA / USGS / Google Earth Engine
- **Identifier**: `USGS/SRTMGL1_003`
- **URL / Reference**: [NASA JPL SRTM Global 1-Arcsecond](https://doi.org/10.5067/MEaSUREs/SRTM/SRTMGL1.003)
- **Spatial Coverage**: Global coverage; clipped to Central Indian Manganese Belt
- **Temporal Coverage**: Static baseline (February 2000 mission, validated through contemporary GNSS)
- **Spatial Resolution**: 1 arc-second (~30 meters horizontal), vertical accuracy $\pm 16\text{ m}$
- **License / Accessibility**: Public Domain (NASA Open Data)
- **Preprocessing Method**:
  - Void-filled elevation raster extracted in EPSG:4326.
  - Second-order derivative computation using Horn's algorithm to generate **Terrain Slope (degrees)** and **Topographic Aspect (azimuth)**.
  - Isolates elevated, erosion-resistant gondite quartzite ridges hosting overturned synclinal manganese ore reefs.
- **Date Accessed / Verified**: 2026-09-28

---

### Dataset 3: ECMWF ERA5-Land Reanalysis (Surface Climate & Hydrology)
- **Source**: European Centre for Medium-Range Weather Forecasts (ECMWF) / Copernicus Climate Change Service (C3S) / Open-Meteo Historical Archive API
- **Identifier**: `ECMWF/ERA5_LAND`
- **URL / Reference**: [Copernicus Climate Data Store ERA5-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land)
- **Spatial Coverage**: Exact GPS coordinates of 10 primary MOIL mining leases
- **Temporal Coverage**: 2022-01-01 to 2024-12-31 (36 continuous months)
- **Spatial Resolution**: 0.1° (~9 km grid), resampled to monthly aggregates
- **License / Accessibility**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Variables Extracted**:
  - `precipitation_sum` $\rightarrow$ Monthly total rainfall (mm)
  - `soil_moisture_0_to_7cm_mean` $\rightarrow$ Volumetric soil moisture ($m^3/m^3$, converted to %)
  - `temperature_2m_max` $\rightarrow$ Monthly peak land surface ambient temperature (°C)
- **Preprocessing Method**: Daily reanalysis records downloaded via REST API with UTC+05:30 alignment, aggregated to calendar months, matched with mine operational target timeframes.
- **Date Accessed / Verified**: 2026-09-28

---

### Dataset 4: Geological Survey of India (GSI) Bhukosh & Published Memoirs
- **Source**: Geological Survey of India (Ministry of Mines, Government of India)
- **URL / Reference**: 
  - [GSI Bhukosh Spatial Data Portal](https://bhukosh.gsi.gov.in)
  - *GSI Memoir Vol. 124*: "Geology and Manganese Ore Deposits of the Sausar Group"
  - *GSI Bulletin Series A, No. 22*: "Manganese Ore Deposits of Madhya Pradesh & Maharashtra"
  - *GSI Special Publication 85*: "Central Indian Precambrian Mineral Provinces"
- **Spatial Coverage**: Toposheet sheets 55 O/10, 55 O/14, 55 O/15, 64 C/1, 64 C/5
- **Temporal Coverage**: Published statutory geological surveys (1965–2022)
- **Resolution**: 1:50,000 scale geological mapping
- **License / Accessibility**: Public access for research and statutory planning
- **Data Points & Ground Truth Integration**:
  - **10 Active MOIL Mine Locations**: Balaghat (Bharveli), Dongri Buzurg, Tirodi, Chikla, Kandri, Ukwa, Munsar, Gumgaon, Beldongri, Sitapatore.
  - **12 GSI Gondite Extension Prospects**: Digitized fold axes and mineralized strike reefs (Garra, Miragpur, Gudma, Pauni, etc.).
  - **62 Background Negative Points**: Sampled with spatial buffer ($>8\text{ km}$ to $>20\text{ km}$) in unmineralized lithologies (Deccan Trap basalts, Wainganga alluvium, Dongargarh Archean granites).
  - **Structural Fault Traces**: Digitized synclinal fault and thrust lines used to compute `fault_proximity_km`.
- **Date Accessed / Verified**: 2026-09-28

---

### Dataset 5: MOIL Limited Statutory Disclosures & Indian Bureau of Mines (IBM) Publications
- **Source**: MOIL Limited (Miniratna PSU, Ministry of Steel) & Indian Bureau of Mines
- **URL / Reference**: 
  - [MOIL Annual Report FY 2023-24 & FY 2024-25](https://moil.nic.in)
  - [IBM Indian Minerals Yearbook: Manganese Ore (2022-2023 editions)](https://ibm.gov.in)
  - MOIL Quarterly Financial & Production Results filed with BSE / NSE
- **Spatial Coverage**: Company-wide and mine-lease specific disclosures
- **Temporal Coverage**: FY 2020-21 through FY 2024-25
- **Resolution**: Mine lease capacity thresholds & quarterly aggregate production
- **License / Accessibility**: Public corporate filings / statutory disclosures
- **Parameters Extracted**:
  - Total annual production benchmark: **18.02 Lakh Tonnes** (FY25).
  - Annual mine rated capacities (Balaghat: 4.5–5.0 LTPA, Dongri Buzurg: 4.0 LTPA, Tirodi: 2.5 LTPA, Chikla: 2.0 LTPA, Kandri: 1.8 LTPA, Ukwa: 1.5 LTPA, etc.).
  - Ore grade categories: High-grade ferro ore ($>42\%$ Mn), medium-grade ($34-40\%$ Mn), dioxide ore, and fines.
  - Benchmark market pricing: Run-of-Mine ore benchmark (₹10,500/Tonne), EMD refined chemical grade (₹1,80,000/Tonne), diamond-core exploration drilling (₹4,500/meter).
- **Date Accessed / Verified**: 2026-09-28

---

### Dataset 6: Proprietary Mine Operational Telemetry (Demonstration Schema)
- **Status**: **PROTOTYPE SCENARIO — AWAITING AUTHENTICATED MINE TELEMETRY**
- **Data Owner**: MOIL Ltd Internal SCADA, Fleet Management, and ERP systems
- **Public Availability**: **NOT PUBLICLY AVAILABLE**. No mining company in India publishes granular, shift-by-shift equipment breakdown logs, blasting delay notices, or workforce roster attendance.
- **System Architecture Handling**:
  - The production shortfall system does **NOT** present synthetic operational logs as historical ground truth.
  - **Mode 1**: Trains the predictive model strictly on public ECMWF ERA5 climate variables and IBM target capacities.
  - **Mode 2 (What-If Simulator)**: Provides a standardized REST ingestion schema ready for future authenticated SCADA/ERP integration, allowing mine engineers to stress-test scenarios (e.g. $+40\text{h}$ excavator breakdown or monsoon delays) without altering the model pipeline.

---

### Dataset 7: Subsurface Borehole Assays (Integration Layer)
- **Status**: **INTEGRATION LAYER — AWAITING AUTHENTICATED DRILL DATA**
- **Data Owner**: MOIL Ltd Geology Division / National Mineral Exploration Trust (NMET)
- **Public Availability**: **RESTRICTED**. Proprietary 3D block models and core assay assays (lithology, % Mn by depth interval) are confidential mining assets.
- **System Architecture Handling**:
  - Provides published stratigraphic reference logs from GSI Memoir Vol. 124 for regional context.
  - Demonstrates the complete enterprise schema:
    `[drill_id, mine_block, lat, lon, depth_m, lithology, ore_intersection_m, manganese_grade_pct, source, confidence]`.
  - The prospectivity model is **NEVER** trained on fabricated drill records.
  - The UI visually flags subsurface drill slots with amber integration badges to prevent misrepresenting mock boreholes as proven reserves.

---

## 3. Data Pipeline & Scientific Workflow Diagram

```
+---------------------------------------------------------------------------------+
|                               AUTHENTIC DATA INPUTS                             |
+------------------------------------+--------------------------------------------+
| Satellite Spectroscopy (Sentinel-2)| Terrain Morphology (SRTM 30m DEM)          |
| - Iron Oxide Ratio (B4/B2)         | - Elevation (m)                            |
| - Ferrous Mineral Index (B11/B8)   | - Slope Gradient (degrees)                 |
| - Clay Mineral Ratio (B11/B12)     | - Aspect (degrees)                         |
| - Manganese Spectral Index         |                                            |
+------------------------------------+--------------------------------------------+
| Structural Geology (GSI Bhukosh)   | Climate Reanalysis (ECMWF ERA5-Land)       |
| - Gondite Horizon Strike Axes      | - Monthly Precipitation (mm)               |
| - Fault Proximity (km)             | - Soil Moisture (0-7cm %)                  |
| - 22 Validated Mineral Occurrences | - Max Land Surface Temperature (°C)        |
+------------------------------------+--------------------------------------------+
                                     |
                                     v
+---------------------------------------------------------------------------------+
|                        STRICT FEATURE DISENTANGLEMENT                           |
+------------------------------------+--------------------------------------------+
| PROSPECTIVITY MODEL FEATURES       | OPERATIONAL SHORTFALL FEATURES             |
| (Precambrian Mineralization)       | (Contemporary Mine Operations)             |
| - Spectral absorption indices      | - Real ERA5 monthly rainfall (mm)          |
| - Terrain slope & elevation        | - Real ERA5 volumetric soil moisture (%)   |
| - Structural fault distance (km)   | - Real ERA5 peak temperature (°C)          |
| [Weather Strictly Excluded]        | - IBM planned monthly target (tonnes)      |
|                                    | - IBM ore grade decline trend (% Mn)       |
+------------------------------------+--------------------------------------------+
                                     |
                                     v
+------------------------------------+--------------------------------------------+
| EXPLORATION INTELLIGENCE           | PRODUCTION INTELLIGENCE                    |
| - Regularized XGBoost Classifier   | - Mode 1: Historical Public Predictive ML  |
| - Random Forest Baseline           | - Mode 2: Mine-Telemetry What-If Simulator |
| - Spatial GroupKFold Cross-Val     | - Cause -> Evidence -> Action Engine       |
| - Output: AI Prospectivity Zones   | - Output: Shortfall Forecasts & Actions    |
+------------------------------------+--------------------------------------------+
```

---

## 4. Verification Checksum & Governance Table

| Requirement Field | Implementation Status | Evidence / Storage Artifact | Provenance Type |
|---|---|---|---|
| **Satellite Inputs** | Verified Live / Preprocessed | `manganese_features.csv` | REAL (Copernicus Sentinel-2 L2A) |
| **Terrain Topography** | Verified Live / Preprocessed | `manganese_features.csv` | REAL (NASA SRTM GL1) |
| **Geological Occurrences** | Verified Digitized Coordinates | `labels.csv` | REAL (GSI Bhukosh & IBM Bulletins) |
| **Historical Weather** | Verified ERA5 Observations | `historical_public_data.csv` | REAL (ECMWF ERA5-Land Reanalysis) |
| **Mine Capacities** | Verified Statutory Filings | `historical_public_data.csv` | REAL / BENCHMARKED (IBM & MOIL Reports) |
| **Operational Telemetry** | Two-Mode Architecture | `shortfall_model.py` (Mode 2) | DEMO SCENARIO / SCHEMA READY |
| **Subsurface Core Assays**| Integration Layer | `geology_adapter.py` | DEMO INTEGRATION / GSI MEMOIR REFS |
| **Reserve Output Claims** | Explicit Prospectivity Labelling| `prospectivity_map.geojson` | COMPLIANT (No False Reserve Claims) |
