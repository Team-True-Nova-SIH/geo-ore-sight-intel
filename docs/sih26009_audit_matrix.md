# SIH26009 Data Authenticity & Compliance Audit

## A. DATA AUTHENTICITY AUDIT MATRIX

| REQUIREMENT | CURRENT SOURCE | REAL/PUBLIC/SYNTHETIC | USED BY MODEL? | EVIDENCE | GAP |
|-------------|----------------|------------------------|----------------|----------|-----|
| Geological data | GSI Bhukosh, MOIL Mine locs | REAL/PUBLIC | YES | `labels.csv` coordinates | Limited spatial density |
| Historical production | Target vs Actual approximations | SYNTHETIC | YES | `shortfall_model.py` simulation | Needs MOIL ERP data |
| Equipment performance | Equipment downtime simulation | SYNTHETIC | NO (Moved to Simulator) | Excluded from historical model | Needs MOIL SCADA/Telemetry |
| Satellite: rainfall | Synthetic / GEE extraction | SYNTHETIC/REAL | YES | Simulated seasonality / GEE logic | |
| Satellite: soil moisture | TerraClimate (GEE) / Synthetic | SYNTHETIC/REAL | YES | `data_pipeline.py` | |
| Satellite: vegetation index| Sentinel-2 NDVI (GEE) / Synthetic | SYNTHETIC/REAL | YES | `data_pipeline.py` | |
| Satellite: land temp | MODIS LST (GEE) / Synthetic | SYNTHETIC/REAL | YES | `data_pipeline.py` | |
| Manganese reserve mapping | Topographical/Spectral indicators | REAL/PUBLIC | YES | `train_prospectivity_model.py` | Cannot be called "Reserves", just "Prospectivity" |
| Production shortfall | Historical environmental factors | SYNTHETIC | YES | `shortfall_model.py` | Awaiting actual production logs |
| Corrective actions | Rule-based decision engine | - | - | Triggered by SHAP/Simulator | |
| User-friendly dashboard | Web Application | REAL | - | `index.html` | Needs clear data provenance flags |

## B. FINAL SIH26009 REQUIREMENT COVERAGE

| SIH REQUIREMENT | DATA SOURCE | IMPLEMENTATION | MODEL | OUTPUT | VALIDATION STATUS |
|-----------------|-------------|----------------|-------|--------|-------------------|
| Identify reserves/prospectivity (surface/subsurface) | Sentinel-2, SRTM, GSI | Prospectivity mapping | XGBoost Spatial | `prospectivity_map.geojson` | 🟨 YELLOW (Awaiting true subsurface drill records) |
| Predict production shortfalls | Historical constraints | Predictive Model | XGBoost Regressor | `shortfall_predictions.json`| 🟨 YELLOW (Awaiting historical telemetry) |
| Suggest corrective actions | Expert rules | Cause->Evidence->Action Engine | SHAP / Rule-based | Dashboard Recommendations | 🟩 GREEN |
| Dashboard visualization | - | Web UI | - | Interactive Maps & Charts | 🟩 GREEN |
| Space Technology Integration | GEE | Automated Pipeline | - | Spectral/Environmental Features | 🟩 GREEN |

## C. FINAL OBJECTIVE SUMMARY

1. **Which SIH26009 requirements are genuinely satisfied:** 
   Space technology integration (NDVI, soil moisture, LST, iron oxide, topography), spatial prospectivity mapping (using real known mine locations), dashboard visualization, rule-based corrective action logic, and "What-If" scenario planning.
2. **Which still require authenticated MOIL/mine data:** 
   Production shortfall forecasting requires real historical production telemetry. Subsurface resource estimation requires actual authenticated drill core assays to upgrade "Prospectivity" to "Reserves".
3. **Exact real datasets used:**
   - Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)
   - SRTM DEM (USGS/SRTMGL1_003)
   - MODIS LST (MOD11A2.061)
   - TerraClimate (IDAHO_EPSCOR/TERRACLIMATE)
   - Known MOIL Mine Locations (labels.csv, sourced from GSI/MOIL Annual Reports)
4. **Exact synthetic/demo datasets remaining:**
   - Simulated operational telemetry (downtime, blasting delays) for the What-If Simulator
   - Mock subsurface drill core assays in `geology_adapter.py`
5. **Final model features:**
   - *Prospectivity Model:* iron_oxide_ratio, ferrous_mineral_index, clay_mineral_ratio, ndvi, mn_indicator, elevation, slope, aspect. (Environmental features removed to prevent false scientific claims).
   - *Shortfall Model:* rainfall_mm, ore_grade_pct, planned_output_tonnes, soil_moisture_pct, land_temperature_c.
6. **Final training/validation methodology:**
   - *Prospectivity:* Spatial GroupKFold CV (DBSCAN clustered), Repeated Stratified 5x5 CV, LOOCV.
   - *Shortfall:* Time-series splitting on environmental features.
7. **Actual validation metrics:**
   - XGBoost Train AUC: ~0.99
   - XGBoost Repeated CV AUC: ~0.94 (depending on noise levels)
   - Shortfall R2: Dependent on environmental variance (synthetic data placeholder).
8. **Whether the system should call its output “prospectivity” or “reserves”:**
   It is strictly called **"AI Prospectivity / Target Zones"**. It does NOT estimate economically mineable reserves.
9. **What changed in the architecture:**
   - Explicit separation of Prospectivity Features (geological/spectral) and Environmental Features (rainfall, temp).
   - The Production module now uses a historical model (weather/grade) and a separate What-If Simulator for unauthenticated telemetry.
   - Added Data Provenance panel to the UI.
10. **What I should show judges during the live demo:**
   - Show the Data Provenance Panel first to establish scientific honesty.
   - Show the Prospectivity Map and explain how spatial CV prevents data leakage.
   - Show the Shortfall Predictions based on environmental constraints.
   - Use the What-If Simulator to demonstrate how future MOIL telemetry (downtime, blasting) will integrate perfectly into the API.
