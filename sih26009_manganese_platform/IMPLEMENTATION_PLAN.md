# Implementation Plan for GeoOreSightIntel - SIH26009

## 1. Missing Satellite / Space Features (Layer 1 - Data Pipeline)
**CURRENT FEATURE**: The GEE data pipeline uses Sentinel-2 and SRTM DEM to generate features like `iron_oxide_ratio`, `ndvi`, `mn_indicator`, `elevation`, `slope`, `aspect`.
**MISSING REQUIREMENT**: Soil Moisture, Land Surface Temperature (LST).
**PROPOSED CHANGE**: Add MODIS/Landsat datasets to Google Earth Engine extraction for `soil_moisture` and `land_temperature`. Update the fallback data generator to include these features realistically.
**FILES TO MODIFY**: `data_pipeline.py`, `train_prospectivity_model.py`, `app.py`
**FILES TO CREATE**: None
**DATA SOURCE**: MOD11A2 (LST) and NASA-USDA SMAP (Soil Moisture) or synthetic fallback.
**VALIDATION METHOD**: Test that GEE correctly fetches the layers and pipeline outputs contain these columns. Run unit tests.
**RISK**: Earth Engine timeout due to increased dataset size.

## 2. Geological + Subsurface Support
**CURRENT FEATURE**: Surface indicators are mapped using remote sensing.
**MISSING REQUIREMENT**: Integration of subsurface/drilling evidence schema. Distinguish surface vs subsurface in UI.
**PROPOSED CHANGE**: Create a new adapter `GeologyDataAdapter` and data schema for geological/drilling evidence. Add mock features if no real data is available, cleanly separated and labeled as demonstrations. Map layers for drill points.
**FILES TO MODIFY**: `data_pipeline.py`, `app.py`, `index.html`
**FILES TO CREATE**: `geology_adapter.py`
**DATA SOURCE**: Demo geological dataset (JSON/CSV) structured for actual subsurface input.
**VALIDATION METHOD**: Validate adapter structure via Pydantic/unit tests. Verify UI displays subsurface evidence separately from surface evidence.
**RISK**: Adding mock data could look misleading if not clearly labeled. Will ensure strict labeling.

## 3. Improved Prospectivity Model
**CURRENT FEATURE**: Model trained using XGBoost.
**MISSING REQUIREMENT**: Stratified CV, ROC-AUC, precision/recall, model report generation. Proper handling of new valid features.
**PROPOSED CHANGE**: Update `train_prospectivity_model.py` to use `StratifiedKFold`. Add Random Forest as a baseline. Generate a comprehensive JSON report with ROC-AUC and Precision/Recall.
**FILES TO MODIFY**: `train_prospectivity_model.py`
**FILES TO CREATE**: None
**DATA SOURCE**: Extracted GEE dataset (or fallback).
**VALIDATION METHOD**: Ensure `model_report.json` outputs the new cross-validated metrics.
**RISK**: Overfitting if data sample size is small.

## 4. Production Shortfall Module Enhancement
**CURRENT FEATURE**: `shortfall_model.py` generates synthetic data with equipment downtime, rainfall, blasting delays.
**MISSING REQUIREMENT**: Add `soil_moisture` and `land_temperature` as operational constraints. Support for What-If scenario simulation via API.
**PROPOSED CHANGE**: Add soil moisture and temperature parameters to `generate_synthetic_operational_data`. Update XGBoost model. Create a new scenario simulation endpoint in FastAPI.
**FILES TO MODIFY**: `shortfall_model.py`, `app.py`
**FILES TO CREATE**: None
**DATA SOURCE**: Augmented synthetic operational dataset.
**VALIDATION METHOD**: Send varied payloads to What-If simulator endpoint and verify predictions change logically.
**RISK**: Minimal risk as data generation logic is controlled.

## 5. Constraint / Driver Analysis & Corrective Actions
**CURRENT FEATURE**: Basic SHAP-based driver extraction and static rules.
**MISSING REQUIREMENT**: Ensure all new features are included in SHAP analysis. Better rule engine that incorporates weather-aware planning and workforce constraints accurately based on the new features.
**PROPOSED CHANGE**: Expand `CORRECTIVE_ACTIONS` rules in `shortfall_model.py`. Ensure SHAP outputs provide clear reasoning.
**FILES TO MODIFY**: `shortfall_model.py`, `explain.py`
**FILES TO CREATE**: None
**DATA SOURCE**: XGBoost Explainer.
**VALIDATION METHOD**: Verify the JSON payload for shortfalls includes accurate reasons matching the simulated inputs.
**RISK**: Conflicting rules, will use a strict priority hierarchy.

## 6. Unified Dashboard Updates & API Requirements
**CURRENT FEATURE**: Basic Leaflet + Chart.js dashboard.
**MISSING REQUIREMENT**: "What-If Simulator", "Ask the Map" / QA capabilities, clear distinction of surface vs subsurface, clearly labeling demo data. Add Swagger UI and Pydantic validation.
**PROPOSED CHANGE**: Add missing APIs (`/api/v1/...`). Add a "What-If" form panel in the dashboard. Ensure all synthetic data is explicitly labeled as "Prototype Operational Telemetry".
**FILES TO MODIFY**: `app.py`, `index.html`
**FILES TO CREATE**: None
**DATA SOURCE**: FastAPI endpoints.
**VALIDATION METHOD**: Test UI manually (verify charts update when scenario sliders change). Test APIs directly via `/docs` (Swagger).
**RISK**: UI layout clutter. Will use modular tabs/panels.

## 7. Documentation
**CURRENT FEATURE**: README exists.
**MISSING REQUIREMENT**: Specific Markdown files for methodologies and limitations.
**PROPOSED CHANGE**: Create required markdown files. Update README with deployment instructions and limitations.
**FILES TO MODIFY**: `README.md`
**FILES TO CREATE**: `docs/data_sources.md`, `docs/model_methodology.md`, `docs/validation.md`, `docs/limitations.md`
**DATA SOURCE**: System implementation logic.
**VALIDATION METHOD**: Review readmes for correctness and completeness.
**RISK**: Low.
