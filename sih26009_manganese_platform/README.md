# Geo Ore Sight Intel

## AI + Earth Observation for Manganese Exploration & Production Intelligence

### Smart India Hackathon 2026 — SIH26009

**Problem Statement:** Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls.  
**Theme:** Smart Automation  
**Category:** Software  
**Sponsor:** Ministry of Steel / MOIL Limited  
**Team:** True Nova  

---

## 1. What is Geo Ore Sight Intel?

Geo Ore Sight Intel is an AI-assisted decision-support platform built specifically for Indian manganese exploration and mine-production planning. 

The platform bridges the gap between raw satellite data and operational mining intelligence through two core workflows:

### 🌍 Exploration Intelligence (Greenfield)
Satellite, terrain, and geological indicators are transformed into machine-learning features. XGBoost and Random Forest models generate manganese prospectivity probabilities for spatial zones across India.
**The system delivers:**
- SHAP-based Explainable AI (XAI) attributions
- Model-agreement confidence scoring
- Ranked prospectivity zones for immediate field investigation
- Interactive map visualization with live geological overlays
- Historical coordinate backtesting

### ⚙️ Production Intelligence (Brownfield)
Operational variables are continuously analyzed to estimate production shortfall risks for existing mines.
**The system evaluates:**
- Equipment downtime & LHD fleet status
- Monsoon rainfall & dewatering capacity
- Workforce availability
- Ore grade decline
- Planned vs. Actual production targets

The output includes predicted shortfall percentages, critical risk drivers, and **automated corrective-action cues** (e.g., "Deploy mechanical repair crews to underground shaft hoist systems").

---

## 2. Evidence-Driven Data Sourcing

To ensure high transparency and trust, here is the explicit breakdown of every data source used in this system:

| Component | Data Source | Status / Type | Description / Origin |
|-----------|-------------|---------------|----------------------|
| **Satellite Imagery** | Google Earth Engine `COPERNICUS/S2_SR` | **Real Live Data** | Sentinel-2 L2A 10-20m surface reflectance imagery. |
| **Terrain / Topography** | GEE `USGS/SRTMGL1_003` | **Real Live Data** | NASA SRTM 30m Digital Elevation Model (DEM). |
| **Mine Coordinates** | Public Disclosures / IBM | **Real Geocoded Data** | Exact GPS coordinates for MOIL mines (Balaghat, Tirodi, Ukwa, etc). |
| **Geological Extension Labels** | GSI Bhukosh | **Real Survey Data** | Digitized Gondite horizon maps from Geological Survey of India. |
| **Annual Production Figures** | MOIL FY 24-25 Disclosures | **Real Statistics** | Total annual output benchmark of **18.02 Lakh Tonnes**. |
| **Manganese Ore Pricing** | MOIL BSE/NSE Disclosures | **Real Prices** | Base prices for EMD and Ferro-grade lump. |
| **Operational Telemetry** | Documented Synthetic | **Synthetic Data** | Month-by-month equipment downtime and workforce attendance. (MOIL mine telemetry is proprietary. Generated based on IMD monsoon records & IBM constraints). |

---

## 3. Core AI Architecture & Validation

The implementation is organized into an 8-layer architecture. 

### Exploration Model (XGBoost)
The primary exploration model utilizes XGBoost, with Random Forest serving as a baseline comparison.
**Features extracted:** Iron oxide ratio, Ferrous mineral index, Clay mineral ratio, NDVI, Manganese spectral indicator, Elevation, Slope, Aspect.

**Current Validation Snapshot:**
- **5-fold CV AUC:** ~0.919
- **Repeated 5×5 CV AUC:** ~0.917 ± 0.109
- **Leave-One-Out (LOOCV) AUC:** ~0.898
- **Spatial/Grouped CV AUC:** ~0.914 ± 0.104

*(Note: These values are evaluation results on the current labeled dataset and are intended for prototype validation. Real-world deployment requires continued geological ground-truthing).*

### Explainability (SHAP)
Instead of a "black box" prediction, the system uses SHAP (SHapley Additive exPlanations). If a zone is flagged as "High Probability", the system provides the exact evidence (e.g., high spectral indicators, specific terrain characteristics) driving that prediction.

---

## 4. API & Endpoints (FastAPI)

The entire platform is powered by a robust REST API backend. 

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/prospectivity/zones` | Ranked prospectivity zones. |
| `GET` | `/api/v1/prospectivity/zones/{id}`| Detailed SHAP feature attributions for a single zone. |
| `GET` | `/api/v1/shortfall/forecast` | Mine-level shortfall predictions. |
| `GET` | `/api/v1/shortfall/forecast/{id}` | Deep-dive telemetry shortfall analysis for a specific mine. |
| `POST` | `/api/v1/backtest` | Tests model predictions against a known historical coordinate. |
| `GET` | `/api/v1/roi/{zone_id}` | Zone-level exploration ROI analysis. |
| `POST` | `/api/v1/chat` | Queries platform information through an interactive assistant. |

---

## 5. Technology Stack

- **Earth Observation:** Sentinel-2, SRTM DEM, Google Earth Engine
- **Machine Learning:** XGBoost, Random Forest, scikit-learn
- **Explainable AI:** SHAP, Ensemble agreement
- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, JavaScript, Leaflet.js, Chart.js
- **Testing:** Pytest

---

## 6. Quick Start & Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Install Dependencies
```bash
git clone https://github.com/Team-True-Nova-SIH/geo-ore-sight-intel.git
cd geo-ore-sight-intel
pip install -r requirements.txt
```

### 3. Launch Backend API & Frontend Dashboard
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open your browser to:
- **Interactive Dashboard**: `http://localhost:8000/`
- **Swagger API Docs**: `http://localhost:8000/docs`

### 4. Run Automated Tests
```bash
pytest test_app.py -v
```

---

## 7. Important Model Limitation & Disclaimer

Geo Ore Sight Intel is a **decision-support system**. A high prospectivity probability does NOT constitute a certified mineral reserve under UNFC/CRIRSCO codes. Satellite-derived predictions must be followed by geological interpretation, field investigation, core sampling, and independent resource estimation. Similarly, production forecasts should be validated against authenticated mine telemetry before operational deployment.
