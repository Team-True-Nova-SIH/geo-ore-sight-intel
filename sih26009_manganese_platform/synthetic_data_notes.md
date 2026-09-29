# Layer 5 — Production Shortfall Model Synthetic Data Documentation

## Overview

This document details the generation methodology, domain constraints, and parameter ranges for the operational dataset used in **Layer 5 (Production Shortfall Forecasting)**.

> [!NOTE]
> **Transparency Statement**: While MOIL Ltd publishes annual aggregate production figures (18.02 lakh tonnes in FY 2024-25) and state-wide grade statistics, detailed month-by-month, mine-specific equipment telemetry, workforce attendance, and local micro-climate metrics are non-public proprietary assets. Therefore, Layer 5 uses a **rigorously constructed synthetic operational dataset** derived from published industry benchmarks, Indian Meteorological Department (IMD) monsoon data, and Indian Bureau of Mines (IBM) technical reports.

---

## Benchmark Anchors & Realistic Constraints

| Metric / Feature | Realistic Base Benchmark | Source / Derivation Basis |
|------------------|--------------------------|---------------------------|
| **Total Annual Target** | 1.80 Million Tonnes (18.02 Lakh Tonnes) | MOIL FY 2024-25 Official Performance Disclosure |
| **Active Mine Count** | 10 Primary Operations (Balaghat, Tirodi, Ukwa, Kandri, Munsar, Dongri Buzurg, Chikla, Gumgaon, Beldongri, Sitapatore) | MOIL Corporate Directory & Lease Maps |
| **Average Monthly Target per Mine** | 12,000 – 25,000 Tonnes / month | Distributed based on mine scale (Balaghat underground = largest, Sitapatore = smaller opencast) |
| **Ore Grade Decline Trend** | Average Mn % falling from 38.7% to 33.4% Mn | Indian Bureau of Mines (IBM) Mineral Yearbook & Under-reporting Disclosures |
| **Seasonal Monsoon Disruption** | June – September (Rainfall: 150mm – 650mm / month) | IMD Central India (Balaghat/Nagpur) Historical Climate Averages |
| **Equipment Downtime** | 20 – 180 Hours / month per mine | Typical underground shaft / dump truck maintenance logs in mining operations |
| **Workforce Availability** | 70% – 95% | Local festival seasons (Diwali, Holi) & summer heatwaves causing attendance drops |

---

## Synthetic Data Schema

Generated across **36 months (3 years)** for **10 MOIL mines** (Total = 360 monthly mine records):

1. `mine_id` & `mine_name`: Target MOIL mine.
2. `month_year`: Monthly timestamp (e.g. `2024-07`).
3. `planned_output_tonnes`: Target output allocated based on annual production target.
4. `actual_output_tonnes`: Output achieved after accounting for downtime, monsoon rain, grade drops, and delays.
5. `equipment_downtime_hours`: Heavy machinery (excavators, dumpers, shaft hoists) breakdown hours.
6. `rainfall_mm`: Monthly rainfall in mm (spikes in June–Sept).
7. `blasting_delays_count`: Number of delayed/cancelled blasting cycles due to explosive supply or weather.
8. `workforce_availability_pct`: Percentage of scheduled shifts fulfilled.
9. `ore_grade_pct`: Extracted ore grade (% Mn content), modeling the historical decline trend.
10. `shortfall_tonnes`: `planned_output_tonnes - actual_output_tonnes` (target variable).
11. `shortfall_pct`: `(shortfall_tonnes / planned_output_tonnes) * 100`.
12. `shortfall_risk_category`: Categorized as `LOW` (<5%), `MODERATE` (5-15%), or `HIGH` (>15%).

---

## Machine Learning & Action Mapping Logic

An **XGBoost Regressor** is trained to predict `shortfall_pct`. SHAP values isolate the top driver per mine-month, mapped directly to corrective recommendations:

| Top SHAP Driver | Corrective Action Recommendation |
|-----------------|----------------------------------|
| `equipment_downtime_hours` | **Deploy preventive maintenance team** & reallocate backup dumpers/hoists to primary working face. |
| `rainfall_mm` | **Activate auxiliary dewatering pumps** & shift excavation from deep opencast pit floor to elevated bench. |
| `workforce_availability_pct` | **Incentivize attendance shifts** & deploy automated haulage options during peak festival/summer periods. |
| `blasting_delays_count` | **Streamline explosive permit logistics** & transition to continuous mechanical hydraulic rock breakers. |
| `ore_grade_pct` | **Ramp up dry magnetic/density beneficiation** & blend low-grade ore with high-grade Balaghat underground reserve. |
