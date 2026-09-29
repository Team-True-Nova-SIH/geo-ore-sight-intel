"""
Layer 8 — Automated Test Suite (pytest)
SIH26009 — AI Manganese Exploration Intelligence Platform

Executes comprehensive integration tests verifying FastAPI backend endpoints,
SHAP explanations, ROI calculator, and historical backtest accuracy.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from app import app, BASE_DIR

client = TestClient(app)

def test_health_endpoint():
    """Verify health check endpoint returns 200 and system metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "SIH26009" in data["system"]
    assert "Ministry of Steel" in data["sponsor"]


def test_prospectivity_zones_endpoint():
    """Verify top prospectivity zones endpoint returns valid ranked zones."""
    response = client.get("/api/v1/prospectivity/zones?min_probability=0.50")
    assert response.status_code == 200
    data = response.json()
    assert "total_zones" in data
    assert "zones" in data
    assert isinstance(data["zones"], list)
    assert len(data["zones"]) > 0
    
    # Check structure of top zone
    top_zone = data["zones"][0]
    assert "zone_id" in top_zone
    assert "probability" in top_zone
    assert top_zone["probability"] >= 0.50
    assert "confidence_level" in top_zone
    assert "top_contributing_features" in top_zone
    assert len(top_zone["top_contributing_features"]) == 3


def test_zone_detail_endpoint_valid():
    """Verify zone detail endpoint returns SHAP explanation for valid zone ID."""
    zones_resp = client.get("/api/v1/prospectivity/zones")
    zones = zones_resp.json()["zones"]
    valid_id = zones[0]["zone_id"]
    
    response = client.get(f"/api/v1/prospectivity/zones/{valid_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["zone_id"] == valid_id
    assert "explanation_text" in detail
    assert len(detail["explanation_text"]) > 10


def test_zone_detail_endpoint_invalid_404():
    """Verify zone detail endpoint returns 404 for invalid zone ID."""
    response = client.get("/api/v1/prospectivity/zones/INVALID_ZONE_999999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_shortfall_forecast_endpoint():
    """Verify shortfall forecast endpoint returns operational predictions for 10 MOIL mines."""
    response = client.get("/api/v1/shortfall/forecast")
    assert response.status_code == 200
    data = response.json()
    assert "forecast_period" in data
    assert "mine_forecasts" in data
    assert len(data["mine_forecasts"]) == 10
    
    first_mine = data["mine_forecasts"][0]
    assert "mine_id" in first_mine
    assert "predicted_shortfall_pct" in first_mine
    assert "recommended_corrective_action" in first_mine


def test_mine_shortfall_detail_valid():
    """Verify mine shortfall detail endpoint for valid mine_id (MINE_01)."""
    response = client.get("/api/v1/shortfall/forecast/MINE_01")
    assert response.status_code == 200
    data = response.json()
    assert data["mine_id"] == "MINE_01"
    assert "mine_name" in data
    assert "top_driving_factor" in data


def test_mine_shortfall_detail_invalid_404():
    """Verify mine shortfall detail returns 404 for non-existent mine_id."""
    response = client.get("/api/v1/shortfall/forecast/MINE_999")
    assert response.status_code == 404


def test_judge_backtest_endpoint_known_deposit():
    """
    CRITICAL JUDGE DEFENSE TEST:
    Verify model correctly scores Balaghat Mine (Bharveli) with high prospectivity (>0.70).
    """
    payload = {
        "deposit_name": "Balaghat Mine (Bharveli)",
        "lat": 21.8464,
        "lon": 80.2281,
        "expected_min_probability": 0.70
    }
    response = client.post("/api/v1/backtest", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "completed"
    assert res["backtest_passed"] is True
    assert res["predicted_probability"] >= 0.70
    assert "SUCCESS" in res["verdict"]


def test_roi_calculator_endpoint():
    """Verify ROI calculator endpoint computes realistic UNFC G3/G4 financial return metrics."""
    zones_resp = client.get("/api/v1/prospectivity/zones")
    zones = zones_resp.json()["zones"]
    valid_id = zones[0]["zone_id"]
    
    response = client.get(f"/api/v1/roi/{valid_id}?drilling_depth_m=150&cost_per_meter_inr=4500")
    assert response.status_code == 200
    roi = response.json()
    assert roi["zone_id"] == valid_id
    assert "drilling_cost_breakdown" in roi
    assert roi["drilling_cost_breakdown"]["total_drilling_cost_inr"] == 675000.0
    assert roi["drilling_cost_breakdown"]["total_drilling_cost_lakhs"] == 6.75
    assert "resource_yield_estimate" in roi
    assert "financial_summary" in roi
    
    # Verify realistic, non-inflated ROI multiplier (3x to 8x, NOT 1998x)
    multiplier = roi["financial_summary"]["estimated_roi_multiplier"]
    assert 2.0 <= multiplier <= 10.0, f"Expected realistic ROI multiplier between 2.0 and 10.0, got {multiplier}"
    assert "economic_model" in roi["financial_summary"]
    assert "UNFC" in roi["financial_summary"]["economic_model"]


def test_shortfall_recommendations_diversity():
    """Verify that mines have diverse, operational-specific recommendations and not uniform dewatering."""
    response = client.get("/api/v1/shortfall/forecast")
    assert response.status_code == 200
    data = response.json()
    forecasts = data["mine_forecasts"]
    
    # Check that not all mines have the same driver or recommendation
    drivers = set(m["top_driving_factor"]["feature"] for m in forecasts)
    recommendations = set(m["recommended_corrective_action"] for m in forecasts)
    
    assert len(drivers) > 1, f"Expected multiple distinct drivers, found only {drivers}"
    assert len(recommendations) > 1, f"Expected diverse corrective actions, found only {recommendations}"
    
    # In winter/dry period, rainfall should NOT be the driver for all mines
    rainfall_count = sum(1 for m in forecasts if m["top_driving_factor"]["feature"] == "rainfall_mm")
    assert rainfall_count < len(forecasts), "Rainfall should not dominate all mines during dry season"


def test_chatbot_endpoint_diverse_queries():
    """Verify 'Ask the Map' AI Chatbot endpoint answers varied domain questions."""
    test_queries = [
        "Which mine has the highest shortfall risk?",
        "Explain the ROI calculation",
        "Why were dewatering pumps recommended?",
        "Tell me about Balaghat mine",
        "What is the geology of the Sausar Group?",
        "What is the top exploration target zone?"
    ]
    
    for q in test_queries:
        response = client.post("/api/v1/chat", json={"query": q})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 30, f"Query '{q}' returned insufficient answer: {data['answer']}"


def test_data_provenance_endpoint():
    """Verify data provenance registry endpoint returns comprehensive metadata for all 7 layers."""
    response = client.get("/api/v1/provenance")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 7
    
    statuses = {d["layer"]: d["status"] for d in data["datasets"]}
    assert "REAL / PUBLIC" in statuses["Satellite Spectroscopy"]
    assert "REAL / PUBLIC" in statuses["Historical Climate Observations"]
    assert "DEMO" in statuses["Mine Operational Telemetry"]
    assert "INTEGRATION" in statuses["Subsurface Borehole Core Assays"]


def test_subsurface_integration_endpoint():
    """Verify subsurface endpoint returns GSI reference logs and integration demonstration schema."""
    response = client.get("/api/v1/exploration/subsurface")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "points" in data
    assert len(data["points"]) >= 6
    assert data["metadata"]["authenticated_gsi_records"] == 4
    
    # Check field schema
    first_pt = data["points"][0]
    required_fields = ["drill_id", "lat", "lon", "depth_m", "lithology", "ore_intersection_m", "manganese_grade_pct", "source", "confidence"]
    for rf in required_fields:
        assert rf in first_pt


def test_predict_point_gap_handling():
    """Verify point prediction endpoint gracefully handles out-of-bounds queries without false claims."""
    # Out of bounds coordinates (Bay of Bengal / South India)
    response = client.post("/api/v1/predict/point?lat=12.5&lon=80.2")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_authenticated_data"
    assert "Insufficient authenticated data" in data["message"]
    assert "data_gap_reasons" in data


def test_scenario_simulator_mode2():
    """Verify Mode 2 What-If scenario simulator returns telemetry impact and Cause->Evidence->Action."""
    payload = {
        "mine_id": "MINE_01",
        "equipment_downtime_hours": 85.0,
        "rainfall_mm": 120.0
    }
    response = client.post("/api/v1/shortfall/scenario", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mine_id"] == "MINE_01"
    assert "scenario_shortfall_pct" in data
    assert "PROTOTYPE SCENARIO" in data["provenance_flag"]
    assert "corrective_recommendation" in data
    assert "cause" in data["corrective_recommendation"]
    assert "evidence" in data["corrective_recommendation"]
    assert "action" in data["corrective_recommendation"]


