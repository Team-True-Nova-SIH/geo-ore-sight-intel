"""
Layer 4 — Explainability & Uncertainty Analysis (SHAP + Ensemble Confidence)
SIH26009 — AI Manganese Exploration Intelligence Platform

Computes per-prediction SHAP feature attributions and ensemble agreement scores
for top-ranked manganese prospectivity zones.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Any

FEATURE_NAME_MAP = {
    'iron_oxide_ratio': 'Iron Oxide Ratio (Band4/Band2)',
    'ferrous_mineral_index': 'Ferrous Mineral Index (SWIR1/NIR)',
    'clay_mineral_ratio': 'Clay Mineral Ratio (SWIR1/SWIR2)',
    'ndvi': 'Vegetation Cover (NDVI)',
    'mn_indicator': 'Manganese Spectral Indicator',
    'elevation': 'Terrain Elevation (m)',
    'slope': 'Slope Gradient (degrees)',
    'aspect': 'Aspect Direction',
    'fault_proximity_km': 'Proximity to Known Fault / Belt Axis (km)'
}

def generate_plain_text_explanation(row: pd.Series, top_features: list) -> str:
    """Translates SHAP feature importances into clear geologist-friendly text."""
    explanations = []
    
    for feat_name, shap_val, feat_val in top_features:
        if feat_name == 'iron_oxide_ratio':
            if feat_val > 0.35:
                explanations.append("Strong iron-oxide spectral signature indicating surface gossan/manganese mineralization")
            else:
                explanations.append("Moderate iron oxide spectral response")
                
        elif feat_name == 'ferrous_mineral_index':
            if feat_val > 1.2:
                explanations.append("High ferrous mineral concentration detected in SWIR reflectance")
            else:
                explanations.append("Elevated ferrous mineral index")
                
        elif feat_name == 'clay_mineral_ratio':
            if feat_val > 1.3:
                explanations.append("Hydrothermal clay alteration envelope surrounding ore horizon")
            else:
                explanations.append("Presence of clay mineral alteration")
                
        elif feat_name == 'ndvi':
            if feat_val < 0.30:
                explanations.append("Low vegetation cover exposing bare rock and outcrop surfaces")
            else:
                explanations.append("Dense vegetation masking surface bedrock signal")
                
        elif feat_name == 'slope':
            if feat_val > 12.0:
                explanations.append("Steep slope topography exposing unweathered ore strike lines")
            else:
                explanations.append("Flat to gentle slope topography")
                
        elif feat_name == 'elevation':
            if feat_val > 350:
                explanations.append("High elevation plateau matching Sausar Group stratigraphic horizon")
            else:
                explanations.append("Moderate elevation valley floor")
                
        elif feat_name == 'mn_indicator':
            explanations.append("High multi-band manganese spectral proxy score")
            
        else:
            explanations.append(f"{FEATURE_NAME_MAP.get(feat_name, feat_name)} value of {feat_val:.2f}")
            
    return ". ".join(explanations[:3]) + "."

def run_explainability_pipeline(output_dir: str = ".") -> Tuple[str, Dict]:
    """Computes SHAP explanations and generates zone_explanations.json."""
    model_path = os.path.join(output_dir, "model.pkl")
    preds_path = os.path.join(output_dir, "full_prospectivity_predictions.csv")
    
    if not os.path.exists(model_path) or not os.path.exists(preds_path):
        from train_prospectivity_model import run_training_pipeline
        run_training_pipeline(output_dir)
        
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        
    xgb_model = artifacts['xgb_model']
    rf_model = artifacts['rf_model']
    feature_cols = artifacts['feature_cols']
    
    df = pd.read_csv(preds_path)
    
    # Select top 50 high-prospectivity zones for detailed SHAP explanation
    top_df = df.sort_values(by='prospectivity_prob', ascending=False).head(50).copy()
    X_top = top_df[feature_cols].values
    
    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_top)
    
    # Handle binary classification output shape differences in SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    zone_explanations = []
    
    for i, (idx, row) in enumerate(top_df.iterrows()):
        row_shap = shap_values[i]
        
        # Sort features by absolute SHAP impact
        top_feat_indices = np.argsort(np.abs(row_shap))[::-1][:3]
        top_features_info = []
        
        for f_idx in top_feat_indices:
            feat_name = feature_cols[f_idx]
            shap_val = float(row_shap[f_idx])
            feat_val = float(row[feat_name])
            top_features_info.append({
                "feature": feat_name,
                "display_name": FEATURE_NAME_MAP.get(feat_name, feat_name),
                "value": round(feat_val, 4),
                "shap_impact": round(shap_val, 4)
            })
            
        plain_text = generate_plain_text_explanation(row, [(f["feature"], f["shap_impact"], f["value"]) for f in top_features_info])
        
        xgb_p = float(row['prospectivity_prob'])
        rf_p = float(row['rf_prob'])
        diff = abs(xgb_p - rf_p)
        
        if diff < 0.15:
            conf_str = "HIGH"
            conf_desc = "XGBoost and Random Forest models strongly agree on high prospectivity."
        elif diff < 0.30:
            conf_str = "MEDIUM"
            conf_desc = "Moderate model agreement; field geological validation recommended."
        else:
            conf_str = "LOW"
            conf_desc = "Model divergence detected; high structural uncertainty."
            
        zone_explanations.append({
            "zone_id": f"ZONE_{int(row['pixel_id'])}",
            "lat": float(row['lat']),
            "lon": float(row['lon']),
            "probability": xgb_p,
            "rf_baseline_probability": rf_p,
            "ensemble_divergence": round(diff, 4),
            "confidence_level": conf_str,
            "confidence_description": conf_desc,
            "top_contributing_features": top_features_info,
            "explanation_text": plain_text
        })
        
    output_json_path = os.path.join(output_dir, "zone_explanations.json")
    with open(output_json_path, 'w') as f:
        json.dump({"total_zones_explained": len(zone_explanations), "zones": zone_explanations}, f, indent=2)
        
    print("\n" + "="*60)
    print("LAYER 4 EXPLAINABILITY & UNCERTAINTY SUMMARY")
    print(f"Total Zones Explained: {len(zone_explanations)}")
    print(f"Top Zone ID          : {zone_explanations[0]['zone_id']} (Prob: {zone_explanations[0]['probability']})")
    print(f"Top Zone Drivers     : {[f['feature'] for f in zone_explanations[0]['top_contributing_features']]}")
    print(f"Explanation Output   : {output_json_path}")
    print("="*60 + "\n")
    
    return output_json_path, {"zones": zone_explanations}

if __name__ == "__main__":
    run_explainability_pipeline()
