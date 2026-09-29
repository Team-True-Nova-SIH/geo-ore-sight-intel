"""
Layer 3 — Prospectivity Model (XGBoost + Random Forest Baseline)
SIH26009 — AI Manganese Exploration Intelligence Platform

SCIENTIFIC PRINCIPLES & DESIGN CORRECTIONS:
1. STRICT FEATURE SEPARATION:
   - Prospectivity Features: Spectral absorption indices (Iron oxide, Ferrous, Clay, Mn spectral index),
     Topography (SRTM DEM elevation, slope, aspect), and Geological structural proximity (fault/thrust axis distance).
   - Operational Weather Features (Rainfall, soil moisture, land surface temperature) are STRICTLY EXCLUDED
     from prospectivity modeling to avoid scientifically spurious claims (e.g. 'soil moisture proves manganese').
2. AUTHENTIC LABELS:
   - 22 positive ground-truth locations (10 active MOIL mines + 12 GSI mineralized block prospects).
   - 62 geologically justified background pseudo-negatives (sampled >8-20km away in Deccan basalts, Wainganga alluvium, and granites).
3. RIGOROUS VALIDATION:
   - Spatial GroupKFold CV (DBSCAN spatial clustering to prevent spatial autocorrelation leakage)
   - Repeated Stratified 5x5 CV (25 evaluations across random seeds)
   - Leave-One-Out CV (LOOCV)
   - Metrics: ROC-AUC, PR-AUC (Average Precision), Sensitivity (Recall), Specificity, Brier Score (Calibration)
4. OUTPUT CLASSIFICATION:
   - Strictly designated as 'AI Prospectivity / Target Zones' — NOT 'Predicted Reserves'.
   - High-prospectivity target zones prioritize ground exploration and core drilling; they are not economically mineable reserve tonnage.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import geojson
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, LeaveOneOut, cross_val_score
)
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    confusion_matrix, brier_score_loss
)
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants & Feature Specifications
# ---------------------------------------------------------------------------
REPEATED_CV_SEEDS: List[int] = [42, 123, 456, 789, 2024]
SPATIAL_CLUSTER_EPS_DEG: float = 0.05  # ~5.5 km at central India latitudes

# Baseline: Spectral Only
SPECTRAL_BASELINE_COLS: List[str] = [
    'iron_oxide_ratio', 'ferrous_mineral_index', 'clay_mineral_ratio', 'mn_indicator'
]

# Defensible Prospectivity: Spectral + Terrain + Structural Fault Proximity
PROSPECTIVITY_FEATURE_COLS: List[str] = [
    'iron_oxide_ratio', 'ferrous_mineral_index', 'clay_mineral_ratio', 'ndvi',
    'mn_indicator', 'elevation', 'slope', 'aspect', 'fault_proximity_km'
]

# Modern weather features (intentionally tracked to demonstrate why they are excluded from prospectivity)
EXCLUDED_OPERATIONAL_WEATHER: List[str] = [
    'rainfall_mm', 'soil_moisture', 'land_temperature'
]


def load_data(
    features_path: str, labels_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join pixel features with spatial ground-truth labels based on nearest proximity."""
    if not os.path.exists(features_path):
        from data_pipeline import run_data_pipeline
        features_path, _ = run_data_pipeline()

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    matched_rows = []
    for _, lbl in labels_df.iterrows():
        dists = (features_df['lat'] - lbl['lat'])**2 + (features_df['lon'] - lbl['lon'])**2
        nearest_idx = dists.idxmin()
        row = features_df.loc[nearest_idx].copy()
        row['label'] = lbl['label']
        row['mine_name'] = lbl['mine_name']
        row['label_type'] = lbl['label_type']
        row['label_lat'] = lbl['lat']
        row['label_lon'] = lbl['lon']
        matched_rows.append(row)

    training_df = pd.DataFrame(matched_rows)
    return features_df, labels_df, training_df


# ---------------------------------------------------------------------------
# Spatial Grouping & Validation Helpers
# ---------------------------------------------------------------------------

def _build_spatial_groups(
    training_df: pd.DataFrame, eps_deg: float = SPATIAL_CLUSTER_EPS_DEG
) -> Tuple[np.ndarray, int, List[Dict]]:
    """Cluster training samples spatially using DBSCAN to guard against autocorrelation leakage."""
    coords = training_df[['label_lat', 'label_lon']].values
    dist_matrix = squareform(pdist(coords, metric='euclidean'))

    proximity_pairs: List[Dict] = []
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < eps_deg:
                proximity_pairs.append({
                    "sample_a": int(i),
                    "sample_b": int(j),
                    "distance_deg": round(float(dist_matrix[i, j]), 5),
                    "approx_km": round(float(dist_matrix[i, j]) * 111, 1),
                    "labels": f"{int(training_df.iloc[i]['label'])}–{int(training_df.iloc[j]['label'])}"
                })

    clustering = DBSCAN(eps=eps_deg, min_samples=1, metric='precomputed')
    groups = clustering.fit_predict(dist_matrix)
    n_groups = len(set(groups))
    return groups, n_groups, proximity_pairs


def _spatial_cv(
    model_cls, model_params: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Tuple[float, float, List[float]]:
    """Grouped k-fold CV using spatial clusters as groups."""
    n_unique = len(set(groups))
    n_splits = min(n_unique, 5)
    if n_splits < 2:
        return float('nan'), float('nan'), []

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: List[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        y_test = y[test_idx]
        if len(set(y_test)) < 2:
            continue
        model = model_cls(**model_params)
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        fold_scores.append(float(roc_auc_score(y_test, probs)))

    if not fold_scores:
        return float('nan'), float('nan'), []
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores


def _repeated_stratified_cv(
    model_cls, model_params: dict, X: np.ndarray, y: np.ndarray,
    n_splits: int = 5, seeds: List[int] = None
) -> Tuple[float, float, List[float]]:
    """Run stratified k-fold CV across multiple random seeds."""
    seeds = seeds or REPEATED_CV_SEEDS
    all_scores: List[float] = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model = model_cls(**{**model_params, "random_state": seed})
        fold_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        all_scores.extend(fold_scores.tolist())
    return float(np.mean(all_scores)), float(np.std(all_scores)), all_scores


def _loocv_predictions(
    model_cls, model_params: dict, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Leave-One-Out CV: train on N-1, predict the held-out sample."""
    loo = LeaveOneOut()
    loo_preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = model_cls(**model_params)
        model.fit(X[train_idx], y[train_idx])
        loo_preds[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return loo_preds


# ---------------------------------------------------------------------------
# Training and Comparative Evaluation
# ---------------------------------------------------------------------------

def train_and_eval_models(
    training_df: pd.DataFrame, feature_cols: list
) -> Tuple[Any, Any, Dict]:
    """
    Trains and compares:
      1. Baseline Model: Random Forest (Spectral Only)
      2. Improved Defensible Model: XGBoost (Geological Structure + Spectral + Topography)
    Computes ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier score, and Spatial CV.
    """
    y = training_df['label'].values.astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    groups, n_groups, proximity_pairs = _build_spatial_groups(training_df)

    # ------------------------------------------------------------------
    # 1. BASELINE MODEL: Random Forest (Spectral Only)
    # ------------------------------------------------------------------
    baseline_cols = [c for c in SPECTRAL_BASELINE_COLS if c in training_df.columns]
    X_base = training_df[baseline_cols].values

    rf_base_params = dict(
        n_estimators=100, max_depth=4, min_samples_leaf=3,
        min_samples_split=5, max_features='sqrt',
        class_weight='balanced', random_state=42
    )
    rf_base_model = RandomForestClassifier(**rf_base_params)

    # Cross-validation for baseline
    rf_base_cv = cross_val_score(rf_base_model, X_base, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    rf_base_rep_mean, rf_base_rep_std, _ = _repeated_stratified_cv(RandomForestClassifier, rf_base_params, X_base, y)
    rf_base_loo_preds = _loocv_predictions(RandomForestClassifier, rf_base_params, X_base, y)
    rf_base_spatial_mean, rf_base_spatial_std, _ = _spatial_cv(RandomForestClassifier, rf_base_params, X_base, y, groups)

    rf_base_model.fit(X_base, y)
    rf_base_train_probs = rf_base_model.predict_proba(X_base)[:, 1]

    # Metrics for baseline
    base_roc_auc = float(roc_auc_score(y, rf_base_loo_preds))
    base_pr_auc = float(average_precision_score(y, rf_base_loo_preds))
    base_binary_preds = (rf_base_loo_preds >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, base_binary_preds).ravel()
    base_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    base_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    base_brier = float(brier_score_loss(y, rf_base_loo_preds))

    # ------------------------------------------------------------------
    # 2. IMPROVED MODEL: XGBoost (Geological Structure + Spectral + Topography)
    # ------------------------------------------------------------------
    X_imp = training_df[feature_cols].values

    xgb_params = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )
    xgb_model = xgb.XGBClassifier(**xgb_params)

    # Cross-validation for improved XGBoost
    xgb_cv = cross_val_score(xgb_model, X_imp, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    xgb_rep_mean, xgb_rep_std, _ = _repeated_stratified_cv(xgb.XGBClassifier, xgb_params, X_imp, y)
    xgb_loo_preds = _loocv_predictions(xgb.XGBClassifier, xgb_params, X_imp, y)
    xgb_spatial_mean, xgb_spatial_std, xgb_spatial_scores = _spatial_cv(xgb.XGBClassifier, xgb_params, X_imp, y, groups)

    xgb_model.fit(X_imp, y)
    xgb_train_probs = xgb_model.predict_proba(X_imp)[:, 1]

    # Metrics for improved model
    imp_roc_auc = float(roc_auc_score(y, xgb_loo_preds))
    imp_pr_auc = float(average_precision_score(y, xgb_loo_preds))
    imp_binary_preds = (xgb_loo_preds >= 0.50).astype(int)
    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, imp_binary_preds).ravel()
    imp_sensitivity = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
    imp_specificity = float(tn_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else 0.0
    imp_brier = float(brier_score_loss(y, xgb_loo_preds))

    # Feature importances
    xgb_importances = dict(zip(feature_cols, [float(x) for x in xgb_model.feature_importances_]))
    rf_base_importances = dict(zip(baseline_cols, [float(x) for x in rf_base_model.feature_importances_]))

    # Measured improvements
    auc_gain = imp_roc_auc - base_roc_auc
    pr_gain = imp_pr_auc - base_pr_auc
    spatial_gain = (xgb_spatial_mean - rf_base_spatial_mean) if not np.isnan(xgb_spatial_mean) and not np.isnan(rf_base_spatial_mean) else 0.0

    report = {
        "scientific_framework": {
            "prospectivity_definition": "AI Prospectivity / Target Zones for Geological Core Drilling",
            "reserve_disclaimer": (
                "Prospectivity scores represent relative favorability for field investigation. "
                "They do NOT constitute UNFC/JORC mineral reserve tonnage."
            ),
            "excluded_features": EXCLUDED_OPERATIONAL_WEATHER,
            "exclusion_rationale": (
                "Modern climate and rainfall do not dictate Proterozoic gondite mineralization. "
                "Weather features are strictly relegated to Layer 5 Production Shortfall Modeling."
            )
        },
        "sample_summary": {
            "total_samples": len(y),
            "positive_occurrences": n_pos,
            "negative_background_points": n_neg,
            "scale_pos_weight": float(scale_pos_weight),
            "spatial_clusters_dbscan": n_groups
        },
        "baseline_model": {
            "name": "Random Forest (Spectral Only)",
            "features_used": baseline_cols,
            "train_auc": float(roc_auc_score(y, rf_base_train_probs)),
            "5fold_cv_auc_mean": float(np.mean(rf_base_cv)),
            "repeated_cv_auc_mean": rf_base_rep_mean,
            "repeated_cv_auc_std": rf_base_rep_std,
            "loocv_roc_auc": base_roc_auc,
            "loocv_pr_auc": base_pr_auc,
            "sensitivity_recall": base_sensitivity,
            "specificity": base_specificity,
            "brier_calibration_score": base_brier,
            "spatial_cv_auc_mean": rf_base_spatial_mean,
            "spatial_cv_auc_std": rf_base_spatial_std,
            "feature_importances": rf_base_importances
        },
        "improved_model": {
            "name": "XGBoost (Geological Structure + Spectral + Topography)",
            "features_used": feature_cols,
            "train_auc": float(roc_auc_score(y, xgb_train_probs)),
            "5fold_cv_auc_mean": float(np.mean(xgb_cv)),
            "repeated_cv_auc_mean": xgb_rep_mean,
            "repeated_cv_auc_std": xgb_rep_std,
            "loocv_roc_auc": imp_roc_auc,
            "loocv_pr_auc": imp_pr_auc,
            "sensitivity_recall": imp_sensitivity,
            "specificity": imp_specificity,
            "brier_calibration_score": imp_brier,
            "spatial_cv_auc_mean": xgb_spatial_mean,
            "spatial_cv_auc_std": xgb_spatial_std,
            "spatial_cv_n_groups": n_groups,
            "feature_importances": xgb_importances,
            "hyperparameters": {
                "max_depth": 3, "min_child_weight": 3, "reg_alpha": 0.5,
                "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8
            }
        },
        "measured_validation_comparison": {
            "roc_auc_gain": round(auc_gain, 4),
            "pr_auc_gain": round(pr_gain, 4),
            "spatial_cv_auc_gain": round(spatial_gain, 4),
            "improved_calibration_brier_diff": round(imp_brier - base_brier, 4),
            "verdict": (
                f"Adding geological structural proximity (fault proximity) and terrain indices to spectral ratios "
                f"improved LOOCV ROC-AUC by {auc_gain:+.4f} (from {base_roc_auc:.4f} to {imp_roc_auc:.4f}) and "
                f"PR-AUC by {pr_gain:+.4f} (from {base_pr_auc:.4f} to {imp_pr_auc:.4f})."
            )
        }
    }

    _print_comparison_table(report)
    return xgb_model, rf_base_model, report


def _print_comparison_table(report: Dict[str, Any]):
    """Prints rigorous scientific comparison table to terminal."""
    b = report["baseline_model"]
    m = report["improved_model"]
    c = report["measured_validation_comparison"]

    print("\n" + "=" * 80)
    print("  SCIENTIFIC MODEL COMPARISON: BASELINE vs IMPROVED PROSPECTIVITY PIPELINE")
    print("=" * 80)
    print(f"{'Validation Metric':<36} {'Baseline (Spectral Only)':>20} {'Improved (Struct+Spectral)':>22}")
    print("-" * 80)
    print(f"{'Model Architecture':<36} {'Random Forest':>20} {'Regularized XGBoost':>22}")
    print(f"{'Feature Count':<36} {len(b['features_used']):>20} {len(m['features_used']):>22}")
    print(f"{'LOOCV ROC-AUC':<36} {b['loocv_roc_auc']:>20.4f} {m['loocv_roc_auc']:>22.4f}")
    print(f"{'LOOCV PR-AUC (Precision-Recall)':<36} {b['loocv_pr_auc']:>20.4f} {m['loocv_pr_auc']:>22.4f}")
    print(f"{'Sensitivity / Recall (Class 1)':<36} {b['sensitivity_recall']:>20.4f} {m['sensitivity_recall']:>22.4f}")
    print(f"{'Specificity (Class 0)':<36} {b['specificity']:>20.4f} {m['specificity']:>22.4f}")
    print(f"{'Calibration (Brier Score, lower=better)':<36} {b['brier_calibration_score']:>20.4f} {m['brier_calibration_score']:>22.4f}")
    print(f"{'Repeated 5x5 Stratified CV':<36} {b['repeated_cv_auc_mean']:>14.4f}+/-{b['repeated_cv_auc_std']:.3f} {m['repeated_cv_auc_mean']:>16.4f}+/-{m['repeated_cv_auc_std']:.3f}")
    print(f"{'Spatial GroupKFold CV (' + str(m['spatial_cv_n_groups']) + ' groups)':<36} {b['spatial_cv_auc_mean']:>14.4f}+/-{b['spatial_cv_auc_std']:.3f} {m['spatial_cv_auc_mean']:>16.4f}+/-{m['spatial_cv_auc_std']:.3f}")
    print("-" * 80)
    print(f"  MEASURED DELTA: ROC-AUC {c['roc_auc_gain']:+.4f} | PR-AUC {c['pr_auc_gain']:+.4f} | Spatial CV {c['spatial_cv_auc_gain']:+.4f}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Prospectivity Map Generation (GeoJSON Target Zones)
# ---------------------------------------------------------------------------

def generate_prospectivity_map(
    xgb_model: Any,
    rf_model: Any,
    features_df: pd.DataFrame,
    feature_cols: list,
    output_dir: str = "."
) -> str:
    """Predict prospectivity for raster pixels and export GeoJSON Target Zones."""
    # Ensure all feature cols exist in features_df
    X_full = features_df[feature_cols].values

    xgb_probs = xgb_model.predict_proba(X_full)[:, 1]
    baseline_cols = [c for c in SPECTRAL_BASELINE_COLS if c in features_df.columns]
    rf_probs = rf_model.predict_proba(features_df[baseline_cols].values)[:, 1]

    features_df['prospectivity_prob'] = np.round(xgb_probs, 4)
    features_df['rf_prob'] = np.round(rf_probs, 4)
    features_df['ensemble_diff'] = np.round(np.abs(xgb_probs - rf_probs), 4)

    # Uncertainty / Agreement Confidence Level
    features_df['confidence'] = np.where(features_df['ensemble_diff'] < 0.15, 'HIGH',
                                np.where(features_df['ensemble_diff'] < 0.30, 'MEDIUM', 'LOW'))

    top_df = features_df.sort_values(by='prospectivity_prob', ascending=False).head(150)

    geojson_features = []
    for idx, row in top_df.iterrows():
        geom = geojson.Point((float(row['lon']), float(row['lat'])))
        props = {
            "zone_id": f"ZONE_{int(row['pixel_id'])}",
            "target_type": "AI Prospectivity / Exploration Target Zone",
            "lat": float(row['lat']),
            "lon": float(row['lon']),
            "probability": float(row['prospectivity_prob']),
            "confidence": str(row['confidence']),
            "scientific_disclaimer": "Prioritizes surface area for diamond core drilling; not an estimate of mineable reserve tonnage.",
            "iron_oxide_ratio": float(row.get('iron_oxide_ratio', 0)),
            "ferrous_mineral_index": float(row.get('ferrous_mineral_index', 0)),
            "clay_mineral_ratio": float(row.get('clay_mineral_ratio', 0)),
            "elevation_m": float(row.get('elevation', 0)),
            "slope_deg": float(row.get('slope', 0)),
            "fault_proximity_km": float(row.get('fault_proximity_km', 0))
        }
        geojson_features.append(geojson.Feature(geometry=geom, properties=props))

    fc = geojson.FeatureCollection(geojson_features)
    geojson_path = os.path.join(output_dir, "prospectivity_map.geojson")

    with open(geojson_path, 'w') as f:
        geojson.dump(fc, f, indent=2)

    features_df.to_csv(os.path.join(output_dir, "full_prospectivity_predictions.csv"), index=False)
    return geojson_path


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

def run_training_pipeline(output_dir: str = ".") -> Tuple[str, str, Dict]:
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "manganese_features.csv")
    labels_path = os.path.join(output_dir, "labels.csv")

    features_df, labels_df, training_df = load_data(features_path, labels_path)

    # Use defensible prospectivity feature list
    feature_cols = [c for c in PROSPECTIVITY_FEATURE_COLS if c in features_df.columns]

    xgb_model, rf_model, report = train_and_eval_models(training_df, feature_cols)

    # Save Model Artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            "xgb_model": xgb_model,
            "rf_model": rf_model,
            "feature_cols": feature_cols,
            "model_type": "Regularized XGBoost Classifier + RF Baseline",
            "target_type": "AI Prospectivity / Target Zones"
        }, f)

    report_path = os.path.join(output_dir, "model_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    geojson_path = generate_prospectivity_map(xgb_model, rf_model, features_df, feature_cols, output_dir)

    print("LAYER 3 PROSPECTIVITY MODEL TRAINING COMPLETE")
    print(f"  Model Serialized To : {model_path}")
    print(f"  Report Serialized To: {report_path}")
    print(f"  GeoJSON Map Output  : {geojson_path}")

    return model_path, report_path, report


if __name__ == "__main__":
    run_training_pipeline()
