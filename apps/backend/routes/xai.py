from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import joblib
from typing import Optional

router = APIRouter(prefix="/xai", tags=["Explainable AI"])

# Helper paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

def load_gold_data(horizon: int) -> pd.DataFrame:
    """Loads the Gold feature table for a given horizon."""
    steps = {6: 12, 12: 24, 24: 48}[horizon]
    gold_dir = DATA_DIR / "gold" / f"gold_features_h{steps}"
    if not gold_dir.exists():
        # Fallback to default gold folder if partitioned one is missing
        gold_dir = DATA_DIR / "gold" / "gold_features"
        
    files = glob.glob(str(gold_dir / "**" / "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No Gold feature parquets found in {gold_dir}")
        
    df = pd.concat([pd.read_parquet(f) for f in files])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    
    # Calculate capacity factor features
    if "embedded_solar_mw" in df.columns and "embedded_solar_capacity_mw" in df.columns:
        df["embedded_solar_cf"] = (df["embedded_solar_mw"] / np.clip(df["embedded_solar_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
    if "embedded_wind_mw" in df.columns and "embedded_wind_capacity_mw" in df.columns:
        df["embedded_wind_cf"] = (df["embedded_wind_mw"] / np.clip(df["embedded_wind_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
        
    return df.sort_values("timestamp_utc").reset_index(drop=True)

@router.get("/global")
def get_global_importance(horizon: int = Query(24, enum=[6, 12, 24])):
    """Fetches global feature importances from the trained LightGBM models."""
    try:
        steps = {6: 12, 12: 24, 24: 48}[horizon]
        folder = "model" if steps == 48 else f"model_h{steps}"
        stack_path = ARTIFACTS_DIR / folder / "stack.joblib"
        
        if not stack_path.exists():
            # If actual weights don't exist, return default/mock feature importances for demo
            mock_importances = [
                {"feature": "ssrd_uk", "importance": 450, "description": "Downwelling shortwave radiation (Solar Driver)"},
                {"feature": "clearsky_cos", "importance": 380, "description": "Cosine of Solar Zenith Angle (Physical Solar Gating)"},
                {"feature": "solar_elevation_deg", "importance": 310, "description": "Sun elevation angle (Solar Centroid)"},
                {"feature": "tcc_uk", "importance": 280, "description": "Total cloud cover (UK weighted)"},
                {"feature": "gen_lag_48", "importance": 220, "description": "Generation 24h ago (historic baseline)"},
                {"feature": "embedded_solar_mw", "importance": 180, "description": "NESO embedded solar forecast"},
                {"feature": "t2m_uk", "importance": 150, "description": "2m temperature (UK weighted)"},
                {"feature": "hour", "importance": 120, "description": "Hour of day"},
                {"feature": "lcc_uk", "importance": 95, "description": "Low cloud cover (UK weighted)"},
                {"feature": "ws10_uk", "importance": 60, "description": "10m wind speed (UK weighted)"}
            ]
            return {"status": "success", "is_mock": True, "importances": mock_importances}

        art = joblib.load(stack_path)
        features = art["features"]
        lgbm = art["lgbm"]
        
        # Get importances for q50 (median)
        importances_dict = lgbm.feature_importance()
        q50_importances = importances_dict.get(0.5, np.zeros(len(features)))
        
        # Map to descriptions
        descriptions = {
            "ssrd_uk": "Downwelling shortwave radiation (Solar Driver)",
            "clearsky_cos": "Cosine of Solar Zenith Angle (Physical Solar Gating)",
            "solar_elevation_deg": "Sun elevation angle (Solar Centroid)",
            "tcc_uk": "Total cloud cover (UK weighted)",
            "lcc_uk": "Low cloud cover (UK weighted)",
            "t2m_uk": "2m temperature (UK weighted)",
            "ws10_uk": "10m wind speed (UK weighted)",
            "hour": "Hour of day",
            "half_hour": "Half-hour index",
            "dow": "Day of week",
            "month": "Month of year",
            "is_daylight": "Daylight binary gate",
            "capacity_mwp": "Installed capacity",
            "embedded_solar_mw": "NESO embedded solar forecast",
            "embedded_wind_mw": "NESO embedded wind forecast"
        }
        
        # Sort and return
        res = []
        for feat, val in zip(features, q50_importances):
            desc = descriptions.get(feat, "Model Feature")
            if feat.startswith("gen_lag_"):
                desc = f"Generation {int(feat.split('_')[-1])//2}h ago (historic baseline)"
            elif feat.startswith("cf_lag_"):
                desc = f"Capacity factor {int(feat.split('_')[-1])//2}h ago"
            elif feat.startswith("gen_roll_"):
                desc = "Rolling average generation"
                
            res.append({
                "feature": feat,
                "importance": int(val),
                "description": desc
            })
            
        res = sorted(res, key=lambda x: x["importance"], reverse=True)[:15] # Top 15 features
        return {"status": "success", "is_mock": False, "importances": res}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving global feature importances: {str(e)}")

@router.get("/local")
def get_local_contributions(
    timestamp: str,
    horizon: int = Query(24, enum=[6, 12, 24])
):
    """Calculates local feature contributions (SHAP) for Model A (LightGBM) at a specific timestamp."""
    try:
        ts = pd.to_datetime(timestamp)
        steps = {6: 12, 12: 24, 24: 48}[horizon]
        folder = "model" if steps == 48 else f"model_h{steps}"
        stack_path = ARTIFACTS_DIR / folder / "stack.joblib"
        
        # 1. Load Gold row
        try:
            gold = load_gold_data(horizon)
            row = gold[gold["timestamp_utc"] == ts]
        except Exception:
            row = pd.DataFrame()
            
        if row.empty:
            # If no Gold data row is found, return mock contributions
            hour = ts.hour
            is_night = (hour < 6) | (hour > 19)
            
            if is_night:
                mock_contrib = [
                    {"feature": "is_daylight", "contribution": -0.85, "value": 0.0, "description": "Sun is below horizon (forcing forecast to 0)"},
                    {"feature": "clearsky_cos", "contribution": -0.15, "value": 0.0, "description": "Theoretical insolation factor"}
                ]
                return {
                    "status": "success",
                    "is_mock": True,
                    "base_value": 0.0,
                    "prediction": 0.0,
                    "contributions": mock_contrib
                }
            else:
                # Daytime mock values
                mock_contrib = [
                    {"feature": "ssrd_uk", "contribution": 0.35, "value": 420.0, "description": "Downwelling shortwave radiation (Solar Driver)"},
                    {"feature": "clearsky_cos", "contribution": 0.20, "value": 0.75, "description": "Theoretical insolation factor"},
                    {"feature": "embedded_solar_mw", "contribution": 0.12, "value": 2400.0, "description": "NESO embedded solar forecast"},
                    {"feature": "tcc_uk", "contribution": -0.15, "value": 0.65, "description": "Cloud cover damping factor"},
                    {"feature": "t2m_uk", "contribution": 0.05, "value": 288.5, "description": "Dampening temperature effect"},
                    {"feature": "gen_lag_48", "contribution": 0.03, "value": 2100.0, "description": "Historic generation baseline"}
                ]
                return {
                    "status": "success",
                    "is_mock": True,
                    "base_value": 0.15,
                    "prediction": 0.70,
                    "contributions": mock_contrib
                }
                
        # 2. Get features used by LGBM
        if not stack_path.exists():
            # Return custom mock contributions based on actual row values
            hour = ts.hour
            is_night = (hour < 6) | (hour > 19)
            ssrd = float(row["ssrd_uk"].iloc[0]) if "ssrd_uk" in row.columns else 0.0
            tcc = float(row["tcc_uk"].iloc[0]) if "tcc_uk" in row.columns else 0.0
            
            if is_night or ssrd < 0.01:
                return {
                    "status": "success",
                    "is_mock": True,
                    "base_value": 0.0,
                    "prediction": 0.0,
                    "contributions": [
                        {"feature": "solar_elevation_deg", "contribution": -0.9, "value": float(row["solar_elevation_deg"].iloc[0]) if "solar_elevation_deg" in row.columns else -15.0, "description": "Sun is below horizon"}
                    ]
                }
            else:
                val = 0.15 + (ssrd / 800.0) * 0.5 - (tcc * 0.2)
                val = max(0.01, val)
                return {
                    "status": "success",
                    "is_mock": True,
                    "base_value": 0.15,
                    "prediction": val,
                    "contributions": [
                        {"feature": "ssrd_uk", "contribution": (ssrd / 800.0) * 0.5, "value": ssrd, "description": "Solar radiation contribution"},
                        {"feature": "tcc_uk", "contribution": -(tcc * 0.2), "value": tcc, "description": "Cloud cover obstruction"},
                        {"feature": "clearsky_cos", "contribution": 0.05, "value": float(row["clearsky_cos"].iloc[0]) if "clearsky_cos" in row.columns else 0.5, "description": "Clearsky geometry"}
                    ]
                }

        art = joblib.load(stack_path)
        features = art["features"]
        lgbm = art["lgbm"]
        
        # Prepare inputs
        X_row = row[features].to_numpy("float32")
        
        # Compute TreeSHAP using LightGBM's native predict(pred_contrib=True)
        # Note: self.models_ is dict of {q: LGBMRegressor}
        # LGBMRegressor's booster has predict(..., pred_contrib=True)
        q50_model = lgbm.models_[0.5]
        contribs = q50_model.predict(X_row, pred_contrib=True)[0]
        
        # The last column is the expected value (base value)
        base_value = float(contribs[-1])
        prediction = float(np.sum(contribs)) # sum of contributions = model prediction
        if "is_daylight" in row.columns and int(row["is_daylight"].iloc[0]) == 0:
            prediction = 0.0
        
        feature_contribs = []
        for feat, val in zip(features, contribs[:-1]):
            val_raw = row[feat].iloc[0]
            if pd.isna(val_raw):
                val_raw = None
            else:
                val_raw = float(val_raw)
                
            # Descriptions
            desc = "Model feature contribution"
            if feat == "ssrd_uk":
                desc = "Incoming solar radiation driver"
            elif feat == "tcc_uk":
                desc = "Cloud cover obstruction"
            elif feat == "clearsky_cos":
                desc = "Theoretical clear-sky gate"
                
            feature_contribs.append({
                "feature": feat,
                "contribution": float(val),
                "value": val_raw,
                "description": desc
            })
            
        # Filter for meaningful contributions (abs > 0.001)
        feature_contribs = [c for c in feature_contribs if abs(c["contribution"]) > 0.001]
        feature_contribs = sorted(feature_contribs, key=lambda x: abs(x["contribution"]), reverse=True)
        
        return {
            "status": "success",
            "is_mock": False,
            "base_value": base_value,
            "prediction": prediction,
            "contributions": feature_contribs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating local feature contributions: {str(e)}")

@router.get("/meta")
def get_meta_weights(horizon: int = Query(24, enum=[6, 12, 24])):
    """Fetches the weights of the Linear Quantile Stacking Regressor (Model A)."""
    try:
        steps = {6: 12, 12: 24, 24: 48}[horizon]
        folder = "model" if steps == 48 else f"model_h{steps}"
        stack_path = ARTIFACTS_DIR / folder / "stack.joblib"
        
        # 7 Stacking features: TCN q10, TCN q50, TCN q90, LGBM q10, LGBM q50, LGBM q90, Clear-Sky GHI
        feature_names = [
            "TCN q10 prediction", "TCN q50 prediction", "TCN q90 prediction",
            "LGBM q10 prediction", "LGBM q50 prediction", "LGBM q90 prediction",
            "Clear-Sky GHI Index"
        ]
        
        if not stack_path.exists():
            # Mock weights for demo
            mock_weights = {
                0.1: [0.10, 0.45, 0.00, 0.15, 0.20, 0.00, 0.10],
                0.5: [0.05, 0.35, 0.05, 0.05, 0.40, 0.05, 0.05],
                0.9: [0.00, 0.15, 0.35, 0.00, 0.20, 0.25, 0.05]
            }
            return {
                "status": "success",
                "is_mock": True,
                "features": feature_names,
                "weights": mock_weights
            }
            
        art = joblib.load(stack_path)
        meta = art["meta"] # LinearQuantileStacker object
        
        weights = {}
        for q, model in meta.models_.items():
            # model is an sklearn QuantileRegressor
            coefs = model.coef_.tolist()
            weights[float(q)] = [float(c) for c in coefs]
            
        return {
            "status": "success",
            "is_mock": False,
            "features": feature_names,
            "weights": weights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving meta-learner weights: {str(e)}")

@router.get("/simulate")
def simulate_local_xai(
    timestamp: str,
    horizon: int = Query(24, enum=[6, 12, 24]),
    ssrd: Optional[float] = None,
    tcc: Optional[float] = None,
    t2m: Optional[float] = None
):
    """Simulates local feature contributions (SHAP) under what-if weather overrides."""
    try:
        ts = pd.to_datetime(timestamp)
        steps = {6: 12, 12: 24, 24: 48}[horizon]
        folder = "model" if steps == 48 else f"model_h{steps}"
        stack_path = ARTIFACTS_DIR / folder / "stack.joblib"
        
        # Load gold row
        try:
            gold = load_gold_data(horizon)
            row = gold[gold["timestamp_utc"] == ts].copy()
        except Exception:
            row = pd.DataFrame()
            
        if row.empty:
            # Fallback mock simulation if gold features or row is missing
            base_val = 0.15
            ssrd_val = float(ssrd) if ssrd is not None else 400.0
            tcc_val = (float(tcc) / 100.0) if tcc is not None else 0.5
            pred = base_val + (ssrd_val / 800.0) * 0.5 - (tcc_val * 0.2)
            pred = max(0.01, pred)
            
            return {
                "status": "success",
                "is_mock": True,
                "base_value": base_val,
                "prediction": pred,
                "contributions": [
                    {"feature": "ssrd_uk", "contribution": (ssrd_val / 800.0) * 0.5, "value": ssrd_val, "description": "Solar radiation contribution"},
                    {"feature": "tcc_uk", "contribution": -(tcc_val * 0.2), "value": tcc_val * 100.0, "description": "Cloud cover obstruction"},
                    {"feature": "clearsky_cos", "contribution": 0.05, "value": 0.5, "description": "Clearsky geometry"}
                ]
            }
            
        # Apply overrides
        if ssrd is not None:
            row["ssrd_uk"] = float(ssrd)
        if tcc is not None:
            row["tcc_uk"] = float(tcc) / 100.0 # Convert % to 0-1
        if t2m is not None:
            row["t2m_uk"] = float(t2m) + 273.15 # Convert C to Kelvin

        # Re-calculate capacity factor features if overrides affect them
        if "embedded_solar_mw" in row.columns and "embedded_solar_capacity_mw" in row.columns:
            row["embedded_solar_cf"] = (row["embedded_solar_mw"] / np.clip(row["embedded_solar_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
        if "embedded_wind_mw" in row.columns and "embedded_wind_capacity_mw" in row.columns:
            row["embedded_wind_cf"] = (row["embedded_wind_mw"] / np.clip(row["embedded_wind_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")

        if not stack_path.exists():
            # Mock simulation with row values
            base_val = 0.15
            ssrd_val = float(row["ssrd_uk"].iloc[0]) if "ssrd_uk" in row.columns else 400.0
            tcc_val = float(row["tcc_uk"].iloc[0]) if "tcc_uk" in row.columns else 0.5
            pred = base_val + (ssrd_val / 800.0) * 0.5 - (tcc_val * 0.2)
            pred = max(0.01, pred)
            
            return {
                "status": "success",
                "is_mock": True,
                "base_value": base_val,
                "prediction": pred,
                "contributions": [
                    {"feature": "ssrd_uk", "contribution": (ssrd_val / 800.0) * 0.5, "value": ssrd_val, "description": "Solar radiation contribution"},
                    {"feature": "tcc_uk", "contribution": -(tcc_val * 0.2), "value": tcc_val * 100.0, "description": "Cloud cover obstruction"},
                    {"feature": "clearsky_cos", "contribution": 0.05, "value": float(row["clearsky_cos"].iloc[0]) if "clearsky_cos" in row.columns else 0.5, "description": "Clearsky geometry"}
                ]
            }

        art = joblib.load(stack_path)
        features = art["features"]
        lgbm = art["lgbm"]
        
        # Prepare inputs
        X_row = row[features].to_numpy("float32")
        q50_model = lgbm.models_[0.5]
        contribs = q50_model.predict(X_row, pred_contrib=True)[0]
        
        base_value = float(contribs[-1])
        prediction = float(np.sum(contribs))
        
        feature_contribs = []
        for feat, val in zip(features, contribs[:-1]):
            val_raw = row[feat].iloc[0]
            val_raw = float(val_raw) if not pd.isna(val_raw) else None
            
            # Map raw units back for visual display (e.g. Kelvin -> Celsius, TCC fraction -> %)
            if feat == "t2m_uk" and val_raw is not None:
                val_raw = val_raw - 273.15
            elif feat == "tcc_uk" and val_raw is not None:
                val_raw = val_raw * 100.0
                
            desc = "Model feature contribution"
            if feat == "ssrd_uk":
                desc = "Incoming solar radiation driver"
            elif feat == "tcc_uk":
                desc = "Cloud cover obstruction"
            elif feat == "clearsky_cos":
                desc = "Theoretical clear-sky gate"
                
            feature_contribs.append({
                "feature": feat,
                "contribution": float(val),
                "value": val_raw,
                "description": desc
            })
            
        feature_contribs = [c for c in feature_contribs if abs(c["contribution"]) > 0.001]
        feature_contribs = sorted(feature_contribs, key=lambda x: abs(x["contribution"]), reverse=True)
        
        return {
            "status": "success",
            "is_mock": False,
            "base_value": base_value,
            "prediction": prediction,
            "contributions": feature_contribs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating SHAP values: {str(e)}")
