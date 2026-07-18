from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .base import ChronosQuantile


def load_chronos_stack(artifacts_dir: str | Path):
    import joblib
    art = joblib.load(Path(artifacts_dir) / "chronos.joblib")
    cfg: ModelConfig = art["cfg"]
    chronos = ChronosQuantile(cfg).load()
    return art, chronos


def predict_chronos_gold(df: pd.DataFrame, artifacts_dir: str | Path = "artifacts/chronos") -> pd.DataFrame:
    art, chronos = load_chronos_stack(artifacts_dir)
    cfg: ModelConfig = art["cfg"]
    calib_factor = art.get("calib_factor", (1.0, 0.5))
    qnames = cfg.quantile_names()
    H = cfg.horizon_steps

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    series = df[cfg.target].to_numpy("float32")
    cap = df["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in df else None
    cos = df["clearsky_cos"].to_numpy("float32") if "clearsky_cos" in df else np.ones(len(df), "float32")
    day = df["is_daylight"].to_numpy() if "is_daylight" in df else np.ones(len(df))

    rows = np.arange(len(df))
    rows = rows[rows >= H]
    rows = rows[day[rows] == 1]

    raw = chronos.forecast_h_ahead(series, rows - H, H)

    w, k = calib_factor if isinstance(calib_factor, (tuple, list)) else (calib_factor, 0.0)
    factor = w / (cos[rows] + k)
    q50 = raw[0.5]
    q10 = q50 - (q50 - raw[0.1]) * factor
    q90 = q50 + (raw[0.9] - q50) * factor
    
    # Apply clear-sky GHI ceiling post-processing to clip median forecast
    q50 = np.minimum(q50, cos[rows] * 0.70)
    q10 = np.minimum(np.maximum(q10, 0), q50)
    q90 = np.maximum(q90, q50)
    preds = {0.1: q10, 0.5: q50, 0.9: q90}

    out = pd.DataFrame({"timestamp_utc": df["timestamp_utc"].to_numpy()})
    cap_full = cap if cap is not None else np.ones(len(df), "float32")
    for q, name in zip(cfg.quantiles, qnames):
        full = np.zeros(len(df), "float32")
        p = preds[q] * cap_full[rows] if cfg.target == "target_cf" else preds[q]
        full[rows] = p
        out[f"pred_{name}"] = full
    return out
