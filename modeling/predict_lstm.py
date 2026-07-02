"""Inference for Standalone LSTM-Q model.

    from modeling.predict_lstm import predict_lstm_gold
    import pandas as pd, glob
    gold = pd.concat([pd.read_parquet(f) for f in
                      glob.glob("data/gold/gold_features/**/*.parquet", recursive=True)])
    out = predict_lstm_gold(gold)       # -> timestamp_utc, pred_q10/q50/q90 (MW)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import make_sequences
from .base import LSTMQuantile


def load_lstm_stack(artifacts_dir: str | Path):
    import joblib
    import torch
    art = joblib.load(Path(artifacts_dir) / "lstm.joblib")
    cfg = art["cfg"]
    lstm = LSTMQuantile(cfg, len(art["features"])).build()
    lstm.model_.load_state_dict(torch.load(Path(artifacts_dir) / "lstm.pt", map_location="cpu"))
    lstm.model_.eval()
    return art, lstm


def predict_lstm_gold(df: pd.DataFrame, artifacts_dir: str | Path = "artifacts/lstm") -> pd.DataFrame:
    art, lstm = load_lstm_stack(artifacts_dir)
    cfg: ModelConfig = art["cfg"]
    feats, std = art["features"], art["standardizer"]
    calib_factor = art.get("calib_factor", 1.0)
    qnames = cfg.quantile_names()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    V = df[feats].to_numpy("float32")
    seqs, end_idx = make_sequences(std.transform(V), cfg.seq_len)

    raw_p = lstm.predict(seqs) # dict of q -> capacity factor
    
    # Apply calibration factor
    q50 = raw_p[0.5]
    if isinstance(calib_factor, (tuple, list)):
        w, k = calib_factor
        if "clearsky_cos" in df.columns:
            cos = df["clearsky_cos"].to_numpy("float32")[end_idx]
            factor = w / (cos + k)
        else:
            factor = w / (1.0 + k)
    else:
        factor = calib_factor

    q10 = q50 - (q50 - raw_p[0.1]) * factor
    q90 = q50 + (raw_p[0.9] - q50) * factor
    
    # Enforce monotonicity
    q10 = np.minimum(np.maximum(q10, 0), q50)
    q90 = np.maximum(q90, q50)

    out = df.iloc[end_idx][["timestamp_utc"]].copy()
    cap = df["capacity_mwp"].to_numpy("float32")[end_idx] if "capacity_mwp" in df else 1.0
    day = df["is_daylight"].to_numpy()[end_idx] if "is_daylight" in df else np.ones(len(end_idx))
    
    preds_scaled = {0.1: q10, 0.5: q50, 0.9: q90}
    for q, name in zip(cfg.quantiles, qnames):
        p = preds_scaled[q] * cap if cfg.target == "target_cf" else preds_scaled[q]
        out[f"pred_{name}"] = np.where(day == 1, p, 0.0).astype("float32")   # night -> 0
        
    return out.reset_index(drop=True)
