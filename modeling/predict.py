"""Inference: load the trained stack and emit q10/q50/q90 forecasts (in MW).

    from modeling.predict import predict_gold
    import pandas as pd, glob
    gold = pd.concat([pd.read_parquet(f) for f in
                      glob.glob("data/gold/gold_features/**/*.parquet", recursive=True)])
    out = predict_gold(gold)            # -> timestamp_utc, pred_q10/q50/q90 (MW)

NOTE: pass a CONTIGUOUS time block that includes at least `seq_len` rows of
history before the slots you want predicted (the TCN needs the window).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import make_sequences
from .clearsky import clearsky_feature
from .base import TCNQuantile, LSTMQuantile
from .stacking import assemble_meta_X


def load_stack(artifacts_dir: str | Path):
    import joblib
    import torch
    art = joblib.load(Path(artifacts_dir) / "stack.joblib")
    cfg = art["cfg"]
    tcn = TCNQuantile(cfg, len(art["features"])).build()
    tcn.model_.load_state_dict(torch.load(Path(artifacts_dir) / "tcn.pt", map_location="cpu"))
    tcn.model_.eval()

    lstm = LSTMQuantile(cfg, len(art["features"])).build()
    lstm.model_.load_state_dict(torch.load(Path(artifacts_dir) / "lstm.pt", map_location="cpu"))
    lstm.model_.eval()
    return art, tcn, lstm


def predict_gold(df: pd.DataFrame, artifacts_dir: str | Path = "artifacts/model") -> pd.DataFrame:
    art, tcn, lstm = load_stack(artifacts_dir)
    cfg: ModelConfig = art["cfg"]
    feats, std, lgbm, meta = art["features"], art["standardizer"], art["lgbm"], art["meta"]
    qnames = cfg.quantile_names()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    V = df[feats].to_numpy("float32")
    seqs, end_idx = make_sequences(std.transform(V), cfg.seq_len)
    clear = clearsky_feature(df)

    lgp = lgbm.predict(V[end_idx])
    tcp = tcn.predict(seqs)
    lstmp = lstm.predict(seqs)
    preds = meta.predict(assemble_meta_X(tcp, lgp, lstmp, clear[end_idx], cfg.quantiles))

    out = df.iloc[end_idx][["timestamp_utc"]].copy()
    cap = df["capacity_mwp"].to_numpy("float32")[end_idx] if "capacity_mwp" in df else 1.0
    day = df["is_daylight"].to_numpy()[end_idx] if "is_daylight" in df else np.ones(len(end_idx))
    for q, name in zip(cfg.quantiles, qnames):
        p = preds[q] * cap if cfg.target == "target_cf" else preds[q]
        out[f"pred_{name}"] = np.where(day == 1, p, 0.0).astype("float32")   # night -> 0
    return out.reset_index(drop=True)
