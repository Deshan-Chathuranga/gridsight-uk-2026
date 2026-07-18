"""Inference: load the trained stack and emit q10/q50/q90 forecasts (in MW).

NOTE: pass a CONTIGUOUS time block that includes at least `seq_len` rows of
history before the slots you want predicted (the TCN needs the window).
"""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import tempfile
import sys

import numpy as np
import pandas as pd
import joblib

from .config import ModelConfig
from .data import make_sequences
from gridsight.models.common.clearsky import clearsky_feature
from .stacking import assemble_meta_X


def run_tcn_predict_subprocess(seqs: np.ndarray, cfg: ModelConfig, n_features: int, artifacts_dir: str | Path) -> dict[float, np.ndarray]:
    """Runs the PyTorch TCN prediction in a separate clean process to avoid OpenMP conflict with LightGBM on macOS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_seqs = Path(tmpdir) / "seqs.npy"
        tmp_preds = Path(tmpdir) / "preds.npy"
        np.save(tmp_seqs, seqs)
        
        cfg_dict = {
            "tcn_channels": list(cfg.tcn_channels),
            "tcn_kernel": int(cfg.tcn_kernel),
            "tcn_dropout": float(cfg.tcn_dropout),
            "quantiles": list(cfg.quantiles),
            "torch_threads": int(cfg.torch_threads)
        }
        tmp_cfg = Path(tmpdir) / "cfg.json"
        with open(tmp_cfg, "w") as f:
            json.dump(cfg_dict, f)
            
        py_code = f"""
import json
import numpy as np
import torch
import torch.nn as nn

# Load config and seqs
with open(r"{tmp_cfg}") as f:
    cfg = json.load(f)
seqs = np.load(r"{tmp_seqs}")

# Define TCN model (clean, no weight norm to avoid macOS PyTorch loading bugs)
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s else x

class TemporalBlock(nn.Module):
    def __init__(self, ci, co, k, d, dropout):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(co, co, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_features, n_quantiles):
        super().__init__()
        layers, ci = [], n_features
        for i, co in enumerate(cfg["tcn_channels"]):
            layers.append(TemporalBlock(ci, co, cfg["tcn_kernel"], 2 ** i, cfg["tcn_dropout"]))
            ci = co
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(ci, n_quantiles)
    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])

# Build model
model = TCN({n_features}, len(cfg["quantiles"]))
state = torch.load(r"{Path(artifacts_dir) / 'tcn.pt'}", map_location=lambda storage, loc: storage)

# Fold weight norm parameters into clean state
clean_state = {{}}
for k, v in state.items():
    if k.endswith(".weight_g"):
        prefix = k[:-9]
        g = v
        v_param = state[prefix + ".weight_v"]
        norm = v_param.norm(dim=(1, 2), keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        weight = g * (v_param / norm)
        clean_state[prefix + ".weight"] = weight
    elif k.endswith(".weight_v"):
        continue
    else:
        clean_state[k] = v
        
model.load_state_dict(clean_state)
model.eval()

# Run prediction
with torch.no_grad():
    x_t = torch.tensor(seqs)
    preds = model(x_t).numpy()
    
np.save(r"{tmp_preds}", preds)
"""
        res = subprocess.run([sys.executable, "-c", py_code], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"TCN subprocess failed: {res.stderr}")
            
        preds = np.load(tmp_preds)
        pred = np.sort(preds, axis=1)               # enforce monotone quantiles
        return {q: pred[:, i].astype("float32") for i, q in enumerate(cfg.quantiles)}


def predict_gold(df: pd.DataFrame, artifacts_dir: str | Path = "artifacts/model") -> pd.DataFrame:
    art = joblib.load(Path(artifacts_dir) / "stack.joblib")
    cfg: ModelConfig = art["cfg"]
    feats, std, lgbm, meta = art["features"], art["standardizer"], art["lgbm"], art["meta"]
    qnames = cfg.quantile_names()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    
    # Calculate capacity factor features
    if "embedded_solar_mw" in df.columns and "embedded_solar_capacity_mw" in df.columns:
        df["embedded_solar_cf"] = (df["embedded_solar_mw"] / np.clip(df["embedded_solar_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
    if "embedded_wind_mw" in df.columns and "embedded_wind_capacity_mw" in df.columns:
        df["embedded_wind_cf"] = (df["embedded_wind_mw"] / np.clip(df["embedded_wind_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")

    V = df[feats].to_numpy("float32")
    seqs, end_idx = make_sequences(std.transform(V), cfg.seq_len)
    clear = clearsky_feature(df)

    lgp = lgbm.predict(V[end_idx])
    tcp = run_tcn_predict_subprocess(seqs, cfg, len(feats), artifacts_dir)
    preds = meta.predict(assemble_meta_X(tcp, lgp, clear[end_idx], cfg.quantiles))

    out = df.iloc[end_idx][["timestamp_utc"]].copy()
    cap = df["capacity_mwp"].to_numpy("float32")[end_idx] if "capacity_mwp" in df else 1.0
    day = df["is_daylight"].to_numpy()[end_idx] if "is_daylight" in df else np.ones(len(end_idx))
    for q, name in zip(cfg.quantiles, qnames):
        p = preds[q] * cap if cfg.target == "target_cf" else preds[q]
        p = np.maximum(p, 0.0)
        out[f"pred_{name}"] = np.where(day == 1, p, 0.0).astype("float32")   # night -> 0
    return out.reset_index(drop=True)
