"""Load the Gold feature store and prepare splits / sequences.

Time-series rules enforced here:
  * Rows are sorted by timestamp; splits are CHRONOLOGICAL (never shuffled).
  * Features exclude targets, the timestamp, and QA flags (config.NON_FEATURE_COLS).
  * Only rows with `has_full_history == 1` and a non-null target are kept.
  * `score_mask` selects the rows we actually train/evaluate on (daylight only by
    default — night is ~0 and handled trivially). IMPORTANT: night rows are kept
    in the frame so the TCN's diurnal sequence context stays continuous; they are
    just excluded from the scored targets.
  * TCN sequences end AT t (causal) — leakage-safe: features at t are forecasts /
    deterministic, and observed-actuals are already lagged inside Gold.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ModelConfig, NON_FEATURE_COLS


@dataclass
class Dataset:
    df: pd.DataFrame                 # full ordered frame (ALL hours), full-history, target known
    feature_cols: list[str]
    cfg: ModelConfig

    def score_mask(self) -> np.ndarray:
        if self.cfg.daylight_only and "is_daylight" in self.df.columns:
            return (self.df["is_daylight"] == 1).to_numpy()
        return np.ones(len(self.df), dtype=bool)

    def split_masks(self):
        ts = self.df["timestamp_utc"]
        val0 = pd.Timestamp(self.cfg.val_start, tz="UTC")
        test0 = pd.Timestamp(self.cfg.test_start, tz="UTC")
        train = (ts < val0).to_numpy()
        val = ((ts >= val0) & (ts < test0)).to_numpy()
        test = (ts >= test0).to_numpy()
        return train, val, test


def load_gold(cfg: ModelConfig) -> pd.DataFrame:
    files = sorted(cfg.gold_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Gold parquet under {cfg.gold_dir}. Build it first: "
            f"python -m data_ingestion.gold --horizon-steps {cfg.horizon_steps}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def prepare(cfg: ModelConfig) -> Dataset:
    df = load_gold(cfg)
    if cfg.target not in df.columns:
        raise KeyError(f"target {cfg.target!r} not in Gold columns")
    df = df[df["has_full_history"] == 1]
    df = df[df[cfg.target].notna()].reset_index(drop=True)   # keep ALL hours (night kept for context)
    
    # Normalize NESO MW features to capacity factors
    if "embedded_solar_mw" in df.columns and "embedded_solar_capacity_mw" in df.columns:
        df["embedded_solar_cf"] = (df["embedded_solar_mw"] / np.clip(df["embedded_solar_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
    if "embedded_wind_mw" in df.columns and "embedded_wind_capacity_mw" in df.columns:
        df["embedded_wind_cf"] = (df["embedded_wind_mw"] / np.clip(df["embedded_wind_capacity_mw"], 1e-6, None)).fillna(0.0).astype("float32")
        
    return Dataset(df=df, feature_cols=feature_columns(df), cfg=cfg)


def make_sequences(values: np.ndarray, seq_len: int):
    """values: (N, F) row-sorted. Returns seqs (M, seq_len, F) and end_idx (row index of each)."""
    n = values.shape[0]
    if n <= seq_len:
        return np.empty((0, seq_len, values.shape[1]), "float32"), np.empty((0,), "int64")
    win = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)  # (M, F, seq_len)
    seqs = np.ascontiguousarray(win.transpose(0, 2, 1)).astype("float32")    # (M, seq_len, F)
    end_idx = np.arange(seq_len - 1, n)
    return seqs, end_idx


class Standardizer:
    """Fit on TRAIN rows only; impute NaN with the train mean, then z-score (for the TCN)."""
    def fit(self, X: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[~np.isfinite(self.std_) | (self.std_ == 0)] = 1.0
        self.mean_ = np.nan_to_num(self.mean_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), self.mean_, X)
        return ((X - self.mean_) / self.std_).astype("float32")
