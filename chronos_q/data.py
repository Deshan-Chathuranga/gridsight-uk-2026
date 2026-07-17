from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ModelConfig

@dataclass
class Dataset:
    df: pd.DataFrame
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

def prepare(cfg: ModelConfig) -> Dataset:
    df = load_gold(cfg)
    if cfg.target not in df.columns:
        raise KeyError(f"target {cfg.target!r} not in Gold columns")
    df = df[df["has_full_history"] == 1]
    df = df[df[cfg.target].notna()].reset_index(drop=True)
    return Dataset(df=df, cfg=cfg)
