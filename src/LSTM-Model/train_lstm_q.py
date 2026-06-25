#!/usr/bin/env python3
"""
⚡ GridSight UK — LSTM-Q Probabilistic Solar Forecasting
Model: LSTM-Q · Baseline Deep Learning Quantile Forecaster

This script implements a complete training and evaluation pipeline for 
generating 80% calibrated prediction intervals (q10, q50, q90) of national 
PV generation using PyTorch Lightning.
"""

import os
import sys
import glob
import json
import random
import argparse
import warnings
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

# Setup environments and styling
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)

# Global configuration constants
SEED = 42
QUANTILES = [0.1, 0.5, 0.9]
LOOK_BACK = 96  # 48 hours of history at 30-min steps

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hardware acceleration setup
DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')
ACCELERATOR = {'cuda': 'gpu', 'mps': 'mps', 'cpu': 'cpu'}[DEVICE]

# ── DYNAMIC FEATURES CONFIGURATION ───────────────────────────────────────────
# Features list is dynamically determined per horizon to ensure leakage safety.


# ── DATASET CLASS ────────────────────────────────────────────────────────────
class SolarWindowDataset(Dataset):
    """Sliding window dataset for time-series forecasting."""
    def __init__(self, X: np.ndarray, y: np.ndarray, look_back: int = LOOK_BACK):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self):
        return len(self.X) - self.look_back

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shift the features window by +1 step relative to the target index,
        # so that the last step of X includes the features at the target step.
        x_seq = self.X[idx + 1 : idx + self.look_back + 1]  # (look_back, n_feats)
        y_tgt = self.y[idx + self.look_back].unsqueeze(0)   # (1,)
        return x_seq, y_tgt


# ── LOSS FUNCTION ─────────────────────────────────────────────────────────────
class PinballLoss(nn.Module):
    """Quantile (pinball) loss for multiple quantiles simultaneously."""
    def __init__(self, quantiles: List[float] = QUANTILES):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds:  (batch, n_quantiles)
        # target: (batch, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            p = preds[:, i].unsqueeze(1)  # (batch, 1)
            e = target - p
            losses.append(torch.mean(torch.where(e >= 0, q * e, (q - 1) * e)))
        return torch.stack(losses).mean()


# ── FORECASTER SYSTEM ─────────────────────────────────────────────────────────
class LSTMForecaster(pl.LightningModule):
    """2-layer LSTM quantile forecaster for GridSight UK solar generation."""
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.20,
        quantiles: List[float] = QUANTILES,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(quantiles)),
        )

        self.criterion = PinballLoss(quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # (batch, look_back, hidden)
        last = out[:, -1, :]         # take last hidden state
        return self.head(last)       # (batch, n_quantiles)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, 'train')

    def validation_step(self, batch, _):
        return self._step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }


# ── EVALUATION METRICS ────────────────────────────────────────────────────────
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))

def mean_pinball(y_true: np.ndarray, preds: Dict[float, np.ndarray]) -> float:
    return float(np.mean([pinball_loss(y_true, preds[q], q) for q in QUANTILES]))

def coverage(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> float:
    return float(np.mean((y_true >= q10) & (y_true <= q90)))

def pi_width(q10: np.ndarray, q90: np.ndarray) -> float:
    return float(np.mean(q90 - q10))

def winkler_score(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray, alpha: float = 0.20) -> float:
    width = q90 - q10
    penalty = np.where(y_true < q10, (2 / alpha) * (q10 - y_true),
                       np.where(y_true > q90, (2 / alpha) * (y_true - q90), 0))
    return float(np.mean(width + penalty))

def evaluate(
    name: str, 
    y_true: np.ndarray, 
    preds: Dict[float, np.ndarray], 
    capacity: np.ndarray, 
    y_persistence: np.ndarray
) -> Dict[str, Any]:
    mpl = mean_pinball(y_true, preds)
    mae = mean_absolute_error(y_true, preds[0.5])
    rmse = float(np.sqrt(mean_squared_error(y_true, preds[0.5])))
    
    # nMAE normalized by contemporaneous capacity
    nmae = float(np.mean(np.abs(y_true - preds[0.5]) / capacity) * 100)
    
    cov = coverage(y_true, preds[0.1], preds[0.9])
    iw = pi_width(preds[0.1], preds[0.9])
    ws = winkler_score(y_true, preds[0.1], preds[0.9])
    
    # Correct Skill Score vs 24-hour persistence (48 steps)
    mae_persistence = mean_absolute_error(y_true, y_persistence)
    ss = float(1 - mae / mae_persistence) if mae_persistence > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  Model          : {name}")
    print(f"  Pinball Loss   : {mpl:.4f} MW")
    print(f"  MAE  (q50)     : {mae:.2f} MW")
    print(f"  RMSE (q50)     : {rmse:.2f} MW")
    print(f"  nMAE (cap-norm): {nmae:.2f}%   (KPI gate ≤ 4.0%)")
    print(f"  Skill Score    : {ss:.3f}     (KPI gate > 0.30)")
    print(f"  PI Coverage    : {cov:.3f}    (target 0.78 – 0.82)")
    print(f"  PI Width       : {iw:.1f} MW")
    print(f"  Winkler Score  : {ws:.3f}")
    
    return {
        'model': name, 'pinball': mpl, 'mae': mae, 'rmse': rmse,
        'nmae': nmae, 'skill': ss, 'coverage': cov, 'pi_width': iw, 'winkler': ws
    }


# ── BASELINE UTILITIES ────────────────────────────────────────────────────────
def neso_baseline_preds(df_split: pd.DataFrame, n_drop_warmup: int) -> Dict[float, np.ndarray]:
    """Generates baseline benchmark predictions based on NESO forecast outputs."""
    y_base = df_split['embedded_solar_mw'].fillna(0).values[n_drop_warmup:]
    y_base = np.clip(y_base, 0, None)
    # Estimate simple static intervals around baseline
    return {0.1: y_base * 0.7, 0.5: y_base, 0.9: y_base * 1.3}


# ── INFERENCE ENGINE ─────────────────────────────────────────────────────────
def predict(
    model: pl.LightningModule, 
    dataloader: DataLoader, 
    capacity: np.ndarray, 
    calib_factor: float = 1.0
) -> Dict[float, np.ndarray]:
    """Run model inference, returning scale-restored MW quantile arrays with post-hoc calibration."""
    model.eval()
    model.to(DEVICE)
    raw_preds = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(DEVICE)
            out = model(x).cpu().numpy()
            raw_preds.append(out)
            
    raw = np.vstack(raw_preds)
    
    # Extract predicted capacity factors for quantiles (q10, q50, q90)
    q10_raw = raw[:, 0]
    q50_raw = raw[:, 1]
    q90_raw = raw[:, 2]
    
    # Apply post-hoc calibration to scale the prediction interval width
    if calib_factor != 1.0:
        q10_raw = q50_raw - (q50_raw - q10_raw) * calib_factor
        q90_raw = q50_raw + (q90_raw - q50_raw) * calib_factor
        
    preds = {}
    preds[0.1] = np.clip(q10_raw * capacity, 0, None)
    preds[0.5] = np.clip(q50_raw * capacity, 0, None)
    preds[0.9] = np.clip(q90_raw * capacity, 0, None)
        
    # Enforce strict quantile monotonicity: q10 <= q50 <= q90
    preds[0.1] = np.minimum(preds[0.1], preds[0.5])
    preds[0.9] = np.maximum(preds[0.9], preds[0.5])
    return preds


def load_and_predict(
    gold_parquet_path: str,
    ckpt_path: str = 'model_artefacts/lstm_q_best.ckpt',
    scaler_path: str = 'model_artefacts/feature_scaler.pkl',
    features_path: str = 'model_artefacts/lstm_features.json',
    look_back: int = LOOK_BACK,
    calib_factor: float = 0.90
) -> pd.DataFrame:
    """Helper method executing end-to-end inference on raw out-of-sample data."""
    model = LSTMForecaster.load_from_checkpoint(ckpt_path)
    model.eval()
    
    scaler_inf = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)
        
    df_new = pd.read_parquet(gold_parquet_path)
    df_new['timestamp_utc'] = pd.to_datetime(df_new['timestamp_utc'], utc=True)
    df_new = df_new.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Apply capacity factor transformations matching training
    df_new['embedded_solar_cf'] = df_new['embedded_solar_mw'] / df_new['capacity_mwp']
    df_new['cf_roll_std_48'] = df_new['gen_roll_std_48'] / df_new['capacity_mwp']
    
    # Dynamically find and normalize the ocf lag column if present
    ocf_lag_cols = [c for c in df_new.columns if c.startswith('ocf_lag_')]
    for col in ocf_lag_cols:
        df_new[f'{col}_cf'] = df_new[col] / df_new['capacity_mwp']
    
    # Fill NA using features median
    df_new[features] = df_new[features].fillna(df_new[features].median())
    
    X_new = scaler_inf.transform(df_new[features]).astype(np.float32)
    
    all_preds = []
    with torch.no_grad():
        for i in range(look_back, len(X_new)):
            # Aligned window ending at i (inclusive)
            window = torch.tensor(X_new[i - look_back + 1 : i + 1]).unsqueeze(0)
            pred = model(window).cpu().numpy()[0]
            all_preds.append(pred)
            
    raw = np.vstack(all_preds)
    out = df_new.iloc[look_back:][['timestamp_utc']].copy().reset_index(drop=True)
    
    q10_raw = raw[:, 0]
    q50_raw = raw[:, 1]
    q90_raw = raw[:, 2]
    
    if calib_factor != 1.0:
        q10_raw = q50_raw - (q50_raw - q10_raw) * calib_factor
        q90_raw = q50_raw + (q90_raw - q50_raw) * calib_factor
        
    capacity_arr = df_new.iloc[look_back:]['capacity_mwp'].values
    out['q10_mw'] = np.clip(q10_raw * capacity_arr, 0, None)
    out['q50_mw'] = np.clip(q50_raw * capacity_arr, 0, None)
    out['q90_mw'] = np.clip(q90_raw * capacity_arr, 0, None)
    
    out['q10_mw'] = np.minimum(out['q10_mw'], out['q50_mw'])
    out['q90_mw'] = np.maximum(out['q90_mw'], out['q50_mw'])
    return out


# ── MAIN PIPELINE CONTROLLER ──────────────────────────────────────────────────
def run_pipeline(gold_dir: str, epochs: int, batch_size: int, lr: float, horizon: int):
    print("🚀 Initializing GridSight LSTM-Q Probabilistic pipeline...")
    
    # 1. Loading data files
    files = sorted(glob.glob(os.path.join(gold_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No Gold layer parquet files found under directory: {gold_dir}")
        
    print(f"Found {len(files)} parquet files. Loading...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    
    # Apply safety lag-history requirements constraint
    if 'has_full_history' in df.columns:
        print("Filtering layout to rows with complete lag/rolling calculations...")
        df = df[df['has_full_history'] == 1].reset_index(drop=True)
        
    # Feature Engineering (Capacity Normalization)
    df['embedded_solar_cf'] = df['embedded_solar_mw'] / df['capacity_mwp']
    df['cf_roll_std_48'] = df['gen_roll_std_48'] / df['capacity_mwp']
    if f'ocf_lag_{horizon}' in df.columns:
        df[f'ocf_lag_{horizon}_cf'] = df[f'ocf_lag_{horizon}'] / df['capacity_mwp']
        
    print(f"Data set loaded successfully: {len(df):,} records across {df.shape[1]} columns.")
    
    # 2. Chronological split configuration
    TRAIN_END = "2023-12-31 23:30:00+00:00"
    VAL_END   = "2024-06-30 23:30:00+00:00"
    
    df_train = df[df['timestamp_utc'] <= TRAIN_END].copy().reset_index(drop=True)
    df_val   = df[(df['timestamp_utc'] > TRAIN_END) & (df['timestamp_utc'] <= VAL_END)].copy().reset_index(drop=True)
    df_test  = df[df['timestamp_utc'] > VAL_END].copy().reset_index(drop=True)
    
    print(f"Train samples: {len(df_train):,} ({df_train['timestamp_utc'].min().date()} to {df_train['timestamp_utc'].max().date()})")
    print(f"Val samples  : {len(df_val):,} ({df_val['timestamp_utc'].min().date()} to {df_val['timestamp_utc'].max().date()})")
    print(f"Test samples : {len(df_test):,} ({df_test['timestamp_utc'].min().date()} to {df_test['timestamp_utc'].max().date()})")
    
    TARGET_CF = 'target_cf'
    CAPACITY = df_train['capacity_mwp'].median()
    print(f"Base Scaling Capacity: {CAPACITY:.1f} MWp")

    # Define features list dynamically based on the active horizon
    features = [
        'solar_elevation_deg', 'clearsky_cos', 'is_daylight',
        'tod_sin', 'tod_cos', 'doy_sin', 'doy_cos', 'is_weekend',
        'ssrd_uk', 'tcc_uk', 'lcc_uk', 't2m_uk', 'ws10_uk', 'nwp_age_h',
        'embedded_solar_cf'
    ]
    if f'cf_lag_{horizon}' in df.columns:
        features.append(f'cf_lag_{horizon}')
    for L in [48, 96,144,336]:          # it have 4 horizon 48, 96, 144 and 366 you are missing 144
        if L >= horizon and L != horizon:
            features.append(f'cf_lag_{L}')
    if 'cf_roll_mean_48' in df.columns:
        features.append('cf_roll_mean_48')
    if 'cf_roll_std_48' in df.columns:
        features.append('cf_roll_std_48')
    if f'ocf_lag_{horizon}_cf' in df.columns:
        features.append(f'ocf_lag_{horizon}_cf')
        
    print(f"Using {len(features)} dynamic features for horizon={horizon}: {features}")

    # Impute and Scale Inputs
    impute_vals = df_train[features].median()
    for split in [df_train, df_val, df_test]:
        split[features] = split[features].fillna(impute_vals)
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features]).astype(np.float32)
    X_val   = scaler.transform(df_val[features]).astype(np.float32)
    X_test  = scaler.transform(df_test[features]).astype(np.float32)
    
    y_train = df_train[TARGET_CF].fillna(0).clip(0, 1).values.astype(np.float32)
    y_val   = df_val[TARGET_CF].fillna(0).clip(0, 1).values.astype(np.float32)
    y_test  = df_test[TARGET_CF].fillna(0).clip(0, 1).values.astype(np.float32)
    
    # 3. Create Datasets
    ds_train = SolarWindowDataset(X_train, y_train)
    ds_val   = SolarWindowDataset(X_val,   y_val)
    ds_test  = SolarWindowDataset(X_test,  y_test)
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. Model Definition
    n_features = len(features)
    lstm_model = LSTMForecaster(
        n_features=n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.20,
        lr=lr,
        weight_decay=1e-4
    )
    
    # Callbacks and Training setup
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1,
        dirpath='checkpoints', filename='lstm_q_best', verbose=True
    )
    es_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    logger_csv = CSVLogger('logs', name='lstm_q')
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=ACCELERATOR,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        logger=logger_csv,
        log_every_n_steps=10
    )
    
    print("\n🏋️ Commencing Quantile Model Training Phase...")
    trainer.fit(lstm_model, dl_train, dl_val)
    print("Training finished successfully.")
    
    # 5. Evaluate the best checkpoint model
    best_path = ckpt_cb.best_model_path
    print(f"Loading optimal checkpoint file found at: {best_path}")
    lstm_best = LSTMForecaster.load_from_checkpoint(best_path)
    
    # Retrieve uncalibrated validation predictions to sweep calibration factor
    lstm_best.eval()
    lstm_best.to(DEVICE)
    raw_val_preds = []
    with torch.no_grad():
        for x, _ in dl_val:
            x = x.to(DEVICE)
            raw_val_preds.append(lstm_best(x).cpu().numpy())
    raw_val = np.vstack(raw_val_preds)
    
    y_val_mw = df_val['target_mw'].values[LOOK_BACK:]
    y_val_cap = df_val['capacity_mwp'].values[LOOK_BACK:]
    y_val_pers = df_val['target_mw'].values[LOOK_BACK - horizon : -horizon]
    
    # Sweep calibration factors to find the one closest to 80% coverage
    best_calib_factor = 1.0
    best_cov_diff = 1.0
    best_val_cov = 0.80
    for factor in np.linspace(0.5, 1.5, 101):
        q50 = np.clip(raw_val[:, 1] * y_val_cap, 0, None)
        q10 = np.clip((raw_val[:, 1] - (raw_val[:, 1] - raw_val[:, 0]) * factor) * y_val_cap, 0, None)
        q90 = np.clip((raw_val[:, 1] + (raw_val[:, 2] - raw_val[:, 1]) * factor) * y_val_cap, 0, None)
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q90, q50)
        cov = np.mean((y_val_mw >= q10) & (y_val_mw <= q90))
        diff = abs(cov - 0.80)
        if diff < best_cov_diff:
            best_cov_diff = diff
            best_calib_factor = factor
            best_val_cov = cov
            
    print(f"🎯 Optimal post-hoc calibration factor selected: {best_calib_factor:.3f} (Val Coverage: {best_val_cov:.3%})")
    
    # Predictions
    preds_val  = predict(lstm_best, dl_val, y_val_cap, calib_factor=best_calib_factor)
    y_val_mw   = df_val['target_mw'].values[LOOK_BACK:]
    metrics_val = evaluate(f'LSTM-Q [val] (H={horizon})', y_val_mw, preds_val, y_val_cap, y_val_pers)
    
    y_test_cap = df_test['capacity_mwp'].values[LOOK_BACK:]
    y_test_pers = df_test['target_mw'].values[LOOK_BACK - horizon : -horizon]
    preds_test  = predict(lstm_best, dl_test, y_test_cap, calib_factor=best_calib_factor)
    y_test_mw   = df_test['target_mw'].values[LOOK_BACK:]
    metrics_test = evaluate(f'LSTM-Q [test]', y_test_mw, preds_test, y_test_cap, y_test_pers)
    
    # Baseline comparison evaluation
    preds_neso = neso_baseline_preds(df_test, LOOK_BACK)
    metrics_neso = evaluate('NESO baseline [test]', y_test_mw, preds_neso, y_test_cap, y_test_pers)
    
    # Compile Comparison Table
    df_metrics = pd.DataFrame([metrics_val, metrics_test, metrics_neso]).set_index('model')
    cols = ['pinball', 'mae', 'rmse', 'nmae', 'skill', 'coverage', 'pi_width', 'winkler']
    df_metrics = df_metrics[cols]
    df_metrics.columns = ['Pinball↓','MAE↓','RMSE↓','nMAE%↓','Skill↑','PICP↑','PI Width↓','Winkler↓']
    
    print("\n=== Comprehensive Model Comparison Matrix ===")
    print(df_metrics.to_string())
    
    # Verification of strict KPI limits
    print("\n=== Verification Checklist ===")
    print(f"  nMAE ≤ 4.0%  : {'✅ PASS' if metrics_test['nmae'] <= 4.0 else '❌ FAIL'} ({metrics_test['nmae']:.2f}%)")
    print(f"  Skill > 0.30 : {'✅ PASS' if metrics_test['skill'] > 0.30 else '❌ FAIL'} ({metrics_test['skill']:.3f})")
    print(f"  PICP in [0.78, 0.82]: {'✅ PASS' if 0.78 <= metrics_test['coverage'] <= 0.82 else '⚠️ WARNING'} ({metrics_test['coverage']:.3f})")
    
    # 6. Save Artifacts for Deployment
    os.makedirs('model_artefacts', exist_ok=True)
    shutil.copy(best_path, f'model_artefacts/lstm_q_best_h{horizon}.ckpt')
    joblib.dump(scaler, f'model_artefacts/feature_scaler_h{horizon}.pkl')
    with open(f'model_artefacts/lstm_features_h{horizon}.json', 'w') as f:
        json.dump(features, f, indent=2)
    df_metrics.reset_index().to_csv(f'model_artefacts/lstm_q_metrics_h{horizon}.csv', index=False)
    
    # Backward compatibility save for default horizon (48 steps = 24h ahead)
    if horizon == 48:
        shutil.copy(best_path, 'model_artefacts/lstm_q_best.ckpt')
        joblib.dump(scaler, 'model_artefacts/feature_scaler.pkl')
        with open('model_artefacts/lstm_features.json', 'w') as f:
            json.dump(features, f, indent=2)
        df_metrics.reset_index().to_csv('model_artefacts/lstm_q_metrics.csv', index=False)
    print("\n💾 Model artifacts saved successfully inside 'model_artefacts/'. Pipeline complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GridSight UK LSTM-Q Training Pipeline")
    parser.add_argument('--gold_dir', type=str, default=None, help='Relative path to Gold dataset Parquet directories.')
    parser.add_argument('--epochs', type=int, default=80, help='Maximum training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Model initial learning rate.')
    parser.add_argument('--horizon', type=int, default=48, help='Forecast horizon in 30-min steps (default 48 = 24h ahead).')
    
    args = parser.parse_args()
    
    # Determine default gold directory if not provided
    gold_dir = args.gold_dir
    if gold_dir is None:
        gold_dir = f'data/gold/gold_features_h{args.horizon}'
        
    try:
        run_pipeline(gold_dir, args.epochs, args.batch_size, args.lr, args.horizon)
    except Exception as e:
        print(f"\n❌ Pipeline execution terminated with error:\n{str(e)}")
        sys.exit(1)