"""Hyperparameter tuning script for Standalone LSTM-Q forecasting.

Performs a Random Search over network architecture, learning rate, regularization,
and sequence history length, using validation split early stopping.
"""
from __future__ import annotations
import json
import argparse
import random
from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd
from loguru import logger

from .config import ModelConfig
from .data import prepare, make_sequences, Standardizer
from .base import LSTMQuantile
from .train import _lstm_fit_with_early_stopping

# Search Space definitions
SEARCH_SPACE = {
    "lstm_hidden": [32, 64, 128],
    "lstm_layers": [1, 2],
    "lstm_dropout": [0.0, 0.1, 0.2, 0.3],
    "lstm_lr": [1e-4, 3e-4, 1e-3],
    "lstm_weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "seq_len": [48, 96, 126]
}


def sample_hyperparameters() -> dict:
    """Randomly sample a set of hyperparameters from the search space."""
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def run_tuning(args) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_cfg = ModelConfig()
    ds = prepare(base_cfg)
    df = ds.df
    V = df[ds.feature_cols].to_numpy("float32")
    y = df[base_cfg.target].to_numpy("float32")

    tr_mask, va_mask, _ = ds.split_masks()
    score = ds.score_mask()

    train_rows = np.where(tr_mask & score)[0]
    val_rows = np.where(va_mask & score)[0]

    # Standardize tabular features
    std = Standardizer().fit(V[tr_mask])
    Vs = std.transform(V)

    n_features = len(ds.feature_cols)

    logger.info(f"Starting Hyperparameter Tuning: trials={args.trials}, max_epochs={args.epochs}")
    logger.info(f"Data summary: train_rows={len(train_rows)}, val_rows={len(val_rows)}, features={n_features}")

    results = []

    for trial_idx in range(1, args.trials + 1):
        params = sample_hyperparameters()
        logger.info(f"--- Trial {trial_idx}/{args.trials} ---")
        logger.info(f"Parameters: {params}")

        # Set up config for this trial
        cfg = dataclasses.replace(
            base_cfg,
            lstm_hidden=params["lstm_hidden"],
            lstm_layers=params["lstm_layers"],
            lstm_dropout=params["lstm_dropout"],
            lstm_lr=params["lstm_lr"],
            lstm_weight_decay=params["lstm_weight_decay"],
            seq_len=params["seq_len"],
            lstm_epochs=args.epochs,
        )

        # Make sequences for the trial's specific seq_len
        seqs, end_idx = make_sequences(Vs, cfg.seq_len)
        seqpos_of_row = np.full(len(df), -1, dtype="int64")
        seqpos_of_row[end_idx] = np.arange(len(end_idx))

        try:
            # Fit LSTM with early stopping on validation split
            logger.info("  Training LSTM model...")
            lstm = _lstm_fit_with_early_stopping(
                cfg, seqs, seqpos_of_row, train_rows, val_rows, y, n_features
            )

            # Predict on validation to evaluate best model performance
            logger.info("  Evaluating validation performance...")
            val_pos_ok = seqpos_of_row[val_rows] >= 0
            val_rows_ok = val_rows[val_pos_ok]
            preds_val = lstm.predict(seqs[seqpos_of_row[val_rows_ok]])

            # Calculate validation pinball loss (mean over all quantiles)
            from .base.lstm_q import pinball_loss_torch
            import torch
            dev = next(lstm.model_.parameters()).device
            lstm.model_.eval()

            with torch.no_grad():
                val_seqs_t = torch.from_numpy(seqs[seqpos_of_row[val_rows_ok]]).to(dev)
                val_y_t = torch.from_numpy(y[val_rows_ok]).to(dev)
                val_loss = float(pinball_loss_torch(lstm.model_(val_seqs_t), val_y_t, cfg.quantiles).item())

            logger.info(f"  Trial {trial_idx} complete. Val Pinball Loss = {val_loss:.5f}")

            results.append({
                "trial": trial_idx,
                "params": params,
                "val_loss": val_loss
            })

        except Exception as e:
            logger.error(f"  Trial {trial_idx} failed: {e}")

    if not results:
        logger.error("All trials failed!")
        return

    # Find the best trial
    results.sort(key=lambda x: x["val_loss"])
    best_trial = results[0]
    logger.success(f"=== TUNING COMPLETE ===")
    logger.info(f"Best Trial: {best_trial['trial']}")
    logger.info(f"Best Parameters: {best_trial['params']}")
    logger.info(f"Best Validation Pinball Loss: {best_trial['val_loss']:.5f}")

    # Save best parameters
    out_dir = Path("configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "best_lstm_params.json"
    out_file.write_text(json.dumps(best_trial["params"], indent=2))
    logger.success(f"Saved best parameters to {out_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tune LSTM-Q hyperparameters")
    p.add_argument("--trials", type=int, default=15, help="Number of search trials")
    p.add_argument("--epochs", type=int, default=40, help="Max epochs per model")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
