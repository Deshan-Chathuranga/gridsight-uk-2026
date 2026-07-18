"""Hyperparameter tuning script for Standalone LSTM-Q forecasting.

Performs hyperparameter search over network architecture, learning rate, regularization,
and sequence history length using Optuna and logs trials using MLflow.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd
import optuna
import mlflow
from loguru import logger

from .config import ModelConfig
from .data import prepare, make_sequences, Standardizer
from .base import LSTMQuantile
from .train import _lstm_fit_with_early_stopping


def run_tuning(args) -> None:
    np.random.seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    logger.info(f"Starting Hyperparameter Tuning with Optuna: trials={args.trials}, max_epochs={args.epochs}")
    logger.info(f"Data summary: train_rows={len(train_rows)}, val_rows={len(val_rows)}, features={n_features}")

    # Set up MLflow experiment
    mlflow.set_experiment("GridSight-LSTM-Tuning")

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.3)
        lstm_lr = trial.suggest_float("lstm_lr", 1e-4, 1e-3, log=True)
        lstm_weight_decay = trial.suggest_float("lstm_weight_decay", 1e-5, 1e-2, log=True)
        seq_len = trial.suggest_categorical("seq_len", [48, 96, 126])

        logger.info(f"--- Trial {trial.number + 1}/{args.trials} ---")
        params = {
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout,
            "lstm_lr": lstm_lr,
            "lstm_weight_decay": lstm_weight_decay,
            "seq_len": seq_len,
        }
        logger.info(f"Parameters: {params}")

        cfg = dataclasses.replace(
            base_cfg,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            lstm_lr=lstm_lr,
            lstm_weight_decay=lstm_weight_decay,
            seq_len=seq_len,
            lstm_epochs=args.epochs,
        )

        # Make sequences for specific seq_len
        seqs, end_idx = make_sequences(Vs, cfg.seq_len)
        seqpos_of_row = np.full(len(df), -1, dtype="int64")
        seqpos_of_row[end_idx] = np.arange(len(end_idx))

        try:
            # Fit LSTM with early stopping
            logger.info("  Training LSTM model...")
            lstm = _lstm_fit_with_early_stopping(
                cfg, seqs, seqpos_of_row, train_rows, val_rows, y, n_features
            )

            # Predict on validation to evaluate best model performance
            logger.info("  Evaluating validation performance...")
            val_pos_ok = seqpos_of_row[val_rows] >= 0
            val_rows_ok = val_rows[val_pos_ok]

            # Calculate validation pinball loss
            from .base.lstm_q import pinball_loss_torch
            import torch
            dev = next(lstm.model_.parameters()).device
            lstm.model_.eval()

            with torch.no_grad():
                val_seqs_t = torch.from_numpy(seqs[seqpos_of_row[val_rows_ok]]).to(dev)
                val_y_t = torch.from_numpy(y[val_rows_ok]).to(dev)
                val_loss = float(pinball_loss_torch(lstm.model_(val_seqs_t), val_y_t, cfg.quantiles).item())

            logger.info(f"  Trial {trial.number + 1} complete. Val Pinball Loss = {val_loss:.5f}")

            # Log to MLflow nested run
            with mlflow.start_run(run_name=f"Trial_{trial.number + 1}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("val_loss", val_loss)

            return val_loss

        except Exception as e:
            logger.error(f"  Trial {trial.number + 1} failed: {e}")
            with mlflow.start_run(run_name=f"Trial_{trial.number + 1}_Failed", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("failed", 1)
            raise e

    # Run Optuna study inside a parent MLflow run
    with mlflow.start_run(run_name="Optuna Study Run"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)

        best_trial = study.best_trial
        logger.success(f"=== TUNING COMPLETE ===")
        logger.info(f"Best Trial: {best_trial.number + 1}")
        logger.info(f"Best Parameters: {best_trial.params}")
        logger.info(f"Best Validation Pinball Loss: {best_trial.value:.5f}")

        # Log study summary to MLflow parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_val_loss", best_trial.value)

        # Save best parameters
        out_dir = Path("configs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "best_lstm_params.json"
        out_file.write_text(json.dumps(best_trial.params, indent=2))
        logger.success(f"Saved best parameters to {out_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tune LSTM-Q hyperparameters using Optuna")
    p.add_argument("--trials", type=int, default=15, help="Number of search trials")
    p.add_argument("--epochs", type=int, default=40, help="Max epochs per model")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
