"""Train the standalone LSTM-Q model and evaluate on the test split.

This pipeline:
  1. Prepares Gold data and chronological train/val/test splits.
  2. Trains LSTM-Q on the train block with early stopping on validation loss.
  3. Sweeps post-hoc calibration factors on validation to optimize 80% coverage.
  4. Evaluates predictions on val & test splits with monotonic sorting.
  5. Saves weights (lstm.pt) and metadata (lstm.joblib) under artifacts/lstm/.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import argparse
import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from .config import ModelConfig
from .data import prepare, make_sequences, Standardizer
from .clearsky import clearsky_feature
from .base import LSTMQuantile
from . import metrics


def _lstm_fit_with_early_stopping(cfg, seqs, seqpos_of_row, tr_rows, val_rows, y, n_features) -> LSTMQuantile:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .base.lstm_q import device, pinball_loss_torch

    dev = device()
    model_wrapper = LSTMQuantile(cfg, n_features)
    model_wrapper.build()
    model = model_wrapper.model_.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)

    # prepare dataloaders
    tr_pos = seqpos_of_row[tr_rows]
    tr_ok = tr_pos >= 0
    ds_tr = TensorDataset(torch.from_numpy(seqs[tr_pos[tr_ok]]), torch.from_numpy(y[tr_rows[tr_ok]]))
    dl_tr = DataLoader(ds_tr, batch_size=cfg.lstm_batch, shuffle=True)

    val_pos = seqpos_of_row[val_rows]
    val_ok = val_pos >= 0
    ds_val = TensorDataset(torch.from_numpy(seqs[val_pos[val_ok]]), torch.from_numpy(y[val_rows[val_ok]]))
    dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 10

    E = cfg.lstm_epochs
    for ep in range(E):
        model.train()
        tr_loss, tr_nb = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = pinball_loss_torch(model(xb), yb, cfg.quantiles)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            tr_nb += 1

        # validation loop
        model.eval()
        val_loss, val_nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = pinball_loss_torch(model(xb), yb, cfg.quantiles)
                val_loss += float(loss.item())
                val_nb += 1

        tr_avg = tr_loss / max(tr_nb, 1)
        val_avg = val_loss / max(val_nb, 1)

        # check early stopping
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if ep == 0 or (ep + 1) % max(1, E // 5) == 0 or ep == E - 1:
            logger.info(f"    LSTM[{dev}] epoch {ep+1}/{E} · train_pinball={tr_avg:.4f} · val_pinball={val_avg:.4f}")

        if patience_counter >= patience:
            logger.info(f"    Early stopping triggered at epoch {ep+1} · Best val loss = {best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model_wrapper


def run_lstm_pipeline(cfg: ModelConfig) -> dict:
    np.random.seed(cfg.seed)
    ds = prepare(cfg)
    qs = cfg.quantiles
    df = ds.df
    V = df[ds.feature_cols].to_numpy("float32")
    y = df[cfg.target].to_numpy("float32")
    cap = df["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in df else None

    tr_mask, va_mask, te_mask = ds.split_masks()
    score = ds.score_mask()

    train_rows = np.where(tr_mask & score)[0]
    val_rows = np.where(va_mask & score)[0]
    test_rows = np.where(te_mask & score)[0]

    logger.info(f"[LSTM] rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} | features={len(ds.feature_cols)}")

    # Standardizer
    std = Standardizer().fit(V[tr_mask])
    Vs = std.transform(V)
    seqs, end_idx = make_sequences(Vs, cfg.seq_len)
    seqpos_of_row = np.full(len(df), -1, dtype="int64")
    seqpos_of_row[end_idx] = np.arange(len(end_idx))

    # 1. Fit LSTM base model on full train block with early stopping
    logger.info("Fitting LSTM model with early stopping on validation split...")
    lstm = _lstm_fit_with_early_stopping(cfg, seqs, seqpos_of_row, train_rows, val_rows, y, len(ds.feature_cols))

    # 2. Get uncalibrated validation predictions to sweep calibration factor
    logger.info("Sweeping validation split to find optimal post-hoc calibration factor...")
    val_pos_ok = seqpos_of_row[val_rows] >= 0
    val_rows_ok = val_rows[val_pos_ok]
    preds_val_raw = lstm.predict(seqs[seqpos_of_row[val_rows_ok]]) # dict of q -> capacity_factor

    y_val_cf = y[val_rows_ok]
    y_val_mw = df["target_mw"].to_numpy("float32")[val_rows_ok] if "target_mw" in df else y_val_cf
    cap_val = cap[val_rows_ok] if cap is not None else np.ones(len(val_rows_ok), dtype="float32")

    cos_val = df["clearsky_cos"].to_numpy("float32")[val_rows_ok]

    best_w, best_k = 1.2, 0.5
    best_diff = 1.0
    best_cov = 0.8
    best_pinball = 999.0

    # Grid search for w (base multiplier) and k (offset)
    W_space = np.linspace(0.1, 3.0, 150)
    K_space = np.linspace(0.01, 1.0, 100)

    for k in K_space:
        for w in W_space:
            factor = w / (cos_val + k)
            q50 = np.clip(preds_val_raw[0.5] * cap_val, 0, None)
            q10 = np.clip((preds_val_raw[0.5] - (preds_val_raw[0.5] - preds_val_raw[0.1]) * factor) * cap_val, 0, None)
            q90 = np.clip((preds_val_raw[0.5] + (preds_val_raw[0.9] - preds_val_raw[0.5]) * factor) * cap_val, 0, None)
            q10 = np.minimum(q10, q50)
            q90 = np.maximum(q90, q50)
            cov = np.mean((y_val_mw >= q10) & (y_val_mw <= q90))
            diff = abs(cov - 0.80)

            # Constraint: validation coverage must be close to 80%
            if diff < 0.005:
                p_dict = {
                    0.1: (preds_val_raw[0.5] - (preds_val_raw[0.5] - preds_val_raw[0.1]) * factor),
                    0.5: preds_val_raw[0.5],
                    0.9: (preds_val_raw[0.5] + (preds_val_raw[0.9] - preds_val_raw[0.5]) * factor)
                }
                p_dict[0.1] = np.minimum(p_dict[0.1], p_dict[0.5])
                p_dict[0.9] = np.maximum(p_dict[0.9], p_dict[0.5])

                rep = metrics.report(y_val_cf, p_dict, qs)
                p_loss = rep["mean_pinball"]
                if p_loss < best_pinball:
                    best_pinball = p_loss
                    best_diff = diff
                    best_w = w
                    best_k = k
                    best_cov = cov

    best_factor = (float(best_w), float(best_k))
    logger.info(f"Optimal post-hoc dynamic calibration selected: factor = {best_w:.3f} / (clearsky_cos + {best_k:.3f}) (Val Coverage: {best_cov:.2%})")

    # Helper function to predict and calibrate a split
    def predict_and_calibrate(rows_split) -> tuple[dict[float, np.ndarray], np.ndarray]:
        pos = seqpos_of_row[rows_split]
        ok = pos >= 0
        r_ok = rows_split[ok]
        raw_p = lstm.predict(seqs[pos[ok]])
        
        q50 = raw_p[0.5]
        w, k = best_factor
        cos_split = df["clearsky_cos"].to_numpy("float32")[r_ok]
        factor = w / (cos_split + k)
        
        q10 = q50 - (q50 - raw_p[0.1]) * factor
        q90 = q50 + (raw_p[0.9] - q50) * factor
        
        # Enforce monotonicity
        q10 = np.minimum(np.maximum(q10, 0), q50)
        q90 = np.maximum(q90, q50)
        
        return {0.1: q10, 0.5: q50, 0.9: q90}, r_ok

    # Evaluate splits
    results = {}
    ts = df["timestamp_utc"].to_numpy()
    for name, rows_split in [("val", val_rows), ("test", test_rows)]:
        preds, r_ok = predict_and_calibrate(rows_split)
        
        # metrics are computed on capacity factor scale
        rep = metrics.report(y[r_ok], preds, qs)
        
        # Skill vs baseline
        base_cf = None
        if "embedded_solar_mw" in df.columns and cap is not None:
            base = df["embedded_solar_mw"].to_numpy("float32")[r_ok]
            base_cf = base / np.clip(cap[r_ok], 1e-6, None) if cfg.target == "target_cf" else base
            rep["skill_vs_neso_q50"] = metrics.skill_vs_baseline(y[r_ok], preds[0.5], base_cf)
            
        results[name] = rep
        logger.info(f"[LSTM {name}] " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()))

        # Save predictions (on capacity factor scale if target_cf)
        cap_split = cap[r_ok] if cap is not None else np.ones(len(r_ok), dtype="float32")
        _save_lstm_predictions(cfg, name, ts[r_ok], y[r_ok], preds,
                               cap_split, base_cf)

    _save_lstm_artifacts(cfg, lstm, std, ds.feature_cols, best_factor, results)
    return results


def _save_lstm_predictions(cfg, name, ts, y_true, preds, cap, base):
    if cfg.horizon_steps != 48:
        out_dir = Path(cfg.artifacts_dir).parent / f"lstm_h{cfg.horizon_steps}"
    else:
        out_dir = Path(cfg.artifacts_dir).parent / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"timestamp_utc": ts, "y_true": y_true})
    for q in cfg.quantiles:
        out[f"q{int(q*100)}"] = preds[q]
    if cap is not None:
        out["capacity_mwp"] = cap
    if base is not None:
        out["neso"] = base
    out["target"] = cfg.target
    out.to_parquet(out_dir / f"pred_{name}.parquet", index=False)


def _save_lstm_artifacts(cfg, lstm, std, feat_cols, calib_factor, results):
    if cfg.horizon_steps != 48:
        out_dir = Path(cfg.artifacts_dir).parent / f"lstm_h{cfg.horizon_steps}"
    else:
        out_dir = Path(cfg.artifacts_dir).parent / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib, torch
        joblib.dump({"standardizer": std, "features": feat_cols,
                     "cfg": cfg, "calib_factor": calib_factor}, out_dir / "lstm.joblib")
        torch.save(lstm.model_.state_dict(), out_dir / "lstm.pt")
    except Exception as e:
        logger.warning(f"LSTM artifact save failed: {e}")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.success(f"LSTM artifacts + metrics -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Standalone LSTM-Q forecasting pipeline")
    p.add_argument("--target", default="target_cf", choices=["target_cf", "target_mw"])
    p.add_argument("--horizon-steps", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="LSTM training epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gold-dir", default=None)
    p.add_argument("--fast", action="store_true", help="quick smoke run")
    args = p.parse_args()

    cfg = ModelConfig()

    # Try loading best params from tuning
    best_params_path = Path("configs/best_lstm_params.json")
    if best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text())
            logger.info(f"Loading tuned hyperparameters from {best_params_path}: {best_params}")
            cfg = dataclasses.replace(
                cfg,
                lstm_hidden=best_params.get("lstm_hidden", cfg.lstm_hidden),
                lstm_layers=best_params.get("lstm_layers", cfg.lstm_layers),
                lstm_dropout=best_params.get("lstm_dropout", cfg.lstm_dropout),
                lstm_lr=best_params.get("lstm_lr", cfg.lstm_lr),
                lstm_weight_decay=best_params.get("lstm_weight_decay", cfg.lstm_weight_decay),
                seq_len=best_params.get("seq_len", cfg.seq_len)
            )
        except Exception as e:
            logger.warning(f"Failed to load tuned parameters: {e}")

    # Override defaults with CLI args if provided
    seq_len = args.seq_len if args.seq_len is not None else cfg.seq_len
    epochs = args.epochs if args.epochs is not None else cfg.lstm_epochs
    lr = args.lr if args.lr is not None else cfg.lstm_lr

    if args.gold_dir:
        cfg = dataclasses.replace(cfg, gold_dir=Path(args.gold_dir))
    cfg = dataclasses.replace(
        cfg,
        target=args.target, horizon_steps=args.horizon_steps,
        seq_len=seq_len, lstm_epochs=epochs,
        lstm_batch=args.batch_size, lstm_lr=lr,
    )
    if args.fast:
        cfg = dataclasses.replace(
            cfg, lstm_epochs=2, seq_len=48, lstm_hidden=32,
        )

    logger.info(f"LSTM config: target={cfg.target} seq_len={cfg.seq_len} epochs={cfg.lstm_epochs} batch={cfg.lstm_batch} lr={cfg.lstm_lr} hidden={cfg.lstm_hidden} layers={cfg.lstm_layers} dropout={cfg.lstm_dropout} weight_decay={cfg.lstm_weight_decay}")
    run_lstm_pipeline(cfg)


if __name__ == "__main__":
    main()
