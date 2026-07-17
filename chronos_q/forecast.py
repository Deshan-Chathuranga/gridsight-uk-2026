from __future__ import annotations
import json
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .config import ModelConfig
from .data import prepare
from .base import ChronosQuantile
from . import metrics


def _target_rows(mask, score, horizon, stride):
    rows = np.where(mask & score)[0]
    rows = rows[rows >= horizon]
    if stride > 1:
        rows = rows[::stride]
    return rows


def _calibrate(raw, cos, w, k):
    factor = w / (cos + k)
    q50 = raw[0.5]
    q10 = q50 - (q50 - raw[0.1]) * factor
    q90 = q50 + (raw[0.9] - q50) * factor
    q10 = np.minimum(np.maximum(q10, 0), q50)
    q90 = np.maximum(q90, q50)
    return {0.1: q10, 0.5: q50, 0.9: q90}


def run_chronos_pipeline(cfg: ModelConfig) -> dict:
    np.random.seed(cfg.seed)
    ds = prepare(cfg)
    qs = cfg.quantiles
    df = ds.df
    H = cfg.horizon_steps

    series = df[cfg.target].to_numpy("float32")
    y = df[cfg.target].to_numpy("float32")
    cap = df["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in df else None
    cos = df["clearsky_cos"].to_numpy("float32") if "clearsky_cos" in df else np.ones(len(df), "float32")
    ts = df["timestamp_utc"].to_numpy()

    tr_mask, va_mask, te_mask = ds.split_masks()
    score = ds.score_mask()

    val_rows = _target_rows(va_mask, score, H, cfg.forecast_stride)
    test_rows = _target_rows(te_mask, score, H, cfg.forecast_stride)

    logger.info(f"[Chronos] model={cfg.model_name} horizon={H} context={cfg.context_len} "
                f"| val_rows={len(val_rows)} test_rows={len(test_rows)}")

    chronos = ChronosQuantile(cfg).load()

    logger.info("Zero-shot forecasting validation origins...")
    raw_val = chronos.forecast_h_ahead(series, val_rows - H, H)

    y_val = y[val_rows]
    cos_val = cos[val_rows]

    best_w, best_k, best_cov, best_pinball = 1.0, 0.5, 0.8, 999.0
    W_space = np.linspace(0.1, 3.0, 150)
    K_space = np.linspace(0.01, 1.0, 100)
    for k in K_space:
        for w in W_space:
            p = _calibrate(raw_val, cos_val, w, k)
            cov = metrics.coverage(y_val, p[0.1], p[0.9])
            if abs(cov - 0.80) < 0.005:
                p_loss = metrics.mean_pinball(y_val, p)
                if p_loss < best_pinball:
                    best_pinball, best_w, best_k, best_cov = p_loss, w, k, cov

    best_factor = (float(best_w), float(best_k))
    logger.info(f"Calibration: factor = {best_w:.3f} / (clearsky_cos + {best_k:.3f}) "
                f"(Val Coverage: {best_cov:.2%})")

    results = {}
    for name, rows, raw in [("val", val_rows, raw_val),
                            ("test", test_rows, None)]:
        if raw is None:
            logger.info(f"Zero-shot forecasting {name} origins...")
            raw = chronos.forecast_h_ahead(series, rows - H, H)
        preds = _calibrate(raw, cos[rows], best_w, best_k)

        rep = metrics.report(y[rows], preds, qs)
        base_cf = None
        if "embedded_solar_mw" in df.columns and cap is not None:
            base = df["embedded_solar_mw"].to_numpy("float32")[rows]
            base_cf = base / np.clip(cap[rows], 1e-6, None) if cfg.target == "target_cf" else base
            rep["skill_vs_neso_q50"] = metrics.skill_vs_baseline(y[rows], preds[0.5], base_cf)
        results[name] = rep
        logger.info(f"[Chronos {name}] " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()))

        cap_split = cap[rows] if cap is not None else np.ones(len(rows), "float32")
        _save_predictions(cfg, name, ts[rows], y[rows], preds, cap_split, base_cf)

    _save_artifacts(cfg, best_factor, results)
    return results


def _save_predictions(cfg, name, ts, y_true, preds, cap, base):
    out_dir = Path(cfg.artifacts_dir)
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


def _save_artifacts(cfg, calib_factor, results):
    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"cfg": cfg, "calib_factor": calib_factor,
                     "model_name": cfg.model_name}, out_dir / "chronos.joblib")
    except Exception as e:
        logger.warning(f"Chronos artifact save failed: {e}")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.success(f"Chronos artifacts + metrics -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot Chronos-Q probabilistic forecasting pipeline")
    p.add_argument("--target", default="target_cf", choices=["target_cf", "target_mw"])
    p.add_argument("--horizon-steps", type=int, default=48)
    p.add_argument("--model-name", default=None)
    p.add_argument("--context-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--gold-dir", default=None)
    p.add_argument("--fast", action="store_true", help="quick smoke run")
    args = p.parse_args()

    cfg = ModelConfig()
    if args.gold_dir:
        cfg = dataclasses.replace(cfg, gold_dir=Path(args.gold_dir))
    cfg = dataclasses.replace(
        cfg,
        target=args.target,
        horizon_steps=args.horizon_steps,
        model_name=args.model_name or cfg.model_name,
        context_len=args.context_len or cfg.context_len,
        chronos_batch=args.batch_size or cfg.chronos_batch,
        forecast_stride=args.stride or cfg.forecast_stride,
    )
    if args.fast:
        cfg = dataclasses.replace(
            cfg, model_name="amazon/chronos-bolt-tiny",
            context_len=128, forecast_stride=24,
        )

    logger.info(f"Chronos config: target={cfg.target} horizon={cfg.horizon_steps} "
                f"model={cfg.model_name} context={cfg.context_len} "
                f"batch={cfg.chronos_batch} stride={cfg.forecast_stride}")
    run_chronos_pipeline(cfg)


if __name__ == "__main__":
    main()
