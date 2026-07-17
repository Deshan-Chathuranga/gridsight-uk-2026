from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def mean_pinball(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    return float(np.mean([pinball_loss(y_true, p, q) for q, p in preds.items()]))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def crossing_rate(preds: dict[float, np.ndarray]) -> float:
    qs = sorted(preds)
    stacked = np.stack([preds[q] for q in qs], axis=1)
    return float(np.mean(np.any(np.diff(stacked, axis=1) < -1e-9, axis=1)))


def skill_vs_baseline(y_true, q50_pred, baseline_pred) -> float:
    mae_model = np.mean(np.abs(y_true - q50_pred))
    mae_base = np.mean(np.abs(y_true - baseline_pred))
    return float(1.0 - mae_model / mae_base) if mae_base > 0 else float("nan")


def report(y_true: np.ndarray, preds: dict[float, np.ndarray],
           quantiles=(0.1, 0.5, 0.9)) -> dict:
    lo, hi = quantiles[0], quantiles[-1]
    out = {f"pinball_q{int(q*100)}": pinball_loss(y_true, preds[q], q) for q in quantiles}
    out["mean_pinball"] = mean_pinball(y_true, preds)
    out["coverage_%d-%d" % (int(lo*100), int(hi*100))] = coverage(y_true, preds[lo], preds[hi])
    out["target_coverage"] = hi - lo
    out["crossing_rate"] = crossing_rate(preds)
    return out
