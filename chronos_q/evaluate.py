from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

QS = (0.1, 0.5, 0.9)
ACCENT, GREEN, BLUE, BG = "#FDB813", "#3DDC97", "#4DA3FF", "#0B1220"


def _load(artifacts_dir: str | Path, split: str):
    f = Path(artifacts_dir) / f"pred_{split}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found — run `python -m chronos_q` first.")
    df = pd.read_parquet(f).sort_values("timestamp_utc").reset_index(drop=True)
    is_cf = ("target" in df and df["target"].iloc[0] == "target_cf")
    scale = df["capacity_mwp"].to_numpy() if (is_cf and "capacity_mwp" in df) else 1.0
    for c in ["y_true", "q10", "q50", "q90"] + (["neso"] if "neso" in df else []):
        df[c + "_mw"] = df[c].to_numpy() * scale
    return df


def _metrics(df: pd.DataFrame) -> dict:
    preds = {q: df[f"q{int(q*100)}"].to_numpy() for q in QS}
    y = df["y_true"].to_numpy()
    rep = metrics.report(y, preds, QS)
    if "neso" in df:
        rep["skill_vs_neso_q50"] = metrics.skill_vs_baseline(y, preds[0.5], df["neso"].to_numpy())
    return rep


def make_charts(artifacts_dir: str | Path = "artifacts/chronos", split: str = "test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "text.color": "#E6EDF7",
                         "axes.labelcolor": "#C7D2E0", "xtick.color": "#9FB0C7",
                         "ytick.color": "#9FB0C7", "axes.edgecolor": "#2A3550",
                         "font.size": 10, "axes.grid": True, "grid.color": "#1B2740"})
    out = Path(artifacts_dir); plots = out / "plots"; plots.mkdir(parents=True, exist_ok=True)
    df = _load(out, split)
    rep = _metrics(df)
    print(f"=== Chronos-Q {split} metrics ===")
    for k, v in rep.items():
        print(f"  {k:24s} {v:.4f}")

    win = df.iloc[:min(len(df), 480)]
    t = pd.to_datetime(win["timestamp_utc"])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(t, win["q10_mw"], win["q90_mw"], color=ACCENT, alpha=0.22,
                    label="q10–q90 (80% interval)")
    ax.plot(t, win["q50_mw"], color=ACCENT, lw=2, label="q50 (median forecast)")
    ax.plot(t, win["y_true_mw"], color=GREEN, lw=1.6, label="Actual")
    if "neso_mw" in win:
        ax.plot(t, win["neso_mw"], color=BLUE, lw=1, ls="--", alpha=0.8, label="NESO baseline")
    ax.set_title(f"Chronos-Q Probabilistic forecast vs actual — {split} set", color="#E6EDF7")
    ax.set_ylabel("Generation (MW)"); ax.legend(loc="upper right", framealpha=0.1, ncol=2)
    fig.tight_layout(); fig.savefig(plots / "evaluation_fan.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    y = df["y_true"].to_numpy()
    preds = {q: df[f"q{int(q*100)}"].to_numpy() for q in QS}

    ax = axes[0, 0]
    emp = [float(np.mean(y <= preds[q])) for q in QS]
    ax.plot([0, 1], [0, 1], color="#5A6B86", ls="--", label="perfect")
    ax.plot(QS, emp, "o-", color=ACCENT, lw=2, ms=8, label="model")
    for q, e in zip(QS, emp):
        ax.annotate(f"{e:.2f}", (q, e), textcoords="offset points", xytext=(6, -10), color="#C7D2E0")
    ax.set_title("Calibration (reliability)"); ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical P(y ≤ q)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(framealpha=0.1)

    ax = axes[0, 1]
    a, p = df["y_true_mw"].to_numpy(), df["q50_mw"].to_numpy()
    ax.scatter(a, p, s=6, alpha=0.25, color=ACCENT)
    lim = max(a.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#5A6B86", ls="--")
    mae = float(np.mean(np.abs(a - p)))
    ax.set_title(f"q50 vs actual  ·  MAE = {mae:,.0f} MW")
    ax.set_xlabel("Actual (MW)"); ax.set_ylabel("Forecast q50 (MW)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = axes[1, 0]
    df["hour"] = pd.to_datetime(df["timestamp_utc"]).dt.hour
    h_mae = df.groupby("hour").apply(lambda g: np.mean(np.abs(g["y_true_mw"] - g["q50_mw"])), include_groups=False)
    ax.bar(h_mae.index, h_mae.values, color=ACCENT, alpha=0.6, width=0.7, label="q50 MAE")
    ax.set_title("Forecast Error by Hour of Day (MW)")
    ax.set_xlabel("Hour of Day (UTC)"); ax.set_ylabel("MAE (MW)")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.5, 23.5)

    ax = axes[1, 1]
    ax.axis("off")
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor="#0E192E", edgecolor="#2A3550", lw=2, transform=ax.transAxes)
    ax.add_patch(rect)
    txt = (
        f"CHRONOS-Q PERFORMANCE CARD\n"
        f"Split: {split.upper()} set\n"
        f"-----------------------------\n"
        f"Pinball loss (q10): {rep['pinball_q10']:.4f}\n"
        f"Pinball loss (q50): {rep['pinball_q50']:.4f}\n"
        f"Pinball loss (q90): {rep['pinball_q90']:.4f}\n"
        f"MEAN PINBALL LOSS:  {rep['mean_pinball']:.4f}\n\n"
        f"Empirical Coverage:  {rep['coverage_10-90']:.2%}\n"
        f"Target Coverage:     {rep['target_coverage']:.2%}\n"
        f"Crossing Rate:       {rep['crossing_rate']:.2%}\n"
    )
    if "skill_vs_neso_q50" in rep:
        txt += f"Skill vs NESO:       {rep['skill_vs_neso_q50']:.2%}\n"
    ax.text(0.15, 0.15, txt, fontfamily="monospace", fontsize=11, color="#E6EDF7", transform=ax.transAxes, verticalalignment="bottom")

    fig.tight_layout(); fig.savefig(plots / "evaluation_dashboard.png", dpi=130); plt.close(fig)
    print(f"Saved evaluation charts under {plots}/")


def main():
    p = argparse.ArgumentParser(description="Evaluate trained Chronos-Q forecasts")
    p.add_argument("--artifacts", default="artifacts/chronos", help="Path to saved pred parquet files")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()
    make_charts(args.artifacts, args.split)


if __name__ == "__main__":
    main()
