"""Central configuration for the probabilistic solar-forecasting stack.

Stack = TCN-Q + LGBM-Q base learners -> Linear-Q meta-learner, all quantile
(q10/q50/q90). Everything reads the LOCAL Gold feature store (no network).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from gridsight.config import settings
    _DATA_DIR = settings.data_dir
except Exception:  # modeling must run even if config import fails
    _DATA_DIR = Path("data")

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Columns that are never model inputs (targets, keys, QA flags).
NON_FEATURE_COLS = {
    "timestamp_utc", "target_mw", "target_cf",
    "pv_flag", "nwp_flag", "neso_flag", "ocf_flag", "has_full_history",
}


@dataclass
class ModelConfig:
    # ---- data ----
    gold_dir: Path = _DATA_DIR / "gold" / "gold_features"
    artifacts_dir: Path = Path("artifacts/model")
    target: str = "target_cf"          # predict capacity factor, convert to MW with capacity_mwp
    quantiles: tuple[float, ...] = QUANTILES
    horizon_steps: int = 48            # must match the Gold build horizon

    # ---- chronological split (no shuffle: this is a time series) ----
    val_start: str = "2024-07-01"
    test_start: str = "2024-10-01"

    # ---- modelling scope ----
    daylight_only: bool = True         # train/score only sun-up slots; night is predicted 0
    seed: int = 42

    # ---- TCN-Q (sequence model, 63h diurnal context) ----
    seq_len: int = 126                 # 63h * 2 (half-hourly)
    tcn_channels: tuple[int, ...] = (64, 64, 64, 64, 64, 64)
    tcn_kernel: int = 3
    tcn_dropout: float = 0.1
    tcn_lr: float = 1e-3
    tcn_epochs: int = 30
    tcn_batch: int = 256
    torch_threads: int = 1             # CPU thread cap. 1 avoids OpenMP oversubscription
                                       # (torch vs LightGBM's libomp) that stalls CPU training.
                                       # Ignored on GPU (Colab) — raise it on a clean CPU box.

    # ---- LGBM-Q (tabular / clear-sky model) ----
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    })

    # ---- stacking ----
    n_folds: int = 5                   # out-of-fold CV for the meta-learner

    def quantile_names(self) -> list[str]:
        return [f"q{int(q*100)}" for q in self.quantiles]
