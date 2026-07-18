"""Central configuration for the standalone LSTM-Q model.

Everything reads the LOCAL Gold feature store (no network).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from gridsight.config import settings
    _DATA_DIR = settings.data_dir
except Exception:  # modeling must run even if config import fails
    _DATA_DIR = Path("data")

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Columns that are never model inputs (targets, keys, QA flags, and raw MW values).
NON_FEATURE_COLS = {
    "timestamp_utc", "target_mw", "target_cf",
    "pv_flag", "nwp_flag", "neso_flag", "ocf_flag", "has_full_history",
    "capacity_mwp",
    "embedded_solar_mw", "embedded_wind_mw",
    "embedded_solar_capacity_mw", "embedded_wind_capacity_mw",
    "gen_lag_48", "gen_lag_96", "gen_lag_144", "gen_lag_336",
    "gen_roll_mean_48", "gen_roll_mean_336", "gen_roll_std_48",
    "ocf_lag_48", "ocf_lag_96", "ocf_lag_144", "ocf_lag_336",
    "ocf_roll_mean_48"
}


@dataclass
class ModelConfig:
    # ---- data ----
    gold_dir: Path = _DATA_DIR / "gold" / "gold_features"
    artifacts_dir: Path = Path("artifacts/lstm")
    target: str = "target_cf"          # predict capacity factor, convert to MW with capacity_mwp
    quantiles: tuple[float, ...] = QUANTILES
    horizon_steps: int = 48            # must match the Gold build horizon

    # ---- chronological split (no shuffle: this is a time series) ----
    val_start: str = "2024-07-01"
    test_start: str = "2024-10-01"

    # ---- modelling scope ----
    daylight_only: bool = True         # train/score only sun-up slots; night is predicted 0
    seed: int = 42

    # ---- LSTM-Q (sequence model) ----
    seq_len: int = 126                 # 63h * 2 (half-hourly)
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.20
    lstm_lr: float = 3e-4
    lstm_epochs: int = 30
    lstm_batch: int = 256
    lstm_weight_decay: float = 1e-4
    torch_threads: int = 1             # CPU thread cap.

    def __post_init__(self):
        # Dynamically separate artifact directories by horizon steps if not default (48)
        if self.horizon_steps != 48:
            self.artifacts_dir = Path(f"artifacts/lstm_h{self.horizon_steps}")

    def quantile_names(self) -> list[str]:
        return [f"q{int(q*100)}" for q in self.quantiles]
