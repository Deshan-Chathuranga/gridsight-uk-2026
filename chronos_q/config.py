from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from gridsight.config import settings
    _DATA_DIR = settings.data_dir
except Exception:
    _DATA_DIR = Path("data")

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

NON_FEATURE_COLS = {
    "timestamp_utc", "target_mw", "target_cf",
    "pv_flag", "nwp_flag", "neso_flag", "ocf_flag", "has_full_history",
}


@dataclass
class ModelConfig:
    gold_dir: Path = _DATA_DIR / "gold" / "gold_features"
    artifacts_dir: Path = Path("artifacts/chronos")
    target: str = "target_cf"
    quantiles: tuple[float, ...] = QUANTILES
    horizon_steps: int = 48

    val_start: str = "2024-07-01"
    test_start: str = "2024-10-01"

    daylight_only: bool = True
    seed: int = 42

    model_name: str = "amazon/chronos-bolt-small"
    context_len: int = 512
    chronos_batch: int = 128
    forecast_stride: int = 1
    torch_dtype: str = "float32"
    torch_threads: int = 1

    def __post_init__(self):
        if self.horizon_steps != 48:
            self.artifacts_dir = Path(f"artifacts/chronos_h{self.horizon_steps}")

    def quantile_names(self) -> list[str]:
        return [f"q{int(q*100)}" for q in self.quantiles]
