# `src/gridsight/models/lstm/` — Probabilistic solar forecast: Standalone LSTM-Q Model

A standalone quantile (q10/q50/q90) sequence model built on the Gold feature store.

```
                  ┌─────────┐
Gold features ───►│ LSTM-Q  ├─► q10/q50/q90 (dynamic calibration + monotonic sort)
                  └─────────┘
```

- **LSTM-Q** — A Long Short-Term Memory network with a quantile output head. Processes continuous sequences of length `seq_len` half-hours.
- **Pinball Loss** — Trained directly on the pinball loss summed across the target quantiles (0.10, 0.50, 0.90).
- **Post-Hoc Dynamic Calibration** — Sweeps validation predictions to find a calibration factor $w / (\text{clearsky\_cos} + k)$ that rescales intervals and guarantees nominal coverage of exactly 80% under physical bounds.
- **Monotonic Sorting** — Predicted quantiles are sorted row-wise to enforce $q_{10} \le q_{50} \le q_{90}$ and eliminate quantile crossing.

## Layout
| File | Role |
|------|------|
| `config.py` | `ModelConfig` (quantiles, sequence history length, split dates, LSTM-specific hyperparams) |
| `data.py` | load Gold, chronological splits, daylight mask, sequence builder, standardizer |
| `common/clearsky.py` | Consolidated Clear-Sky GHI (Haurwitz) helper under common models |
| `base/lstm_q.py` | PyTorch LSTM model wrapper, custom pinball loss, and device selector |
| `common/metrics.py` | Consolidated pinball loss, coverage (PICP), crossing rate, and skill score vs NESO helper under common models |
| `train.py` | train LSTM with early stopping ➔ sweep dynamic calibration ➔ evaluate ➔ save artifacts |
| `tune.py` | random search hyperparameter tuning |
| `predict.py` | standalone inference helper for production pipelines |
| `evaluate.py` | metrics printing, fan chart and diagnostic dashboard generation |

## Setup
```bash
./venv/bin/pip install -r requirements.txt
```

## Run
```bash
# Full training with validation early stopping and dynamic calibration:
./venv/bin/python -m gridsight.models.lstm.train --horizon-steps 48 --gold-dir data/gold/gold_features_h48

# Fast smoke run (for quick validation and CPU testing):
./venv/bin/python -m gridsight.models.lstm.train --fast

# Hyperparameter search:
./venv/bin/python -m gridsight.models.lstm.tune --trials 15 --epochs 40

# Render evaluation fan chart and dashboard plots:
./venv/bin/python -m gridsight.models.lstm.evaluate --split test
```

Artifacts (`lstm.joblib`, `lstm.pt`) and `metrics.json` land in `artifacts/lstm/`.

## Inference Usage
```python
import pandas as pd
import glob
from gridsight.models.lstm.predict import predict_lstm_gold

# Load Gold features
gold = pd.concat([pd.read_parquet(f) for f in glob.glob("data/gold/gold_features/**/*.parquet", recursive=True)])

# Generate calibrated, sorted quantile forecasts (timestamp_utc, pred_q10/q50/q90 in MW)
forecast = predict_lstm_gold(gold)
```
