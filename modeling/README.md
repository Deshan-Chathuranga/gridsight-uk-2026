# `modeling/` — Probabilistic solar forecast: TCN-Q + LGBM-Q → Linear-Q stack

A quantile (q10/q50/q90) forecasting stack built on the Gold feature store.

```
                 ┌─ TCN-Q  (sequence, 63h diurnal context)  ─┐  q10/q50/q90
Gold features ───┤                                            ├─► Linear-Q meta ─► q10/q50/q90
                 └─ LGBM-Q (tabular / clear-sky features)   ─┘   + Clear-Sky GHI    (sorted)
```

- **TCN-Q** — Temporal Convolutional Network (dilated *causal* convs) over the last
  `seq_len` half-hours; one forward pass emits all quantiles (pinball loss).
- **LGBM-Q** — one LightGBM per quantile on the tabular features (weather, NESO,
  calendar, solar, lags). Native NaN handling.
- **Linear-Q meta** — Linear Quantile Regression over the 6 base quantiles **+
  Clear-Sky GHI**; outputs are **sorted per row** so q10 ≤ q50 ≤ q90 never cross.
- **OOF stacking** — base predictions for the meta-learner are generated
  out-of-fold (`TimeSeriesSplit`, `n_folds`), so the meta-learner never trains on
  a base prediction made by a model that saw that row.

## Layout
| File | Role |
|------|------|
| `config.py` | `ModelConfig` (quantiles, horizon, split dates, TCN/LGBM hyperparams) |
| `data.py` | load Gold, chronological splits, daylight mask, sequence builder, standardizer |
| `clearsky.py` | Clear-Sky GHI (Haurwitz) from solar elevation |
| `base/lgbm_q.py` | LGBM-Q base learner |
| `base/tcn_q.py` | TCN-Q base learner (PyTorch) |
| `stacking.py` | Linear-Q meta-learner (+ monotonic sort) |
| `metrics.py` | pinball, coverage (PICP), crossing rate, skill vs NESO |
| `train.py` | OOF → meta fit → refit → evaluate → save artifacts |
| `cli.py` | command-line entry |

## Setup
```bash
./venv/bin/pip install -r requirements-model.txt
# macOS only: LightGBM needs OpenMP -> brew install libomp
```

## Run
```bash
# rebuild Gold first if needed:
./venv/bin/python -m data_ingestion.gold --horizon-steps 48

./venv/bin/python -m modeling --fast          # quick smoke run
./venv/bin/python -m modeling                 # full train + evaluate
./venv/bin/python -m modeling --target target_mw --epochs 40
```
Artifacts (`stack.joblib`, `tcn.pt`) and `metrics.json` land in `artifacts/model/`.

## Run on Google Colab (GPU) — recommended for the TCN

The TCN auto-uses CUDA (`"cuda" if torch.cuda.is_available()`), so on a Colab GPU
runtime training is fast and the CPU OpenMP issue below does not apply.

```python
# Colab cell (Runtime -> Change runtime type -> GPU)
!pip -q install lightgbm scikit-learn joblib
# bring the repo + the Gold data (e.g. clone repo, then pull gold from team HF):
!pip -q install huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download("gridsight-team/gridsight-gold", repo_type="dataset",
                  local_dir="data/gold", token="hf_xxx")

!python -m modeling --epochs 40          # full GPU training
```

Inference later:
```python
from modeling.predict import predict_gold
import pandas as pd, glob
gold = pd.concat([pd.read_parquet(f) for f in
                  glob.glob("data/gold/gold_features/**/*.parquet", recursive=True)])
forecast = predict_gold(gold)            # timestamp_utc, pred_q10/q50/q90 (MW)
```

> ⚠️ **CPU caveat (macOS/local):** LightGBM's `libomp` and PyTorch both use OpenMP;
> on CPU they oversubscribe threads and TCN training can stall. `torch_threads`
> defaults to 1 to mitigate, but CPU training is still slow — use the GPU. (GPU
> ignores `torch_threads`.) The non-TCN parts (LGBM, meta-learner) are fast on CPU.

## Design notes
- **Leakage-safe**: chronological splits (no shuffle), OOF meta training, and Gold
  itself enforces feature-level anti-leakage (observed actuals only via lags).
- **Target**: defaults to `target_cf` (capacity factor — robust to the +26%
  capacity growth); multiply by the known future `capacity_mwp` to get MW.
- **Daylight scoring**: night slots (~0) are excluded from training/metrics but
  kept in the frame so the TCN's diurnal context stays continuous. At inference,
  predict 0 for night.
- **Benchmark**: `skill_vs_neso_q50` compares the q50 forecast to NESO's
  `embedded_solar_mw` — the operator baseline to beat.
