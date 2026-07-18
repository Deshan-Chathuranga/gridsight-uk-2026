# chronos_q

Model C. Chronos foundation model, zero-shot. No training here — pull a checkpoint
off HF, run it on the target series, calibrate, done.

Point of it: a feature-free baseline you can stand up in one command, plus a third
leg for the ensemble. If it beats A/B on some horizon, that's a signal worth chasing.

## what it actually does

- reads only `target_cf` history. no weather, no calendar, no lags. univariate.
- Chronos already spits out quantiles, so we just ask for q10/q50/q90.
- then the same `w / (clearsky_cos + k)` calibration everything else uses, swept on val
  to pin coverage at 80%.
- night slots -> 0, only daylight is scored. same as lstm_q.

nothing is trained. `chronos.joblib` only holds cfg + the calibration factor + the
checkpoint name. weights live in the HF cache.

## the forecast trick (don't skip this)

Chronos is a multi-step forecaster, but Gold's leakage rule is "slot t can only see
data from >= horizon steps back". So for each scored row t:

```
origin  = t - horizon
context = series[origin-C+1 : origin]      # C = context_len, default 512
pred_t  = Chronos(context, prediction_length=horizon)[step = horizon]   # keep the LAST step only
```

we throw away steps 1..horizon-1 and keep the horizon-th. that makes slot t depend
only on data `horizon` back — same constraint the lag features live under. yes it's
wasteful (forecast 48, use 1), but horizon is small and bolt is fast.

`--stride N` skips origins if you just want a quick number. `--fast` = tiny checkpoint.

## run it

### Running for the 3 Horizons

Ensure you have rebuilt the corresponding Gold feature store for each horizon first. Then, run the Chronos-Q pipeline by pointing to the correct gold directory and specifying `--horizon-steps`:

```bash
# 1. Horizon 12 (6-hour ahead)
./venv/bin/python -m gridsight.models.chronos --horizon-steps 12 --gold-dir data/gold/gold_features_h12

# 2. Horizon 24 (12-hour ahead)
./venv/bin/python -m gridsight.models.chronos --horizon-steps 24 --gold-dir data/gold/gold_features_h24

# 3. Horizon 48 (24-hour ahead, default)
./venv/bin/python -m gridsight.models.chronos --horizon-steps 48 --gold-dir data/gold/gold_features_h48
```

*Note: For testing/smoke runs, you can append `--fast` to use a tiny checkpoint (`chronos-bolt-tiny`), a shorter context, and a stride of 24 (subsampling origins for speed).*

### Checking the Results

Run the evaluation script for each horizon to print the metrics and generate visualization charts:

```bash
# 1. Evaluate Horizon 12
./venv/bin/python -m gridsight.models.chronos.evaluate --split test --artifacts artifacts/chronos_h12

# 2. Evaluate Horizon 24
./venv/bin/python -m gridsight.models.chronos.evaluate --split test --artifacts artifacts/chronos_h24

# 3. Evaluate Horizon 48 (default artifacts directory is artifacts/chronos)
./venv/bin/python -m gridsight.models.chronos.evaluate --split test --artifacts artifacts/chronos
```

The first run downloads the default checkpoint (`chronos-bolt-small`, ~50MB, not gated) and caches it.


## flags

```
--target        target_cf | target_mw        (default target_cf)
--horizon-steps 48 = day-ahead               (default 48)
--model-name    any chronos / chronos-bolt HF id
--context-len   history window fed in         (default 512)
--batch-size    origins per chronos batch     (default 128)
--stride        subsample origins             (default 1 = every scored row)
--gold-dir      path to gold features
--fast          bolt-tiny + short ctx + stride 24
```

checkpoints to try: `chronos-bolt-{tiny,mini,small,base}`, or the og
`chronos-t5-{small,base}`.

## outputs -> artifacts/chronos/

```
chronos.joblib              cfg + calib factor + checkpoint name (NO weights)
metrics.json                val/test: pinball, coverage, crossing, skill
pred_{val,test}.parquet     actuals + calibrated quantiles
plots/evaluation_fan.png    q10-q90 band vs actual (MW)
plots/evaluation_dashboard.png  calibration + q50 scatter + err-by-hour + kpi card
```

non-default horizon H -> `artifacts/chronos_h{H}/`.

## inference from python

```python
from chronos_q.predict import predict_chronos_gold
import pandas as pd, glob

gold = pd.concat([pd.read_parquet(f) for f in
                  glob.glob("data/gold/gold_features/**/*.parquet", recursive=True)])
out = predict_chronos_gold(gold)     # timestamp_utc, pred_q10/q50/q90 (MW), night=0
```

## gotchas

- needs `chronos-forecasting>=1.5.0` (older ones don't have `predict_quantiles`).
- it eats the *observed* target as context, so this is eval-framing, not true live
  forecasting — same deal as feeding lag features.
- univariate means it ignores NWP entirely. don't expect it to beat the supervised
  models on cloudy-day sharpness; expect it to be a solid, honest floor.
