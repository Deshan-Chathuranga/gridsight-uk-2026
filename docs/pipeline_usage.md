# GridSight Pipeline — Usage (Silver & Gold from the team's HF Bronze)

How to rebuild **Silver** and **Gold** on your machine by pulling **Bronze from the
team's HuggingFace repo** (`gridsight-team/gridsight-bronze`) — **not** from the
original upstream datasets (OCF / Met Office / NESO).

> Why pull from the team repo? Re-fetching the raw sources is slow and the Met
> Office bronze alone is point-extracted from ~100 GB/month of `.zarr.zip`. The
> team Bronze on HF is the agreed single source of truth — sync it once, then
> everything is local and offline.

---

## 0. Prerequisites

- Use the project **venv** (pyarrow 24 / pandas 3). Reading the parquet with an
older pyarrow (e.g. anaconda's 19) fails with *"Repetition level histogram size
mismatch"*.
  ```bash
  which python    # must be .../test_gridsight-uk/venv/bin/python
  ```
- `.env` at the repo root with a HF token that can **read** the team repos:
  ```
  GRIDSIGHT_HF_TOKEN=hf_xxx
  GRIDSIGHT_BRONZE_HF_REPO=gridsight-team/gridsight-bronze
  ```

Team repos:


| Layer  | HF dataset repo                   |
| ------ | --------------------------------- |
| Bronze | `gridsight-team/gridsight-bronze` |
| Silver | `gridsight-team/gridsight-silver` |
| Gold   | `gridsight-team/gridsight-gold`   |


---

## 1. Pull Bronze from the team HF → local `data/bronze/`

```bash
# all sources (met_office_nwp, neso, ocf_pv, pv_live)
./venv/bin/python -m data_ingestion.sync_bronze --source all

# or a single source
./venv/bin/python -m data_ingestion.sync_bronze --source met_office_nwp
```

- Downloads from `gridsight-team/gridsight-bronze` (**team repo, not upstream**).
- Incremental & resumable: files already present are skipped.
- **Live NWP Ingestion (2026+)**: When ingesting `met_office_nwp` for 2026 or later, the pipeline automatically fetches live UKMO regional deterministic forecasts from Open-Meteo via `data_ingestion.bronze.fetch_met_office_live`.
- **PV Live (Current Month)**: Partition for the current month is never skipped, enabling dynamic updates of live actuals.

---

## 2. Build Silver from local Bronze (no network)

```bash
./venv/bin/python -m data_ingestion.silver --source all
# single table: --source {pv_live|ocf_pv|met_office_nwp|neso}
```

Writes `data/silver/<table>/year=YYYY/month=MM/`. Rebuildable offline — Silver
reads **local Bronze only**, never HF.

---

## 3. Build Gold from local Silver (no network)

```bash
# Day-ahead (horizon=48 steps = 24h, default)
./venv/bin/python -m data_ingestion.gold --horizon-steps 48

# 12-hour ahead (horizon=24 steps = 12h)
./venv/bin/python -m data_ingestion.gold --horizon-steps 24

# 6-hour ahead (horizon=12 steps = 6h)
./venv/bin/python -m data_ingestion.gold --horizon-steps 12
```

Writes the feature tables under `data/gold/gold_features_h{horizon}/`. Gold
reads **local Silver only**.

During the Gold build:
- **Spine Extension**: The index spine is extended to cover the future forecast horizon from NWP forecasts. `capacity_mwp` is filled using forward/backward fill.
- **Weather Imputation & Fallbacks**: Missing NWP slots are dynamically imputed using clear-sky physical models (e.g., `ssrd_uk = clearsky_cos * 900.0` and scaled `t2m_uk`).
- **OCF Proxying**: Missing OCF observations (`ocf_total_mw`) are automatically proxied using scaled NESO actuals (`generation_mw * 0.004026`).

**Full local rebuild in one go:**

```bash
./venv/bin/python -m data_ingestion.sync_bronze --source all
./venv/bin/python -m data_ingestion.silver --source all
./venv/bin/python -m data_ingestion.gold
```

---

## 4. (Optional) Skip rebuilding — pull Silver/Gold directly from team HF

If you only need the data and don't want to recompute (uses the `hf` CLI; for a
private team repo, log in once with `./venv/bin/hf auth login` or append
`--token <your_hf_token>`):

```bash
# Silver
./venv/bin/hf download gridsight-team/gridsight-silver \
    --repo-type dataset --local-dir data/silver
# Gold
./venv/bin/hf download gridsight-team/gridsight-gold \
    --repo-type dataset --local-dir data/gold
```

Push your local layers back up (after building):

```bash
./venv/bin/python -c "from data_ingestion.silver import upload_silver_to_hf; upload_silver_to_hf()"
./venv/bin/python -m data_ingestion.gold --upload
```

---

## 5. Gold feature reference (`gold_features`, 30-min UTC, one row per slot)

Key column: `**timestamp_utc**` (tz-aware UTC, 30-min grid). 47 columns total.

### Targets


| Column         | Meaning                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `target_mw`    | National PV generation at t (MW) — primary target                            |
| `target_cf`    | Capacity factor = generation / capacity (0–1.5) — capacity-normalised target |
| `capacity_mwp` | Installed PV capacity at t (MW, known/planned → usable as feature)           |


### Weather — Met Office NWP (forecast, known ahead → used at t directly)


| Column      | Meaning                                                                        |
| ----------- | ------------------------------------------------------------------------------ |
| `ssrd_uk`   | Surface downwelling shortwave radiation, UK weighted (W/m²) — top solar driver |
| `tcc_uk`    | Total cloud cover (0–1)                                                        |
| `lcc_uk`    | Low cloud cover (0–1)                                                          |
| `t2m_uk`    | 2 m air temperature (K)                                                        |
| `ws10_uk`   | 10 m wind speed (m/s)                                                          |
| `nwp_age_h` | Forecast lead/age in hours (0–15; NWP source caps at 15 h)                     |

> [!NOTE]
> Weather slots that are missing are automatically imputed during the Gold build phase using clear-sky physical models (e.g. estimating shortwave radiation as a fraction of `clearsky_cos`).


### Operator baseline — NESO (forecast, known ahead)


| Column                                                     | Meaning                                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `embedded_solar_mw`                                        | NESO embedded solar forecast (MW) — strong feature **and** benchmark to beat (MAE ≈ 317 MW) |
| `embedded_wind_mw`                                         | NESO embedded wind forecast (MW)                                                            |
| `embedded_solar_capacity_mw` / `embedded_wind_capacity_mw` | NESO capacity context (MW)                                                                  |


### Calendar (deterministic, always known)


| Column                                                          | Meaning                                  |
| --------------------------------------------------------------- | ---------------------------------------- |
| `hour`, `half_hour` (0–47), `dow`, `month`, `doy`, `is_weekend` | Calendar fields                          |
| `tod_sin`, `tod_cos`                                            | Cyclical time-of-day encoding            |
| `doy_sin`, `doy_cos`                                            | Cyclical day-of-year (seasonal) encoding |


### Solar geometry (deterministic — strongest physical prior)


| Column                | Meaning                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `solar_elevation_deg` | Sun elevation angle at UK centroid (NOAA formula)                   |
| `clearsky_cos`        | cos(zenith) clipped ≥0 — theoretical insolation factor (0 at night) |
| `is_daylight`         | 1 if sun above horizon                                              |


### Lagged / rolling — OBSERVED actuals (leakage-safe, shifted ≥ horizon)


| Column                                          | Meaning                                                     |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `gen_lag_{48,96,144,336}`                       | Generation 1 / 2 / 3 / 7 days ago (MW)                      |
| `cf_lag_{48,96,144,336}`                        | Capacity factor at the same lags                            |
| `gen_roll_mean_48` / `gen_roll_mean_336`        | Trailing 1-day / 1-week mean generation (ends at t−horizon) |
| `gen_roll_std_48`                               | Trailing 1-day generation volatility                        |
| `cf_roll_mean_48` / `cf_roll_mean_336`          | Trailing 1-day / 1-week mean capacity factor                |
| `ocf_lag_{48,96,144,336}`, `ocf_roll_mean_48`   | OCF rooftop-fleet index, lagged / trailing mean (MW)        |


### Quality / bookkeeping (not model inputs)


| Column                                         | Meaning                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `pv_flag`, `nwp_flag`, `neso_flag`, `ocf_flag` | Per-source data-quality flag (`ok` / `ffill` / `gap` / `long_gap`) |
| `has_full_history`                             | 1 once enough history exists for all lag/rolling features          |


---

## 6. Anti-leakage rules (read before modelling)

- **Forecasts** (NWP `*_uk`, NESO `embedded_`*) and **deterministic** features
(calendar, solar, capacity) are aligned at valid time `t` and used **as-is** —
they are known when a forecast is issued.
- **Observed actuals** (generation, OCF) appear **only via lags shifted ≥ horizon**.
The raw contemporaneous `generation_mw`, `ocf_total_mw`, etc. are intentionally
**not** in Gold (they would leak). Lags are picked so every `_lag_N` has
`N ≥ horizon-steps`.
- When training: `y = target_mw` (or `target_cf`); exclude `timestamp_utc`, the  
other target, and the `*_flag` columns from `X`. Filter `has_full_history == 1`.

