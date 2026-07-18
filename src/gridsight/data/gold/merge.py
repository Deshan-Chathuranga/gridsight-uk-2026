"""Time-based merge of the 4 Silver tables -> one wide 30-min table.
Output: one row per 30-min slot with target + all source columns, sorted by time.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from .common import TS, read_silver
from ..silver.common import canonical_index


def _norm_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Force timestamp_utc to a single resolution/tz so joins match exactly."""
    df = df.copy()
    df[TS] = pd.to_datetime(df[TS], utc=True).astype("datetime64[us, UTC]")
    return df


def merge_silver() -> pd.DataFrame:
    pv = read_silver("silver_pv_live")
    nwp = read_silver("silver_met_office_nwp")
    neso = read_silver("silver_neso")
    ocf = read_silver("silver_ocf_pv")
    if pv.empty:
        logger.error("merge: silver_pv_live is empty - build Silver first")
        return pd.DataFrame()

    if nwp.empty:
        logger.warning("merge: silver_met_office_nwp is empty - using empty schema fallback")
        nwp = pd.DataFrame(columns=[TS, "ssrd_uk", "tcc_uk", "lcc_uk", "t2m_uk", "ws10_uk", "init_time", "nwp_age_h", "data_quality_flag"])
    if neso.empty:
        logger.warning("merge: silver_neso is empty - using empty schema fallback")
        neso = pd.DataFrame(columns=[TS, "embedded_solar_mw", "embedded_wind_mw", "data_quality_flag"])
    if ocf.empty:
        logger.warning("merge: silver_ocf_pv is empty - using empty schema fallback")
        ocf = pd.DataFrame(columns=[TS, "ocf_total_mw", "ocf_mean_wh", "ocf_n_systems", "data_quality_flag"])

    # --- spine = pv_live, deduped & sorted on the canonical grid ---
    pv = (_norm_ts(pv)
          .rename(columns={"data_quality_flag": "pv_flag"})
          .drop_duplicates(TS).sort_values(TS).reset_index(drop=True))

    # --- prepare each source: normalise time, rename flags, drop non-features ---
    nwp_norm = _norm_ts(nwp)
    
    # Extend the spine to cover the future forecast horizon from NWP (weather forecasts)
    max_ts = max(pv[TS].max(), nwp_norm[TS].max())
    grid = canonical_index(pv[TS].min(), max_ts)
    pv = pv.set_index(TS).reindex(grid).reset_index().rename(columns={"index": TS})
    pv["capacity_mwp"] = pv["capacity_mwp"].ffill().bfill()
    pv["pv_flag"] = pv["pv_flag"].fillna("ok")

    nwp = (nwp_norm
           .drop(columns=["init_time"], errors="ignore")
           .rename(columns={"data_quality_flag": "nwp_flag"})
           .drop_duplicates(TS))
    neso = (_norm_ts(neso)
            .rename(columns={"data_quality_flag": "neso_flag"})
            .drop_duplicates(TS))
    ocf = (_norm_ts(ocf)
           .rename(columns={"data_quality_flag": "ocf_flag"})
           .drop_duplicates(TS))

    # --- LEFT JOIN everything onto the pv_live spine ---
    out = pv
    for src in (nwp, neso, ocf):
        out = out.merge(src, on=TS, how="left")

    out = out.sort_values(TS).reset_index(drop=True)

    # Proxy ocf_total_mw if missing (NaN) using 0.004026 * generation_mw
    if "ocf_total_mw" in out.columns and "generation_mw" in out.columns:
        is_missing = out["ocf_total_mw"].isna()
        if is_missing.any():
            logger.warning(f"merge: Proxying {is_missing.sum()} missing OCF solar actuals using scaled NESO actuals.")
            out.loc[is_missing, "ocf_total_mw"] = out.loc[is_missing, "generation_mw"] * 0.004026
            if "ocf_flag" in out.columns:
                out.loc[is_missing, "ocf_flag"] = "ok"

    # join coverage report (catches silent misalignment early)
    for col, name in [("ssrd_uk", "nwp"), ("embedded_solar_mw", "neso"),
                      ("ocf_total_mw", "ocf")]:
        if col in out:
            logger.info(f"merge: {name} coverage = {out[col].notna().mean()*100:.1f}% "
                        f"of {len(out):,} spine slots")
    return out
