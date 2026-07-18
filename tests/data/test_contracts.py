import pandas as pd
import pytest
from gridsight.data.silver.contracts import (
    validate_pv_live,
    validate_met_office_nwp,
    validate_ocf_pv,
    validate_neso,
)
from gridsight.data.gold.contracts import validate_gold

def test_silver_contracts_valid():
    # Construct a valid dataframe structure for PV live
    time_spine = pd.date_range("2026-07-15 00:00", periods=5, freq="30min", tz="UTC")
    df_pv = pd.DataFrame({
        "timestamp_utc": time_spine,
        "generation_mw": [10.0, 50.0, 100.0, 20.0, 0.0],
        "capacity_mwp": [500.0, 500.0, 500.0, 500.0, 500.0]
    })
    # Should not raise any assertion error
    validate_pv_live(df_pv)

    # Valid dataframe for NESO
    df_neso = pd.DataFrame({
        "timestamp_utc": time_spine,
        "embedded_solar_mw": [50.0, 150.0, 250.0, 50.0, 0.0]
    })
    validate_neso(df_neso)

def test_silver_contracts_invalid_tz():
    # timezone is EST instead of UTC
    time_spine = pd.date_range("2026-07-15 00:00", periods=5, freq="30min", tz="EST")
    df_pv = pd.DataFrame({
        "timestamp_utc": time_spine,
        "generation_mw": [10.0, 50.0, 100.0, 20.0, 0.0]
    })
    with pytest.raises(AssertionError, match="not tz-aware UTC"):
        validate_pv_live(df_pv)

def test_silver_contracts_invalid_range():
    # generation out of bounds (negative)
    time_spine = pd.date_range("2026-07-15 00:00", periods=3, freq="30min", tz="UTC")
    df_pv = pd.DataFrame({
        "timestamp_utc": time_spine,
        "generation_mw": [10.0, -50.0, 100.0],
        "capacity_mwp": [100.0, 100.0, 100.0]
    })
    with pytest.raises(AssertionError, match="generation_mw out of range"):
        validate_pv_live(df_pv)

def test_gold_contracts_valid():
    from gridsight.data.gold.lag_features import lag_set
    horizon = 24
    time_spine = pd.date_range("2026-07-15 00:00", periods=5, freq="30min", tz="UTC")
    
    # Dynamically build df with all expected lag columns for the given horizon
    data = {
        "timestamp_utc": time_spine,
        "target_mw": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target_cf": [0.1, 0.2, 0.3, 0.4, 0.5],
        "solar_elevation_deg": [10.0, 15.0, 20.0, 15.0, 10.0]
    }
    for L in lag_set(horizon):
        data[f"gen_lag_{L}"] = [5.0, 6.0, 7.0, 8.0, 9.0]
        
    df_gold = pd.DataFrame(data)
    validate_gold(df_gold, horizon=horizon)

def test_gold_contracts_leakage():
    from gridsight.data.gold.lag_features import lag_set
    horizon = 24
    time_spine = pd.date_range("2026-07-15 00:00", periods=5, freq="30min", tz="UTC")
    
    # 1. Raw generation_mw feature present (direct leakage)
    data_leak1 = {
        "timestamp_utc": time_spine,
        "target_mw": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target_cf": [0.1, 0.2, 0.3, 0.4, 0.5],
        "generation_mw": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    for L in lag_set(horizon):
        data_leak1[f"gen_lag_{L}"] = [5.0, 6.0, 7.0, 8.0, 9.0]
        
    df_leak1 = pd.DataFrame(data_leak1)
    with pytest.raises(AssertionError, match="LEAKAGE: raw observed columns present"):
        validate_gold(df_leak1, horizon=horizon)

    # 2. Lag too small (leakage of future info relative to horizon)
    # Horizon is 24, lag is 12 (12 < 24)
    df_leak2 = pd.DataFrame({
        "timestamp_utc": time_spine,
        "target_mw": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target_cf": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gen_lag_12": [5.0, 6.0, 7.0, 8.0, 9.0]
    })
    with pytest.raises(AssertionError, match="LEAKAGE: .* uses lag 12 < horizon 24"):
        validate_gold(df_leak2, horizon=horizon)
