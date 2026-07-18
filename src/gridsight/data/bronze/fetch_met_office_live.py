"""Fetch live weather forecasts from Open-Meteo (UKMET model) to populate Bronze met_office_nwp for 2026.
"""
from __future__ import annotations
import os
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from .met_office import _load_uk_points

def fetch_and_save_live_nwp(start_date: str = "2026-06-01", end_date: str | None = None):
    if end_date is None:
        # Fetch up to 2 days in the future to cover forecast horizons
        end_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
        end_date = end_dt.strftime("%Y-%m-%d")
        
    points = _load_uk_points()
    logger.info(f"Fetching Open-Meteo UKMO regional forecast from {start_date} to {end_date} for {len(points)} UK points...")
    
    # Open-Meteo Multi-location Query
    url = "https://api.open-meteo.com/v1/forecast"
    lats = ",".join(str(p[1]) for p in points)
    lons = ",".join(str(p[2]) for p in points)
    
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": "shortwave_radiation,cloud_cover,cloud_cover_low,temperature_2m,wind_speed_10m",
        "models": "ukmo_uk_deterministic_2km",
        "wind_speed_unit": "ms",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logger.error(f"Failed to fetch forecast from Open-Meteo: {response.text}")
        return False
        
    data = response.json()
    if not isinstance(data, list):
        data = [data]
        
    extracted_at = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    
    # Process hourly timestamps
    times = pd.to_datetime(data[0]["hourly"]["time"], utc=True)
    
    # We will generate daily runs: 00Z and 12Z for each day in the range
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    
    written_count = 0
    for day in days:
        for hour in [0, 12]:
            init_time = pd.Timestamp(day).replace(hour=hour, minute=0, second=0).tz_localize("UTC")
            # We forecast for the next 24 hours (forecast hours 0 to 23)
            forecast_hours = list(range(24))
            valid_times = [init_time + datetime.timedelta(hours=h) for h in forecast_hours]
            
            # Filter API times
            valid_times_set = set(valid_times)
            mask = [t in valid_times_set for t in times]
            if not any(mask):
                continue
                
            # Build rows for this run
            run_rows = []
            for i, (region, lat, lon) in enumerate(points):
                loc_data = data[i]["hourly"]
                for h in forecast_hours:
                    vt = init_time + datetime.timedelta(hours=h)
                    try:
                        idx = loc_data["time"].index(vt.strftime("%Y-%m-%dT%H:00"))
                        val_ssrd = loc_data["shortwave_radiation"][idx]
                        val_tcc = loc_data["cloud_cover"][idx]
                        val_lcc = loc_data["cloud_cover_low"][idx]
                        val_t2m = loc_data["temperature_2m"][idx]
                        val_ws10 = loc_data["wind_speed_10m"][idx]
                        
                        run_rows.append({
                            "init_time": init_time,
                            "valid_time": vt,
                            "forecast_hour": h,
                            "region": region,
                            "lat": lat,
                            "lon": lon,
                            "ssrd": float(val_ssrd) if val_ssrd is not None else 0.0,
                            "tcc": (float(val_tcc) / 100.0) if val_tcc is not None else 0.0,
                            "lcc": (float(val_lcc) / 100.0) if val_lcc is not None else 0.0,
                            "t2m": (float(val_t2m) + 273.15) if val_t2m is not None else 288.15,
                            "ws10": float(val_ws10) if val_ws10 is not None else 0.0,
                            "source_file": "open_meteo_api",
                            "extracted_at": extracted_at
                        })
                    except (ValueError, IndexError):
                        continue
                        
            if not run_rows:
                continue
                
            run_df = pd.DataFrame(run_rows)
            
            # Save to partitioned bronze structure
            y = init_time.year
            m = init_time.month
            out_dir = PROJECT_ROOT / "data" / "bronze" / "met_office_nwp" / f"year={y}" / f"month={m:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"nwp_{init_time.strftime('%Y%m%d')}-{hour:02d}Z.parquet"
            out_path = out_dir / file_name
            run_df.to_parquet(out_path, index=False)
            written_count += 1
            
    logger.success(f"Successfully generated {written_count} Met Office bronze live forecast parquets -> data/bronze/met_office_nwp")
    return True

if __name__ == "__main__":
    fetch_and_save_live_nwp()
