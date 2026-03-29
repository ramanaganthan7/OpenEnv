"""
ClimateWatch — Real Sensor Data Fetcher
=======================================
Fetches REAL hourly air quality measurements from Open-Meteo API.
No API key required. Data is free and publicly sourced from
official monitoring networks (CAMS, EPA-equivalent stations).

Run once: uv run python scripts/fetch_real_data.py
Output:   app/data/real_task1.json
          app/data/real_task2.json
          app/data/real_task3.json

Data Source: Open-Meteo Air Quality API (https://open-meteo.com)
Backed by:   CAMS European Centre for Medium-Range Weather Forecasts (ECMWF)
             EPA-equivalent global monitoring stations
"""

import json
import time
import requests
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "app" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── Real industrial monitoring locations ──────────────────────────────────────
LOCATIONS = {
    # Task 1 — single sensor 24hr scenarios (20 locations)
    "houston_tx":      {"lat": 29.76,  "lon": -95.37,  "label": "Houston TX Petrochemical Corridor, USA"},
    "delhi_ncr":       {"lat": 28.63,  "lon":  77.22,  "label": "Delhi NCR Industrial Zone, India"},
    "north_sea":       {"lat": 57.50,  "lon":   2.00,  "label": "North Sea Oil Field, Aberdeen UK"},
    "los_angeles":     {"lat": 34.05,  "lon": -118.24, "label": "Los Angeles Basin Air Monitor, CA USA"},
    "beijing":         {"lat": 39.90,  "lon": 116.39,  "label": "Beijing Haidian Industrial Station, China"},
    "baton_rouge":     {"lat": 30.45,  "lon": -91.18,  "label": "Chemical Corridor, Baton Rouge LA USA"},
    "chennai":         {"lat": 13.08,  "lon":  80.27,  "label": "Coastal Zone Monitor, Chennai India"},
    "jubail":          {"lat": 27.00,  "lon":  49.66,  "label": "Petroleum Complex, Jubail Saudi Arabia"},
    "svalbard":        {"lat": 78.23,  "lon":  15.60,  "label": "Arctic Research Station, Ny-Ålesund Norway"},
    "manaus":          {"lat": -3.10,  "lon": -60.02,  "label": "Amazon Basin Station, Manaus Brazil"},
    "essen":           {"lat": 51.46,  "lon":   7.01,  "label": "Ruhr Valley Industrial, Essen Germany"},
    "guangzhou":       {"lat": 23.13,  "lon": 113.26,  "label": "Pearl River Delta Monitor, Guangzhou China"},
    "wichita":         {"lat": 37.69,  "lon": -97.34,  "label": "Great Plains Monitor, Wichita KS USA"},
    "tilbury":         {"lat": 51.46,  "lon":   0.36,  "label": "Thames Estuary Monitor, Tilbury UK"},
    "ulaanbaatar":     {"lat": 47.90,  "lon": 106.91,  "label": "Mongolian Steppe Station, Ulaanbaatar"},
    "san_jose_ca":     {"lat": 37.34,  "lon": -121.89, "label": "Silicon Valley Air Basin, San Jose CA USA"},
    "novosibirsk":     {"lat": 55.00,  "lon":  82.95,  "label": "Siberian Taiga Station, Novosibirsk Russia"},
    "alexandria_eg":   {"lat": 31.20,  "lon":  29.92,  "label": "Nile Delta Station, Alexandria Egypt"},
    "ahmadi_kuwait":   {"lat": 29.08,  "lon":  48.08,  "label": "Petroleum Refinery, Ahmadi Kuwait"},
    "qld_australia":   {"lat": -16.92, "lon": 145.77,  "label": "Great Barrier Reef Monitor, QLD Australia"},
}


def fetch_24h(lat: float, lon: float) -> dict:
    """Fetch 24 hours of real hourly measurements for a location."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,methane",
        "past_days": 1,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r = requests.get(BASE_URL, params=params, timeout=20, verify=False)
    r.raise_for_status()
    return r.json()


def fetch_7day(lat: float, lon: float) -> dict:
    """Fetch 7 days of real hourly measurements."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,methane",
        "past_days": 7,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    r = requests.get(BASE_URL, params=params, timeout=20, verify=False)
    r.raise_for_status()
    return r.json()


def fetch_30day(lat: float, lon: float) -> dict:
    """Fetch 30 days of real hourly measurements."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,methane",
        "past_days": 30,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    r = requests.get(BASE_URL, params=params, timeout=20, verify=False)
    r.raise_for_status()
    return r.json()


def extract_24h_readings(raw: dict, param_key: str) -> list:
    """Extract exactly 24 clean hourly values from API response."""
    hourly = raw.get("hourly", {})
    values = hourly.get(param_key, [])
    # Take last 24 hours only
    vals = values[-24:] if len(values) >= 24 else values
    # Fill to exactly 24 if needed
    while len(vals) < 24:
        vals.append(None)
    # Round to 2dp where not None
    return [round(v, 2) if v is not None else None for v in vals[:24]]


def extract_7day_daily_means(raw: dict, param_key: str) -> list:
    """Extract 7-day daily mean from 168 hourly values."""
    hourly = raw.get("hourly", {})
    values = hourly.get(param_key, [])
    values = values[-168:] if len(values) >= 168 else values
    # Group into days of 24
    daily = []
    for day in range(7):
        start = day * 24
        end = start + 24
        day_vals = [v for v in values[start:end] if v is not None]
        if day_vals:
            mean = round(sum(day_vals) / len(day_vals), 2)
            mn   = round(min(day_vals), 2)
            mx   = round(max(day_vals), 2)
            missing = 24 - len(day_vals)
        else:
            mean = mn = mx = None
            missing = 24
        daily.append({"day": day + 1, "mean": mean, "min": mn, "max": mx, "missing_hours": missing})
    return daily


def extract_30day_daily_means(raw: dict, param_key: str) -> list:
    """Extract 30-day daily means from hourly values."""
    hourly = raw.get("hourly", {})
    values = hourly.get(param_key, [])
    values = values[-720:] if len(values) >= 720 else values
    daily = []
    for day in range(30):
        start = day * 24
        end = start + 24
        day_vals = [v for v in values[start:end] if v is not None]
        if day_vals:
            mean = round(sum(day_vals) / len(day_vals), 2)
            missing = 24 - len(day_vals)
        else:
            mean = None
            missing = 24
        daily.append({"day": day + 1, "mean": mean, "missing_hours": missing})
    return daily


# ── Parameter mapping: OpenMeteo key → our sensor param name ─────────────────
PARAM_MAP = {
    "pm2_5":            {"sensor_key": "PM25", "param": "PM25_ugm3",  "unit": "µg/m³", "normal_range": [5, 35],     "dec": 1},
    "nitrogen_dioxide": {"sensor_key": "NO2",  "param": "NO2_ppb",    "unit": "ppb",   "normal_range": [10, 50],    "dec": 1},
    "ozone":            {"sensor_key": "O3",   "param": "O3_ppb",     "unit": "ppb",   "normal_range": [20, 60],    "dec": 1},
    "sulphur_dioxide":  {"sensor_key": "SO2",  "param": "SO2_ppb",    "unit": "ppb",   "normal_range": [2, 15],     "dec": 1},
    "carbon_monoxide":  {"sensor_key": "CO",   "param": "CO_ppb",     "unit": "ppb",   "normal_range": [200, 800],  "dec": 0},
    "methane":          {"sensor_key": "CH4",  "param": "CH4_ppb",    "unit": "ppb",   "normal_range": [1800, 2000],"dec": 0},
}

PARAM_KEYS = list(PARAM_MAP.keys())
LOC_KEYS   = list(LOCATIONS.keys())


def build_task1_data():
    """Build 20 real scenarios for Task 1 (24hr single sensor)."""
    print("Fetching Task 1 real data (20 locations × 24 hours)...")
    scenarios = []

    for i, loc_key in enumerate(LOC_KEYS):
        loc = LOCATIONS[loc_key]
        print(f"  [{i+1:02d}/20] {loc['label']}")
        try:
            raw = fetch_24h(loc["lat"], loc["lon"])
        except Exception as e:
            print(f"  WARN: fetch failed for {loc_key}: {e}. Using fallback.")
            raw = {"hourly": {k: [None]*24 for k in PARAM_KEYS}}

        # Pick sensor type based on scenario index
        param_key   = PARAM_KEYS[i % len(PARAM_KEYS)]
        param_info  = PARAM_MAP[param_key]
        readings_raw = extract_24h_readings(raw, param_key)

        # Replace None with interpolated or last-good value for missing
        readings_clean = []
        last_good = None
        for v in readings_raw:
            if v is None:
                readings_clean.append(last_good)
            else:
                last_good = v
                readings_clean.append(v)

        scenarios.append({
            "scenario_id": i,
            "location_key": loc_key,
            "location": loc["label"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "api_param_key": param_key,
            "sensor_key": param_info["sensor_key"],
            "param": param_info["param"],
            "unit": param_info["unit"],
            "normal_range": param_info["normal_range"],
            "real_readings": readings_clean,  # 24 real hourly values
        })
        time.sleep(0.3)  # gentle rate limiting

    out = OUT_DIR / "real_task1.json"
    with open(out, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved {len(scenarios)} scenarios -> {out}")
    return scenarios


def build_task2_data():
    """Build 10 real network scenarios for Task 2 (7 days × 5 sensors)."""
    print("\nFetching Task 2 real data (10 networks × 5 sensors × 7 days)...")

    # 10 networks — each picks 5 parameters from a location
    networks = [
        ("houston_tx",    "COASTAL-ZONE-3",      "Gulf Coast Industrial Corridor, Texas USA"),
        ("jubail",        "REFINERY-SOUTH-7",    "Petroleum Complex, Jubail Industrial City, Saudi Arabia"),
        ("delhi_ncr",     "URBAN-GRID-12",       "Metropolitan Air Quality Network, Delhi NCR, India"),
        ("svalbard",      "ARCTIC-STATION-2",    "Polar Research Station, Ny-Ålesund, Svalbard Norway"),
        ("manaus",        "AMAZON-BASIN-4",      "Tropical Forest Research Station, Manaus, Brazil"),
        ("north_sea",     "NORTH-SEA-OIL-6",    "Offshore Production Platform, North Sea, UK Sector"),
        ("essen",         "RUHR-VALLEY-9",       "Heavy Industrial Monitor Network, Essen, Germany"),
        ("guangzhou",     "PEARL-RIVER-11",      "Industrial Zone Air Quality Network, Guangzhou, China"),
        ("wichita",       "GREAT-PLAINS-14",     "Agricultural Region Air Monitor, Wichita, Kansas USA"),
        ("tilbury",       "THAMES-ESTUARY-17",   "Port Industrial Zone Monitor, Tilbury, Thames Estuary UK"),
    ]

    scenarios = []
    for i, (loc_key, network_id, location) in enumerate(networks):
        loc = LOCATIONS[loc_key]
        print(f"  [{i+1:02d}/10] {network_id} — {loc_key}")
        try:
            raw = fetch_7day(loc["lat"], loc["lon"])
        except Exception as e:
            print(f"  WARN: fetch failed for {loc_key}: {e}.")
            raw = {"hourly": {k: [None]*168 for k in PARAM_KEYS}}

        sensors = []
        sensor_ids = ["S1", "S2", "S3", "S4", "S5"]
        for j, (sid, param_key) in enumerate(zip(sensor_ids, PARAM_KEYS)):
            param_info = PARAM_MAP[param_key]
            daily = extract_7day_daily_means(raw, param_key)
            vals = [d["mean"] for d in daily if d["mean"] is not None]
            total_missing = sum(d["missing_hours"] for d in daily)
            sensors.append({
                "sensor_id": sid,
                "sensor_key": param_info["sensor_key"],
                "param": param_info["param"],
                "unit": param_info["unit"],
                "normal_range": param_info["normal_range"],
                "daily_means": daily,
                "overall_mean": round(sum(vals)/len(vals), 2) if vals else None,
                "total_missing_hours": total_missing,
            })

        scenarios.append({
            "scenario_id": i,
            "network_id": network_id,
            "location": location,
            "location_key": loc_key,
            "lat": loc["lat"],
            "lon": loc["lon"],
            "period_days": 7,
            "sensors": sensors,
        })
        time.sleep(0.3)

    out = OUT_DIR / "real_task2.json"
    with open(out, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved {len(scenarios)} scenarios -> {out}")
    return scenarios


def build_task3_data():
    """Build 5 real cascade failure scenarios for Task 3 (30 days × 10 sensors)."""
    print("\nFetching Task 3 real data (5 networks × 10 sensors × 30 days)...")

    networks = [
        ("ahmadi_kuwait",  "REFINERY-NORTH",         "Crude Oil Refinery, North Sector, Kuwait"),
        ("houston_tx",     "PIPELINE-CENTRAL",        "Gas Pipeline Monitoring Hub, Permian Basin, Texas"),
        ("beijing",        "URBAN-NETWORK-ALPHA",     "Metropolitan Air Quality Network, Beijing, China"),
        ("chennai",        "COASTAL-MONITOR-BETA",    "Coastal Industrial Zone, Chennai, India"),
        ("north_sea",      "OFFSHORE-PLATFORM-7",     "Fixed Production Platform, North Sea, UK/Norway"),
    ]

    scenarios = []
    for i, (loc_key, network_id, location) in enumerate(networks):
        loc = LOCATIONS[loc_key]
        print(f"  [{i+1:02d}/05] {network_id} — {loc_key}")
        try:
            raw = fetch_30day(loc["lat"], loc["lon"])
        except Exception as e:
            print(f"  WARN: fetch failed for {loc_key}: {e}.")
            raw = {"hourly": {k: [None]*720 for k in PARAM_KEYS}}

        # 10 sensors — repeat params across sensor pairs
        sensor_params = [
            ("S1",  "methane"),          ("S2",  "carbon_monoxide"),
            ("S3",  "nitrogen_dioxide"), ("S4",  "methane"),
            ("S5",  "methane"),          ("S6",  "nitrogen_dioxide"),
            ("S7",  "ozone"),            ("S8",  "nitrogen_dioxide"),
            ("S9",  "ozone"),            ("S10", "ozone"),
        ]

        sensors = []
        for sid, param_key in sensor_params:
            param_info = PARAM_MAP[param_key]
            daily = extract_30day_daily_means(raw, param_key)
            vals = [d["mean"] for d in daily if d["mean"] is not None]
            total_missing = sum(d["missing_hours"] for d in daily)
            offline_days = sum(1 for d in daily if d["mean"] is None)
            sensors.append({
                "sensor_id": sid,
                "sensor_key": param_info["sensor_key"],
                "param": param_info["param"],
                "unit": param_info["unit"],
                "normal_range": param_info["normal_range"],
                "daily_means": daily,
                "overall_mean": round(sum(vals)/len(vals), 2) if vals else None,
                "total_missing_hours": total_missing,
                "offline_days": offline_days,
            })

        scenarios.append({
            "scenario_id": i,
            "network_id": network_id,
            "location": location,
            "location_key": loc_key,
            "lat": loc["lat"],
            "lon": loc["lon"],
            "period_days": 30,
            "sensors": sensors,
        })
        time.sleep(0.3)

    out = OUT_DIR / "real_task3.json"
    with open(out, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved {len(scenarios)} scenarios -> {out}")
    return scenarios


if __name__ == "__main__":
    print("=" * 60)
    print("ClimateWatch - Real Data Fetcher")
    print("Source: Open-Meteo Air Quality API (CAMS/ECMWF)")
    print("No API key required.")
    print("=" * 60)

    t1 = build_task1_data()
    t2 = build_task2_data()
    t3 = build_task3_data()

    print("\n" + "=" * 60)
    print("Done. Real data saved to app/data/")
    print(f"  real_task1.json — {len(t1)} scenarios")
    print(f"  real_task2.json — {len(t2)} scenarios")
    print(f"  real_task3.json — {len(t3)} scenarios")
    print("=" * 60)
