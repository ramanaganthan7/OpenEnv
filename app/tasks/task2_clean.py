"""
Task 2 — Multi-Sensor Data Stream Cleaning (MEDIUM)
====================================================
Agent receives 7 days of data from 5 sensors in a network.
Each sensor has a different fault (or is clean).
Agent must DIAGNOSE each sensor and RECOMMEND the correct fix.

Action schema:
  { "diagnoses": [
      { "sensor_id": str,
        "fault_type": "drift|missing|bias|noise|stuck|spike|valid",
        "severity":   "none|low|medium|high|critical",
        "fix":        "no_action|interpolate|recalibrate|offset_correction|smooth|flag_only|replace",
        "fix_params": {} }
    ]
  }

Harder than Task 1 because:
  - 5 sensors simultaneously, each with different fault
  - Must recommend the correct FIX, not just identify the fault
  - Severity must match
  - One sensor is always "valid" (agent must not over-diagnose)

Grader: weighted average of (fault_type_accuracy × 0.60) + (fix_accuracy × 0.40)
"""

import random
import math
from typing import Optional, Tuple, Dict, Any, List

# ── Sensor parameter catalogue ────────────────────────────────────────────────
SENSOR_PARAMS: Dict[str, Dict] = {
    "NO2":  {"param": "NO2_ppb",    "unit": "ppb",   "normal_range": [10, 50],
             "center": 28.0,  "std": 2.5,  "dec": 1},
    "PM25": {"param": "PM25_ugm3",  "unit": "µg/m³", "normal_range": [5, 35],
             "center": 18.0,  "std": 2.0,  "dec": 1},
    "O3":   {"param": "O3_ppb",     "unit": "ppb",   "normal_range": [20, 60],
             "center": 38.0,  "std": 3.0,  "dec": 1},
    "CH4":  {"param": "CH4_ppm",    "unit": "ppm",   "normal_range": [1.7, 2.0],
             "center": 1.87,  "std": 0.02, "dec": 3},
    "CO2":  {"param": "CO2_ppm",    "unit": "ppm",   "normal_range": [400, 450],
             "center": 420.0, "std": 3.0,  "dec": 1},
    "SO2":  {"param": "SO2_ppb",    "unit": "ppb",   "normal_range": [2, 15],
             "center": 8.0,   "std": 1.0,  "dec": 1},
    "CO":   {"param": "CO_ppm",     "unit": "ppm",   "normal_range": [0.1, 2.0],
             "center": 0.8,   "std": 0.05, "dec": 3},
}

DAYS = 7
HOURS = DAYS * 24  # 168 total readings

# ── 10 scenario templates ─────────────────────────────────────────────────────
# Each template: network_id, location, sensors (5 sensors with fault specs), ground_truth
TEMPLATES: List[Dict] = [
    {
        "network_id": "COASTAL-ZONE-3",
        "location": "Gulf Coast Industrial Corridor, Texas USA",
        "reference_id": "REF-GC-01",
        "sensors": [
            {"id": "S1", "key": "NO2",  "fault": "drift",   "drift_rate": 0.8,   "severity": "high"},
            {"id": "S2", "key": "PM25", "fault": "missing", "missing_pct": 0.20, "severity": "medium"},
            {"id": "S3", "key": "O3",   "fault": "bias",    "offset": 12.0,      "severity": "high"},
            {"id": "S4", "key": "CH4",  "fault": "noise",   "noise_scale": 7.0,  "severity": "medium"},
            {"id": "S5", "key": "CO2",  "fault": "valid",                        "severity": "none"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.8, "direction": "positive"}},
            {"sensor_id": "S2", "fault_type": "missing", "severity": "medium",  "fix": "interpolate",       "fix_params": {"method": "linear", "max_gap_hours": 4}},
            {"sensor_id": "S3", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": -12.0}},
            {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 3, "method": "rolling_mean"}},
            {"sensor_id": "S5", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
        ],
    },
    {
        "network_id": "REFINERY-SOUTH-7",
        "location": "Petroleum Complex, Jubail Industrial City, Saudi Arabia",
        "reference_id": "REF-JS-02",
        "sensors": [
            {"id": "S1", "key": "CH4",  "fault": "stuck",   "stuck_day": 3,      "severity": "critical"},
            {"id": "S2", "key": "SO2",  "fault": "drift",   "drift_rate": 0.5,   "severity": "medium"},
            {"id": "S3", "key": "CO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S4", "key": "NO2",  "fault": "bias",    "offset": -8.0,      "severity": "high"},
            {"id": "S5", "key": "O3",   "fault": "missing", "missing_pct": 0.15, "severity": "low"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "stuck",   "severity": "critical","fix": "replace",           "fix_params": {"start_day": 3}},
            {"sensor_id": "S2", "fault_type": "drift",   "severity": "medium",  "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.5, "direction": "positive"}},
            {"sensor_id": "S3", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S4", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": 8.0}},
            {"sensor_id": "S5", "fault_type": "missing", "severity": "low",     "fix": "interpolate",       "fix_params": {"method": "linear", "max_gap_hours": 4}},
        ],
    },
    {
        "network_id": "URBAN-GRID-12",
        "location": "Metropolitan Air Quality Network, Delhi NCR, India",
        "reference_id": "REF-DL-03",
        "sensors": [
            {"id": "S1", "key": "PM25", "fault": "bias",    "offset": 22.0,      "severity": "critical"},
            {"id": "S2", "key": "NO2",  "fault": "noise",   "noise_scale": 5.0,  "severity": "medium"},
            {"id": "S3", "key": "CO",   "fault": "drift",   "drift_rate": 0.03,  "severity": "high"},
            {"id": "S4", "key": "SO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S5", "key": "O3",   "fault": "missing", "missing_pct": 0.25, "severity": "high"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "bias",    "severity": "critical","fix": "offset_correction", "fix_params": {"offset": -22.0}},
            {"sensor_id": "S2", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 5, "method": "rolling_mean"}},
            {"sensor_id": "S3", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.03}},
            {"sensor_id": "S4", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S5", "fault_type": "missing", "severity": "high",    "fix": "interpolate",       "fix_params": {"method": "linear", "max_gap_hours": 6}},
        ],
    },
    {
        "network_id": "ARCTIC-STATION-2",
        "location": "Polar Research Station, Ny-Ålesund, Svalbard Norway",
        "reference_id": "REF-SV-04",
        "sensors": [
            {"id": "S1", "key": "CO2",  "fault": "drift",   "drift_rate": 2.1,   "severity": "high"},
            {"id": "S2", "key": "CH4",  "fault": "bias",    "offset": 0.08,      "severity": "medium"},
            {"id": "S3", "key": "NO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S4", "key": "O3",   "fault": "noise",   "noise_scale": 6.0,  "severity": "medium"},
            {"id": "S5", "key": "PM25", "fault": "missing", "missing_pct": 0.30, "severity": "high"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "drift",   "severity": "high",   "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 2.1}},
            {"sensor_id": "S2", "fault_type": "bias",    "severity": "medium", "fix": "offset_correction", "fix_params": {"offset": -0.08}},
            {"sensor_id": "S3", "fault_type": "valid",   "severity": "none",   "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium", "fix": "smooth",            "fix_params": {"window": 3}},
            {"sensor_id": "S5", "fault_type": "missing", "severity": "high",   "fix": "interpolate",       "fix_params": {"method": "spline"}},
        ],
    },
    {
        "network_id": "AMAZON-BASIN-4",
        "location": "Tropical Forest Research Station, Manaus, Brazil",
        "reference_id": "REF-AM-05",
        "sensors": [
            {"id": "S1", "key": "CO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S2", "key": "O3",   "fault": "stuck",   "stuck_day": 2,      "severity": "high"},
            {"id": "S3", "key": "CH4",  "fault": "drift",   "drift_rate": 0.005, "severity": "medium"},
            {"id": "S4", "key": "PM25", "fault": "noise",   "noise_scale": 4.0,  "severity": "low"},
            {"id": "S5", "key": "NO2",  "fault": "bias",    "offset": -6.0,      "severity": "medium"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S2", "fault_type": "stuck",   "severity": "high",    "fix": "replace",           "fix_params": {"start_day": 2}},
            {"sensor_id": "S3", "fault_type": "drift",   "severity": "medium",  "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.005}},
            {"sensor_id": "S4", "fault_type": "noise",   "severity": "low",     "fix": "smooth",            "fix_params": {"window": 3}},
            {"sensor_id": "S5", "fault_type": "bias",    "severity": "medium",  "fix": "offset_correction", "fix_params": {"offset": 6.0}},
        ],
    },
    {
        "network_id": "NORTH-SEA-OIL-6",
        "location": "Offshore Production Platform, North Sea, UK Sector",
        "reference_id": "REF-NS-06",
        "sensors": [
            {"id": "S1", "key": "CH4",  "fault": "bias",    "offset": 0.12,      "severity": "critical"},
            {"id": "S2", "key": "SO2",  "fault": "missing", "missing_pct": 0.35, "severity": "high"},
            {"id": "S3", "key": "NO2",  "fault": "noise",   "noise_scale": 8.0,  "severity": "high"},
            {"id": "S4", "key": "CO2",  "fault": "drift",   "drift_rate": 3.0,   "severity": "high"},
            {"id": "S5", "key": "PM25", "fault": "valid",                        "severity": "none"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "bias",    "severity": "critical","fix": "offset_correction", "fix_params": {"offset": -0.12}},
            {"sensor_id": "S2", "fault_type": "missing", "severity": "high",    "fix": "flag_only",         "fix_params": {"reason": "gap_too_large_for_interpolation"}},
            {"sensor_id": "S3", "fault_type": "noise",   "severity": "high",    "fix": "smooth",            "fix_params": {"window": 5}},
            {"sensor_id": "S4", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 3.0}},
            {"sensor_id": "S5", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
        ],
    },
    {
        "network_id": "RUHR-VALLEY-9",
        "location": "Heavy Industrial Monitor Network, Essen, Germany",
        "reference_id": "REF-RV-07",
        "sensors": [
            {"id": "S1", "key": "SO2",  "fault": "drift",   "drift_rate": 0.4,   "severity": "medium"},
            {"id": "S2", "key": "CO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S3", "key": "PM25", "fault": "stuck",   "stuck_day": 4,      "severity": "critical"},
            {"id": "S4", "key": "O3",   "fault": "bias",    "offset": -9.0,      "severity": "high"},
            {"id": "S5", "key": "CH4",  "fault": "missing", "missing_pct": 0.10, "severity": "low"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "drift",   "severity": "medium",  "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.4}},
            {"sensor_id": "S2", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S3", "fault_type": "stuck",   "severity": "critical","fix": "replace",           "fix_params": {"start_day": 4}},
            {"sensor_id": "S4", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": 9.0}},
            {"sensor_id": "S5", "fault_type": "missing", "severity": "low",     "fix": "interpolate",       "fix_params": {"method": "linear"}},
        ],
    },
    {
        "network_id": "PEARL-RIVER-11",
        "location": "Industrial Zone Air Quality Network, Guangzhou, China",
        "reference_id": "REF-PR-08",
        "sensors": [
            {"id": "S1", "key": "NO2",  "fault": "missing", "missing_pct": 0.40, "severity": "high"},
            {"id": "S2", "key": "O3",   "fault": "drift",   "drift_rate": 1.5,   "severity": "high"},
            {"id": "S3", "key": "SO2",  "fault": "noise",   "noise_scale": 6.0,  "severity": "medium"},
            {"id": "S4", "key": "CH4",  "fault": "valid",                        "severity": "none"},
            {"id": "S5", "key": "PM25", "fault": "bias",    "offset": 30.0,      "severity": "critical"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "missing", "severity": "high",    "fix": "flag_only",         "fix_params": {"reason": "gap_too_large_for_interpolation"}},
            {"sensor_id": "S2", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 1.5}},
            {"sensor_id": "S3", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 3}},
            {"sensor_id": "S4", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S5", "fault_type": "bias",    "severity": "critical","fix": "offset_correction", "fix_params": {"offset": -30.0}},
        ],
    },
    {
        "network_id": "GREAT-PLAINS-14",
        "location": "Agricultural Region Air Monitor, Wichita, Kansas USA",
        "reference_id": "REF-GP-09",
        "sensors": [
            {"id": "S1", "key": "O3",   "fault": "valid",                        "severity": "none"},
            {"id": "S2", "key": "PM25", "fault": "drift",   "drift_rate": 1.2,   "severity": "high"},
            {"id": "S3", "key": "NO2",  "fault": "bias",    "offset": 15.0,      "severity": "critical"},
            {"id": "S4", "key": "CH4",  "fault": "noise",   "noise_scale": 5.5,  "severity": "medium"},
            {"id": "S5", "key": "CO",   "fault": "missing", "missing_pct": 0.18, "severity": "medium"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S2", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 1.2}},
            {"sensor_id": "S3", "fault_type": "bias",    "severity": "critical","fix": "offset_correction", "fix_params": {"offset": -15.0}},
            {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 3}},
            {"sensor_id": "S5", "fault_type": "missing", "severity": "medium",  "fix": "interpolate",       "fix_params": {"method": "linear", "max_gap_hours": 3}},
        ],
    },
    {
        "network_id": "THAMES-ESTUARY-17",
        "location": "Port Industrial Zone Monitor, Tilbury, Thames Estuary UK",
        "reference_id": "REF-TE-10",
        "sensors": [
            {"id": "S1", "key": "SO2",  "fault": "noise",   "noise_scale": 9.0,  "severity": "high"},
            {"id": "S2", "key": "NO2",  "fault": "valid",                        "severity": "none"},
            {"id": "S3", "key": "CO2",  "fault": "bias",    "offset": -18.0,     "severity": "high"},
            {"id": "S4", "key": "PM25", "fault": "stuck",   "stuck_day": 5,      "severity": "critical"},
            {"id": "S5", "key": "CH4",  "fault": "drift",   "drift_rate": 0.007, "severity": "medium"},
        ],
        "gt": [
            {"sensor_id": "S1", "fault_type": "noise",   "severity": "high",    "fix": "smooth",            "fix_params": {"window": 5}},
            {"sensor_id": "S2", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
            {"sensor_id": "S3", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": 18.0}},
            {"sensor_id": "S4", "fault_type": "stuck",   "severity": "critical","fix": "replace",           "fix_params": {"start_day": 5}},
            {"sensor_id": "S5", "fault_type": "drift",   "severity": "medium",  "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.007}},
        ],
    },
]


def _r(v: float, dec: int) -> float:
    return round(v, dec)


def _build_sensor_summary(sensor_spec: Dict, template_sensor: Dict,
                           rng: random.Random) -> Dict[str, Any]:
    """Build a 7-day summary for one sensor with the appropriate fault injected."""
    key = template_sensor["key"]
    spec = SENSOR_PARAMS[key]
    dec = spec["dec"]
    center = spec["center"]
    std = spec["std"]
    fault = template_sensor["fault"]

    # Generate 7 × 24 = 168 hourly readings (stored as daily stats for brevity)
    daily_summaries = []

    for day in range(1, 8):
        # Generate 24 hourly readings for this day
        hourly = [_r(center + rng.gauss(0, std * 0.6), dec) for _ in range(24)]

        if fault == "drift":
            rate = template_sensor.get("drift_rate", 1.0)
            for h in range(24):
                hourly[h] = _r(hourly[h] + rate * (day - 1) + rate * h / 24, dec)

        elif fault == "bias":
            offset = template_sensor.get("offset", 0)
            hourly = [_r(v + offset, dec) for v in hourly]

        elif fault == "noise":
            scale = template_sensor.get("noise_scale", 3.0)
            hourly = [_r(v + rng.gauss(0, std * scale), dec) for v in hourly]

        elif fault == "missing":
            pct = template_sensor.get("missing_pct", 0.20)
            n_missing = int(24 * pct)
            missing_hrs = rng.sample(range(24), n_missing)
            for h in missing_hrs:
                hourly[h] = None

        elif fault == "stuck":
            stuck_day = template_sensor.get("stuck_day", 2)
            if day >= stuck_day:
                frozen = _r(center, dec)
                hourly = [frozen] * 24

        valid_vals = [v for v in hourly if v is not None]
        n_missing = hourly.count(None)
        mean_val = _r(sum(valid_vals) / len(valid_vals), dec) if valid_vals else None
        max_val  = _r(max(valid_vals), dec) if valid_vals else None
        min_val  = _r(min(valid_vals), dec) if valid_vals else None
        if len(valid_vals) > 1:
            var = sum((v - mean_val) ** 2 for v in valid_vals) / len(valid_vals)
            std_val = _r(math.sqrt(var), dec)
        else:
            std_val = 0.0

        daily_summaries.append({
            "day": day,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "missing_hours": n_missing,
        })

    # Overall stats
    all_vals = [d["mean"] for d in daily_summaries if d["mean"] is not None]
    overall_mean = _r(sum(all_vals) / len(all_vals), dec) if all_vals else None
    total_missing = sum(d["missing_hours"] for d in daily_summaries)

    # Compute trend (day7_mean - day1_mean if available)
    if daily_summaries[0]["mean"] and daily_summaries[-1]["mean"]:
        trend_total = _r(daily_summaries[-1]["mean"] - daily_summaries[0]["mean"], dec)
        trend_str = f"{'+' if trend_total > 0 else ''}{trend_total} {spec['unit']} over 7 days"
    else:
        trend_str = "unavailable"

    return {
        "sensor_id": template_sensor["id"],
        "parameter": spec["param"],
        "unit": spec["unit"],
        "normal_range": spec["normal_range"],
        "daily_summaries": daily_summaries,
        "stats": {
            "overall_mean": overall_mean,
            "total_missing_hours": total_missing,
            "missing_pct": _r(total_missing / HOURS * 100, 1),
            "trend_7day": trend_str,
        },
    }


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Generate a deterministic Task 2 scenario from seed."""
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(TEMPLATES)) if seed is not None else rng.randint(0, len(TEMPLATES) - 1)
    tmpl = TEMPLATES[idx]

    sensors_obs = []
    for ts in tmpl["sensors"]:
        sensors_obs.append(_build_sensor_summary(SENSOR_PARAMS[ts["key"]], ts, rng))

    observation = {
        "network_id": tmpl["network_id"],
        "location": tmpl["location"],
        "period_days": DAYS,
        "reference_station": {
            "id": tmpl["reference_id"],
            "note": "Clean reference readings available — compare against it to detect bias.",
        },
        "sensors": sensors_obs,
        "instructions": (
            "Diagnose each sensor's fault type and severity, then recommend the correct fix. "
            "One sensor may be fully valid — do not over-diagnose."
        ),
    }
    return observation, {"diagnoses": tmpl["gt"]}


def load_task2(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    return generate_scenario(seed)


# ── Grader ────────────────────────────────────────────────────────────────────

# Fault families for partial credit
_FAULT_FAMILY: Dict[str, set] = {
    "drift":   {"drift", "bias"},
    "bias":    {"bias", "drift"},
    "noise":   {"noise", "spike"},
    "spike":   {"spike", "noise"},
    "stuck":   {"stuck"},
    "missing": {"missing"},
    "valid":   {"valid"},
}

# Fix families for partial credit
_FIX_FAMILY: Dict[str, set] = {
    "recalibrate":       {"recalibrate", "offset_correction"},
    "offset_correction": {"offset_correction", "recalibrate"},
    "interpolate":       {"interpolate", "flag_only"},
    "flag_only":         {"flag_only", "interpolate"},
    "smooth":            {"smooth"},
    "replace":           {"replace", "flag_only"},
    "no_action":         {"no_action"},
}

# Severity adjacency for partial credit
_SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]


def _same_family(f1: str, f2: str) -> bool:
    return f2 in _FAULT_FAMILY.get(f1, {f1})


def _fix_related(fix1: str, fix2: str) -> bool:
    return fix2 in _FIX_FAMILY.get(fix1, {fix1})


def _severity_dist(s1: str, s2: str) -> int:
    try:
        return abs(_SEVERITY_ORDER.index(s1) - _SEVERITY_ORDER.index(s2))
    except ValueError:
        return 2


def grade_task2(action: dict, ground_truth: dict) -> float:
    """
    Weighted score:
      60% — fault_type accuracy (exact=1.0, same family=0.4, wrong=0.0)
      40% — fix accuracy (exact=1.0, related=0.3, wrong=0.0)
    Severity distance penalises by up to 0.1 per severity level off.
    Returns float in [0.0, 1.0].
    """
    diagnoses = action.get("diagnoses", [])
    gt_list   = ground_truth.get("diagnoses", [])

    if not gt_list:
        return 0.0

    gt_map = {d["sensor_id"]: d for d in gt_list}
    sensor_count = len(gt_map)

    fault_scores: List[float] = []
    fix_scores:   List[float] = []

    for sensor_id, gt in gt_map.items():
        # Find matching prediction
        pred = next((d for d in diagnoses
                     if d.get("sensor_id") == sensor_id), None)
        if pred is None:
            fault_scores.append(0.0)
            fix_scores.append(0.0)
            continue

        # Fault type score
        pft = pred.get("fault_type", "")
        gft = gt["fault_type"]
        if pft == gft:
            fs = 1.0
        elif _same_family(pft, gft):
            fs = 0.4
        else:
            fs = 0.0

        # Severity penalty
        sev_dist = _severity_dist(pred.get("severity", "none"), gt["severity"])
        sev_penalty = min(0.2, sev_dist * 0.08)
        fs = max(0.0, fs - sev_penalty)
        fault_scores.append(fs)

        # Fix score
        pfix = pred.get("fix", "")
        gfix = gt["fix"]
        if pfix == gfix:
            fxs = 1.0
        elif _fix_related(pfix, gfix):
            fxs = 0.3
        else:
            fxs = 0.0
        fix_scores.append(fxs)

    fault_avg = sum(fault_scores) / sensor_count
    fix_avg   = sum(fix_scores)   / sensor_count

    score = 0.60 * fault_avg + 0.40 * fix_avg
    return round(max(0.0, min(1.0, score)), 4)
