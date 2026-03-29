"""
Task 2 - Multi-Sensor Data Stream Cleaning (MEDIUM)
====================================================
Agent receives 7 days of REAL measured air quality data
from 5 sensors in an actual industrial network location.
Faults are injected on top of the real baseline per sensor.

Real data source: Open-Meteo Air Quality API (CAMS/ECMWF)
  - 10 real industrial network locations
  - 5 sensors per network (different pollutants)
  - 7 days x 24 hours = 168 real hourly readings per sensor

Agent must diagnose each sensor and recommend the correct fix.
One sensor is always valid (no faults) -- agent must not over-diagnose.

Grader: 60% fault type accuracy + 40% fix appropriateness.
"""

import json
import copy
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

_DATA_PATH = Path(__file__).parent.parent / "data" / "real_task2.json"

def _load_real_data() -> List[Dict]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_REAL_SCENARIOS = _load_real_data()

# ── Normal ranges (EPA/WHO standards) ────────────────────────────────────────
SENSOR_META: Dict[str, Dict] = {
    "PM25": {"param": "PM25_ugm3",  "unit": "ug/m3",  "normal_range": [5,  35]},
    "NO2":  {"param": "NO2_ppb",    "unit": "ppb",    "normal_range": [10, 50]},
    "O3":   {"param": "O3_ppb",     "unit": "ppb",    "normal_range": [20, 60]},
    "SO2":  {"param": "SO2_ppb",    "unit": "ppb",    "normal_range": [2,  15]},
    "CO":   {"param": "CO_ppb",     "unit": "ppb",    "normal_range": [200, 800]},
    "CH4":  {"param": "CH4_ppb",    "unit": "ppb",    "normal_range": [1800, 2000]},
}

# ── 10 fault injection templates ──────────────────────────────────────────────
# Each maps to one real scenario. Faults are injected on real 7-day data.
# Format per sensor: (fault_type, severity, fix, fix_params, injection_params)
FAULT_TEMPLATES: List[List[Dict]] = [
    # 0: COASTAL-ZONE-3 (Houston TX)
    [
        {"sid": "S1", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.8,   "direction": "positive"}, "inj": {"pct_per_day": 0.05}},
        {"sid": "S2", "fault": "missing", "sev": "medium",   "fix": "interpolate",       "fp": {"method": "linear"},         "inj": {"missing_pct": 0.20}},
        {"sid": "S3", "fault": "bias",    "sev": "high",     "fix": "offset_correction", "fp": {"offset": -12.0},            "inj": {"bias_pct": 0.30}},
        {"sid": "S4", "fault": "noise",   "sev": "medium",   "fix": "smooth",            "fp": {"window": 3},                "inj": {"noise_scale": 6.0}},
        {"sid": "S5", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
    ],
    # 1: REFINERY-SOUTH-7 (Jubail)
    [
        {"sid": "S1", "fault": "stuck",   "sev": "critical", "fix": "replace",           "fp": {"start_day": 3},             "inj": {"from_day": 3}},
        {"sid": "S2", "fault": "drift",   "sev": "medium",   "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.5},  "inj": {"pct_per_day": 0.03}},
        {"sid": "S3", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S4", "fault": "bias",    "sev": "high",     "fix": "offset_correction", "fp": {"offset": 8.0},              "inj": {"bias_pct": -0.20}},
        {"sid": "S5", "fault": "missing", "sev": "low",      "fix": "interpolate",       "fp": {"method": "linear"},         "inj": {"missing_pct": 0.10}},
    ],
    # 2: URBAN-GRID-12 (Delhi NCR)
    [
        {"sid": "S1", "fault": "bias",    "sev": "critical", "fix": "offset_correction", "fp": {"offset": -22.0},            "inj": {"bias_pct": 0.55}},
        {"sid": "S2", "fault": "noise",   "sev": "medium",   "fix": "smooth",            "fp": {"window": 5},                "inj": {"noise_scale": 5.0}},
        {"sid": "S3", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.03}, "inj": {"pct_per_day": 0.04}},
        {"sid": "S4", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S5", "fault": "missing", "sev": "high",     "fix": "interpolate",       "fp": {"method": "linear"},         "inj": {"missing_pct": 0.25}},
    ],
    # 3: ARCTIC-STATION-2 (Svalbard)
    [
        {"sid": "S1", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 2.1},  "inj": {"pct_per_day": 0.06}},
        {"sid": "S2", "fault": "bias",    "sev": "medium",   "fix": "offset_correction", "fp": {"offset": -0.08},            "inj": {"bias_pct": 0.15}},
        {"sid": "S3", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S4", "fault": "noise",   "sev": "medium",   "fix": "smooth",            "fp": {"window": 3},                "inj": {"noise_scale": 4.0}},
        {"sid": "S5", "fault": "missing", "sev": "high",     "fix": "interpolate",       "fp": {"method": "spline"},         "inj": {"missing_pct": 0.30}},
    ],
    # 4: AMAZON-BASIN-4 (Manaus)
    [
        {"sid": "S1", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S2", "fault": "stuck",   "sev": "high",     "fix": "replace",           "fp": {"start_day": 2},             "inj": {"from_day": 2}},
        {"sid": "S3", "fault": "drift",   "sev": "medium",   "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.005},"inj": {"pct_per_day": 0.02}},
        {"sid": "S4", "fault": "noise",   "sev": "low",      "fix": "smooth",            "fp": {"window": 3},                "inj": {"noise_scale": 3.5}},
        {"sid": "S5", "fault": "bias",    "sev": "medium",   "fix": "offset_correction", "fp": {"offset": 6.0},              "inj": {"bias_pct": -0.18}},
    ],
    # 5: NORTH-SEA-OIL-6 (North Sea)
    [
        {"sid": "S1", "fault": "bias",    "sev": "critical", "fix": "offset_correction", "fp": {"offset": -0.12},            "inj": {"bias_pct": 0.35}},
        {"sid": "S2", "fault": "missing", "sev": "high",     "fix": "flag_only",         "fp": {"reason": "gap_too_large"},  "inj": {"missing_pct": 0.35}},
        {"sid": "S3", "fault": "noise",   "sev": "high",     "fix": "smooth",            "fp": {"window": 5},                "inj": {"noise_scale": 8.0}},
        {"sid": "S4", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 3.0},  "inj": {"pct_per_day": 0.07}},
        {"sid": "S5", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
    ],
    # 6: RUHR-VALLEY-9 (Essen)
    [
        {"sid": "S1", "fault": "drift",   "sev": "medium",   "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.4},  "inj": {"pct_per_day": 0.03}},
        {"sid": "S2", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S3", "fault": "stuck",   "sev": "critical", "fix": "replace",           "fp": {"start_day": 4},             "inj": {"from_day": 4}},
        {"sid": "S4", "fault": "bias",    "sev": "high",     "fix": "offset_correction", "fp": {"offset": 9.0},              "inj": {"bias_pct": -0.25}},
        {"sid": "S5", "fault": "missing", "sev": "low",      "fix": "interpolate",       "fp": {"method": "linear"},         "inj": {"missing_pct": 0.10}},
    ],
    # 7: PEARL-RIVER-11 (Guangzhou)
    [
        {"sid": "S1", "fault": "missing", "sev": "high",     "fix": "flag_only",         "fp": {"reason": "gap_too_large"},  "inj": {"missing_pct": 0.40}},
        {"sid": "S2", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 1.5},  "inj": {"pct_per_day": 0.05}},
        {"sid": "S3", "fault": "noise",   "sev": "medium",   "fix": "smooth",            "fp": {"window": 3},                "inj": {"noise_scale": 6.0}},
        {"sid": "S4", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S5", "fault": "bias",    "sev": "critical", "fix": "offset_correction", "fp": {"offset": -30.0},            "inj": {"bias_pct": 0.60}},
    ],
    # 8: GREAT-PLAINS-14 (Wichita)
    [
        {"sid": "S1", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S2", "fault": "drift",   "sev": "high",     "fix": "recalibrate",       "fp": {"drift_rate_per_day": 1.2},  "inj": {"pct_per_day": 0.05}},
        {"sid": "S3", "fault": "bias",    "sev": "critical", "fix": "offset_correction", "fp": {"offset": -15.0},            "inj": {"bias_pct": 0.40}},
        {"sid": "S4", "fault": "noise",   "sev": "medium",   "fix": "smooth",            "fp": {"window": 3},                "inj": {"noise_scale": 5.5}},
        {"sid": "S5", "fault": "missing", "sev": "medium",   "fix": "interpolate",       "fp": {"method": "linear"},         "inj": {"missing_pct": 0.18}},
    ],
    # 9: THAMES-ESTUARY-17 (Tilbury)
    [
        {"sid": "S1", "fault": "noise",   "sev": "high",     "fix": "smooth",            "fp": {"window": 5},                "inj": {"noise_scale": 9.0}},
        {"sid": "S2", "fault": "valid",   "sev": "none",     "fix": "no_action",         "fp": {},                           "inj": {}},
        {"sid": "S3", "fault": "bias",    "sev": "high",     "fix": "offset_correction", "fp": {"offset": 18.0},             "inj": {"bias_pct": -0.35}},
        {"sid": "S4", "fault": "stuck",   "sev": "critical", "fix": "replace",           "fp": {"start_day": 5},             "inj": {"from_day": 5}},
        {"sid": "S5", "fault": "drift",   "sev": "medium",   "fix": "recalibrate",       "fp": {"drift_rate_per_day": 0.007},"inj": {"pct_per_day": 0.02}},
    ],
]

import random

def _inject_faults(daily_means: List[Dict], fault_info: Dict, rng: random.Random) -> List[Dict]:
    """Inject a fault into real 7-day daily means. Returns modified copy."""
    days = copy.deepcopy(daily_means)
    fault = fault_info["fault"]
    inj   = fault_info["inj"]

    if fault == "valid":
        return days

    elif fault == "drift":
        pct_pd = inj.get("pct_per_day", 0.04)
        for d in days:
            if d["mean"] is not None:
                d["mean"] = round(d["mean"] * (1 + pct_pd * d["day"]), 2)

    elif fault == "bias":
        bias_pct = inj.get("bias_pct", 0.25)
        for d in days:
            if d["mean"] is not None:
                d["mean"] = round(d["mean"] * (1 + bias_pct), 2)

    elif fault == "noise":
        scale = inj.get("noise_scale", 4.0)
        for d in days:
            if d["mean"] is not None:
                noise = rng.gauss(0, scale)
                d["mean"] = round(max(0.01, d["mean"] + noise), 2)
                d["std"]  = round(scale * 1.2, 2)

    elif fault == "missing":
        pct = inj.get("missing_pct", 0.20)
        n_days_missing = max(1, int(7 * pct))
        missing_days = rng.sample(range(7), n_days_missing)
        for i in missing_days:
            days[i]["mean"] = None
            days[i]["missing_hours"] = 24

    elif fault == "stuck":
        from_day = inj.get("from_day", 3) - 1  # 0-indexed
        frozen = days[from_day - 1]["mean"] if from_day > 0 and days[from_day - 1]["mean"] else days[0]["mean"]
        for i in range(from_day, 7):
            days[i]["mean"] = frozen
            days[i]["std"]  = 0.0

    return days


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Build a Task 2 scenario from REAL 7-day air quality data."""
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(_REAL_SCENARIOS)) if seed is not None else rng.randint(0, len(_REAL_SCENARIOS) - 1)

    real_sc      = _REAL_SCENARIOS[idx]
    fault_tmpl   = FAULT_TEMPLATES[idx]

    sensors_obs = []
    for real_s, ft in zip(real_sc["sensors"], fault_tmpl):
        sk   = real_s["sensor_key"]
        meta = SENSOR_META.get(sk, SENSOR_META["NO2"])

        daily = _inject_faults(real_s["daily_means"], ft, rng)

        ok_means = [d["mean"] for d in daily if d["mean"] is not None]
        total_missing = sum(d["missing_hours"] for d in daily)

        if len(ok_means) >= 2:
            overall_mean = round(sum(ok_means) / len(ok_means), 2)
            trend_delta  = round(ok_means[-1] - ok_means[0], 2) if ok_means else 0
            trend_str    = f"{'+' if trend_delta >= 0 else ''}{trend_delta} {meta['unit']} over 7 days"
        else:
            overall_mean = ok_means[0] if ok_means else None
            trend_str    = "unavailable"

        sensors_obs.append({
            "sensor_id":     ft["sid"],
            "parameter":     meta["param"],
            "unit":          meta["unit"],
            "normal_range":  meta["normal_range"],
            "data_source":   "Open-Meteo CAMS/ECMWF (real measured baseline)",
            "daily_summaries": daily,
            "stats": {
                "overall_mean":         overall_mean,
                "total_missing_hours":  total_missing,
                "missing_pct":          round(total_missing / 168 * 100, 1),
                "trend_7day":           trend_str,
            },
        })

    observation = {
        "network_id":   real_sc["network_id"],
        "location":     real_sc["location"],
        "lat":          real_sc["lat"],
        "lon":          real_sc["lon"],
        "period_days":  7,
        "data_source":  "Open-Meteo CAMS/ECMWF (real measured baseline + fault injection)",
        "reference_station": {
            "id": f"REF-{real_sc['network_id'][:4]}",
            "note": "Clean reference readings -- compare for bias detection.",
        },
        "sensors": sensors_obs,
        "instructions": (
            "Diagnose each sensor fault and recommend the correct fix. "
            "One sensor is fully valid. Do not over-diagnose."
        ),
    }

    gt_diagnoses = [
        {
            "sensor_id": ft["sid"],
            "fault_type": ft["fault"],
            "severity":   ft["sev"],
            "fix":        ft["fix"],
            "fix_params": ft["fp"],
        }
        for ft in fault_tmpl
    ]

    return observation, {"diagnoses": gt_diagnoses}


def load_task2(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    return generate_scenario(seed)


# ── Grader ────────────────────────────────────────────────────────────────────

_FAULT_FAMILY: Dict[str, set] = {
    "drift":   {"drift", "bias"},
    "bias":    {"bias", "drift"},
    "noise":   {"noise", "spike"},
    "spike":   {"spike", "noise"},
    "stuck":   {"stuck"},
    "missing": {"missing"},
    "valid":   {"valid"},
}

_FIX_FAMILY: Dict[str, set] = {
    "recalibrate":       {"recalibrate", "offset_correction"},
    "offset_correction": {"offset_correction", "recalibrate"},
    "interpolate":       {"interpolate", "flag_only"},
    "flag_only":         {"flag_only", "interpolate"},
    "smooth":            {"smooth"},
    "replace":           {"replace", "flag_only"},
    "no_action":         {"no_action"},
}

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
    diagnoses = action.get("diagnoses", [])
    gt_list   = ground_truth.get("diagnoses", [])
    if not gt_list:
        return 0.0

    gt_map = {d["sensor_id"]: d for d in gt_list}
    sensor_count = len(gt_map)

    fault_scores: List[float] = []
    fix_scores:   List[float] = []

    for sensor_id, gt in gt_map.items():
        pred = next((d for d in diagnoses if d.get("sensor_id") == sensor_id), None)
        if pred is None:
            fault_scores.append(0.0)
            fix_scores.append(0.0)
            continue

        pft = pred.get("fault_type", "")
        gft = gt["fault_type"]
        if pft == gft:
            fs = 1.0
        elif _same_family(pft, gft):
            fs = 0.4
        else:
            fs = 0.0

        sev_dist = _severity_dist(pred.get("severity", "none"), gt["severity"])
        fs = max(0.0, fs - min(0.2, sev_dist * 0.08))
        fault_scores.append(fs)

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
    return round(max(0.0, min(1.0, 0.60 * fault_avg + 0.40 * fix_avg)), 4)
