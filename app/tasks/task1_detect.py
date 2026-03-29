"""
Task 1 - Single Sensor Anomaly Detection (EASY)
================================================
Agent receives 24 hours of REAL measured air quality readings
from an actual monitoring location (sourced from Open-Meteo/CAMS).
Faults are injected on top of the real baseline.

Real data source: Open-Meteo Air Quality API (CAMS/ECMWF)
  - PM2.5, NO2, O3, SO2, CO, CH4
  - 20 real global industrial monitoring locations

Fault injection on real data:
  outlier  -> single reading spiked far outside normal (sensor malfunction)
  stuck    -> repeating same value (firmware freeze)
  missing  -> null reading (network dropout)
  drift    -> gradual systematic shift added to real baseline
  spike    -> short burst anomaly injected (EMI event)
  bias     -> constant offset added to real readings (calibration error)

Grader: F1 score between predicted and actual fault flags + calibration bonus.
"""

import json
import random
import copy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# ── Load real sensor data ─────────────────────────────────────────────────────
_DATA_PATH = Path(__file__).parent.parent / "data" / "real_task1.json"

def _load_real_data() -> List[Dict]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_REAL_SCENARIOS = _load_real_data()

# ── Sensor parameter metadata (normal ranges from EPA/WHO standards) ──────────
SENSOR_META: Dict[str, Dict] = {
    "PM25": {"param": "PM25_ugm3",  "unit": "ug/m3", "normal_range": [5,  35],    "dec": 1,
             "outlier_val": 999.9,  "spike_val": 380.0},
    "NO2":  {"param": "NO2_ppb",    "unit": "ppb",   "normal_range": [10, 50],    "dec": 1,
             "outlier_val": 500.0,  "spike_val": 220.0},
    "O3":   {"param": "O3_ppb",     "unit": "ppb",   "normal_range": [20, 60],    "dec": 1,
             "outlier_val": 800.0,  "spike_val": 320.0},
    "SO2":  {"param": "SO2_ppb",    "unit": "ppb",   "normal_range": [2,  15],    "dec": 1,
             "outlier_val": 300.0,  "spike_val": 120.0},
    "CO":   {"param": "CO_ppb",     "unit": "ppb",   "normal_range": [200, 800],  "dec": 0,
             "outlier_val": 9999.0, "spike_val": 4500.0},
    "CH4":  {"param": "CH4_ppb",    "unit": "ppb",   "normal_range": [1800, 2000],"dec": 0,
             "outlier_val": 9500.0, "spike_val": 4800.0},
}

# ── 20 fault injection templates (applied on top of real data) ────────────────
# Each matches one real scenario by index (0-19)
FAULT_TEMPLATES: List[List[Dict]] = [
    # 0: stuck block + outlier + missing
    [{"type": "stuck",   "start": 3,  "end": 5},
     {"type": "outlier", "hour": 6},
     {"type": "missing", "hours": [8, 9]}],
    # 1: full-day upward drift
    [{"type": "drift", "start": 0, "end": 23, "pct": 0.35}],
    # 2: daytime spike + isolated missing
    [{"type": "spike",   "start": 10, "end": 12},
     {"type": "missing", "hours": [18]}],
    # 3: afternoon bias
    [{"type": "bias", "start": 8, "end": 23, "pct": 0.30}],
    # 4: overnight stuck + afternoon outlier
    [{"type": "stuck",   "start": 0, "end": 8},
     {"type": "outlier", "hour": 15}],
    # 5: missing block + evening spike
    [{"type": "missing", "hours": [3, 4, 5, 6, 7]},
     {"type": "spike",   "start": 18, "end": 20}],
    # 6: midday stuck + late missing
    [{"type": "stuck",   "start": 12, "end": 16},
     {"type": "missing", "hours": [22, 23]}],
    # 7: three scattered outliers
    [{"type": "outlier", "hour": 2},
     {"type": "outlier", "hour": 14},
     {"type": "outlier", "hour": 21}],
    # 8: partial drift + missing gap
    [{"type": "drift",   "start": 4, "end": 23, "pct": 0.28},
     {"type": "missing", "hours": [5, 6]}],
    # 9: ALL VALID (no faults -- agent must resist over-flagging)
    [],
    # 10: outlier + stuck + missing
    [{"type": "outlier", "hour": 5},
     {"type": "stuck",   "start": 8, "end": 11},
     {"type": "missing", "hours": [15, 16, 17]}],
    # 11: strong afternoon drift
    [{"type": "drift", "start": 6, "end": 23, "pct": 0.60}],
    # 12: early-morning spike
    [{"type": "spike", "start": 0, "end": 3}],
    # 13: overnight data loss
    [{"type": "missing", "hours": list(range(0, 8))}],
    # 14: stuck mid-morning + single outlier
    [{"type": "stuck",   "start": 5, "end": 9},
     {"type": "outlier", "hour": 20}],
    # 15: single isolated outlier
    [{"type": "outlier", "hour": 13}],
    # 16: single missing hour
    [{"type": "missing", "hours": [12]}],
    # 17: overnight bias
    [{"type": "bias", "start": 8, "end": 23, "pct": 0.22}],
    # 18: end-of-day stuck
    [{"type": "stuck", "start": 20, "end": 23}],
    # 19: four-fault stress test
    [{"type": "outlier", "hour": 3},
     {"type": "stuck",   "start": 10, "end": 13},
     {"type": "missing", "hours": [19, 20]},
     {"type": "spike",   "start": 22, "end": 23}],
]


def _r(v: float, dec: int) -> float:
    return round(v, dec)


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """
    Build a Task 1 scenario from REAL measured air quality data.
    Faults are injected on top of the real baseline readings.
    Same seed always produces the same result (deterministic).
    """
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(_REAL_SCENARIOS)) if seed is not None else rng.randint(0, len(_REAL_SCENARIOS) - 1)

    real_scenario = _REAL_SCENARIOS[idx]
    faults        = FAULT_TEMPLATES[idx]
    sensor_key    = real_scenario["sensor_key"]
    meta          = SENSOR_META.get(sensor_key, SENSOR_META["NO2"])
    dec           = meta["dec"]

    # Start with real measured values (deep copy)
    readings: List[Optional[float]] = copy.deepcopy(real_scenario["real_readings"])

    # Fill any None values in real data with interpolated neighbour
    for h in range(24):
        if readings[h] is None:
            # Find nearest valid neighbour
            for delta in range(1, 24):
                if h - delta >= 0 and readings[h - delta] is not None:
                    readings[h] = readings[h - delta]
                    break
                if h + delta < 24 and readings[h + delta] is not None:
                    readings[h] = readings[h + delta]
                    break
            if readings[h] is None:
                readings[h] = _r(sum(v for v in readings if v is not None) /
                                  max(1, sum(1 for v in readings if v is not None)), dec)

    gt_flags: List[Dict] = []

    # Inject faults on top of real baseline
    for fault in faults:
        ft = fault["type"]

        if ft == "stuck":
            s, e = fault["start"], fault["end"]
            frozen = readings[s]
            for h in range(s, e + 1):
                readings[h] = frozen
                gt_flags.append({"hour": h, "fault": "stuck"})

        elif ft == "outlier":
            h = fault["hour"]
            readings[h] = meta["outlier_val"]
            gt_flags.append({"hour": h, "fault": "outlier"})

        elif ft == "missing":
            for h in fault["hours"]:
                readings[h] = None
                gt_flags.append({"hour": h, "fault": "missing"})

        elif ft == "drift":
            s, e = fault["start"], fault["end"]
            pct  = fault.get("pct", 0.30)  # % of current value added as drift
            span = max(1, e - s)
            for h in range(s, e + 1):
                frac = (h - s) / span
                drift_add = _r(readings[h] * pct * frac, dec)
                readings[h] = _r(readings[h] + drift_add, dec)
                gt_flags.append({"hour": h, "fault": "drift"})

        elif ft == "spike":
            s, e = fault["start"], fault["end"]
            sv = meta["spike_val"]
            for h in range(s, e + 1):
                readings[h] = sv
                gt_flags.append({"hour": h, "fault": "spike"})

        elif ft == "bias":
            s, e = fault.get("start", 0), fault.get("end", 23)
            pct  = fault.get("pct", 0.25)
            for h in range(s, e + 1):
                readings[h] = _r(readings[h] + readings[h] * pct, dec)
                gt_flags.append({"hour": h, "fault": "bias"})

    sensor_id = f"{sensor_key}-{100 + idx:03d}"

    observation = {
        "sensor_id":    sensor_id,
        "parameter":    meta["param"],
        "unit":         meta["unit"],
        "location":     real_scenario["location"],
        "lat":          real_scenario["lat"],
        "lon":          real_scenario["lon"],
        "normal_range": meta["normal_range"],
        "data_source":  "Open-Meteo CAMS/ECMWF (real measured data)",
        "readings":     [{"hour": h, "value": readings[h]} for h in range(24)],
        "hint":         "Identify faulty readings. Baseline is real CAMS/ECMWF data. Faults injected on top.",
    }
    ground_truth = {
        "sensor_id": sensor_id,
        "flags": sorted(gt_flags, key=lambda x: x["hour"]),
    }
    return observation, ground_truth


def load_task1(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    return generate_scenario(seed)


# ── Grader ────────────────────────────────────────────────────────────────────

def grade_task1(action: dict, ground_truth: dict) -> float:
    """
    F1 score between predicted and actual fault flags.
    Bonus +0.05 for varied confidence values (calibration reward).
    Always returns float in [0.0, 1.0].
    """
    flags    = action.get("flags", [])
    gt_flags = ground_truth.get("flags", [])

    predicted = {(int(f["hour"]), str(f["fault"])) for f in flags if f.get("fault") != "valid"}
    actual    = {(int(f["hour"]), str(f["fault"])) for f in gt_flags}

    # All-valid scenario: agent must NOT flag anything
    if not actual:
        non_valid_count = len(predicted)
        return max(0.0, round(1.0 - non_valid_count * 0.2, 4))

    if not predicted:
        return 0.0

    correct   = predicted & actual
    precision = len(correct) / len(predicted)
    recall    = len(correct) / len(actual)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Calibration bonus: reward expressing varied confidence
    confidences = [f.get("confidence", 1.0) for f in flags]
    cal_bonus = 0.05 if len(set(round(c, 1) for c in confidences)) > 1 else 0.0

    return round(max(0.0, min(1.0, f1 + cal_bonus)), 4)
