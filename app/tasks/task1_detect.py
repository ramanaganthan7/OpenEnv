"""
Task 1 — Single Sensor Anomaly Detection (EASY)
================================================
Agent receives 24 hours of readings from ONE sensor.
Some readings contain injected faults.
Agent must identify WHICH hours are faulty and WHAT fault type.

Action schema:
  { "sensor_id": str, "flags": [{"hour": int, "fault": str, "confidence": float}] }

Fault types: outlier | stuck | missing | drift | spike | bias
  outlier  — single extreme reading
  stuck    — repeating same value for multiple hours
  missing  — null reading (sensor dropout)
  drift    — gradual systematic shift from baseline
  spike    — short burst of bad readings (2-4 hours)
  bias     — all readings shifted by a fixed offset

Grader: F1 score between predicted and actual fault flags, +calibration bonus.
"""

import random
from typing import Optional, Tuple, Dict, Any, List

# ── Sensor catalogue ──────────────────────────────────────────────────────────
# dec = decimal places for rounding
SENSOR_CATALOG: Dict[str, Dict] = {
    "CO2":  {"param": "CO2_ppm",     "unit": "ppm",    "normal_range": [400, 450],
             "center": 420.0, "std": 3.0,  "outlier_val": 9999.0, "spike_val": 2100.0, "dec": 1},
    "NO2":  {"param": "NO2_ppb",     "unit": "ppb",    "normal_range": [10, 50],
             "center": 28.0,  "std": 2.5,  "outlier_val": 500.0,  "spike_val": 220.0,  "dec": 1},
    "CH4":  {"param": "CH4_ppm",     "unit": "ppm",    "normal_range": [1.7, 2.0],
             "center": 1.87,  "std": 0.02, "outlier_val": 9.5,    "spike_val": 4.8,    "dec": 3},
    "O3":   {"param": "O3_ppb",      "unit": "ppb",    "normal_range": [20, 60],
             "center": 38.0,  "std": 3.0,  "outlier_val": 800.0,  "spike_val": 320.0,  "dec": 1},
    "PM25": {"param": "PM25_ugm3",   "unit": "µg/m³",  "normal_range": [5, 35],
             "center": 18.0,  "std": 2.0,  "outlier_val": 999.0,  "spike_val": 380.0,  "dec": 1},
    "SO2":  {"param": "SO2_ppb",     "unit": "ppb",    "normal_range": [2, 15],
             "center": 8.0,   "std": 1.0,  "outlier_val": 300.0,  "spike_val": 120.0,  "dec": 1},
    "TEMP": {"param": "Temp_C",      "unit": "°C",     "normal_range": [15, 35],
             "center": 23.0,  "std": 0.5,  "outlier_val": 99.9,   "spike_val": 72.0,   "dec": 1},
    "HUM":  {"param": "Humidity_pct","unit": "%",      "normal_range": [30, 80],
             "center": 55.0,  "std": 2.0,  "outlier_val": 99.9,   "spike_val": 97.5,   "dec": 1},
}

LOCATIONS: List[str] = [
    "Industrial Zone A, Houston TX",
    "Delhi NCR Monitoring Station #12",
    "North Sea Oil Field, Aberdeen UK",
    "Los Angeles Basin Air Monitor, CA",
    "Beijing Haidian Air Quality Station",
    "Chemical Corridor, Baton Rouge LA",
    "Sahara Weather Station, Tamanrasset Algeria",
    "Coastal Zone Monitor, Chennai India",
    "Petroleum Refinery, Ahmadi Kuwait",
    "Arctic Research Station, Svalbard Norway",
    "Great Barrier Reef Monitor, QLD Australia",
    "Silicon Valley Air Basin, San Jose CA",
    "Amazon Basin Station, Manaus Brazil",
    "Thames Estuary Monitor, Tilbury UK",
    "Mongolian Steppe Station, Ulaanbaatar",
    "Pearl River Delta, Guangzhou China",
    "Ruhr Valley Industrial, Essen Germany",
    "Nile Delta Station, Alexandria Egypt",
    "Siberian Taiga Station, Novosibirsk Russia",
    "Great Plains Monitor, Wichita KS",
]

# 20 scenario templates: (sensor_key, location_idx, fault_list)
# Each fault: dict with "type" and additional params
TEMPLATES: List[Tuple] = [
    # 0: CO2 — stuck block + single outlier + gap
    ("CO2", 0, [
        {"type": "stuck",   "start": 3,  "end": 5},
        {"type": "outlier", "hour": 6},
        {"type": "missing", "hours": [8, 9]},
    ]),
    # 1: NO2 — full-day upward drift
    ("NO2", 1, [
        {"type": "drift", "start": 0, "end": 23, "delta": 28.0},
    ]),
    # 2: CH4 — daytime spike + isolated missing
    ("CH4", 2, [
        {"type": "spike",   "start": 10, "end": 12},
        {"type": "missing", "hours": [18]},
    ]),
    # 3: O3 — afternoon bias
    ("O3", 3, [
        {"type": "bias", "start": 8, "end": 23, "offset": 18.0},
    ]),
    # 4: PM25 — overnight stuck + afternoon outlier
    ("PM25", 4, [
        {"type": "stuck",   "start": 0, "end": 8},
        {"type": "outlier", "hour": 15},
    ]),
    # 5: SO2 — missing block + evening spike
    ("SO2", 5, [
        {"type": "missing", "hours": [3, 4, 5, 6, 7]},
        {"type": "spike",   "start": 18, "end": 20},
    ]),
    # 6: TEMP — midday stuck + late missing
    ("TEMP", 6, [
        {"type": "stuck",   "start": 12, "end": 16},
        {"type": "missing", "hours": [22, 23]},
    ]),
    # 7: CO2 — three scattered outliers
    ("CO2", 7, [
        {"type": "outlier", "hour": 2},
        {"type": "outlier", "hour": 14},
        {"type": "outlier", "hour": 21},
    ]),
    # 8: NO2 — partial drift + missing gap in drift window
    ("NO2", 8, [
        {"type": "drift",   "start": 4, "end": 23, "delta": 22.0},
        {"type": "missing", "hours": [5, 6]},
    ]),
    # 9: CH4 — ALL VALID (no faults — agent must resist over-flagging)
    ("CH4", 9, []),
    # 10: O3 — three-fault combo (outlier + stuck + missing gap)
    ("O3", 10, [
        {"type": "outlier", "hour": 5},
        {"type": "stuck",   "start": 8, "end": 11},
        {"type": "missing", "hours": [15, 16, 17]},
    ]),
    # 11: PM25 — strong afternoon drift
    ("PM25", 11, [
        {"type": "drift", "start": 6, "end": 23, "delta": 45.0},
    ]),
    # 12: SO2 — early-morning spike (hours 0-3)
    ("SO2", 12, [
        {"type": "spike", "start": 0, "end": 3},
    ]),
    # 13: TEMP — overnight data loss
    ("TEMP", 13, [
        {"type": "missing", "hours": list(range(0, 8))},
    ]),
    # 14: HUM — stuck mid-morning + single outlier
    ("HUM", 14, [
        {"type": "stuck",   "start": 5, "end": 9},
        {"type": "outlier", "hour": 20},
    ]),
    # 15: CO2 — single isolated outlier
    ("CO2", 15, [
        {"type": "outlier", "hour": 13},
    ]),
    # 16: NO2 — single missing hour (noon)
    ("NO2", 16, [
        {"type": "missing", "hours": [12]},
    ]),
    # 17: CH4 — overnight bias
    ("CH4", 17, [
        {"type": "bias", "start": 8, "end": 23, "offset": 0.18},
    ]),
    # 18: O3 — end-of-day stuck
    ("O3", 18, [
        {"type": "stuck", "start": 20, "end": 23},
    ]),
    # 19: PM25 — four-fault stress test
    ("PM25", 19, [
        {"type": "outlier", "hour": 3},
        {"type": "stuck",   "start": 10, "end": 13},
        {"type": "missing", "hours": [19, 20]},
        {"type": "spike",   "start": 22, "end": 23},
    ]),
]


def _r(val: float, dec: int) -> float:
    return round(val, dec)


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate a deterministic Task 1 scenario.
    Same seed always produces the same scenario.
    """
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(TEMPLATES)) if seed is not None else rng.randint(0, len(TEMPLATES) - 1)

    sensor_key, loc_idx, faults = TEMPLATES[idx]
    spec = SENSOR_CATALOG[sensor_key]
    dec = spec["dec"]
    center = spec["center"]
    std = spec["std"]

    # Generate 24 clean baseline readings with realistic natural variation
    readings: List[Optional[float]] = [
        _r(center + rng.gauss(0, std * 0.55), dec)
        for _ in range(24)
    ]

    gt_flags: List[Dict] = []

    for fault in faults:
        ft = fault["type"]

        if ft == "stuck":
            s, e = fault["start"], fault["end"]
            frozen = readings[s]   # freeze at value present at fault start
            for h in range(s, e + 1):
                readings[h] = frozen
            for h in range(s, e + 1):
                gt_flags.append({"hour": h, "fault": "stuck"})

        elif ft == "outlier":
            h = fault["hour"]
            readings[h] = spec["outlier_val"]
            gt_flags.append({"hour": h, "fault": "outlier"})

        elif ft == "missing":
            for h in fault["hours"]:
                readings[h] = None
                gt_flags.append({"hour": h, "fault": "missing"})

        elif ft == "drift":
            s, e = fault["start"], fault["end"]
            total = fault["delta"]
            span = max(1, e - s)
            for h in range(s, e + 1):
                frac = (h - s) / span
                readings[h] = _r(readings[h] + total * frac, dec)
            for h in range(s, e + 1):
                gt_flags.append({"hour": h, "fault": "drift"})

        elif ft == "spike":
            s, e = fault["start"], fault["end"]
            sv = spec["spike_val"]
            for h in range(s, e + 1):
                readings[h] = sv
            for h in range(s, e + 1):
                gt_flags.append({"hour": h, "fault": "spike"})

        elif ft == "bias":
            s, e = fault.get("start", 0), fault.get("end", 23)
            off = fault["offset"]
            for h in range(s, e + 1):
                readings[h] = _r(readings[h] + off, dec)
            for h in range(s, e + 1):
                gt_flags.append({"hour": h, "fault": "bias"})

    sensor_id = f"{sensor_key}-{100 + idx:03d}"

    observation = {
        "sensor_id": sensor_id,
        "parameter": spec["param"],
        "unit": spec["unit"],
        "location": LOCATIONS[loc_idx],
        "normal_range": spec["normal_range"],
        "readings": [{"hour": h, "value": readings[h]} for h in range(24)],
        "hint": "Identify each faulty reading by hour. Faults: outlier, stuck, missing, drift, spike, bias.",
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
    Bonus +0.05 for varied confidence values (calibration).
    Never returns NaN or values outside [0.0, 1.0].
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

    # Calibration bonus: reward expressing varied confidence (not all 1.0)
    confidences = [f.get("confidence", 1.0) for f in flags]
    cal_bonus = 0.05 if len(set(round(c, 1) for c in confidences)) > 1 else 0.0

    return round(max(0.0, min(1.0, f1 + cal_bonus)), 4)
