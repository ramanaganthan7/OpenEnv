"""
Task 3 - Cascade Failure Diagnosis & Compliance Audit (HARD)
=============================================================
Agent receives 30 days of REAL measured air quality data
from a 10-sensor industrial network.
Reference sensors are artificially taken offline, corrupting
their dependent sensors (cascade failure simulation on real data).

Real data source: Open-Meteo Air Quality API (CAMS/ECMWF)
  - 5 real industrial network locations
  - 10 sensors per network (30 days x 24 hours = 720 real readings)

Agent must:
  1. Identify ROOT CAUSE sensors (references that failed)
  2. Determine correct REPAIR ORDER (topology-aware)
  3. Identify FAULT WINDOW (start/end day)
  4. Check EPA REGULATORY COMPLIANCE under uncertainty

Grader: 4-component weighted score.
  35% root cause (Jaccard)
  30% repair order (dependency violations)
  20% fault window (temporal accuracy)
  15% compliance accuracy
"""

import json
import copy
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Set

_DATA_PATH = Path(__file__).parent.parent / "data" / "real_task3.json"

def _load_real_data() -> List[Dict]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_REAL_SCENARIOS = _load_real_data()

# ── EPA regulatory thresholds (NAAQS) ─────────────────────────────────────────
EPA_THRESHOLDS: Dict[str, Dict] = {
    "CH4_ppb":  {"warning": 1800, "violation": 2000},
    "NO2_ppb":  {"warning": 80,   "violation": 100},
    "PM25_ugm3":{"warning": 35,   "violation": 65},
    "SO2_ppb":  {"warning": 70,   "violation": 100},
    "O3_ppb":   {"warning": 70,   "violation": 85},
}

# ── 5 cascade scenario templates ─────────────────────────────────────────────
TEMPLATES: List[Dict] = [
    # 0: REFINERY-NORTH (Ahmadi Kuwait)
    {
        "dependency_graph": {
            "S1":  {"calibrates": ["S4", "S5"], "role": "reference"},
            "S2":  {"independent": True},
            "S3":  {"calibrates": ["S6", "S7", "S8"], "role": "reference"},
            "S4":  {"calibrated_by": "S1", "role": "dependent"},
            "S5":  {"calibrated_by": "S1", "role": "dependent"},
            "S6":  {"calibrated_by": "S3", "role": "dependent"},
            "S7":  {"calibrated_by": "S3", "role": "dependent"},
            "S8":  {"calibrated_by": "S3", "role": "dependent"},
            "S9":  {"independent": True},
            "S10": {"independent": True},
        },
        "faults": {
            "S1": {"type": "offline",   "start": 8,  "end": 21},
            "S3": {"type": "offline",   "start": 12, "end": 21},
            "S4": {"type": "corrupted", "start": 8,  "end": 21, "corrupt_pct": 1.38},
            "S5": {"type": "corrupted", "start": 8,  "end": 21, "corrupt_pct": 1.38},
            "S6": {"type": "corrupted", "start": 12, "end": 21, "corrupt_pct": 1.42},
            "S7": {"type": "corrupted", "start": 12, "end": 21, "corrupt_pct": 1.42},
            "S8": {"type": "corrupted", "start": 12, "end": 21, "corrupt_pct": 1.42},
        },
        "regulatory_thresholds": {
            "CH4_ppb": {"warning": 1800, "violation": 2000},
            "NO2_ppb": {"warning": 80,   "violation": 100},
        },
        "known_facts": [
            "S1 (CH4 reference) was offline days 8-21 due to calibration gas depletion",
            "S4 and S5 (CH4 dependents) show systematic +38% error matching S1 outage period",
            "S3 (NO2 reference) went offline day 12 after power surge",
            "S9 and S10 (NOx independent) readings were stable throughout -- no anomalies",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S3"],
            "repair_order": ["S1", "S3", "S4", "S5", "S6", "S7", "S8"],
            "fault_window": {"start": "day_8", "end": "day_21"},
            "compliance": {"CH4_ppb": "POSSIBLE_VIOLATION", "NO2_ppb": "CLEAN"},
        },
    },
    # 1: PIPELINE-CENTRAL (Houston TX)
    {
        "dependency_graph": {
            "S1":  {"independent": True},
            "S2":  {"calibrates": ["S5", "S6", "S7"], "role": "reference"},
            "S3":  {"independent": True},
            "S4":  {"calibrates": ["S8", "S9"], "role": "reference"},
            "S5":  {"calibrated_by": "S2", "role": "dependent"},
            "S6":  {"calibrated_by": "S2", "role": "dependent"},
            "S7":  {"calibrated_by": "S2", "role": "dependent"},
            "S8":  {"calibrated_by": "S4", "role": "dependent"},
            "S9":  {"calibrated_by": "S4", "role": "dependent"},
            "S10": {"independent": True},
        },
        "faults": {
            "S2": {"type": "offline",   "start": 5,  "end": 18},
            "S5": {"type": "corrupted", "start": 5,  "end": 18, "corrupt_pct": 1.45},
            "S6": {"type": "corrupted", "start": 5,  "end": 18, "corrupt_pct": 1.45},
            "S7": {"type": "corrupted", "start": 5,  "end": 18, "corrupt_pct": 1.45},
        },
        "regulatory_thresholds": {
            "PM25_ugm3": {"warning": 35, "violation": 65},
            "SO2_ppb":   {"warning": 70, "violation": 100},
        },
        "known_facts": [
            "S2 (PM25 reference) went offline days 5-18 after membrane fouling",
            "Pre-failure S2 readings (days 1-4) showed PM25 averaging 68 ug/m3 -- above violation threshold",
            "S5, S6, S7 (PM25 dependents) show systematic corruption matching S2 outage",
            "S1 and S10 (SO2 independent) remained stable -- readings 4-9 ppb throughout",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S2"],
            "repair_order": ["S2", "S5", "S6", "S7"],
            "fault_window": {"start": "day_5", "end": "day_18"},
            "compliance": {"PM25_ugm3": "CONFIRMED_VIOLATION", "SO2_ppb": "CLEAN"},
        },
    },
    # 2: URBAN-NETWORK-ALPHA (Beijing)
    {
        "dependency_graph": {
            "S1":  {"calibrates": ["S4", "S5"], "role": "reference"},
            "S2":  {"calibrates": ["S6", "S7"], "role": "reference"},
            "S3":  {"calibrates": ["S8", "S9"], "role": "reference"},
            "S4":  {"calibrated_by": "S1", "role": "dependent"},
            "S5":  {"calibrated_by": "S1", "role": "dependent"},
            "S6":  {"calibrated_by": "S2", "role": "dependent"},
            "S7":  {"calibrated_by": "S2", "role": "dependent"},
            "S8":  {"calibrated_by": "S3", "role": "dependent"},
            "S9":  {"calibrated_by": "S3", "role": "dependent"},
            "S10": {"independent": True},
        },
        "faults": {
            "S1": {"type": "offline",   "start": 3,  "end": 15},
            "S2": {"type": "offline",   "start": 7,  "end": 20},
            "S3": {"type": "offline",   "start": 10, "end": 25},
            "S4": {"type": "corrupted", "start": 3,  "end": 15, "corrupt_pct": 1.35},
            "S5": {"type": "corrupted", "start": 3,  "end": 15, "corrupt_pct": 1.35},
            "S6": {"type": "corrupted", "start": 7,  "end": 20, "corrupt_pct": 1.40},
            "S7": {"type": "corrupted", "start": 7,  "end": 20, "corrupt_pct": 1.40},
            "S8": {"type": "corrupted", "start": 10, "end": 25, "corrupt_pct": 1.50},
            "S9": {"type": "corrupted", "start": 10, "end": 25, "corrupt_pct": 1.50},
        },
        "regulatory_thresholds": {
            "CH4_ppb": {"warning": 1800, "violation": 2000},
            "NO2_ppb": {"warning": 80,   "violation": 100},
        },
        "known_facts": [
            "S1 offline days 3-15, S2 offline days 7-20, S3 offline days 10-25",
            "Together S1-S9 cover ALL CH4 measurement in this network",
            "During combined outage window no reliable CH4 data exists",
            "S10 (NOx independent) functioned throughout -- stable readings below threshold",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S2", "S3"],
            "repair_order": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
            "fault_window": {"start": "day_3", "end": "day_25"},
            "compliance": {"CH4_ppb": "INSUFFICIENT_DATA", "NO2_ppb": "CLEAN"},
        },
    },
    # 3: COASTAL-MONITOR-BETA (Chennai)
    {
        "dependency_graph": {
            "S1":  {"calibrates": ["S4", "S5"], "role": "reference"},
            "S2":  {"independent": True},
            "S3":  {"calibrates": ["S6", "S7", "S8"], "role": "reference"},
            "S4":  {"calibrated_by": "S1", "role": "dependent"},
            "S5":  {"calibrated_by": "S1", "role": "dependent"},
            "S6":  {"calibrated_by": "S3", "role": "dependent"},
            "S7":  {"calibrated_by": "S3", "role": "dependent"},
            "S8":  {"calibrated_by": "S3", "role": "dependent"},
            "S9":  {"independent": True},
            "S10": {"independent": True},
        },
        "faults": {
            "S3":  {"type": "offline",   "start": 16, "end": 28},
            "S6":  {"type": "corrupted", "start": 16, "end": 28, "corrupt_pct": 1.55},
            "S7":  {"type": "corrupted", "start": 16, "end": 28, "corrupt_pct": 1.55},
            "S8":  {"type": "corrupted", "start": 16, "end": 28, "corrupt_pct": 1.55},
        },
        "regulatory_thresholds": {
            "O3_ppb":    {"warning": 70, "violation": 85},
            "PM25_ugm3": {"warning": 35, "violation": 65},
        },
        "known_facts": [
            "S3 (O3 reference) went offline day 16 due to sea salt electrode fouling",
            "S6, S7, S8 (O3 dependents) show readings 40-55% above normal during fault window",
            "S1 and PM25 network (S4, S5) operated normally throughout",
            "S9, S10 (PM25 independent) readings averaged within NAAQS limits",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S3"],
            "repair_order": ["S3", "S6", "S7", "S8"],
            "fault_window": {"start": "day_16", "end": "day_28"},
            "compliance": {"O3_ppb": "POSSIBLE_VIOLATION", "PM25_ugm3": "CLEAN"},
        },
    },
    # 4: OFFSHORE-PLATFORM-7 (North Sea)
    {
        "dependency_graph": {
            "S1":  {"calibrates": ["S4", "S5"], "role": "reference"},
            "S2":  {"calibrates": ["S6", "S7"], "role": "reference"},
            "S3":  {"independent": True},
            "S4":  {"calibrated_by": "S1", "role": "dependent"},
            "S5":  {"calibrated_by": "S1", "role": "dependent"},
            "S6":  {"calibrated_by": "S2", "role": "dependent"},
            "S7":  {"calibrated_by": "S2", "role": "dependent"},
            "S8":  {"independent": True},
            "S9":  {"independent": True},
            "S10": {"independent": True},
        },
        "faults": {
            "S1":  {"type": "offline",   "start": 5,  "end": 10},
            "S2":  {"type": "offline",   "start": 8,  "end": 15},
            "S4":  {"type": "corrupted", "start": 5,  "end": 10, "corrupt_pct": 1.32},
            "S5":  {"type": "corrupted", "start": 5,  "end": 10, "corrupt_pct": 1.32},
            "S6":  {"type": "corrupted", "start": 8,  "end": 15, "corrupt_pct": 1.36},
            "S7":  {"type": "corrupted", "start": 8,  "end": 15, "corrupt_pct": 1.36},
        },
        "regulatory_thresholds": {
            "CH4_ppb": {"warning": 1800, "violation": 2000},
            "SO2_ppb": {"warning": 70,   "violation": 100},
        },
        "known_facts": [
            "S1 (CH4 reference) lost power days 5-10 during planned maintenance",
            "S4, S5 (CH4 dependents) show erratic readings exactly matching S1 outage",
            "S2 (NO2 reference) cable severed by wave action on day 8",
            "S3, S8, S10 (SO2 independent) stable -- readings 4-9 ppb throughout",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S2"],
            "repair_order": ["S1", "S2", "S4", "S5", "S6", "S7"],
            "fault_window": {"start": "day_5", "end": "day_15"},
            "compliance": {"CH4_ppb": "POSSIBLE_VIOLATION", "SO2_ppb": "CLEAN"},
        },
    },
]

# Sensor param assignment per scenario (which real param each sensor ID uses)
SENSOR_PARAMS_MAP: List[Dict[str, str]] = [
    {"S1":"CH4","S2":"CO","S3":"NO2","S4":"CH4","S5":"CH4","S6":"NO2","S7":"NO2","S8":"NO2","S9":"NO2","S10":"NO2"},
    {"S1":"SO2","S2":"PM25","S3":"CO","S4":"NO2","S5":"PM25","S6":"PM25","S7":"PM25","S8":"NO2","S9":"NO2","S10":"SO2"},
    {"S1":"CH4","S2":"CH4","S3":"CH4","S4":"CH4","S5":"CH4","S6":"CH4","S7":"CH4","S8":"CH4","S9":"CH4","S10":"NO2"},
    {"S1":"PM25","S2":"CO","S3":"O3","S4":"PM25","S5":"PM25","S6":"O3","S7":"O3","S8":"O3","S9":"PM25","S10":"PM25"},
    {"S1":"CH4","S2":"NO2","S3":"SO2","S4":"CH4","S5":"CH4","S6":"NO2","S7":"NO2","S8":"SO2","S9":"CO","S10":"SO2"},
]

SENSOR_META: Dict[str, Dict] = {
    "PM25": {"param": "PM25_ugm3",  "unit": "ug/m3",  "normal_range": [5,  35]},
    "NO2":  {"param": "NO2_ppb",    "unit": "ppb",    "normal_range": [10, 50]},
    "O3":   {"param": "O3_ppb",     "unit": "ppb",    "normal_range": [20, 60]},
    "SO2":  {"param": "SO2_ppb",    "unit": "ppb",    "normal_range": [2,  15]},
    "CO":   {"param": "CO_ppb",     "unit": "ppb",    "normal_range": [200, 800]},
    "CH4":  {"param": "CH4_ppb",    "unit": "ppb",    "normal_range": [1800, 2000]},
}

# Real API param key -> sensor_key mapping
API_PARAM_TO_SENSOR = {
    "pm2_5": "PM25", "nitrogen_dioxide": "NO2", "ozone": "O3",
    "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "methane": "CH4",
}


def _build_sensor_timeline(sid: str, sensor_key: str,
                            fault_info: Optional[Dict],
                            real_daily: List[Dict]) -> Dict:
    """Build 30-day sensor status from real data with fault applied."""
    meta = SENSOR_META.get(sensor_key, SENSOR_META["NO2"])
    daily_out = []

    for d in real_daily:
        day = d["day"]
        mean = d["mean"]

        if fault_info:
            ftype = fault_info["type"]
            start = fault_info["start"]
            end   = fault_info["end"]

            if ftype == "offline" and start <= day <= end:
                daily_out.append({"day": day, "mean": None, "status": "SENSOR_OFFLINE"})
                continue
            elif ftype == "corrupted" and start <= day <= end:
                pct = fault_info.get("corrupt_pct", 1.38)
                corrupted_val = round(mean * pct, 2) if mean is not None else None
                daily_out.append({"day": day, "mean": corrupted_val, "status": "CORRUPTED"})
                continue

        daily_out.append({"day": day, "mean": mean, "status": "OK"})

    ok_vals = [d["mean"] for d in daily_out if d["status"] == "OK" and d["mean"] is not None]
    offline_days   = sum(1 for d in daily_out if d["status"] == "SENSOR_OFFLINE")
    corrupted_days = sum(1 for d in daily_out if d["status"] == "CORRUPTED")
    mean_clean = round(sum(ok_vals) / len(ok_vals), 2) if ok_vals else None

    return {
        "sensor_id":     sid,
        "parameter":     meta["param"],
        "unit":          meta["unit"],
        "normal_range":  meta["normal_range"],
        "data_source":   "Open-Meteo CAMS/ECMWF (real measured baseline)",
        "daily_readings": daily_out,
        "stats": {
            "mean_clean_days":  mean_clean,
            "offline_days":     offline_days,
            "corrupted_days":   corrupted_days,
            "data_quality_pct": round((30 - offline_days - corrupted_days) / 30 * 100, 1),
        },
    }


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Build a Task 3 cascade scenario from REAL 30-day data."""
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(_REAL_SCENARIOS)) if seed is not None else rng.randint(0, len(_REAL_SCENARIOS) - 1)

    real_sc   = _REAL_SCENARIOS[idx]
    tmpl      = TEMPLATES[idx]
    param_map = SENSOR_PARAMS_MAP[idx]

    # Build a lookup: sensor_key -> real 30-day daily means from API data
    # Each real_sc has 10 sensors in order S1..S10
    real_sensor_lookup = {s["sensor_id"]: s for s in real_sc["sensors"]}

    sensors_obs = []
    for i, sid in enumerate([f"S{n}" for n in range(1, 11)]):
        sensor_key  = param_map.get(sid, "NO2")
        fault_info  = tmpl["faults"].get(sid)

        # Use matching real sensor's daily data
        real_s = real_sc["sensors"][i % len(real_sc["sensors"])]
        real_daily = real_s["daily_means"]

        sensors_obs.append(
            _build_sensor_timeline(sid, sensor_key, fault_info, real_daily)
        )

    observation = {
        "task_id":       "task3_cascade",
        "network_id":    real_sc["network_id"],
        "location":      real_sc["location"],
        "lat":           real_sc["lat"],
        "lon":           real_sc["lon"],
        "period_days":   30,
        "data_source":   "Open-Meteo CAMS/ECMWF (real measured baseline + cascade simulation)",
        "dependency_graph": tmpl["dependency_graph"],
        "sensors":          sensors_obs,
        "regulatory_thresholds": tmpl["regulatory_thresholds"],
        "known_facts":      tmpl["known_facts"],
        "instructions": (
            "Identify root cause sensors (references that ACTUALLY failed), "
            "correct repair order (references before dependents), fault window, "
            "and EPA compliance status. Corrupted sensors are VICTIMS not causes."
        ),
    }
    return observation, tmpl["ground_truth"]


def load_task3(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    return generate_scenario(seed)


# ── Grader ────────────────────────────────────────────────────────────────────

_COMPLIANCE_ADJACENT: Dict[str, Set[str]] = {
    "CLEAN":                {"POSSIBLE_VIOLATION"},
    "POSSIBLE_VIOLATION":   {"CLEAN", "CONFIRMED_VIOLATION", "INSUFFICIENT_DATA"},
    "CONFIRMED_VIOLATION":  {"POSSIBLE_VIOLATION"},
    "INSUFFICIENT_DATA":    {"POSSIBLE_VIOLATION"},
}

def _adjacent_status(s1: str, s2: str) -> bool:
    return s2 in _COMPLIANCE_ADJACENT.get(s1, set())

def _parse_day(day_str: str) -> Optional[int]:
    try:
        return int(str(day_str).lower().replace("day_","").replace("day","").strip())
    except (ValueError, AttributeError):
        return None

def _count_dependency_violations(repair_order: List[str], dep_graph: Dict) -> int:
    violations = 0
    order_pos = {s: i for i, s in enumerate(repair_order)}
    for sensor_id, info in dep_graph.items():
        if "calibrates" in info:
            ref_pos = order_pos.get(sensor_id, -1)
            for dep in info["calibrates"]:
                dep_pos = order_pos.get(dep, -1)
                if ref_pos == -1 and dep_pos != -1:
                    violations += 1
                elif dep_pos != -1 and ref_pos > dep_pos:
                    violations += 1
    return violations

def _get_all_affected(root_causes: List[str], dep_graph: Dict) -> Set[str]:
    affected: Set[str] = set(root_causes)
    for rc in root_causes:
        info = dep_graph.get(rc, {})
        for dep in info.get("calibrates", []):
            affected.add(dep)
    return affected


def grade_task3(action: dict, ground_truth: dict) -> float:
    gt = ground_truth

    # 1. Root cause (35%)
    pred_roots = set(action.get("root_cause_sensors", []))
    true_roots = set(gt.get("root_cause_sensors", []))
    if true_roots:
        inter = pred_roots & true_roots
        union = pred_roots | true_roots
        root_score = len(inter) / len(union) if union else 0.0
    else:
        root_score = 1.0 if not pred_roots else 0.0

    # 2. Repair order (30%)
    repair_order = action.get("repair_order", [])
    template = next(
        (t for t in TEMPLATES
         if set(t["ground_truth"]["root_cause_sensors"]) == true_roots
         and t["ground_truth"]["fault_window"]["start"] == gt.get("fault_window", {}).get("start")),
        None
    )
    if template:
        dep_graph = template["dependency_graph"]
        violations = _count_dependency_violations(repair_order, dep_graph)
        order_base = max(0.0, 1.0 - violations * 0.25)
        affected = _get_all_affected(list(true_roots), dep_graph)
        completeness = len(set(repair_order) & affected) / len(affected) if affected else 1.0
        order_score = 0.65 * order_base + 0.35 * completeness
    else:
        gt_repair = gt.get("repair_order", [])
        correct_set = set(gt_repair) & set(repair_order)
        order_score = len(correct_set) / len(gt_repair) if gt_repair else 0.0

    # 3. Fault window (20%)
    pred_start_n = _parse_day(action.get("fault_window_start", ""))
    pred_end_n   = _parse_day(action.get("fault_window_end", ""))
    true_window  = gt.get("fault_window", {})
    true_start_n = _parse_day(true_window.get("start", ""))
    true_end_n   = _parse_day(true_window.get("end", ""))

    if true_start_n is not None and true_end_n is not None:
        start_score = max(0.0, 1.0 - abs(pred_start_n - true_start_n) * 0.3) if pred_start_n is not None else 0.0
        end_score   = max(0.0, 1.0 - abs(pred_end_n   - true_end_n)   * 0.2) if pred_end_n   is not None else 0.0
        window_score = (start_score + end_score) / 2.0
    else:
        window_score = 0.0

    # 4. Compliance (15%)
    checks = action.get("compliance_checks", [])
    gt_compliance = gt.get("compliance", {})
    pred_compliance = {c["parameter"]: c["status"] for c in checks}

    if gt_compliance:
        comp_total = 0.0
        for param, true_status in gt_compliance.items():
            pred_status = pred_compliance.get(param, "INSUFFICIENT_DATA")
            if pred_status == true_status:
                comp_total += 1.0
            elif _adjacent_status(pred_status, true_status):
                comp_total += 0.3
        compliance_score = comp_total / len(gt_compliance)
    else:
        compliance_score = 1.0

    final = (0.35 * root_score + 0.30 * order_score +
             0.20 * window_score + 0.15 * compliance_score)
    return round(max(0.01, min(0.99, final)), 4)
