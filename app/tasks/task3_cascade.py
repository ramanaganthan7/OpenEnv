"""
Task 3 — Cascade Failure Diagnosis & Compliance Audit (HARD)
=============================================================
The hardest task. Designed to challenge frontier LLMs (GPT-4, Nemotron).

Scenario:
  A 10-sensor network runs for 30 days.
  Some reference sensors FAIL, causing downstream sensors (calibrated from them)
  to report CORRUPTED data.
  The agent must:
    1. Identify ROOT CAUSE sensors (references that actually failed)
    2. Determine correct REPAIR ORDER (must fix reference before dependents)
    3. Identify FAULT WINDOW (start/end day)
    4. Check REGULATORY COMPLIANCE — did emissions violate EPA thresholds?

Why it's genuinely hard:
  - Must distinguish root causes from downstream victims (the corrupted sensors
    look broken too, but are NOT the root cause)
  - Repair order requires understanding a dependency graph (topological reasoning)
  - Compliance check requires handling UNCERTAIN data: if the sensor network
    measuring CH4 was corrupted, you CANNOT confirm or deny compliance
  - Fault window requires reading patterns across 30 days of data

Action schema:
  {
    "root_cause_sensors": ["S1", "S3"],
    "repair_order": ["S1", "S3", "S4", "S5", "S6", "S7", "S8"],
    "fault_window_start": "day_8",
    "fault_window_end": "day_21",
    "compliance_checks": [
      {"parameter": "CH4_ppm", "status": "POSSIBLE_VIOLATION",
       "confidence": 0.75, "reasoning": "..."},
      {"parameter": "NOx_ppb", "status": "CLEAN",
       "confidence": 0.9, "reasoning": "..."}
    ],
    "recommended_action": "flag_for_review"
  }

Grader (4 components):
  35% — Root cause identification (Jaccard similarity)
  30% — Repair order (penalty per dependency violation)
  20% — Fault window accuracy (within ±2 days = partial credit)
  15% — Compliance check accuracy
"""

import random
import math
from typing import Optional, Tuple, Dict, Any, List, Set

SENSOR_PARAMS: Dict[str, Dict] = {
    "CH4":  {"param": "CH4_ppm",    "unit": "ppm",   "normal_range": [1.7, 2.0],  "center": 1.87, "std": 0.02, "dec": 3},
    "NO2":  {"param": "NO2_ppb",    "unit": "ppb",   "normal_range": [10, 50],    "center": 28.0, "std": 2.5,  "dec": 1},
    "NOx":  {"param": "NOx_ppb",    "unit": "ppb",   "normal_range": [15, 65],    "center": 35.0, "std": 3.0,  "dec": 1},
    "CO2":  {"param": "CO2_ppm",    "unit": "ppm",   "normal_range": [400, 450],  "center": 420.0,"std": 3.0,  "dec": 1},
    "O3":   {"param": "O3_ppb",     "unit": "ppb",   "normal_range": [20, 60],    "center": 38.0, "std": 3.0,  "dec": 1},
    "PM25": {"param": "PM25_ugm3",  "unit": "µg/m³", "normal_range": [5, 35],     "center": 18.0, "std": 2.0,  "dec": 1},
    "SO2":  {"param": "SO2_ppb",    "unit": "ppb",   "normal_range": [2, 15],     "center": 8.0,  "std": 1.0,  "dec": 1},
}

# EPA regulatory thresholds (NAAQS standards, simplified)
EPA_THRESHOLDS: Dict[str, Dict] = {
    "CH4_ppm":  {"warning": 1.8, "violation": 2.0},
    "NOx_ppb":  {"warning": 80,  "violation": 100},
    "PM25_ugm3":{"warning": 35,  "violation": 65},
    "SO2_ppb":  {"warning": 70,  "violation": 100},
    "O3_ppb":   {"warning": 70,  "violation": 85},
}

# ── 5 scenario templates ──────────────────────────────────────────────────────
# dependency_graph: { sensor_id: {"calibrates": [...]} or {"independent": true} }
# sensors: each sensor has role (reference/dependent/independent), param, and fault info
# compliance: what EPA check results are ground truth

TEMPLATES: List[Dict] = [
    # ── Scenario 0: REFINERY-NORTH ──
    # Classic dual cascade: S1→[S4,S5], S3→[S6,S7,S8]
    # S1 fails d8-21, S3 fails d12-21 → CH4 uncertain, NOx clean
    {
        "network_id": "REFINERY-NORTH",
        "location": "Crude Oil Refinery, North Sector, Kuwait",
        "period_days": 30,
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
        "sensor_params": {
            "S1": "CH4",  "S2": "CO2",  "S3": "NO2",
            "S4": "CH4",  "S5": "CH4",  "S6": "NO2",
            "S7": "NOx",  "S8": "NO2",  "S9": "NOx", "S10": "NOx",
        },
        "faults": {
            "S1":  {"type": "offline",    "start": 8,  "end": 21},
            "S3":  {"type": "offline",    "start": 12, "end": 21},
            "S4":  {"type": "corrupted",  "start": 8,  "end": 21},
            "S5":  {"type": "corrupted",  "start": 8,  "end": 21},
            "S6":  {"type": "corrupted",  "start": 12, "end": 21},
            "S7":  {"type": "corrupted",  "start": 12, "end": 21},
            "S8":  {"type": "corrupted",  "start": 12, "end": 21},
        },
        "regulatory_thresholds": {
            "CH4_ppm": {"warning": 1.8,  "violation": 2.0},
            "NOx_ppb": {"warning": 80,   "violation": 100},
        },
        "known_facts": [
            "S1 (CH4 reference) was offline days 8-21 — calibration signal lost",
            "S4 and S5 (CH4 dependents) show systematic +35% error matching S1 outage period",
            "S3 (NO2 reference) went offline day 12 due to power failure",
            "S9 and S10 (NOx independent) readings were stable throughout — no anomalies",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S3"],
            "repair_order": ["S1", "S3", "S4", "S5", "S6", "S7", "S8"],
            "fault_window": {"start": "day_8", "end": "day_21"},
            "compliance": {
                "CH4_ppm":  "POSSIBLE_VIOLATION",
                "NOx_ppb":  "CLEAN",
            },
        },
    },

    # ── Scenario 1: PIPELINE-CENTRAL ──
    # Single cascade: S2→[S5,S6,S7], S4→[S8,S9] (S4 healthy)
    # S2 fails d5-18 → PM25 confirmed violation (pre-failure data shows it)
    {
        "network_id": "PIPELINE-CENTRAL",
        "location": "Gas Pipeline Monitoring Hub, Permian Basin, Texas",
        "period_days": 30,
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
        "sensor_params": {
            "S1": "SO2",  "S2": "PM25", "S3": "CO2",
            "S4": "NOx",  "S5": "PM25", "S6": "PM25",
            "S7": "PM25", "S8": "NOx",  "S9": "NOx", "S10": "SO2",
        },
        "faults": {
            "S2":  {"type": "offline",   "start": 5,  "end": 18},
            "S5":  {"type": "corrupted", "start": 5,  "end": 18},
            "S6":  {"type": "corrupted", "start": 5,  "end": 18},
            "S7":  {"type": "corrupted", "start": 5,  "end": 18},
        },
        "regulatory_thresholds": {
            "PM25_ugm3": {"warning": 35, "violation": 65},
            "SO2_ppb":   {"warning": 70, "violation": 100},
        },
        "known_facts": [
            "S2 (PM25 reference) went offline days 5-18 after membrane fouling",
            "Pre-failure S2 readings (days 1-4) showed PM25 averaging 68 µg/m³ — above violation threshold",
            "S5, S6, S7 (PM25 dependents) show systematic corruption matching S2 outage",
            "S1 and S10 (SO2 independent) remained functional throughout — readings within NAAQS limits",
            "S4 and its network (S8, S9) were unaffected — S4 is a separate NOx reference",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S2"],
            "repair_order": ["S2", "S5", "S6", "S7"],
            "fault_window": {"start": "day_5", "end": "day_18"},
            "compliance": {
                "PM25_ugm3": "CONFIRMED_VIOLATION",
                "SO2_ppb":   "CLEAN",
            },
        },
    },

    # ── Scenario 2: URBAN-NETWORK-ALPHA ──
    # Triple cascade: S1→[S4,S5], S2→[S6,S7], S3→[S8,S9], S10 independent
    # All three references fail at different times → CH4 insufficient data, NOx clean
    {
        "network_id": "URBAN-NETWORK-ALPHA",
        "location": "Metropolitan Air Quality Network, Beijing, China",
        "period_days": 30,
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
        "sensor_params": {
            "S1": "CH4",  "S2": "CH4",  "S3": "CH4",
            "S4": "CH4",  "S5": "CH4",  "S6": "CH4",
            "S7": "CH4",  "S8": "CH4",  "S9": "CH4", "S10": "NOx",
        },
        "faults": {
            "S1":  {"type": "offline",   "start": 3,  "end": 15},
            "S2":  {"type": "offline",   "start": 7,  "end": 20},
            "S3":  {"type": "offline",   "start": 10, "end": 25},
            "S4":  {"type": "corrupted", "start": 3,  "end": 15},
            "S5":  {"type": "corrupted", "start": 3,  "end": 15},
            "S6":  {"type": "corrupted", "start": 7,  "end": 20},
            "S7":  {"type": "corrupted", "start": 7,  "end": 20},
            "S8":  {"type": "corrupted", "start": 10, "end": 25},
            "S9":  {"type": "corrupted", "start": 10, "end": 25},
        },
        "regulatory_thresholds": {
            "CH4_ppm": {"warning": 1.8, "violation": 2.0},
            "NOx_ppb": {"warning": 80,  "violation": 100},
        },
        "known_facts": [
            "S1 offline days 3-15, S2 offline days 7-20, S3 offline days 10-25",
            "Together, S1-S9 cover ALL CH4 measurement in this network",
            "During the combined outage window, NO reliable CH4 data exists",
            "S10 (NOx independent) functioned throughout — stable readings well below threshold",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S2", "S3"],
            "repair_order": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
            "fault_window": {"start": "day_3", "end": "day_25"},
            "compliance": {
                "CH4_ppm": "INSUFFICIENT_DATA",
                "NOx_ppb": "CLEAN",
            },
        },
    },

    # ── Scenario 3: COASTAL-MONITOR-BETA ──
    # Single late cascade: S3→[S6,S7,S8]. S1 and its dependents are healthy.
    # S3 fails d16-28. O3 possible violation, PM25 clean.
    {
        "network_id": "COASTAL-MONITOR-BETA",
        "location": "Coastal Industrial Zone, Chennai, India",
        "period_days": 30,
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
        "sensor_params": {
            "S1": "PM25", "S2": "CO2",  "S3": "O3",
            "S4": "PM25", "S5": "PM25", "S6": "O3",
            "S7": "O3",   "S8": "O3",   "S9": "PM25", "S10": "PM25",
        },
        "faults": {
            "S3":  {"type": "offline",   "start": 16, "end": 28},
            "S6":  {"type": "corrupted", "start": 16, "end": 28},
            "S7":  {"type": "corrupted", "start": 16, "end": 28},
            "S8":  {"type": "corrupted", "start": 16, "end": 28},
        },
        "regulatory_thresholds": {
            "O3_ppb":    {"warning": 70, "violation": 85},
            "PM25_ugm3": {"warning": 35, "violation": 65},
        },
        "known_facts": [
            "S3 (O3 reference) went offline day 16 — electrode fouling from sea salt",
            "S6, S7, S8 (O3 dependents) show readings 40-60% above normal during fault window",
            "S1 and its PM25 network (S4, S5) operated normally throughout the period",
            "S9, S10 (independent PM25) readings averaged 18-22 µg/m³ — well within NAAQS",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S3"],
            "repair_order": ["S3", "S6", "S7", "S8"],
            "fault_window": {"start": "day_16", "end": "day_28"},
            "compliance": {
                "O3_ppb":    "POSSIBLE_VIOLATION",
                "PM25_ugm3": "CLEAN",
            },
        },
    },

    # ── Scenario 4: OFFSHORE-PLATFORM-7 ──
    # Two cascades, overlapping windows: S1→[S4,S5] (d5-10), S2→[S6,S7] (d8-15)
    # CH4 possible violation (S1 window), SO2 clean (S3 independent)
    {
        "network_id": "OFFSHORE-PLATFORM-7",
        "location": "Fixed Production Platform, North Sea, UK/Norway",
        "period_days": 30,
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
        "sensor_params": {
            "S1": "CH4",  "S2": "NO2",  "S3": "SO2",
            "S4": "CH4",  "S5": "CH4",  "S6": "NO2",
            "S7": "NO2",  "S8": "SO2",  "S9": "CO2", "S10": "SO2",
        },
        "faults": {
            "S1":  {"type": "offline",   "start": 5,  "end": 10},
            "S2":  {"type": "offline",   "start": 8,  "end": 15},
            "S4":  {"type": "corrupted", "start": 5,  "end": 10},
            "S5":  {"type": "corrupted", "start": 5,  "end": 10},
            "S6":  {"type": "corrupted", "start": 8,  "end": 15},
            "S7":  {"type": "corrupted", "start": 8,  "end": 15},
        },
        "regulatory_thresholds": {
            "CH4_ppm": {"warning": 1.8, "violation": 2.0},
            "SO2_ppb": {"warning": 70,  "violation": 100},
        },
        "known_facts": [
            "S1 (CH4 reference) lost power days 5-10 during maintenance window",
            "S4, S5 (CH4 dependents) show erratic readings matching S1 outage exactly",
            "S2 (NO2 reference) began failing day 8 when wave damage severed its cable",
            "S3, S8, S10 (SO2 independent) remained stable — readings 4-9 ppb throughout",
        ],
        "ground_truth": {
            "root_cause_sensors": ["S1", "S2"],
            "repair_order": ["S1", "S2", "S4", "S5", "S6", "S7"],
            "fault_window": {"start": "day_5", "end": "day_15"},
            "compliance": {
                "CH4_ppm": "POSSIBLE_VIOLATION",
                "SO2_ppb": "CLEAN",
            },
        },
    },
]


def _r(v: float, dec: int) -> float:
    return round(v, dec)


def _build_sensor_timeline(sensor_id: str, param_key: str,
                            fault_info: Optional[Dict],
                            rng: random.Random) -> Dict[str, Any]:
    """Build 30-day daily summary for one sensor."""
    spec = SENSOR_PARAMS[param_key]
    dec  = spec["dec"]
    ctr  = spec["center"]
    std  = spec["std"]

    daily = []
    for day in range(1, 31):
        if fault_info and fault_info["type"] == "offline":
            if fault_info["start"] <= day <= fault_info["end"]:
                daily.append({"day": day, "mean": None, "status": "SENSOR_OFFLINE"})
                continue
        if fault_info and fault_info["type"] == "corrupted":
            if fault_info["start"] <= day <= fault_info["end"]:
                corrupt_val = _r(ctr * 1.38 + rng.gauss(0, std * 4), dec)
                daily.append({"day": day, "mean": corrupt_val, "status": "CORRUPTED"})
                continue
        # Normal reading
        val = _r(ctr + rng.gauss(0, std * 0.5), dec)
        daily.append({"day": day, "mean": val, "status": "OK"})

    ok_vals = [d["mean"] for d in daily if d["status"] == "OK"]
    offline_days = sum(1 for d in daily if d["status"] == "SENSOR_OFFLINE")
    corrupted_days = sum(1 for d in daily if d["status"] == "CORRUPTED")
    mean_clean = _r(sum(ok_vals) / len(ok_vals), dec) if ok_vals else None

    return {
        "sensor_id": sensor_id,
        "parameter": spec["param"],
        "unit": spec["unit"],
        "normal_range": spec["normal_range"],
        "daily_readings": daily,
        "stats": {
            "mean_clean_days": mean_clean,
            "offline_days": offline_days,
            "corrupted_days": corrupted_days,
            "data_quality_pct": _r((30 - offline_days - corrupted_days) / 30 * 100, 1),
        },
    }


def generate_scenario(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Generate a deterministic Task 3 scenario from seed."""
    rng = random.Random(seed if seed is not None else random.randint(0, 99999))
    idx = (seed % len(TEMPLATES)) if seed is not None else rng.randint(0, len(TEMPLATES) - 1)
    tmpl = TEMPLATES[idx]

    sensors_obs = []
    for sid in [f"S{i}" for i in range(1, 11)]:
        param_key = tmpl["sensor_params"][sid]
        fault_info = tmpl["faults"].get(sid)
        sensors_obs.append(_build_sensor_timeline(sid, param_key, fault_info, rng))

    observation = {
        "task_id": "task3_cascade",
        "network_id": tmpl["network_id"],
        "location": tmpl["location"],
        "period_days": tmpl["period_days"],
        "dependency_graph": tmpl["dependency_graph"],
        "sensors": sensors_obs,
        "regulatory_thresholds": tmpl["regulatory_thresholds"],
        "known_facts": tmpl["known_facts"],
        "instructions": (
            "Identify root cause sensors (references that ACTUALLY failed), "
            "the correct repair order (references before dependents), "
            "the fault window (start/end day), and regulatory compliance status. "
            "Corrupted sensors are VICTIMS of cascade — they are NOT root causes."
        ),
    }
    ground_truth = tmpl["ground_truth"]
    return observation, ground_truth


def load_task3(seed: Optional[int] = None) -> Tuple[Dict, Dict]:
    return generate_scenario(seed)


# ── Grader ────────────────────────────────────────────────────────────────────

# Compliance status adjacency for partial credit
_COMPLIANCE_ADJACENT: Dict[str, Set[str]] = {
    "CLEAN":                {"POSSIBLE_VIOLATION"},
    "POSSIBLE_VIOLATION":   {"CLEAN", "CONFIRMED_VIOLATION", "INSUFFICIENT_DATA"},
    "CONFIRMED_VIOLATION":  {"POSSIBLE_VIOLATION"},
    "INSUFFICIENT_DATA":    {"POSSIBLE_VIOLATION"},
}


def _adjacent_status(s1: str, s2: str) -> bool:
    return s2 in _COMPLIANCE_ADJACENT.get(s1, set())


def _parse_day(day_str: str) -> Optional[int]:
    """Parse 'day_N' → N. Returns None if unparseable."""
    try:
        return int(str(day_str).lower().replace("day_", "").replace("day", "").strip())
    except (ValueError, AttributeError):
        return None


def _count_dependency_violations(repair_order: List[str],
                                  dep_graph: Dict[str, Any]) -> int:
    """
    Count times a reference sensor appears AFTER one of its dependents
    in the repair order.
    """
    violations = 0
    order_pos = {s: i for i, s in enumerate(repair_order)}

    for sensor_id, info in dep_graph.items():
        if "calibrates" in info:
            ref_pos = order_pos.get(sensor_id, -1)
            for dep in info["calibrates"]:
                dep_pos = order_pos.get(dep, -1)
                if ref_pos == -1 and dep_pos != -1:
                    violations += 1  # reference missing but dependent present
                elif dep_pos != -1 and ref_pos > dep_pos:
                    violations += 1  # dependent repaired before reference
    return violations


def _get_all_affected(root_causes: List[str], dep_graph: Dict[str, Any]) -> Set[str]:
    """Get all sensors that need repair (root causes + their dependents)."""
    affected: Set[str] = set(root_causes)
    for rc in root_causes:
        info = dep_graph.get(rc, {})
        for dep in info.get("calibrates", []):
            affected.add(dep)
    return affected


def grade_task3(action: dict, ground_truth: dict) -> float:
    """
    4-component grader for cascade failure diagnosis.

    35% root_cause (Jaccard)
    30% repair_order (dependency violations + completeness)
    20% fault_window (temporal proximity)
    15% compliance (exact=1.0, adjacent=0.3)

    Returns float in [0.0, 1.0].
    """
    gt = ground_truth

    # ── 1. Root cause score (35%) ─────────────────────────────────
    pred_roots = set(action.get("root_cause_sensors", []))
    true_roots = set(gt.get("root_cause_sensors", []))

    if true_roots:
        inter = pred_roots & true_roots
        union = pred_roots | true_roots
        jaccard = len(inter) / len(union) if union else 0.0
    else:
        jaccard = 1.0 if not pred_roots else 0.0
    root_score = jaccard

    # ── 2. Repair order score (30%) ───────────────────────────────
    repair_order = action.get("repair_order", [])
    dep_graph = {}
    # Try to get dep graph from action context or fall back to GT info
    # We need to reconstruct the dep graph from ground truth
    # (the agent got it in the observation)
    # We'll find it via a stored reference
    gt_repair = gt.get("repair_order", [])

    # Score based on: no dependency violations + completeness
    # We need the dependency graph — stored in the scenario template
    # Find the template that matches this ground truth
    template = next(
        (t for t in TEMPLATES if t["ground_truth"]["root_cause_sensors"] == gt.get("root_cause_sensors")
         and t["ground_truth"]["fault_window"]["start"] == gt.get("fault_window", {}).get("start")),
        None
    )
    if template:
        dep_graph_full = template["dependency_graph"]
        violations = _count_dependency_violations(repair_order, dep_graph_full)
        order_score_base = max(0.0, 1.0 - violations * 0.25)

        # Completeness: does repair_order include all affected sensors?
        affected = _get_all_affected(list(true_roots), dep_graph_full)
        included = set(repair_order) & affected
        completeness = len(included) / len(affected) if affected else 1.0
        order_score = 0.65 * order_score_base + 0.35 * completeness
    else:
        # Fallback: compare directly to GT repair order
        correct_set = set(gt_repair) & set(repair_order)
        order_score = len(correct_set) / len(gt_repair) if gt_repair else 0.0

    # ── 3. Fault window score (20%) ───────────────────────────────
    pred_start_n = _parse_day(action.get("fault_window_start", ""))
    pred_end_n   = _parse_day(action.get("fault_window_end", ""))
    true_window  = gt.get("fault_window", {})
    true_start_n = _parse_day(true_window.get("start", ""))
    true_end_n   = _parse_day(true_window.get("end", ""))

    if true_start_n is not None and true_end_n is not None:
        if pred_start_n is not None:
            start_err = abs(pred_start_n - true_start_n)
            start_score = max(0.0, 1.0 - start_err * 0.3)
        else:
            start_score = 0.0
        if pred_end_n is not None:
            end_err = abs(pred_end_n - true_end_n)
            end_score = max(0.0, 1.0 - end_err * 0.2)
        else:
            end_score = 0.0
        window_score = (start_score + end_score) / 2.0
    else:
        window_score = 0.0

    # ── 4. Compliance score (15%) ─────────────────────────────────
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

    # ── Final weighted sum ────────────────────────────────────────
    final = (0.35 * root_score +
             0.30 * order_score +
             0.20 * window_score +
             0.15 * compliance_score)

    return round(max(0.0, min(1.0, final)), 4)
