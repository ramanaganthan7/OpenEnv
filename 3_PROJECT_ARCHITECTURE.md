# ClimateWatch — Complete Project Architecture
> What it is, what it does, who benefits, how every piece fits together

---

# PART 1 — WHAT IS CLIMATEWATCH?

## One Sentence
> ClimateWatch is an AI training environment where an agent learns to detect faults in climate sensor data, clean corrupted readings, and check environmental regulatory compliance — exactly what data engineers at BP, EPA, and NOAA do every day.

## The Problem It Solves

```
Real-world situation:
  Oil company (BP) has 10,000 sensors across oil fields measuring:
  → Methane (CH4) — leak detection
  → Nitrogen Oxide (NOx) — air quality
  → CO2 — emissions monitoring
  → Temperature, pressure, humidity

  These sensors constantly:
  → Break and repeat the same value (STUCK)
  → Drift and give readings 15% too high (DRIFT/BIAS)
  → Drop out entirely (MISSING)
  → Spike randomly (OUTLIER/SPIKE)

  When data is wrong:
  → Dangerous methane leak goes undetected → explosion risk
  → Wrong emissions report filed → $93,750/day EPA fine
  → Climate models trained on bad data → wrong policy decisions

  Currently: a human data engineer reviews this manually. Expensive. Slow.
  With ClimateWatch: an AI agent learns to do this automatically.
```

---

## How ClimateWatch Works — The Full Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIMATEWATCH ENVIRONMENT                       │
│                   (running on HF Spaces)                         │
│                                                                   │
│  1. Judge calls POST /reset                                       │
│     → Environment loads a sensor scenario (24hr data)            │
│     → Returns sensor readings to agent                           │
│                                                                   │
│  2. Agent (LLM) reads the sensor data                            │
│     → "Hour 6 is 9999.0 — that's an outlier"                    │
│     → "Hours 3-5 all same value — that's stuck sensor"          │
│     → Agent sends JSON action with its analysis                  │
│                                                                   │
│  3. Environment receives action                                   │
│     → Compares to ground truth                                   │
│     → Computes reward (partial credit per correct detection)     │
│     → Returns reward + feedback to agent                         │
│                                                                   │
│  4. Repeat until episode ends (5 steps max)                      │
│                                                                   │
│  5. Grader scores final performance 0.0–1.0                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## The 8 Fault Types Your Environment Simulates

```
OUTLIER   → Single reading spikes far outside normal range
            Example: CO2 reading of 9999 ppm when normal is ~415
            Real cause: sensor electrical interference

STUCK     → Sensor freezes — same value repeats for hours
            Example: temperature reads 22.3 all day
            Real cause: sensor memory failure or frozen firmware

MISSING   → Gap in data — null/None values
            Example: no readings for hours 8-12
            Real cause: network dropout, battery failure

DRIFT     → Gradual baseline shift over days
            Example: NO2 readings creep up 0.5 ppb per day
            Real cause: sensor electrode degradation over months

SPIKE     → Short burst of bad readings then back to normal
            Example: 3 hours of wrong values then normal resumes
            Real cause: electromagnetic interference event

BIAS      → Systematic offset — reads consistently high/low
            Example: O3 sensor reads exactly 12 ppb too high always
            Real cause: factory calibration error or contamination

NOISE     → Random fluctuation obscuring real signal
            Example: ±5 ppm random variation on clean signal
            Real cause: low-cost sensor, vibration, humidity

CASCADE   → One sensor failure causes downstream sensors to misread
            Example: reference sensor S1 fails → S4, S5 calibrated
                     from S1 now read wrong too
            Real cause: shared calibration dependency
```

---

# PART 2 — WHO BENEFITS

## Direct Users

| Organization | How They Use ClimateWatch | Impact |
|---|---|---|
| **BP, Shell, ExxonMobil** | Train AI to monitor methane sensors on 10,000+ oil field sites | Catch compliance violations before regulators do |
| **US EPA** | Validate AI agents for national air quality monitoring networks | Ensure policy is based on clean data |
| **NOAA** | Test agents on ocean buoy / weather station data quality | Better weather forecasting accuracy |
| **Climate Researchers** | Benchmark LLMs on real-world data quality tasks | Avoid training models on corrupted datasets |
| **Smart City Operators** | Automate quality control on IoT sensor networks | Scale monitoring without adding staff |
| **Renewable Energy Cos** | Quality-check solar irradiance / wind speed sensors | Prevent wrong energy dispatch decisions |

## Indirect Beneficiaries

```
General public:
  → Better air quality monitoring → more accurate pollution alerts
  → Caught methane leaks → reduced greenhouse gas emissions
  → Valid emissions reports → effective climate regulation

The RL Research Community:
  → New real-world benchmark for LLM agents
  → First environmental data quality environment in OpenEnv
  → Researchers can fork and extend it for other sensor domains
```

## The Numbers That Matter

```
$14.4 billion   — environmental monitoring market size (2024)
$41.4 billion   — projected market size by 2029
$93,750/day     — EPA clean air act penalty per violation
$56.6 billion   — total EPA oil & gas penalties since 2000
116%            — worst-case sensor calibration error documented
80%             — industrial downtime caused by equipment/sensor failures
$50 billion/yr  — annual cost of unplanned downtime globally
```

---

# PART 3 — COMPLETE FILE STRUCTURE

```
climatewatch-env/                   ← root
│
├── inference.py                    ← MANDATORY: baseline inference script
├── openenv.yaml                    ← MANDATORY: environment metadata
├── Dockerfile                      ← MANDATORY: container definition
├── requirements.txt                ← Python dependencies
├── pyproject.toml                  ← uv + taskipy dev config
├── README.md                       ← Full documentation + HF Spaces header
├── DEPLOYMENT_GUIDE.md             ← Step-by-step HF deployment guide
├── TESTING.md                      ← All curl commands for live verification
├── architecture.svg                ← System architecture diagram (embedded in README)
│
├── scripts/
│   ├── fetch_real_data.py          ← Fetches real data from Open-Meteo CAMS/ECMWF API
│   ├── kill_port.py                ← Kills port 7860 before server start
│   └── check_live.py              ← Tests all endpoints against live HF Space
│
├── app/
│   ├── __init__.py
│   ├── main.py                     ← FastAPI app, all HTTP endpoints
│   ├── environment.py              ← Core: reset(), step(), state()
│   ├── models.py                   ← All Pydantic models
│   ├── reward.py                   ← Reward function
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── task1_detect.py         ← Task 1 scenario loader + grader
│   │   ├── task2_clean.py          ← Task 2 scenario loader + grader
│   │   └── task3_cascade.py        ← Task 3 scenario loader + grader
│   │
│   └── data/
│       ├── real_task1.json         ← 20 REAL scenarios (Open-Meteo CAMS/ECMWF)
│       │                              24hr PM2.5/NO2/O3/SO2/CO/CH4 readings
│       │                              20 global industrial monitoring locations
│       ├── real_task2.json         ← 10 REAL network scenarios
│       │                              7-day multi-sensor daily means
│       │                              10 industrial network locations
│       └── real_task3.json         ← 5 REAL cascade network scenarios
│                                      30-day sensor status timelines
│                                      5 industrial complexes (Kuwait, TX, Beijing...)
│
└── tests/
    ├── __init__.py
    ├── test_graders.py             ← 22 tests: graders return different scores
    └── test_endpoints.py           ← 41 tests: all endpoints work correctly
```

## REAL DATA ARCHITECTURE

```
Open-Meteo Air Quality API (free, no key, CAMS/ECMWF backed)
  https://air-quality-api.open-meteo.com/v1/air-quality
  ?latitude=29.76&longitude=-95.37   <- Houston TX (BP refinery corridor)
  &hourly=pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,methane
  &past_days=30

Returns actual measured values:
  NO2: [14.2, 16.8, 22.4, 18.7, 12.3, ...]  <- real ppb readings
  PM2.5: [11.3, 10.6, 9.5, 9.1, 9.0, ...]   <- real ug/m3 readings

Pre-fetched by: scripts/fetch_real_data.py (run once, committed to repo)
Stored in: app/data/real_task1.json, real_task2.json, real_task3.json

At episode time:
  1. Load real baseline from JSON
  2. Inject fault at specific hours/days (deterministic by seed)
  3. Ground truth = only the injected hours/sensors
  4. Grader compares agent flags against known injections
```

---

# PART 4 — THE 3 TASKS IN FULL DETAIL

## Task 1 — EASY: Single Sensor Anomaly Detection

**What it simulates:**
A data engineer receives 24 hours of readings from one sensor.
Some readings are faulty. They must identify which ones and why.

**Input (what agent sees):**
```json
{
  "task_id": "task1_detect",
  "step_count": 0,
  "done": false,
  "reward": 0.0,
  "sensor_data": {
    "sensor_id": "AQ-007",
    "location": "Industrial Zone A, Chennai",
    "parameter": "CO2_ppm",
    "unit": "ppm",
    "normal_range": [400, 450],
    "readings": [
      {"hour": 0, "value": 412.3},
      {"hour": 1, "value": 413.1},
      {"hour": 2, "value": 412.8},
      {"hour": 3, "value": 412.8},
      {"hour": 4, "value": 412.8},
      {"hour": 5, "value": 412.8},
      {"hour": 6, "value": 9999.0},
      {"hour": 7, "value": 413.5},
      {"hour": 8, "value": null},
      {"hour": 9, "value": null},
      {"hour": 10, "value": 414.2}
    ]
  },
  "feedback": "Analyse the 24-hour sensor data and identify all faults."
}
```

**Output (what agent must send):**
```json
{
  "sensor_id": "AQ-007",
  "flags": [
    {"hour": 3,  "fault": "stuck",   "confidence": 0.9},
    {"hour": 4,  "fault": "stuck",   "confidence": 0.9},
    {"hour": 5,  "fault": "stuck",   "confidence": 0.9},
    {"hour": 6,  "fault": "outlier", "confidence": 1.0},
    {"hour": 8,  "fault": "missing", "confidence": 1.0},
    {"hour": 9,  "fault": "missing", "confidence": 1.0}
  ]
}
```

**Grader logic:**
```python
def grade_task1(action, ground_truth):
    predicted = {(f.hour, f.fault) for f in action.flags}
    actual    = {(f.hour, f.fault) for f in ground_truth.flags}

    correct = predicted & actual
    fp = predicted - actual   # false positives
    fn = actual - predicted   # false negatives

    precision = len(correct) / len(predicted) if predicted else 0
    recall    = len(correct) / len(actual) if actual else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Bonus for calibrated confidence (not all 1.0)
    confidences = [f.confidence for f in action.flags]
    calibration_bonus = 0.05 if len(set(confidences)) > 1 else 0

    return min(1.0, f1 + calibration_bonus)

# Score ranges:
#   Detects all correctly   →  ~0.9–1.0
#   Detects most correctly  →  ~0.6–0.8
#   Detects some correctly  →  ~0.3–0.5
#   Gets it all wrong       →  ~0.0–0.1
```

---

## Task 2 — MEDIUM: Multi-Sensor Data Stream Cleaning

**What it simulates:**
A data engineer receives 7 days of data from 5 different sensors.
Each sensor has a different fault. They must diagnose each sensor
and recommend the correct fix.

**Input (what agent sees):**
```json
{
  "task_id": "task2_clean",
  "network_id": "COASTAL-ZONE-3",
  "period_days": 7,
  "sensors": [
    {
      "sensor_id": "S1", "parameter": "NO2_ppb",
      "readings": ["...168 readings showing gradual upward drift..."],
      "stats": {"mean": 42.3, "std": 8.1, "trend": "+0.8ppb/day"}
    },
    {
      "sensor_id": "S2", "parameter": "PM2.5_ugm3",
      "readings": ["...168 readings with 20% null values..."],
      "missing_count": 34
    },
    {
      "sensor_id": "S3", "parameter": "O3_ppb",
      "readings": ["...168 readings systematically 12ppb too high..."],
      "stats": {"vs_reference": "+12.0ppb constant offset"}
    },
    {
      "sensor_id": "S4", "parameter": "CH4_ppm",
      "readings": ["...168 readings with noisy variance..."],
      "stats": {"std": 15.2, "expected_std": 2.1}
    },
    {
      "sensor_id": "S5", "parameter": "CO_ppm",
      "readings": ["...clean 168 readings..."]
    }
  ],
  "reference_station": {
    "id": "REF-42", "distance_km": 2.1,
    "readings": ["...clean reference data..."]
  }
}
```

**Output (what agent must send):**
```json
{
  "diagnoses": [
    {
      "sensor_id": "S1",
      "fault_type": "drift",
      "severity": "high",
      "fix": "recalibrate",
      "fix_params": {"drift_rate_per_day": 0.8, "direction": "positive"}
    },
    {
      "sensor_id": "S2",
      "fault_type": "missing",
      "severity": "medium",
      "fix": "interpolate",
      "fix_params": {"method": "linear", "max_gap_hours": 4}
    },
    {
      "sensor_id": "S3",
      "fault_type": "bias",
      "severity": "high",
      "fix": "offset_correction",
      "fix_params": {"offset": -12.0}
    },
    {
      "sensor_id": "S4",
      "fault_type": "noise",
      "severity": "medium",
      "fix": "smooth",
      "fix_params": {"window": 3, "method": "rolling_mean"}
    },
    {
      "sensor_id": "S5",
      "fault_type": "valid",
      "severity": "none",
      "fix": "no_action",
      "fix_params": {}
    }
  ]
}
```

**Grader logic:**
```python
def grade_task2(action, ground_truth):
    fault_scores = []
    fix_scores   = []

    for diagnosis in action.diagnoses:
        gt = get_ground_truth_for_sensor(diagnosis.sensor_id, ground_truth)

        # Fault type score
        if diagnosis.fault_type == gt.fault_type:
            fault_scores.append(1.0)
        elif same_family(diagnosis.fault_type, gt.fault_type):
            fault_scores.append(0.4)   # partial credit
        else:
            fault_scores.append(0.0)

        # Fix appropriateness score
        if diagnosis.fix == gt.fix:
            fix_scores.append(1.0)
        elif fix_direction_correct(diagnosis.fix, gt.fix):
            fix_scores.append(0.3)
        else:
            fix_scores.append(0.0)

    fault_avg = sum(fault_scores) / len(fault_scores)
    fix_avg   = sum(fix_scores)   / len(fix_scores)

    return 0.6 * fault_avg + 0.4 * fix_avg

# Score ranges:
#   All correct     →  ~0.8–1.0
#   Most correct    →  ~0.5–0.7
#   Partially right →  ~0.2–0.4
#   All wrong       →  ~0.0–0.1
```

---

## Task 3 — HARD: Cascade Failure Diagnosis & Compliance Check

**What it simulates:**
A data engineer investigates a 30-day failure in a 10-sensor network.
Three sensors failed, but because sensors share calibration dependencies,
the failures cascaded. The engineer must find the ROOT CAUSE (not just symptoms),
order the repairs correctly, and check if emissions breached EPA limits.

**The dependency structure:**
```
S1 (calibration reference) → calibrates S4, S5
S3 (calibration reference) → calibrates S6, S7, S8
S2 (independent)
S9, S10 (independent)
```

**Input (what agent sees):**
```json
{
  "task_id": "task3_cascade",
  "network_id": "REFINERY-NORTH",
  "period_days": 30,
  "dependency_graph": {
    "S1": {"calibrates": ["S4", "S5"]},
    "S3": {"calibrates": ["S6", "S7", "S8"]},
    "S2": {"independent": true},
    "S9": {"independent": true},
    "S10": {"independent": true}
  },
  "sensors": ["...30 days of data, S1 failed day 8-21, S3 failed day 12-21..."],
  "regulatory_thresholds": {
    "CH4_ppm":  {"warning": 1.5, "violation": 2.0},
    "NOx_ppb":  {"warning": 80,  "violation": 100}
  },
  "known_facts": [
    "S1 was offline days 8-21",
    "S4 and S5 show systematic error matching S1 outage period",
    "CH4 readings are unreliable during the fault window"
  ]
}
```

**Output (what agent must send):**
```json
{
  "root_cause_sensors": ["S1", "S3"],
  "repair_order": ["S1", "S3", "S4", "S5", "S6", "S7", "S8"],
  "fault_window_start": "day_8",
  "fault_window_end": "day_21",
  "compliance_checks": [
    {
      "parameter": "CH4_ppm",
      "status": "POSSIBLE_VIOLATION",
      "confidence": 0.75,
      "reasoning": "CH4 data is unreliable during fault window. Cannot confirm compliance."
    },
    {
      "parameter": "NOx_ppb",
      "status": "CLEAN",
      "confidence": 0.9,
      "reasoning": "NOx sensors S9, S10 were independent and functioning. Readings clean."
    }
  ],
  "recommended_action": "flag_for_manual_review"
}
```

**Grader logic:**
```python
def grade_task3(action, ground_truth):
    # Root cause identification
    predicted_roots = set(action.root_cause_sensors)
    actual_roots    = set(ground_truth.root_cause_sensors)
    root_score = len(predicted_roots & actual_roots) / len(actual_roots)

    # Repair order — dependency violations
    violations = count_dependency_violations(action.repair_order, dependency_graph)
    order_score = max(0.0, 1.0 - violations * 0.25)

    # Fault window accuracy
    window_score = 1.0 if (action.fault_window_start == ground_truth.fault_window_start
                        and action.fault_window_end   == ground_truth.fault_window_end) else 0.3

    # Compliance check accuracy
    compliance_score = 0.0
    for check in action.compliance_checks:
        gt_check = get_gt_compliance(check.parameter, ground_truth)
        if check.status == gt_check.status:
            compliance_score += 1.0 / len(action.compliance_checks)
        elif adjacent_status(check.status, gt_check.status):
            compliance_score += 0.3 / len(action.compliance_checks)

    final = (0.35 * root_score + 0.30 * order_score +
             0.20 * compliance_score + 0.15 * window_score)
    return round(min(1.0, final), 4)

# Score ranges:
#   Expert-level analysis   →  ~0.7–1.0
#   Good but partial        →  ~0.4–0.6
#   Some correct parts      →  ~0.2–0.3
#   Completely wrong        →  ~0.0–0.1
```

---

# PART 5 — COMPLETE CODE ARCHITECTURE

## models.py — All Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any

# ── Sensor data ──────────────────────────────────────────────────
class Reading(BaseModel):
    hour: int
    value: Optional[float] = None

class SensorData(BaseModel):
    sensor_id: str
    parameter: str
    unit: str
    location: str
    normal_range: List[float]
    readings: List[Reading]

# ── Task 1: Anomaly Detection ────────────────────────────────────
class FaultFlag(BaseModel):
    hour: int
    fault: Literal["outlier", "missing", "stuck", "drift", "spike", "bias", "valid"]
    confidence: float = Field(ge=0.0, le=1.0)

class DetectAction(BaseModel):
    sensor_id: str
    flags: List[FaultFlag]

# ── Task 2: Data Cleaning ────────────────────────────────────────
class SensorDiagnosis(BaseModel):
    sensor_id: str
    fault_type: Literal["drift", "missing", "bias", "noise", "stuck", "spike", "valid"]
    severity:   Literal["none", "low", "medium", "high", "critical"]
    fix:        Literal["no_action", "interpolate", "recalibrate",
                        "offset_correction", "smooth", "flag_only", "replace"]
    fix_params: Dict[str, Any] = {}

class CleanAction(BaseModel):
    diagnoses: List[SensorDiagnosis]

# ── Task 3: Cascade Failure ──────────────────────────────────────
class ComplianceCheck(BaseModel):
    parameter: str
    status: Literal["CLEAN", "POSSIBLE_VIOLATION", "CONFIRMED_VIOLATION", "INSUFFICIENT_DATA"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class CascadeAction(BaseModel):
    root_cause_sensors: List[str]
    repair_order: List[str]
    fault_window_start: str
    fault_window_end: str
    compliance_checks: List[ComplianceCheck]
    recommended_action: Literal["no_action", "flag_for_review",
                                "file_compliance_report", "emergency_shutdown"]

# ── Observation ──────────────────────────────────────────────────
class SensorObservation(BaseModel):
    done: bool = False
    reward: float = 0.0
    task_id: str = ""
    step_count: int = 0
    sensor_data: Any = None
    feedback: str = ""
    metadata: Dict[str, Any] = {}

# ── State ────────────────────────────────────────────────────────
class SensorState(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False
```

---

## environment.py — Core Logic

```python
import uuid
from typing import Optional
from app.models import SensorObservation, SensorState
from app.tasks.task1_detect import load_task1, grade_task1
from app.tasks.task2_clean  import load_task2, grade_task2
from app.tasks.task3_cascade import load_task3, grade_task3
from app.reward import compute_reward

TASK_LOADERS = {
    "task1_detect":  load_task1,
    "task2_clean":   load_task2,
    "task3_cascade": load_task3,
}
TASK_GRADERS = {
    "task1_detect":  grade_task1,
    "task2_clean":   grade_task2,
    "task3_cascade": grade_task3,
}
MAX_STEPS = 5

class ClimateWatchEnvironment:

    def __init__(self):
        self._clear()

    def _clear(self):
        self.episode_id    = None
        self.task_id       = None
        self.step_count    = 0
        self.total_reward  = 0.0
        self.done          = False
        self.scenario      = None
        self.ground_truth  = None
        self.history       = []

    def reset(self, task_id: str = "task1_detect",
              seed: Optional[int] = None) -> SensorObservation:
        self._clear()
        self.episode_id  = str(uuid.uuid4())
        self.task_id     = task_id
        loader           = TASK_LOADERS[task_id]
        self.scenario, self.ground_truth = loader(seed=seed)

        return SensorObservation(
            done=False, reward=0.0,
            task_id=task_id, step_count=0,
            sensor_data=self.scenario,
            feedback=f"New episode started for {task_id}. Analyse the sensor data."
        )

    def step(self, raw_action: dict) -> SensorObservation:
        reward = compute_reward(
            raw_action, self.ground_truth, self.task_id, self.history
        )
        self.total_reward += reward
        self.step_count   += 1
        self.history.append(raw_action)

        grader  = TASK_GRADERS[self.task_id]
        episode_score = grader(raw_action, self.ground_truth)
        self.done = (self.step_count >= MAX_STEPS) or (episode_score >= 0.8)

        return SensorObservation(
            done=self.done,
            reward=round(reward, 4),
            task_id=self.task_id,
            step_count=self.step_count,
            sensor_data=self.scenario,
            feedback=self._generate_feedback(reward, raw_action),
            metadata={"episode_score": round(episode_score, 4),
                      "total_reward": round(self.total_reward, 4)}
        )

    def state(self) -> SensorState:
        return SensorState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            total_reward=round(self.total_reward, 4),
            done=self.done
        )

    def _generate_feedback(self, reward: float, action: dict) -> str:
        if reward >= 0.8:   return "Excellent analysis. High accuracy."
        if reward >= 0.5:   return "Good analysis. Some faults missed or misidentified."
        if reward >= 0.2:   return "Partial analysis. Review the sensor patterns more carefully."
        return "Low accuracy. Re-examine the readings for statistical patterns."
```

---

## app/main.py — All Endpoints

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import subprocess, json

from app.models import SensorObservation, SensorState
from app.environment import ClimateWatchEnvironment

app = FastAPI(title="ClimateWatch — Environmental Sensor Environment",
              version="1.0.0")
env = ClimateWatchEnvironment()

# ── Reset ─────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "task1_detect"
    seed: Optional[int] = None

@app.post("/reset", response_model=SensorObservation)
def reset(req: ResetRequest):
    return env.reset(task_id=req.task_id, seed=req.seed)

# ── Step ──────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: dict

@app.post("/step", response_model=SensorObservation)
def step(req: StepRequest):
    if env.episode_id is None:
        raise HTTPException(400, "Call /reset first")
    return env.step(req.action)

# ── State ─────────────────────────────────────────────────────────
@app.get("/state", response_model=SensorState)
def state():
    return env.state()

# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy"}

# ── Tasks ─────────────────────────────────────────────────────────
@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "task1_detect", "name": "Single Sensor Anomaly Detection",
         "difficulty": "easy",
         "action_schema": {
             "sensor_id": "string",
             "flags": [{"hour": "int", "fault": "outlier|missing|stuck|drift|spike|bias|valid",
                        "confidence": "float 0.0–1.0"}]
         }},
        {"id": "task2_clean", "name": "Multi-Sensor Data Cleaning",
         "difficulty": "medium",
         "action_schema": {
             "diagnoses": [{"sensor_id": "string",
                            "fault_type": "drift|missing|bias|noise|stuck|spike|valid",
                            "severity": "none|low|medium|high|critical",
                            "fix": "no_action|interpolate|recalibrate|offset_correction|smooth",
                            "fix_params": "object"}]
         }},
        {"id": "task3_cascade", "name": "Cascade Failure & Compliance Check",
         "difficulty": "hard",
         "action_schema": {
             "root_cause_sensors": ["list of sensor IDs"],
             "repair_order": ["ordered list of sensor IDs"],
             "fault_window_start": "day_N",
             "fault_window_end": "day_N",
             "compliance_checks": [{"parameter": "string",
                                    "status": "CLEAN|POSSIBLE_VIOLATION|CONFIRMED_VIOLATION|INSUFFICIENT_DATA",
                                    "confidence": "float 0.0–1.0",
                                    "reasoning": "string"}],
             "recommended_action": "no_action|flag_for_review|file_compliance_report|emergency_shutdown"
         }}
    ]}

# ── Grader ────────────────────────────────────────────────────────
@app.post("/grader")
def grader():
    if env.episode_id is None:
        raise HTTPException(400, "No active episode")
    from app.tasks import get_grader
    score = get_grader(env.task_id)(env.last_action, env.ground_truth)
    return {"episode_id": env.episode_id, "task_id": env.task_id,
            "final_score": round(score, 4), "step_count": env.step_count}

# ── Baseline ──────────────────────────────────────────────────────
@app.post("/baseline")
def baseline():
    result = subprocess.run(["python", "inference.py"],
                            capture_output=True, text=True, timeout=1200)
    return {"stdout": result.stdout, "stderr": result.stderr,
            "returncode": result.returncode}

# ── Dashboard UI ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html><html>
    <head><title>ClimateWatch Environment</title>
    <style>
      body{font-family:monospace;background:#0a0f0a;color:#00ff88;padding:2rem;margin:0}
      h1{color:#00ccff;border-bottom:1px solid #00ff88;padding-bottom:1rem}
      .card{border:1px solid #00ff88;padding:1rem;margin:1rem 0;border-radius:4px}
      button{background:#00ff88;color:#000;border:none;padding:.5rem 1rem;
             margin:.3rem;cursor:pointer;border-radius:3px;font-family:monospace}
      button:hover{background:#00ccff}
      pre{background:#0d1a0d;padding:1rem;border-radius:4px;overflow-x:auto;
          font-size:0.85rem;max-height:300px;overflow-y:auto}
      a{color:#00ccff}
    </style></head>
    <body>
      <h1>🌍 ClimateWatch — Environmental Sensor Environment</h1>
      <div class="card">
        <h3>Quick Start</h3>
        <button onclick="resetTask('task1_detect')">▶ Task 1 (Easy)</button>
        <button onclick="resetTask('task2_clean')">▶ Task 2 (Medium)</button>
        <button onclick="resetTask('task3_cascade')">▶ Task 3 (Hard)</button>
      </div>
      <div class="card">
        <h3>Current State</h3>
        <pre id="state">Click a task to start...</pre>
      </div>
      <div class="card">
        <h3>Links</h3>
        <a href="/docs">→ Interactive API Docs (Swagger UI)</a><br>
        <a href="/tasks">→ View all tasks</a><br>
        <a href="/health">→ Health check</a>
      </div>
      <script>
        async function resetTask(t){
          const r=await fetch('/reset',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({task_id:t})});
          document.getElementById('state').textContent=
            JSON.stringify(await r.json(),null,2);
        }
        async function loadState(){
          try{const r=await fetch('/state');
          const d=await r.json();
          if(d.episode_id)
            document.getElementById('state').textContent=
              JSON.stringify(d,null,2);}catch(e){}
        }
        setInterval(loadState,5000);
      </script>
    </body></html>"""
```

---

## inference.py — Root Directory

```python
"""
ClimateWatch — Baseline Inference Script
========================================
MANDATORY:
- Named: inference.py
- Located in: root directory
- Uses: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Uses: OpenAI client with HF router
- Runtime: < 20 minutes total
"""

import os, json, requests, time
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS    = 5
TEMPERATURE  = 0.0   # deterministic = reproducible scores

# ── OpenAI client pointing to HF router ──────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert environmental data engineer.
You specialise in detecting faults in climate sensor networks.
Analyse the sensor data and respond with valid JSON matching the schema."""

def ask_llm(observation: dict, task_id: str) -> dict:
    tasks_context = {
        "task1_detect": "Detect faults. Return: {sensor_id, flags:[{hour,fault,confidence}]}",
        "task2_clean":  "Diagnose sensors. Return: {diagnoses:[{sensor_id,fault_type,severity,fix,fix_params}]}",
        "task3_cascade": "Find root cause. Return: {root_cause_sensors,repair_order,fault_window_start,fault_window_end,compliance_checks,recommended_action}"
    }

    prompt = f"""Task: {tasks_context.get(task_id, '')}

Sensor data:
{json.dumps(observation.get('sensor_data', observation), indent=2)}

Respond with valid JSON only."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        max_tokens=800
    )
    return json.loads(response.choices[0].message.content)

def run_task(task_id: str) -> float:
    obs = requests.post(f"{ENV_URL}/reset",
                        json={"task_id": task_id},
                        timeout=30).json()
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = ask_llm(obs, task_id)
        result = requests.post(f"{ENV_URL}/step",
                               json={"action": action},
                               timeout=30).json()
        total_reward += result.get("reward", 0.0)
        done  = result.get("done", False)
        obs   = result
        steps += 1
        time.sleep(0.5)   # rate limiting

    return round(total_reward, 4)

if __name__ == "__main__":
    tasks = ["task1_detect", "task2_clean", "task3_cascade"]
    print("=" * 50)
    print("ClimateWatch — Baseline Scores")
    print("=" * 50)

    scores = {}
    for task in tasks:
        print(f"\nRunning {task}...", flush=True)
        score = run_task(task)
        scores[task] = score
        print(f"  Score: {score:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n{'='*50}")
    print(f"Average: {avg:.4f}")
    print(json.dumps(scores, indent=2))
```

---

# PART 6 — HOW SCORING MAPS TO YOUR PROJECT

```
Real-world utility (30%)
  ClimateWatch target: 26–30/30
  Why: $14.4B market, BP/EPA/NOAA use this daily, fills real gap in OpenEnv

Task quality (25%)
  ClimateWatch target: 20–25/25
  Why: 3 clear tasks, F1/Kendall-tau graders, hard task challenges GPT-4

Environment design (20%)
  ClimateWatch target: 16–20/20
  Why: clean _clear() in reset, partial rewards, sensible 5-step episodes

Code quality (15%)
  ClimateWatch target: 12–15/15
  Why: Pydantic models, fastapi, docker, openenv validate passes

Creativity (10%)
  ClimateWatch target: 8–10/10
  Why: zero existing OpenEnv environments in this domain

TOTAL TARGET: 82–100 points
```

---

*ClimateWatch — Environmental Sensor Data Quality & Compliance Monitoring*
