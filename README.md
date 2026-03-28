---
title: ClimateWatch Environment
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# ClimateWatch — Environmental Sensor AI Environment

> An OpenEnv-compatible reinforcement learning environment where AI agents learn to detect faults in climate sensor data, clean corrupted readings, and verify EPA regulatory compliance — exactly what data engineers at BP, EPA, and NOAA do every day.

**Team:** ALGORITHMIC AVENGERS — Ramana Ganthan S + Sre Sandhya K

---

## Table of Contents

1. [What Is ClimateWatch?](#what-is-climatewatch)
2. [The Problem It Solves](#the-problem-it-solves)
3. [How to Run It](#how-to-run-it)
4. [How to Test It](#how-to-test-it)
5. [The 3 Tasks — What They Do](#the-3-tasks)
6. [API Reference — Every Endpoint](#api-reference)
7. [How the Scoring Works](#how-the-scoring-works)
8. [Tech Stack — What Is Used For What](#tech-stack)
9. [File Structure — What Each File Does](#file-structure)
10. [Baseline Scores](#baseline-scores)

---

## What Is ClimateWatch?

ClimateWatch is a **reinforcement learning environment** where an AI agent interacts with simulated industrial sensor networks. The agent receives sensor data, analyses it, submits a JSON action with its findings, and receives a reward score (0.0–1.0) based on how accurate its analysis was.

It works exactly like a gym environment — but for real-world environmental data tasks:

```
Agent                          ClimateWatch Server
  │                                    │
  │── POST /reset ──────────────────►  │  loads a sensor scenario
  │◄─────────────────── observation ── │  returns 24hrs of sensor data
  │                                    │
  │── POST /step (action: flags) ────► │  grades the action
  │◄─────────────────── reward 0.8 ─── │  returns score + feedback
  │                                    │
  │── POST /grader ──────────────────► │  final score for episode
  │◄─────────────────── score 0.85 ─── │
```

---

## The Problem It Solves

Real industrial sensor networks constantly break in ways that are dangerous if undetected:

| Fault Type | What Happens | Real-World Risk |
|---|---|---|
| **Stuck** | Sensor freezes at one value for hours | Firmware crash — looks like everything is fine when it isn't |
| **Outlier** | Single extreme spike reading | Electrical interference — triggers false alarms |
| **Drift** | Gradual baseline shift over days | Electrode degradation — slowly growing measurement error |
| **Bias** | All readings constantly too high/low | Calibration error — systematic wrong reporting |
| **Missing** | Null values, data gaps | Network dropout, battery failure |
| **Noise** | Random fluctuation beyond normal variance | Low-cost sensor, vibration, humidity |
| **Spike** | Short burst of bad readings then back | Electromagnetic interference event |
| **Cascade** | One reference sensor fails → all sensors calibrated from it misread | Shared calibration dependency — worst kind of failure |

**When sensor data is wrong:**
- Dangerous methane leaks go undetected → **explosion risk**
- Wrong emissions reports are filed → **$93,750/day EPA fines**
- Climate models trained on bad data → **wrong policy decisions**

**Market:** $14.4B environmental monitoring market (2024), growing to $41.4B by 2029.

---

## How to Run It

### Prerequisites

Install **uv** (one time only — replaces pip + venv):

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 1 — Start the server

```bash
cd C:\Users\2471999\Documents\BuildVerse

uv run task serve
```

This single command does everything:
1. Creates `.venv/` automatically (first run only)
2. Installs all 41 packages from `pyproject.toml`
3. Kills anything already running on port 7860
4. Starts the server at **http://localhost:7860**

### Step 2 — Open the dashboard

Go to **http://localhost:7860** in your browser. You'll see an interactive dashboard with buttons to start each task.

### Step 3 — Try the API directly

```bash
# Check server is alive
curl http://localhost:7860/health

# Start a Task 1 episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": [{"hour": 6, "fault": "outlier", "confidence": 1.0}]}}'

# Get your final score
curl -X POST http://localhost:7860/grader
```

### Step 4 — Run the LLM baseline (needs HuggingFace key)

```bash
# Set your credentials
set HF_TOKEN=hf_your_token_here
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

# Run all 3 tasks with an LLM agent
uv run task infer
```

### Docker (for deployment)

```bash
# Build the image
docker build -t climatewatch .

# Run
docker run -p 7860:7860 climatewatch

# Run with LLM inference
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  climatewatch
```

---

## How to Test It

### Run all tests (63 tests)

```bash
uv run task test
```

### Run specific test files

```bash
# Test the graders only (do they score correctly?)
uv run pytest tests/test_graders.py -v

# Test the API endpoints only (do all HTTP routes work?)
uv run pytest tests/test_endpoints.py -v
```

### Run a single test class

```bash
uv run pytest tests/test_graders.py::TestTask1Grader -v
uv run pytest tests/test_endpoints.py::TestFullEpisode -v
```

### What the tests verify

**`tests/test_graders.py`** — 22 tests covering:
- Perfect action scores ≥ 0.80 on all 3 tasks
- Empty / wrong action scores near 0.0
- Partial answer scores between wrong and perfect
- Different actions give different scores (critical requirement — graders that always return the same score = disqualified)
- Score is always in [0.0, 1.0], never NaN
- Calibration bonus is awarded for varied confidence values
- Anti-loop penalty reduces reward for repeated actions

**`tests/test_endpoints.py`** — 41 tests covering:
- `GET /health` returns 200 + `{"status": "healthy"}`
- `GET /tasks` returns exactly 3 tasks with correct difficulties
- `POST /reset` works for all 3 tasks, returns correct structure, is deterministic with seed
- `POST /step` increments step count, returns reward in [0.0, 1.0], gives feedback
- `POST /grader` returns final score after episode
- Full episode flow completes without error for all 3 tasks
- Invalid requests return 400

### Expected test output

```
63 passed in 2.51s
```

---

## The 3 Tasks

### Task 1 — Single Sensor Anomaly Detection `[EASY]`

**What it does:** Gives the agent 24 hours of hourly readings from one sensor. Some hours contain injected faults. The agent must find every faulty hour and classify the fault.

**Why it's easy:** Only one sensor, only 24 data points, fault types are visually obvious in the data.

**What the agent sees:**
```json
{
  "sensor_id": "CO2-100",
  "parameter": "CO2_ppm",
  "unit": "ppm",
  "location": "Industrial Zone A, Houston TX",
  "normal_range": [400, 450],
  "readings": [
    {"hour": 0, "value": 412.3},
    {"hour": 3, "value": 413.1},
    {"hour": 4, "value": 413.1},
    {"hour": 5, "value": 413.1},
    {"hour": 6, "value": 9999.0},
    {"hour": 8, "value": null},
    ...
  ]
}
```

**What the agent must send:**
```json
{
  "sensor_id": "CO2-100",
  "flags": [
    {"hour": 3, "fault": "stuck",   "confidence": 0.95},
    {"hour": 4, "fault": "stuck",   "confidence": 0.95},
    {"hour": 5, "fault": "stuck",   "confidence": 0.95},
    {"hour": 6, "fault": "outlier", "confidence": 1.0},
    {"hour": 8, "fault": "missing", "confidence": 1.0}
  ]
}
```

**Fault types:** `outlier` | `stuck` | `missing` | `drift` | `spike` | `bias`

**20 scenarios** — sensors: CO2, NO2, CH4, O3, PM2.5, SO2, Temperature, Humidity across 20 real industrial locations.

**Grader:** F1 score between predicted and actual flags. Bonus +0.05 for varied confidence values (calibration reward).

---

### Task 2 — Multi-Sensor Data Stream Cleaning `[MEDIUM]`

**What it does:** Gives the agent 7 days of hourly data from 5 sensors in a network. Each sensor has a *different* fault. One sensor is always clean. The agent must diagnose each sensor AND recommend the correct fix.

**Why it's harder than Task 1:**
- 5 sensors at once instead of 1
- Must choose the correct *fix*, not just identify the fault
- One sensor is always valid — agent must not over-diagnose
- Severity must match (none/low/medium/high/critical)

**What the agent sees:** Daily summaries + statistics for each sensor over 7 days, including trend, missing %, and mean values.

**What the agent must send:**
```json
{
  "diagnoses": [
    {"sensor_id": "S1", "fault_type": "drift",   "severity": "high",     "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.8}},
    {"sensor_id": "S2", "fault_type": "missing", "severity": "medium",   "fix": "interpolate",       "fix_params": {"method": "linear"}},
    {"sensor_id": "S3", "fault_type": "bias",    "severity": "high",     "fix": "offset_correction", "fix_params": {"offset": -12.0}},
    {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium",   "fix": "smooth",            "fix_params": {"window": 3}},
    {"sensor_id": "S5", "fault_type": "valid",   "severity": "none",     "fix": "no_action",         "fix_params": {}}
  ]
}
```

**Fix types:** `recalibrate` | `offset_correction` | `interpolate` | `smooth` | `replace` | `flag_only` | `no_action`

**10 scenarios** across Gulf Coast, Jubail Saudi Arabia, Delhi NCR, Svalbard Norway, Amazon Basin, North Sea, and more.

**Grader:** 60% fault type accuracy + 40% fix appropriateness. Partial credit for same-family faults (drift ↔ bias) and related fixes (recalibrate ↔ offset_correction).

---

### Task 3 — Cascade Failure & Compliance Audit `[HARD]`

**What it does:** Gives the agent 30 days of data from a 10-sensor network where reference sensors fail and corrupt downstream sensors. The agent must perform multi-dimensional reasoning.

**Why it genuinely challenges frontier LLMs (GPT-4, Nemotron):**

1. **Graph reasoning** — must understand sensor dependency relationships (S1 calibrates S4 and S5 → repairing S4 before S1 is wrong)
2. **Cause vs symptom** — corrupted sensors *look* broken but are NOT the root cause. Must identify the references that actually failed
3. **Temporal analysis** — fault window spans 30 days, must identify exact start/end day
4. **Epistemic compliance** — if the sensors measuring CH4 were corrupted, you *cannot* confirm or deny EPA compliance — must reason about what you *can't* know

**What the agent sees:** 30-day daily summaries per sensor, the full dependency graph, EPA thresholds, and known facts about the network.

**What the agent must send:**
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
      "reasoning": "CH4 sensors S1, S4, S5 were corrupted during fault window. Cannot confirm compliance."
    },
    {
      "parameter": "NOx_ppb",
      "status": "CLEAN",
      "confidence": 0.90,
      "reasoning": "NOx sensors S9, S10 are independent and showed clean readings throughout."
    }
  ],
  "recommended_action": "flag_for_review"
}
```

**Compliance statuses:** `CLEAN` | `POSSIBLE_VIOLATION` | `CONFIRMED_VIOLATION` | `INSUFFICIENT_DATA`

**5 scenarios** — different network topologies, failure patterns, and compliance situations.

**Grader (4 components):**
| Component | Weight | What it checks |
|---|---|---|
| Root cause | 35% | Jaccard similarity between predicted and actual root causes |
| Repair order | 30% | Dependency violations (-0.25 each) + completeness of repair list |
| Fault window | 20% | Temporal accuracy (partial credit within ±3 days) |
| Compliance | 15% | Correct status per parameter (adjacent status = 0.3 partial credit) |

---

## API Reference

All endpoints, inputs, and outputs:

### `POST /reset`
Start a new episode.

**Input:**
```json
{"task_id": "task1_detect", "seed": 42}
```
- `task_id`: `task1_detect` | `task2_clean` | `task3_cascade`
- `seed`: integer for deterministic scenario (null = random)

**Output:** `SensorObservation`
```json
{
  "done": false,
  "reward": 0.0,
  "task_id": "task1_detect",
  "step_count": 0,
  "sensor_data": { ... },
  "feedback": "Episode started. Analyse the sensor data.",
  "metadata": {"episode_id": "uuid", "max_steps": 5}
}
```

---

### `POST /step`
Submit an action and receive reward.

**Input:**
```json
{"action": { ...your task-specific action... }}
```

**Output:** `SensorObservation` (same structure as reset, but with non-zero reward)
```json
{
  "done": false,
  "reward": 0.72,
  "step_count": 1,
  "feedback": "Good analysis — score 0.72. Most faults correctly identified.",
  "metadata": {"episode_score": 0.72, "total_reward": 0.72, "steps_left": 4}
}
```

---

### `GET /state`
Current episode state — no input needed.

**Output:**
```json
{"episode_id": "uuid", "task_id": "task1_detect", "step_count": 2, "total_reward": 1.44, "done": false}
```

---

### `GET /health`
Liveness probe. Returns `200 OK` when server is running.

**Output:**
```json
{"status": "healthy"}
```

---

### `GET /tasks`
Full task catalog with action schemas and examples.

**Output:** List of 3 tasks, each with `id`, `name`, `difficulty`, `description`, `action_schema`, `example_action`.

---

### `POST /grader`
Final score for the completed episode. Call after `done=true`.

**Output:**
```json
{
  "episode_id": "uuid",
  "task_id": "task1_detect",
  "final_score": 0.85,
  "step_count": 3,
  "breakdown": {"total_reward": 2.55, "steps_used": 3, "done": true}
}
```

---

### `POST /baseline`
Runs `inference.py` and returns scores for all 3 tasks. Requires `HF_TOKEN` and `MODEL_NAME` environment variables to be set.

**Output:**
```json
{"stdout": "task1_detect: 0.72\ntask2_clean: 0.58\n...", "stderr": "", "returncode": 0}
```

---

## How the Scoring Works

### Per-step reward

Every `POST /step` returns a reward immediately — not just at the end:

```
reward = grader_score(action, ground_truth)
       - 0.30  if action == previous action  (anti-loop penalty)
       - 0.05  if score got worse vs previous step  (regression penalty)
```

### Early stop

Episode ends automatically if `episode_score >= 0.80` — the problem is considered solved.

### Final grade

`POST /grader` returns the grader score for the **last** action submitted. This is the score that counts.

### Score ranges by task

| Task | Random agent | Decent LLM | Expert analysis |
|---|---|---|---|
| task1_detect | 0.0–0.1 | 0.5–0.75 | 0.85–1.0 |
| task2_clean | 0.0–0.15 | 0.4–0.65 | 0.75–1.0 |
| task3_cascade | 0.0–0.1 | 0.3–0.55 | 0.65–0.9 |

---

## Tech Stack

### What each technology does in this project

| Technology | Version | Used For |
|---|---|---|
| **Python** | 3.11+ | Primary language |
| **FastAPI** | 0.115.6 | Web framework — defines all HTTP endpoints, handles request/response validation |
| **Pydantic** | 2.10.4 | Data models — validates all JSON inputs/outputs, defines action schemas |
| **Uvicorn** | 0.32.1 | ASGI server — runs the FastAPI app, handles HTTP connections |
| **OpenAI SDK** | ≥1.58 | LLM client in `inference.py` — connects to HuggingFace router to call the LLM |
| **Requests** | 2.32.3 | HTTP client in `inference.py` — calls the ClimateWatch server from the inference script |
| **NumPy** | 2.2.1 | Scenario generation — statistical operations for creating realistic sensor data |
| **python-dotenv** | 1.0.1 | Loads `.env` file locally so you don't have to set env vars manually |
| **uv** | ≥0.9 | Package manager + task runner — replaces pip, venv, and make |
| **taskipy** | ≥1.13 | Task definitions in `pyproject.toml` — enables `uv run task serve` |
| **pytest** | ≥8.0 | Test framework — runs all 63 tests |
| **httpx** | ≥0.27 | HTTP client used by FastAPI's `TestClient` in tests |
| **pytest-asyncio** | ≥0.23 | Async test support for FastAPI endpoints |
| **ruff** | ≥0.3 | Linter and formatter |
| **Docker** | — | Containerisation — packages the whole server for HuggingFace Spaces deployment |
| **HuggingFace Spaces** | — | Cloud deployment — hosts the Docker container publicly |

### Why these choices

- **FastAPI over Flask**: automatic request validation via Pydantic, auto-generates `/docs` Swagger UI, async-native
- **Pydantic v2 over dataclasses**: field-level validation, clear error messages when agents send wrong JSON
- **uv over pip**: 10–100× faster installs, lockfile for reproducibility, built-in task runner via taskipy
- **No database**: all scenarios are generated in-memory from seeded RNG — keeps the container lightweight (no Postgres, no Redis, runs on 2 vCPU / 8 GB RAM)
- **No ML models at startup**: no PyTorch, no TensorFlow — server starts in <2 seconds, uses <200 MB RAM

---

## File Structure

```
climatewatch-env/
│
├── inference.py              LLM baseline agent — runs all 3 tasks, prints scores
│                             Uses OpenAI client → HuggingFace router (not openai.com)
│                             Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from env
│
├── openenv.yaml              Environment metadata — name, version, 3 tasks with difficulties
│                             Required by openenv validate tool
│
├── Dockerfile                Container definition — python:3.11-slim, installs deps, runs uvicorn
│                             Uses uv inside Docker for faster builds
│
├── requirements.txt          Package list — required by hackathon judges
│                             Kept in sync with pyproject.toml dependencies
│
├── pyproject.toml            Single source of truth for dev — all packages, task shortcuts, test config
│                             Run: uv run task serve / test / lint / infer
│
├── README.md                 This file
│
├── scripts/
│   └── kill_port.py          Utility — kills any process on port 7860 before server starts
│                             Prevents "port in use" errors on restart
│
├── app/
│   ├── __init__.py
│   │
│   ├── main.py               All HTTP endpoints — /reset /step /state /health /tasks /grader /baseline
│                             Also serves the interactive dashboard at /
│   │
│   ├── environment.py        Episode manager — reset(), step(), state(), final_grade()
│                             Thread-safe (RLock), supports early-stop at score ≥ 0.80
│   │
│   ├── models.py             All Pydantic schemas — request/response models for all 3 tasks
│                             DetectAction, CleanAction, CascadeAction, SensorObservation, etc.
│   │
│   ├── reward.py             Per-step reward function
│                             Applies grader score + anti-loop penalty + regression penalty
│   │
│   └── tasks/
│       ├── __init__.py       Task registry — maps task_id → loader and grader functions
│       │
│       ├── task1_detect.py   20 deterministic scenarios, F1 + calibration bonus grader
│                             Sensors: CO2, NO2, CH4, O3, PM2.5, SO2, Temp, Humidity
│                             Fault types: outlier, stuck, missing, drift, spike, bias
│       │
│       ├── task2_clean.py    10 scenarios (5 sensors × 7 days each)
│                             Grader: 60% fault type + 40% fix appropriateness + severity penalty
│       │
│       └── task3_cascade.py  5 cascade failure scenarios (10 sensors × 30 days each)
│                             Grader: 35% root cause + 30% repair order + 20% window + 15% compliance
│
└── tests/
    ├── __init__.py
    ├── test_graders.py       22 tests — grader correctness, score ranges, no constant scores
    └── test_endpoints.py     41 tests — all HTTP endpoints, full episode flows, edge cases
```

---

## Environment Variables

| Variable | Default | Required | Where Used |
|---|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | For inference | `inference.py` — LLM endpoint |
| `MODEL_NAME` | — | For inference | `inference.py` — which model to call |
| `HF_TOKEN` | — | For inference | `inference.py` — HuggingFace API key |
| `API_KEY` | — | Optional | `inference.py` — fallback if HF_TOKEN not set |
| `ENV_URL` | `http://localhost:7860` | For inference | `inference.py` — where ClimateWatch server is |

The server itself needs **no environment variables** to run. Only `inference.py` needs them.

---

## Baseline Scores

Scores obtained with `meta-llama/Llama-3.3-70B-Instruct` at `temperature=0.0`, `seed=42`:

| Task | Score | What the LLM got right / wrong |
|---|---|---|
| task1_detect | ~0.72 | Most fault types detected. Occasionally misses stuck sensors or mislabels drift as bias |
| task2_clean | ~0.58 | Good fault identification. Fix parameters (offset values, drift rates) often approximate |
| task3_cascade | ~0.41 | Partially identifies root causes. Repair order sometimes violates dependencies. Compliance reasoning is the weakest area |
| **Average** | **~0.57** | Typical for a capable 70B model with no fine-tuning |

*GPT-4 and Nemotron 3 Super expected to score 0.65–0.85 on task1, 0.55–0.75 on task2, 0.35–0.60 on task3.*

---

## Quick Command Reference

```bash
uv run task serve      # kill port + start server at http://localhost:7860
uv run task test       # run all 63 tests
uv run task lint       # check code style with ruff
uv run task fmt        # auto-format code with ruff
uv run task kill       # kill anything on port 7860
uv run task infer      # run inference.py (needs HF_TOKEN + MODEL_NAME)
```
