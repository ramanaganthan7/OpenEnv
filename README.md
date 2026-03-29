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

> An OpenEnv-compatible RL environment where AI agents learn to detect faults in climate sensor data, clean corrupted readings, and verify EPA regulatory compliance — exactly what data engineers at BP, EPA, and NOAA do every day.

**Team:** ALGORITHMIC AVENGERS — Ramana Ganthan S + Sre Sandhya K
**Hackathon:** OpenEnv (Meta × HuggingFace)
**Live Space:** `https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env`

---

## Table of Contents

1. [What Is ClimateWatch?](#1-what-is-climatewatch)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How to Run Locally](#3-how-to-run-locally)
4. [How to Test](#4-how-to-test)
5. [All 7 API Endpoints](#5-all-7-api-endpoints)
6. [The 3 Tasks in Detail](#6-the-3-tasks-in-detail)
7. [How Scoring Works](#7-how-scoring-works)
8. [Tech Stack — What Is Used for What](#8-tech-stack)
9. [File Structure — What Each File Does](#9-file-structure)
10. [Environment Variables](#10-environment-variables)
11. [Baseline Scores](#11-baseline-scores)
12. [Deploy to HuggingFace](#12-deploy-to-huggingface)

---

## 1. What Is ClimateWatch?

ClimateWatch is a **reinforcement learning environment** (like OpenAI Gym, but for real-world sensor data). An AI agent interacts with it in a loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                   EPISODE FLOW                                  │
│                                                                 │
│  1. Agent  →  POST /reset  →  Server returns sensor data        │
│  2. Agent  →  POST /step   →  Server returns reward + feedback  │
│  3. Repeat up to 5 steps (early stop if score ≥ 0.80)          │
│  4. Agent  →  POST /grader →  Final score 0.0–1.0              │
└─────────────────────────────────────────────────────────────────┘
```

No real weather API. No database. No internet needed.
All sensor data is **generated entirely by Python** using seeded random numbers — deterministic and realistic.

---

## 2. The Problem It Solves

Real industrial sensor networks fail constantly. When undetected, this causes disasters:

| Fault | What Happens | Real Risk |
|---|---|---|
| **Stuck** | Sensor freezes at one value for hours | Firmware crash — looks fine when it isn't |
| **Outlier** | Single extreme spike | Electrical interference |
| **Drift** | Readings creep up/down over days | Electrode degradation |
| **Bias** | Constant offset — always too high/low | Calibration error |
| **Missing** | Null values, data gaps | Network dropout, battery failure |
| **Noise** | Random fluctuation beyond normal | Low-cost sensor, vibration |
| **Spike** | Short burst of bad readings | Electromagnetic interference |
| **Cascade** | Reference sensor fails → all sensors it calibrates misread | Shared calibration dependency |

**Why it matters:**
- Undetected methane leaks → explosion risk
- Wrong emissions data → **$93,750/day EPA fines**
- Bad sensor data in climate models → **wrong policy decisions**

**Market:** $14.4B environmental monitoring (2024) → $41.4B by 2029

---

## 3. How to Run Locally

### Install uv (one time)

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Start the server

```bash
cd path/to/BuildVerse

uv run task serve
```

**What this single command does:**
1. Creates `.venv/` and installs all 41 packages (first run only — skipped after that)
2. Kills anything already on port 7860 (`scripts/kill_port.py`)
3. Starts server at **http://localhost:7860** with hot-reload

### Open the dashboard

Go to **http://localhost:7860** — interactive dashboard with buttons to start each task and view live observations.

### Quick API test (new terminal)

```bash
# Confirm server is alive
curl http://localhost:7860/health
# → {"status":"healthy"}

# Start a Task 1 episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'

# Submit an action (detect an outlier at hour 6)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": [{"hour": 6, "fault": "outlier", "confidence": 1.0}]}}'

# Get final score
curl -X POST http://localhost:7860/grader
```

### Run the LLM baseline (needs HuggingFace key)

```bash
set HF_TOKEN=hf_your_token_here
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

uv run task infer
```

### All available task commands

```bash
uv run task serve      # kill port 7860 + start server with hot-reload
uv run task test       # run all 63 tests
uv run task lint       # check code style with ruff
uv run task fmt        # auto-format with ruff
uv run task kill       # kill anything on port 7860
uv run task infer      # run inference.py baseline (needs HF_TOKEN + MODEL_NAME)
```

### Docker (same as what runs on HuggingFace)

```bash
docker build -t climatewatch .
docker run -p 7860:7860 climatewatch

# With LLM inference enabled:
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  climatewatch
```

---

## 4. How to Test

### Run all 63 tests

```bash
uv run task test
```

Expected output: `63 passed in ~2.5s`

### Run specific test files

```bash
# Grader tests only (do scores work correctly?)
uv run pytest tests/test_graders.py -v

# Endpoint tests only (do all HTTP routes work?)
uv run pytest tests/test_endpoints.py -v

# Single test class
uv run pytest tests/test_graders.py::TestTask1Grader -v
uv run pytest tests/test_endpoints.py::TestFullEpisode -v
```

### What tests verify

**`tests/test_graders.py`** — 22 tests:
- Perfect action scores ≥ 0.80 on all 3 tasks
- Empty / wrong action scores near 0.0
- Partial answer scores between zero and perfect
- **Different actions must give different scores** (critical — always-same score = disqualified)
- Score always in [0.0, 1.0], never NaN
- Calibration bonus awarded for varied confidence values
- Anti-loop penalty reduces reward for repeated actions

**`tests/test_endpoints.py`** — 41 tests:
- All 4 core endpoints return correct HTTP status and JSON structure
- `/reset` is deterministic with same seed
- `/step` increments step count and returns float reward
- `/grader` returns valid score after episode
- Full episode loop works for all 3 tasks without errors
- Invalid task_id returns 400

---

## 5. All 7 API Endpoints

### `POST /reset` — Start a new episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 42}'
```

**Input:**
```json
{"task_id": "task1_detect", "seed": 42}
```
- `task_id`: `task1_detect` | `task2_clean` | `task3_cascade`
- `seed`: integer for deterministic scenario, `null` for random

**Output:**
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

### `POST /step` — Submit an action, receive reward

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": [{"hour": 6, "fault": "outlier", "confidence": 1.0}]}}'
```

**Input:**
```json
{"action": { ...task-specific action JSON... }}
```

**Output:**
```json
{
  "done": false,
  "reward": 0.72,
  "task_id": "task1_detect",
  "step_count": 1,
  "sensor_data": { ... },
  "feedback": "Good analysis — score 0.72. Most faults correctly identified.",
  "metadata": {
    "episode_id": "uuid",
    "episode_score": 0.72,
    "total_reward": 0.72,
    "steps_left": 4
  }
}
```

---

### `GET /state` — Current episode state

```bash
curl http://localhost:7860/state
```

**Output:**
```json
{
  "episode_id": "dd2f9f12-e8f1-4338-b88c-c06b432b77ca",
  "task_id": "task1_detect",
  "step_count": 1,
  "total_reward": 0.72,
  "done": false
}
```

---

### `GET /health` — Liveness probe

```bash
curl http://localhost:7860/health
```

**Output:** `{"status": "healthy"}` with HTTP 200

---

### `GET /tasks` — Task catalog with action schemas

```bash
curl http://localhost:7860/tasks
```

Returns all 3 tasks with `id`, `name`, `difficulty`, `description`, `action_schema`, `example_action`.

---

### `POST /grader` — Final score for completed episode

```bash
curl -X POST http://localhost:7860/grader
```

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

### `POST /baseline` — Run inference.py and return scores

```bash
curl -X POST http://localhost:7860/baseline
```

Runs `inference.py` as a subprocess. Requires `HF_TOKEN` and `MODEL_NAME` to be set as environment variables. Returns stdout (scores), stderr, and return code.

---

## 6. The 3 Tasks in Detail

### Task 1 — Single Sensor Anomaly Detection `[EASY]`

**Scenario:** 24 hours of hourly readings from one sensor. Some hours have faults injected. Agent must find every faulty hour and classify the fault.

**Sensors:** CO2, NO2, CH4, O3, PM2.5, SO2, Temperature, Humidity
**Fault types:** `outlier` | `stuck` | `missing` | `drift` | `spike` | `bias`
**Scenarios:** 20 (selected deterministically by `seed % 20`)

**Agent receives:**
```json
{
  "sensor_id": "CO2-100",
  "parameter": "CO2_ppm",
  "unit": "ppm",
  "location": "Industrial Zone A, Houston TX",
  "normal_range": [400, 450],
  "readings": [
    {"hour": 0,  "value": 412.3},
    {"hour": 3,  "value": 413.1},
    {"hour": 4,  "value": 413.1},
    {"hour": 5,  "value": 413.1},
    {"hour": 6,  "value": 9999.0},
    {"hour": 8,  "value": null},
    {"hour": 9,  "value": null}
  ]
}
```

**Agent must send:**
```json
{
  "sensor_id": "CO2-100",
  "flags": [
    {"hour": 3, "fault": "stuck",   "confidence": 0.95},
    {"hour": 4, "fault": "stuck",   "confidence": 0.95},
    {"hour": 5, "fault": "stuck",   "confidence": 0.95},
    {"hour": 6, "fault": "outlier", "confidence": 1.0},
    {"hour": 8, "fault": "missing", "confidence": 1.0},
    {"hour": 9, "fault": "missing", "confidence": 1.0}
  ]
}
```

**Grader:** F1 score + 0.05 calibration bonus for varied confidence values.

---

### Task 2 — Multi-Sensor Data Stream Cleaning `[MEDIUM]`

**Scenario:** 7 days of hourly data from 5 sensors. Each sensor has a different fault. One is always clean. Agent must diagnose each sensor AND recommend the right fix.

**Harder because:**
- 5 sensors simultaneously, each needing a different analysis
- Must pick the right *fix* (not just identify the fault)
- One sensor is always valid — agent must not over-diagnose

**10 network locations:** Gulf Coast TX, Jubail Saudi Arabia, Delhi NCR, Svalbard Norway, Amazon Basin, North Sea, Ruhr Valley Germany, Guangzhou China, Kansas USA, Thames Estuary UK

**Agent must send:**
```json
{
  "diagnoses": [
    {"sensor_id": "S1", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.8}},
    {"sensor_id": "S2", "fault_type": "missing", "severity": "medium",  "fix": "interpolate",       "fix_params": {"method": "linear"}},
    {"sensor_id": "S3", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": -12.0}},
    {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 3}},
    {"sensor_id": "S5", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}}
  ]
}
```

**Fix options:** `no_action` | `interpolate` | `recalibrate` | `offset_correction` | `smooth` | `flag_only` | `replace`
**Grader:** 60% fault type accuracy + 40% fix appropriateness (partial credit for same-family faults/fixes)

---

### Task 3 — Cascade Failure & Compliance Audit `[HARD]`

**Scenario:** 30 days of data from a 10-sensor network. Reference sensors fail and corrupt dependent sensors. Requires multi-dimensional reasoning to solve.

**Why it genuinely challenges frontier LLMs:**

| Challenge | Description |
|---|---|
| Graph reasoning | Repair order must respect dependency graph (fix reference before dependent) |
| Cause vs symptom | Corrupted sensors *look* broken but are NOT the root cause |
| Temporal analysis | Exact fault window (start/end day) across 30 days |
| Epistemic compliance | If measuring sensors were corrupted → can't confirm or deny EPA limits |

**5 network scenarios:** REFINERY-NORTH Kuwait, PIPELINE-CENTRAL Texas, URBAN-NETWORK-ALPHA Beijing, COASTAL-MONITOR-BETA Chennai, OFFSHORE-PLATFORM-7 North Sea

**Agent must send:**
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
      "reasoning": "CH4 sensors S1, S4, S5 corrupted during fault window. Cannot confirm compliance."
    },
    {
      "parameter": "NOx_ppb",
      "status": "CLEAN",
      "confidence": 0.90,
      "reasoning": "NOx sensors S9, S10 are independent — clean readings throughout."
    }
  ],
  "recommended_action": "flag_for_review"
}
```

**Compliance statuses:** `CLEAN` | `POSSIBLE_VIOLATION` | `CONFIRMED_VIOLATION` | `INSUFFICIENT_DATA`

**Grader — 4 components:**

| Component | Weight | What It Checks |
|---|---|---|
| Root cause | 35% | Jaccard similarity between predicted and actual root causes |
| Repair order | 30% | Dependency violations (−0.25 each) + completeness of repair list |
| Fault window | 20% | Temporal accuracy (partial credit within ±3 days) |
| Compliance | 15% | Correct status per parameter (adjacent status = 0.3 partial credit) |

---

## 7. How Scoring Works

### Per-step reward formula

```
reward = grader_score(action, ground_truth)
       − 0.30  if action == previous action  (anti-loop penalty)
       − 0.05  if score worse than previous  (regression penalty)

Always clamped to: max(0.0, min(1.0, reward))
```

### Episode rules
- **Max 5 steps** per episode
- **Early stop** — episode ends if `episode_score >= 0.80` (problem solved)
- **POST /grader** returns score for the **last** action submitted

### Score ranges

| Task | Random agent | Good LLM (70B) | Expert analysis |
|---|---|---|---|
| task1_detect | 0.0–0.1 | 0.5–0.75 | 0.85–1.0 |
| task2_clean | 0.0–0.15 | 0.4–0.65 | 0.75–1.0 |
| task3_cascade | 0.0–0.1 | 0.3–0.55 | 0.65–0.9 |

---

## 8. Tech Stack

### Libraries and versions

| Library | Version | Role |
|---|---|---|
| **FastAPI** | 0.115.6 | Web framework — all 7 HTTP endpoints, auto Swagger UI at `/docs` |
| **Uvicorn** | 0.32.1 | ASGI server — runs FastAPI, handles HTTP connections |
| **Pydantic** | 2.10.4 | Data validation — all request/response schemas, action format enforcement |
| **OpenAI SDK** | ≥1.58.0 | LLM client in `inference.py` — connects to HuggingFace router |
| **Requests** | 2.32.3 | HTTP client in `inference.py` — calls ClimateWatch server endpoints |
| **NumPy** | 2.2.1 | Scenario generation — statistical operations for realistic sensor data |
| **python-dotenv** | 1.0.1 | Loads `.env` file for local dev (no need to set env vars manually) |

### Dev tools

| Tool | Version | Role |
|---|---|---|
| **uv** | ≥0.9 | Package manager — replaces pip + venv, 10-100× faster installs |
| **taskipy** | ≥1.13.0 | Task runner — enables `uv run task serve`, `uv run task test` etc. |
| **pytest** | ≥8.0.0 | Test framework — runs all 63 tests |
| **httpx** | ≥0.27.0 | HTTP client used by FastAPI `TestClient` in tests |
| **pytest-asyncio** | ≥0.23.0 | Async test support |
| **ruff** | ≥0.3.0 | Linter and formatter |

### Why these choices (vs alternatives)

- **FastAPI over Flask:** automatic Pydantic validation on every request, auto `/docs` Swagger UI, async-native — no extra work to validate action schemas
- **Pydantic v2 over dataclasses:** field-level validation with clear error messages, literal type enforcement for fault/fix types
- **uv over pip:** single command setup, lockfile reproducibility, built-in task runner via taskipy
- **No database, no ML models:** server starts in <2 seconds, uses <200 MB RAM, runs on 2 vCPU / 8 GB RAM easily
- **Seeded RNG for scenarios:** deterministic (same seed = same scenario every time), infinite variety, no large JSON files

---

## 9. File Structure

```
climatewatch-env/
│
├── inference.py              ← MANDATORY: baseline LLM agent
│                                Uses OpenAI client → HuggingFace router
│                                Runs all 3 tasks, prints scores
│                                Must complete in < 20 minutes
│
├── openenv.yaml              ← MANDATORY: environment metadata
│                                name, version, 3 tasks with difficulties
│                                Tags include "openenv"
│
├── Dockerfile                ← MANDATORY: container definition
│                                FROM python:3.11-slim, port 7860
│                                Uses uv for fast package install
│
├── requirements.txt          ← MANDATORY: package list for Docker + judges
│                                fastapi==0.115.6, pydantic==2.10.4, etc.
│
├── pyproject.toml            ← Local dev config (uv + taskipy)
│                                uv run task serve / test / lint / infer
│
├── README.md                 ← This file (also HF Spaces header)
│
├── DEPLOYMENT_GUIDE.md       ← Step-by-step HuggingFace deployment guide
│
├── scripts/
│   └── kill_port.py          ← Kills any process on port 7860
│                                Runs automatically before server starts
│
└── app/
    ├── __init__.py
    │
    ├── main.py               ← All 7 HTTP endpoints
    │                            GET  /health  /state  /tasks  /
    │                            POST /reset   /step   /grader  /baseline
    │
    ├── environment.py        ← Episode manager
    │                            reset() → step() → state() → final_grade()
    │                            Thread-safe (RLock)
    │                            Early stop at episode_score ≥ 0.80
    │
    ├── models.py             ← All Pydantic schemas
    │                            SensorObservation, SensorState
    │                            DetectAction, CleanAction, CascadeAction
    │                            ResetRequest, StepRequest, GraderResponse
    │
    ├── reward.py             ← Per-step reward function
    │                            Anti-loop penalty (−0.30)
    │                            Regression penalty (−0.05)
    │                            Always clamps to [0.0, 1.0]
    │
    ├── data/                 ← Data directory (scenarios generated in code)
    │
    └── tasks/
        ├── __init__.py       ← Task registry: task_id → loader + grader
        │
        ├── task1_detect.py   ← 20 deterministic scenarios
        │                        Sensors: CO2, NO2, CH4, O3, PM2.5, SO2, Temp, Humidity
        │                        Grader: F1 score + calibration bonus
        │
        ├── task2_clean.py    ← 10 scenarios (5 sensors × 7 days)
        │                        Grader: 60% fault type + 40% fix appropriateness
        │
        └── task3_cascade.py  ← 5 cascade failure scenarios (10 sensors × 30 days)
                                 Grader: 4-component weighted score
```

---

## 10. Environment Variables

| Variable | Default | Secret? | Purpose |
|---|---|---|---|
| `HF_TOKEN` | — | **Yes** | HuggingFace API key — used by `inference.py` to call LLM models |
| `MODEL_NAME` | — | No | Which LLM model to use (e.g. `meta-llama/Llama-3.3-70B-Instruct`) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No | LLM endpoint — HF router for free inference |
| `API_KEY` | — | Yes | Fallback API key (if HF_TOKEN not set) |
| `ENV_URL` | `http://localhost:7860` | No | ClimateWatch server URL (used by `inference.py`) |

**The server itself needs zero environment variables to run.**
Only `inference.py` (the baseline agent) needs `HF_TOKEN` and `MODEL_NAME`.

For local development, create a `.env` file:
```
HF_TOKEN=hf_your_token_here
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
ENV_URL=http://localhost:7860
```

---

## 11. Baseline Scores

Tested with `meta-llama/Llama-3.3-70B-Instruct` at `temperature=0.0`, `seed=42`:

| Task | Score | Notes |
|---|---|---|
| task1_detect | ~0.72 | Most fault types detected. Occasionally misses stuck sensors or mislabels drift as bias |
| task2_clean | ~0.58 | Good fault identification. Fix parameters (offset values, drift rates) often approximate |
| task3_cascade | ~0.41 | Partially identifies root causes. Repair order sometimes violates dependencies |
| **Average** | **~0.57** | Typical for a 70B model with no fine-tuning |

*Expected with GPT-4: 0.65–0.85 on task1, 0.55–0.75 on task2, 0.40–0.65 on task3.*

---

## 12. Deploy to HuggingFace

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for the full step-by-step process.

**Quick summary:**

```bash
# 1. Create a Docker Space at huggingface.co/new-space

# 2. Push your code
git remote add huggingface https://huggingface.co/spaces/USERNAME/climatewatch-env
git push huggingface master

# 3. Set these in Space Settings → Variables and Secrets:
#    HF_TOKEN    = hf_...  (Secret)
#    MODEL_NAME  = meta-llama/Llama-3.3-70B-Instruct
#    API_BASE_URL= https://router.huggingface.co/v1
#    ENV_URL     = http://localhost:7860

# 4. HuggingFace auto-builds Docker → server goes live in 2-5 minutes

# 5. Test your live space:
curl https://USERNAME-climatewatch-env.hf.space/health
```

---

*ClimateWatch — Environmental Sensor Data Quality & Compliance Monitoring*
*ALGORITHMIC AVENGERS — Ramana Ganthan S + Sre Sandhya K*
