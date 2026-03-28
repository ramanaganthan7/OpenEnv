"""
ClimateWatch — FastAPI Application
====================================
All required OpenEnv endpoints:
  POST /reset      → start episode
  POST /step       → submit action
  GET  /state      → current episode state
  GET  /health     → liveness probe
  GET  /tasks      → task catalog with action schemas
  POST /grader     → final score for completed episode
  POST /baseline   → run inference.py and return scores
  GET  /           → interactive dashboard UI
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.models import (
    SensorObservation, SensorState,
    ResetRequest, StepRequest,
    GraderResponse, BaselineResponse,
)
from app.environment import ClimateWatchEnvironment

# ── App & shared environment instance ────────────────────────────────────────
app = FastAPI(
    title="ClimateWatch — Environmental Sensor AI Environment",
    description=(
        "An OpenEnv-compatible reinforcement learning environment for training "
        "AI agents on real-world environmental sensor data quality tasks. "
        "Simulates fault detection, data cleaning, and EPA compliance auditing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
env = ClimateWatchEnvironment()


# ── POST /reset ───────────────────────────────────────────────────────────────

@app.post(
    "/reset",
    response_model=SensorObservation,
    summary="Start a new episode",
    description="Initialise a fresh episode for the given task. Returns the first observation.",
)
def reset(req: ResetRequest) -> SensorObservation:
    try:
        return env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── POST /step ────────────────────────────────────────────────────────────────

@app.post(
    "/step",
    response_model=SensorObservation,
    summary="Submit an action",
    description=(
        "Send the agent's analysis/action for the current step. "
        "Returns reward, feedback, and updated observation."
    ),
)
def step(req: StepRequest) -> SensorObservation:
    try:
        return env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── GET /state ────────────────────────────────────────────────────────────────

@app.get(
    "/state",
    response_model=SensorState,
    summary="Current episode state",
    description="Returns episode_id, task_id, step_count, total_reward, done.",
)
def state() -> SensorState:
    return env.state()


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 + {status: healthy} when server is running.",
)
def health():
    return {"status": "healthy"}


# ── GET /tasks ────────────────────────────────────────────────────────────────

@app.get(
    "/tasks",
    summary="Task catalog",
    description="List all 3 tasks with difficulty, description, and full action schema.",
)
def tasks():
    return {
        "tasks": [
            {
                "id": "task1_detect",
                "name": "Single Sensor Anomaly Detection",
                "difficulty": "easy",
                "description": (
                    "Receive 24 hours of readings from one sensor. "
                    "Identify every faulty hour and classify the fault type. "
                    "Faults: outlier, stuck, missing, drift, spike, bias."
                ),
                "max_steps": 5,
                "action_schema": {
                    "sensor_id": "string — must match the sensor_id in the observation",
                    "flags": [
                        {
                            "hour":       "integer 0–23",
                            "fault":      "outlier | stuck | missing | drift | spike | bias",
                            "confidence": "float 0.0–1.0",
                        }
                    ],
                },
                "example_action": {
                    "sensor_id": "CO2-100",
                    "flags": [
                        {"hour": 3,  "fault": "stuck",   "confidence": 0.95},
                        {"hour": 6,  "fault": "outlier", "confidence": 1.0},
                        {"hour": 8,  "fault": "missing", "confidence": 1.0},
                        {"hour": 9,  "fault": "missing", "confidence": 1.0},
                    ],
                },
            },
            {
                "id": "task2_clean",
                "name": "Multi-Sensor Data Stream Cleaning",
                "difficulty": "medium",
                "description": (
                    "Receive 7 days of data from 5 sensors in a network. "
                    "Diagnose each sensor's fault type, severity, and the correct fix. "
                    "One sensor is always clean — do not over-diagnose."
                ),
                "max_steps": 5,
                "action_schema": {
                    "diagnoses": [
                        {
                            "sensor_id":  "string — e.g. S1, S2, ..., S5",
                            "fault_type": "drift | missing | bias | noise | stuck | spike | valid",
                            "severity":   "none | low | medium | high | critical",
                            "fix":        "no_action | interpolate | recalibrate | offset_correction | smooth | flag_only | replace",
                            "fix_params": "object — e.g. {offset: -12.0} or {drift_rate_per_day: 0.8}",
                        }
                    ]
                },
                "example_action": {
                    "diagnoses": [
                        {"sensor_id": "S1", "fault_type": "drift",   "severity": "high",    "fix": "recalibrate",       "fix_params": {"drift_rate_per_day": 0.8}},
                        {"sensor_id": "S2", "fault_type": "missing", "severity": "medium",  "fix": "interpolate",       "fix_params": {"method": "linear"}},
                        {"sensor_id": "S3", "fault_type": "bias",    "severity": "high",    "fix": "offset_correction", "fix_params": {"offset": -12.0}},
                        {"sensor_id": "S4", "fault_type": "noise",   "severity": "medium",  "fix": "smooth",            "fix_params": {"window": 3}},
                        {"sensor_id": "S5", "fault_type": "valid",   "severity": "none",    "fix": "no_action",         "fix_params": {}},
                    ]
                },
            },
            {
                "id": "task3_cascade",
                "name": "Cascade Failure & Compliance Audit",
                "difficulty": "hard",
                "description": (
                    "Analyse a 30-day 10-sensor network with cascade failures. "
                    "Reference sensors fail and corrupt downstream (dependent) sensors. "
                    "Find root causes, correct repair order, fault window, "
                    "and EPA regulatory compliance status."
                ),
                "max_steps": 5,
                "action_schema": {
                    "root_cause_sensors": "list[string] — sensor IDs that ACTUALLY failed",
                    "repair_order":       "list[string] — ordered repair sequence (references before dependents)",
                    "fault_window_start": "string — format: 'day_N' e.g. 'day_8'",
                    "fault_window_end":   "string — format: 'day_N' e.g. 'day_21'",
                    "compliance_checks": [
                        {
                            "parameter":  "string — e.g. CH4_ppm, NOx_ppb",
                            "status":     "CLEAN | POSSIBLE_VIOLATION | CONFIRMED_VIOLATION | INSUFFICIENT_DATA",
                            "confidence": "float 0.0–1.0",
                            "reasoning":  "string — justify your compliance judgement",
                        }
                    ],
                    "recommended_action": "no_action | flag_for_review | file_compliance_report | emergency_shutdown",
                },
                "example_action": {
                    "root_cause_sensors": ["S1", "S3"],
                    "repair_order":       ["S1", "S3", "S4", "S5", "S6", "S7", "S8"],
                    "fault_window_start": "day_8",
                    "fault_window_end":   "day_21",
                    "compliance_checks": [
                        {"parameter": "CH4_ppm",  "status": "POSSIBLE_VIOLATION",
                         "confidence": 0.75,
                         "reasoning": "CH4 sensors S1,S4,S5 were corrupted during fault window. Cannot confirm compliance."},
                        {"parameter": "NOx_ppb",  "status": "CLEAN",
                         "confidence": 0.90,
                         "reasoning": "NOx sensors S9,S10 are independent and showed clean readings throughout."},
                    ],
                    "recommended_action": "flag_for_review",
                },
            },
        ]
    }


# ── POST /grader ──────────────────────────────────────────────────────────────

@app.post(
    "/grader",
    response_model=GraderResponse,
    summary="Score the completed episode",
    description=(
        "Call after the episode ends (done=true). "
        "Returns the final grader score 0.0–1.0 for the last action submitted."
    ),
)
def grader() -> GraderResponse:
    s = env.state()
    if s.episode_id is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    final_score = env.final_grade()
    return GraderResponse(
        episode_id=s.episode_id,
        task_id=s.task_id,
        final_score=round(final_score, 4),
        step_count=s.step_count,
        breakdown={
            "total_reward":    s.total_reward,
            "steps_used":      s.step_count,
            "done":            s.done,
        },
    )


# ── POST /baseline ────────────────────────────────────────────────────────────

@app.post(
    "/baseline",
    response_model=BaselineResponse,
    summary="Run the baseline inference script",
    description=(
        "Runs inference.py with the configured LLM and returns scores for all 3 tasks. "
        "Requires ENV_URL, MODEL_NAME, HF_TOKEN environment variables."
    ),
)
def baseline() -> BaselineResponse:
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        timeout=1200,   # 20 minute hard limit
        cwd="/app",
    )
    return BaselineResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


# ── GET / — Interactive Dashboard ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ClimateWatch — Environmental Sensor AI Environment</title>
  <style>
    :root {
      --bg: #080e08; --bg2: #0d160d; --bg3: #112011;
      --green: #00ff88; --blue: #00ccff; --amber: #ffaa00;
      --red: #ff4466; --text: #c8e6c9; --border: #1e3a1e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Courier New', monospace; background: var(--bg);
           color: var(--text); padding: 1.5rem; line-height: 1.6; }
    h1 { color: var(--green); font-size: 1.5rem; margin-bottom: 0.3rem; }
    .subtitle { color: var(--blue); font-size: 0.85rem; margin-bottom: 1.5rem; opacity: 0.8; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .card { border: 1px solid var(--border); border-radius: 6px;
            padding: 1rem; background: var(--bg2); }
    .card h3 { color: var(--blue); margin-bottom: 0.7rem; font-size: 0.95rem; }
    .task-btn {
      display: block; width: 100%; padding: 0.6rem 1rem;
      margin: 0.3rem 0; background: var(--bg3); color: var(--green);
      border: 1px solid var(--border); border-radius: 4px;
      cursor: pointer; font-family: monospace; font-size: 0.88rem;
      text-align: left; transition: all 0.15s;
    }
    .task-btn:hover { background: var(--border); border-color: var(--green); }
    .task-btn .badge {
      float: right; font-size: 0.75rem; padding: 0.1rem 0.5rem;
      border-radius: 10px; font-weight: bold;
    }
    .easy   { background: #1a3a1a; color: var(--green); }
    .medium { background: #3a2a00; color: var(--amber); }
    .hard   { background: #3a1020; color: var(--red); }
    pre {
      background: var(--bg); padding: 0.8rem; border-radius: 4px;
      overflow: auto; max-height: 350px; font-size: 0.78rem;
      border: 1px solid var(--border); white-space: pre-wrap;
    }
    .links a {
      display: inline-block; color: var(--blue); text-decoration: none;
      margin: 0.2rem 0.4rem 0.2rem 0; padding: 0.3rem 0.7rem;
      border: 1px solid var(--border); border-radius: 4px; font-size: 0.83rem;
    }
    .links a:hover { border-color: var(--blue); }
    .status-bar {
      display: flex; gap: 1rem; padding: 0.5rem 1rem;
      background: var(--bg3); border: 1px solid var(--border);
      border-radius: 4px; margin-bottom: 1rem; font-size: 0.82rem;
    }
    .status-bar span { color: var(--green); }
    .full { grid-column: 1 / -1; }
    @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>🌍 ClimateWatch — Environmental Sensor AI Environment</h1>
  <p class="subtitle">OpenEnv benchmark for AI agents on real-world climate sensor data quality tasks</p>

  <div class="status-bar" id="statusBar">
    <span>Episode: —</span>
    <span>Task: —</span>
    <span>Step: 0</span>
    <span>Reward: 0.0000</span>
    <span>Done: false</span>
  </div>

  <div class="grid">
    <div class="card">
      <h3>▶ Start Episode</h3>
      <button class="task-btn" onclick="startTask('task1_detect')">
        Task 1 — Single Sensor Anomaly Detection
        <span class="badge easy">EASY</span>
      </button>
      <button class="task-btn" onclick="startTask('task2_clean')">
        Task 2 — Multi-Sensor Data Cleaning
        <span class="badge medium">MEDIUM</span>
      </button>
      <button class="task-btn" onclick="startTask('task3_cascade')">
        Task 3 — Cascade Failure &amp; Compliance
        <span class="badge hard">HARD</span>
      </button>
    </div>

    <div class="card">
      <h3>🔗 Quick Links</h3>
      <div class="links">
        <a href="/docs">Swagger UI</a>
        <a href="/redoc">ReDoc</a>
        <a href="/tasks">Task Catalog</a>
        <a href="/health">Health</a>
        <a href="/state">State</a>
      </div>
      <br>
      <h3>📊 Environment Stats</h3>
      <pre id="envStats">Loading...</pre>
    </div>

    <div class="card full">
      <h3>📡 Last Observation</h3>
      <pre id="output">Click a task to start an episode...</pre>
    </div>
  </div>

<script>
const out = () => document.getElementById('output');
const stats = () => document.getElementById('envStats');
const bar = () => document.getElementById('statusBar');

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

async function startTask(tid) {
  const obs = await api('POST', '/reset', {task_id: tid});
  out().textContent = JSON.stringify(obs, null, 2);
  await refreshState();
}

async function refreshState() {
  try {
    const s = await api('GET', '/state');
    const spans = bar().querySelectorAll('span');
    spans[0].textContent = 'Episode: ' + (s.episode_id ? s.episode_id.slice(0,8)+'...' : '—');
    spans[1].textContent = 'Task: ' + (s.task_id || '—');
    spans[2].textContent = 'Step: ' + s.step_count;
    spans[3].textContent = 'Reward: ' + s.total_reward;
    spans[4].textContent = 'Done: ' + s.done;
  } catch(e) {}
}

async function loadStats() {
  try {
    const t = await api('GET', '/tasks');
    const info = t.tasks.map(tk =>
      tk.id + ' [' + tk.difficulty + '] max_steps=' + tk.max_steps
    ).join('\\n');
    stats().textContent = 'Tasks:\\n' + info + '\\n\\nHit /docs for full API reference.';
  } catch(e) { stats().textContent = 'Error loading stats.'; }
}

loadStats();
refreshState();
setInterval(refreshState, 5000);
</script>
</body>
</html>"""
