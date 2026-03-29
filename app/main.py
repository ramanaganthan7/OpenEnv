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
import pathlib
from typing import Optional

# Project root = parent of the app/ directory (works on Windows, Linux, Docker)
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

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
        cwd=str(_PROJECT_ROOT),   # works on Windows locally and /app in Docker
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
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #f4f6f8;
      --surface: #ffffff;
      --raised:  #f0f4f0;
      --border:  #d8e4d8;
      --border2: #e4ede4;
      --green:   #1a7a40;
      --green2:  #2e9e58;
      --blue:    #1565a8;
      --amber:   #b45d00;
      --red:     #c0323e;
      --muted:   #7a927a;
      --text:    #1a2e1a;
      --text2:   #4a6a4a;
    }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ── */
    header {
      background: var(--green);
      border-bottom: 3px solid #155e32;
      padding: 0.9rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .brand { display: flex; flex-direction: column; }
    .brand-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.02em;
    }
    .brand-sub {
      font-size: 0.73rem;
      color: #b2dfc2;
      margin-top: 1px;
    }
    .header-links { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .header-links a {
      font-size: 0.78rem;
      color: #ffffff;
      text-decoration: none;
      padding: 0.25rem 0.7rem;
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 4px;
      background: rgba(255,255,255,0.12);
      transition: background 0.15s;
    }
    .header-links a:hover { background: rgba(255,255,255,0.25); }

    /* ── Episode bar ── */
    #episodeBar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0.45rem 1.5rem;
      display: flex;
      gap: 2rem;
      font-size: 0.78rem;
      color: var(--text2);
      flex-wrap: wrap;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .ep-item { display: flex; gap: 0.4rem; align-items: center; }
    .ep-label { color: var(--muted); }
    .ep-val   { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
    .ep-val.good { color: var(--green); }
    .ep-val.done { color: var(--red); }

    /* ── Main layout ── */
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      flex: 1;
      min-height: 0;
    }

    /* ── Sidebar ── */
    aside {
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      overflow-y: auto;
    }
    .section-label {
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }

    /* Task buttons */
    .task-btn {
      display: block;
      width: 100%;
      padding: 0.65rem 0.9rem;
      margin-bottom: 0.4rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      cursor: pointer;
      text-align: left;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
      color: var(--text);
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .task-btn:hover { border-color: var(--green2); background: #f0faf4; box-shadow: 0 2px 6px rgba(26,122,64,0.1); }
    .task-btn.active { border-color: var(--green); background: #e8f5ee; border-left: 3px solid var(--green); }
    .task-btn-name { font-size: 0.82rem; font-weight: 600; display: block; margin-bottom: 0.2rem; color: var(--text); }
    .task-btn-desc { font-size: 0.71rem; color: var(--text2); display: block; }
    .badge {
      display: inline-block;
      font-size: 0.62rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 0.1rem 0.45rem;
      border-radius: 3px;
      margin-left: 0.4rem;
      vertical-align: middle;
    }
    .easy   { background: #d4f0df; color: #1a7a40; }
    .medium { background: #fdebd0; color: #b45d00; }
    .hard   { background: #fad4d8; color: #c0323e; }

    /* Score meters */
    .score-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      margin-bottom: 0.5rem;
    }
    .score-label { color: var(--text2); }
    .score-num   { font-weight: 700; color: var(--green); font-variant-numeric: tabular-nums; }
    .meter {
      height: 6px;
      background: var(--border2);
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 0.8rem;
    }
    .meter-fill {
      height: 100%;
      border-radius: 3px;
      background: var(--green2);
      transition: width 0.4s ease;
    }

    /* Sidebar info box */
    .info-box {
      background: var(--raised);
      border: 1px solid var(--border2);
      border-radius: 6px;
      padding: 0.7rem;
      font-size: 0.75rem;
      color: var(--text2);
      line-height: 1.65;
    }
    .info-box strong { color: var(--text); }

    /* ── Main panel ── */
    main {
      padding: 1.2rem 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
      background: var(--bg);
    }

    /* Panel cards */
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .panel-header {
      padding: 0.6rem 1rem;
      background: var(--raised);
      border-bottom: 1px solid var(--border);
      font-size: 0.73rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .panel-body { padding: 1rem; }

    /* Sensor info strip */
    .sensor-meta {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
      font-size: 0.78rem;
      padding: 0.7rem 0.8rem;
      background: var(--raised);
      border: 1px solid var(--border2);
      border-radius: 6px;
    }
    .sensor-meta-item { display: flex; flex-direction: column; gap: 2px; }
    .sensor-meta-item .k { color: var(--muted); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .sensor-meta-item .v { color: var(--text); font-weight: 600; }

    /* Readings table */
    .readings-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
    }
    .readings-table th {
      text-align: left;
      padding: 0.4rem 0.7rem;
      color: var(--muted);
      font-size: 0.67rem;
      font-weight: 700;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      border-bottom: 2px solid var(--border);
      background: var(--raised);
    }
    .readings-table td {
      padding: 0.32rem 0.7rem;
      border-bottom: 1px solid var(--border2);
      font-variant-numeric: tabular-nums;
    }
    .readings-table tr:hover td { background: #f8fbf8; }
    .readings-table tr:last-child td { border-bottom: none; }
    .val-normal  { color: var(--text); }
    .val-missing { color: var(--muted); font-style: italic; }
    .val-anomaly { color: var(--amber); font-weight: 600; }
    .val-outlier { color: var(--red);   font-weight: 700; }

    /* Inline bar */
    .bar-cell { width: 120px; }
    .inline-bar {
      height: 8px;
      background: var(--border2);
      border-radius: 2px;
      overflow: hidden;
    }
    .inline-bar-fill {
      height: 100%;
      border-radius: 2px;
      background: var(--green2);
      transition: width 0.3s;
    }
    .inline-bar-fill.warn { background: var(--amber); }
    .inline-bar-fill.crit { background: var(--red); }

    /* Feedback box */
    .feedback-box {
      padding: 0.7rem 1rem;
      border-left: 4px solid var(--green2);
      background: #f0faf4;
      border-radius: 0 6px 6px 0;
      font-size: 0.82rem;
      color: var(--text);
      line-height: 1.55;
    }
    .feedback-box.warn { border-left-color: var(--amber); background: #fff8ee; }
    .feedback-box.bad  { border-left-color: var(--red);   background: #fff0f0; }

    /* Multi-sensor grid */
    .sensor-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 0.8rem;
    }
    .sensor-card {
      background: var(--raised);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .sensor-card-title { font-size: 0.8rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; border-bottom: 1px solid var(--border2); padding-bottom: 0.4rem; }
    .sensor-stat { display: flex; justify-content: space-between; font-size: 0.73rem; margin-bottom: 0.25rem; }
    .sensor-stat .sk { color: var(--muted); }
    .sensor-stat .sv { color: var(--text); font-weight: 600; }
    .fault-tag {
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: 3px;
      margin-top: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .ft-valid   { background: #d4f0df; color: #1a7a40; }
    .ft-drift   { background: #fdebd0; color: #b45d00; }
    .ft-bias    { background: #fad4d8; color: #c0323e; }
    .ft-noise   { background: #dce8f8; color: #1565a8; }
    .ft-missing { background: #ede8e0; color: #7a5a30; }
    .ft-stuck   { background: #ece0f0; color: #7a3a9a; }
    .ft-unknown { background: #e8e8e8; color: #555555; }

    /* Raw JSON toggle */
    .raw-toggle {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text2);
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      font-size: 0.7rem;
      cursor: pointer;
    }
    .raw-toggle:hover { border-color: var(--green2); color: var(--green); }
    .raw-json {
      background: #f8fbf8;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.8rem;
      font-family: 'Courier New', monospace;
      font-size: 0.71rem;
      white-space: pre-wrap;
      overflow-x: auto;
      max-height: 300px;
      overflow-y: auto;
      margin-top: 0.8rem;
      display: none;
      color: #3a5a3a;
    }

    /* Placeholder */
    .placeholder {
      padding: 3rem;
      text-align: center;
      color: var(--muted);
      font-size: 0.85rem;
    }
    .placeholder strong { display: block; font-size: 1.05rem; color: var(--text2); margin-bottom: 0.5rem; font-weight: 600; }

    /* Loader */
    .loading { color: var(--muted); font-size: 0.82rem; padding: 1.5rem; }

    @media (max-width: 800px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-title">ClimateWatch</span>
    <span class="brand-sub">Environmental Sensor AI Environment &mdash; OpenEnv Hackathon</span>
  </div>
  <div class="header-links">
    <a href="/docs">API Docs</a>
    <a href="/tasks">Task Catalog</a>
    <a href="/health">Health</a>
    <a href="/state">State</a>
  </div>
</header>

<div id="episodeBar">
  <div class="ep-item"><span class="ep-label">Episode</span><span class="ep-val" id="ep-id">No active episode</span></div>
  <div class="ep-item"><span class="ep-label">Task</span><span class="ep-val" id="ep-task">—</span></div>
  <div class="ep-item"><span class="ep-label">Step</span><span class="ep-val" id="ep-step">0 / 5</span></div>
  <div class="ep-item"><span class="ep-label">Total Reward</span><span class="ep-val" id="ep-reward">0.0000</span></div>
  <div class="ep-item"><span class="ep-label">Status</span><span class="ep-val" id="ep-done">Idle</span></div>
</div>

<div class="layout">
  <aside>
    <div>
      <div class="section-label">Start Episode</div>
      <button class="task-btn" id="btn-t1" onclick="startTask('task1_detect')">
        <span class="task-btn-name">Task 1 <span class="badge easy">EASY</span></span>
        <span class="task-btn-desc">Single Sensor Anomaly Detection</span>
      </button>
      <button class="task-btn" id="btn-t2" onclick="startTask('task2_clean')">
        <span class="task-btn-name">Task 2 <span class="badge medium">MEDIUM</span></span>
        <span class="task-btn-desc">Multi-Sensor Data Cleaning</span>
      </button>
      <button class="task-btn" id="btn-t3" onclick="startTask('task3_cascade')">
        <span class="task-btn-name">Task 3 <span class="badge hard">HARD</span></span>
        <span class="task-btn-desc">Cascade Failure &amp; Compliance</span>
      </button>
    </div>

    <div>
      <div class="section-label">Episode Score</div>
      <div class="score-row"><span class="score-label">Last Step</span><span class="score-num" id="sc-step">—</span></div>
      <div class="meter"><div class="meter-fill" id="mtr-step" style="width:0%"></div></div>
      <div class="score-row"><span class="score-label">Total Reward</span><span class="score-num" id="sc-total">0.0000</span></div>
      <div class="meter"><div class="meter-fill" id="mtr-total" style="width:0%"></div></div>
    </div>

    <div>
      <div class="section-label">Environment</div>
      <div class="info-box" id="envInfo">
        <strong>ClimateWatch v1.0</strong><br>
        3 tasks &mdash; 35 total scenarios<br>
        Max 5 steps per episode<br>
        Early stop at score &ge; 0.80<br>
        Reward: F1 + partial credit
      </div>
    </div>

    <div>
      <div class="section-label">Fault Types</div>
      <div class="info-box">
        <strong>outlier</strong> &mdash; single extreme spike<br>
        <strong>stuck</strong> &mdash; frozen repeated value<br>
        <strong>missing</strong> &mdash; null data gap<br>
        <strong>drift</strong> &mdash; gradual baseline shift<br>
        <strong>spike</strong> &mdash; short burst anomaly<br>
        <strong>bias</strong> &mdash; constant offset error
      </div>
    </div>
  </aside>

  <main id="mainPanel">
    <div class="panel">
      <div class="panel-body">
        <div class="placeholder">
          <strong>No active episode</strong>
          Select a task from the sidebar to start an episode.<br>
          Real sensor data will appear here with live readings.
        </div>
      </div>
    </div>
  </main>
</div>

<script>
let currentObs = null;
let currentTaskId = null;

/* ── API helper ── */
async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

/* ── Start episode ── */
async function startTask(tid) {
  currentTaskId = tid;
  ['t1','t2','t3'].forEach(x => document.getElementById('btn-'+x).classList.remove('active'));
  const map = {task1_detect:'t1', task2_clean:'t2', task3_cascade:'t3'};
  document.getElementById('btn-'+map[tid]).classList.add('active');

  renderMain('<div class="loading">Loading scenario...</div>');
  const obs = await api('POST', '/reset', {task_id: tid});
  currentObs = obs;
  renderObs(obs);
  await refreshEpisodeBar();
}

/* ── Refresh episode bar ── */
async function refreshEpisodeBar() {
  try {
    const s = await api('GET', '/state');
    document.getElementById('ep-id').textContent =
      s.episode_id ? s.episode_id.slice(0,8) + '...' : 'No active episode';
    document.getElementById('ep-task').textContent = s.task_id || '—';
    document.getElementById('ep-step').textContent = s.step_count + ' / 5';
    document.getElementById('ep-reward').textContent =
      typeof s.total_reward === 'number' ? s.total_reward.toFixed(4) : '0.0000';

    const doneEl = document.getElementById('ep-done');
    if (s.done) {
      doneEl.textContent = 'Done';
      doneEl.className = 'ep-val done';
    } else if (s.episode_id) {
      doneEl.textContent = 'Running';
      doneEl.className = 'ep-val good';
    } else {
      doneEl.textContent = 'Idle';
      doneEl.className = 'ep-val';
    }

    document.getElementById('sc-total').textContent =
      typeof s.total_reward === 'number' ? s.total_reward.toFixed(4) : '0.0000';
    document.getElementById('mtr-total').style.width =
      (Math.min(1, s.total_reward / 5) * 100) + '%';
  } catch(e) {}
}

/* ── Render main panel ── */
function renderMain(html) {
  document.getElementById('mainPanel').innerHTML =
    '<div class="panel"><div class="panel-body">' + html + '</div></div>';
}

/* ── Render observation ── */
function renderObs(obs) {
  const sd = obs.sensor_data;
  if (!sd) { renderMain('<div class="loading">No sensor data returned.</div>'); return; }
  const tid = obs.task_id;

  if (tid === 'task1_detect') renderTask1(obs, sd);
  else if (tid === 'task2_clean') renderTask2(obs, sd);
  else renderTask3(obs, sd);
}

/* ── Task 1: single sensor readings table ── */
function renderTask1(obs, sd) {
  const readings = sd.readings || [];
  const normal = sd.normal_range || [0, 1000];
  const nMin = normal[0], nMax = normal[1];
  const range = nMax - nMin || 1;

  // Determine anomalies by comparing to normal range + detecting stuck values
  const values = readings.map(r => r.value);
  const validVals = values.filter(v => v !== null && v !== undefined);
  const valueCounts = {};
  values.forEach(v => { if (v !== null) valueCounts[v] = (valueCounts[v]||0)+1; });

  let rows = '';
  readings.forEach((r, i) => {
    const v = r.value;
    const isMissing = v === null || v === undefined;
    const isOutlier = !isMissing && (v < nMin * 0.5 || v > nMax * 2.5 || Math.abs(v) > 9000);
    const isStuck = !isMissing && valueCounts[v] >= 3 && readings.filter(x => x.value === v).length >= 3;
    const isHigh  = !isMissing && !isOutlier && (v > nMax || v < nMin);

    let cls = 'val-normal', barCls = '', label = '';
    let pct = 0;
    if (isMissing)      { cls = 'val-missing'; label = 'MISSING'; }
    else if (isOutlier) { cls = 'val-outlier'; barCls = 'crit'; pct = 100; label = 'OUTLIER'; }
    else if (isStuck)   { cls = 'val-anomaly'; barCls = 'warn'; pct = 50; label = 'STUCK'; }
    else if (isHigh)    { cls = 'val-anomaly'; barCls = 'warn';
                          pct = Math.min(100, Math.abs(v - nMax) / range * 100 + 60);
                        }
    else { pct = Math.min(100, Math.max(0, (v - nMin) / range * 100)); }

    rows += '<tr>' +
      '<td>' + r.hour.toString().padStart(2,'0') + ':00</td>' +
      '<td class="' + cls + '">' + (isMissing ? 'null' : v) + '</td>' +
      '<td>' + sd.unit + '</td>' +
      '<td class="bar-cell"><div class="inline-bar"><div class="inline-bar-fill ' + barCls + '" style="width:' + pct.toFixed(0) + '%"></div></div></td>' +
      '<td style="font-size:0.68rem;color:var(--' + (label?'amber':'muted') + ')">' + label + '</td>' +
    '</tr>';
  });

  const feedCls = obs.reward > 0.7 ? '' : obs.reward > 0.3 ? 'warn' : 'bad';
  const epScore = obs.metadata && obs.metadata.episode_score !== undefined
    ? obs.metadata.episode_score : null;

  if (epScore !== null) {
    document.getElementById('sc-step').textContent = epScore.toFixed(4);
    document.getElementById('mtr-step').style.width = (epScore * 100) + '%';
  }

  const main = document.getElementById('mainPanel');
  main.innerHTML = `
    <div class="panel">
      <div class="panel-header">
        <span>Sensor Readings &mdash; ${sd.sensor_id || ''}</span>
        <button class="raw-toggle" onclick="toggleRaw(this)">Show Raw JSON</button>
      </div>
      <div class="panel-body">
        <div class="sensor-meta">
          <div class="sensor-meta-item"><span class="k">Sensor ID</span><span class="v">${sd.sensor_id||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Parameter</span><span class="v">${sd.parameter||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Unit</span><span class="v">${sd.unit||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Normal Range</span><span class="v">${nMin} &ndash; ${nMax} ${sd.unit||''}</span></div>
          <div class="sensor-meta-item"><span class="k">Location</span><span class="v">${sd.location||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Readings</span><span class="v">${readings.length} hours</span></div>
        </div>
        <table class="readings-table">
          <thead><tr><th>Time</th><th>Value</th><th>Unit</th><th>Level</th><th>Flag</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        ${obs.feedback ? '<div style="margin-top:1rem"><div class="feedback-box ' + feedCls + '">' + obs.feedback + '</div></div>' : ''}
        <pre class="raw-json">${JSON.stringify(obs, null, 2)}</pre>
      </div>
    </div>`;
}

/* ── Task 2: multi-sensor cards ── */
function renderTask2(obs, sd) {
  const sensors = sd.sensors || [];
  const epScore = obs.metadata && obs.metadata.episode_score !== undefined
    ? obs.metadata.episode_score : null;
  if (epScore !== null) {
    document.getElementById('sc-step').textContent = epScore.toFixed(4);
    document.getElementById('mtr-step').style.width = (epScore * 100) + '%';
  }

  let cards = '';
  sensors.forEach(s => {
    const st = s.stats || {};
    const ds = s.daily_summaries || [];
    const means = ds.map(d => d.mean).filter(v => v !== null);
    const minV = means.length ? Math.min(...means).toFixed(2) : '—';
    const maxV = means.length ? Math.max(...means).toFixed(2) : '—';
    const nr = s.normal_range || [];

    // Guess fault from stats
    let faultHint = 'unknown';
    if (st.total_missing_hours > 20) faultHint = 'missing';
    else if (st.missing_pct > 10) faultHint = 'missing';
    else if (parseFloat(st.std) > 8) faultHint = 'noise';
    else {
      const trend = (st.trend_7day || '').toLowerCase();
      if (trend.includes('+') && parseFloat(trend) > 1) faultHint = 'drift';
      else if (trend.includes('-') && Math.abs(parseFloat(trend)) > 1) faultHint = 'drift';
    }
    // Check if any day shows constant value
    const dsMeans = ds.map(d => d.mean).filter(v=>v!==null);
    const uniqueMeans = new Set(dsMeans.map(v => v.toFixed(2)));
    if (uniqueMeans.size <= 2 && dsMeans.length > 3) faultHint = 'stuck';

    const ftClass = 'ft-' + faultHint;

    cards += `<div class="sensor-card">
      <div class="sensor-card-title">${s.sensor_id} &mdash; ${s.parameter||''}</div>
      <div class="sensor-stat"><span class="sk">Unit</span><span class="sv">${s.unit||'—'}</span></div>
      <div class="sensor-stat"><span class="sk">Normal range</span><span class="sv">${nr[0]||'—'} &ndash; ${nr[1]||'—'}</span></div>
      <div class="sensor-stat"><span class="sk">7-day mean</span><span class="sv">${st.overall_mean !== undefined ? st.overall_mean : '—'}</span></div>
      <div class="sensor-stat"><span class="sk">7-day min / max</span><span class="sv">${minV} / ${maxV}</span></div>
      <div class="sensor-stat"><span class="sk">Missing hours</span><span class="sv">${st.total_missing_hours !== undefined ? st.total_missing_hours : '—'} (${st.missing_pct !== undefined ? st.missing_pct : '—'}%)</span></div>
      <div class="sensor-stat"><span class="sk">Trend</span><span class="sv">${st.trend_7day || '—'}</span></div>
      <span class="fault-tag ${ftClass}">${faultHint}</span>
    </div>`;
  });

  const feedCls = obs.reward > 0.7 ? '' : obs.reward > 0.3 ? 'warn' : 'bad';
  document.getElementById('mainPanel').innerHTML = `
    <div class="panel">
      <div class="panel-header">
        <span>Network &mdash; ${sd.network_id||''} &mdash; ${sd.location||''}</span>
        <button class="raw-toggle" onclick="toggleRaw(this)">Show Raw JSON</button>
      </div>
      <div class="panel-body">
        <div class="sensor-meta" style="margin-bottom:1rem">
          <div class="sensor-meta-item"><span class="k">Network</span><span class="v">${sd.network_id||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Period</span><span class="v">${sd.period_days||7} days</span></div>
          <div class="sensor-meta-item"><span class="k">Sensors</span><span class="v">${sensors.length}</span></div>
          <div class="sensor-meta-item"><span class="k">Reference Station</span><span class="v">${(sd.reference_station||{}).id||'—'}</span></div>
        </div>
        <div class="sensor-grid">${cards}</div>
        ${obs.feedback ? '<div style="margin-top:1rem"><div class="feedback-box ' + feedCls + '">' + obs.feedback + '</div></div>' : ''}
        <pre class="raw-json">${JSON.stringify(obs, null, 2)}</pre>
      </div>
    </div>`;
}

/* ── Task 3: cascade network ── */
function renderTask3(obs, sd) {
  const sensors = sd.sensors || [];
  const dg = sd.dependency_graph || {};
  const thresholds = sd.regulatory_thresholds || {};
  const facts = sd.known_facts || [];

  const epScore = obs.metadata && obs.metadata.episode_score !== undefined
    ? obs.metadata.episode_score : null;
  if (epScore !== null) {
    document.getElementById('sc-step').textContent = epScore.toFixed(4);
    document.getElementById('mtr-step').style.width = (epScore * 100) + '%';
  }

  // Build dependency rows
  let depRows = '';
  Object.entries(dg).forEach(([sid, info]) => {
    const role = info.role || (info.independent ? 'independent' : 'dependent');
    const cals = (info.calibrates || []).join(', ') || '—';
    const calBy = info.calibrated_by || '—';
    depRows += `<tr>
      <td>${sid}</td>
      <td style="color:var(--${role==='reference'?'amber':role==='independent'?'blue':'text2'})">${role}</td>
      <td>${cals}</td>
      <td>${calBy}</td>
    </tr>`;
  });

  // Sensor status
  let sensorRows = '';
  sensors.forEach(s => {
    const st = s.stats || {};
    const offline = st.offline_days || 0;
    const corrupted = st.corrupted_days || 0;
    const quality = st.data_quality_pct || 0;
    const cls = offline > 0 ? 'val-outlier' : corrupted > 0 ? 'val-anomaly' : 'val-normal';
    const status = offline > 0 ? 'OFFLINE' : corrupted > 0 ? 'CORRUPTED' : 'OK';
    sensorRows += `<tr>
      <td>${s.sensor_id}</td>
      <td>${s.parameter||'—'}</td>
      <td>${st.offline_days||0} days</td>
      <td>${st.corrupted_days||0} days</td>
      <td class="bar-cell"><div class="inline-bar"><div class="inline-bar-fill ${quality<70?'crit':quality<90?'warn':''}" style="width:${quality}%"></div></div></td>
      <td class="${cls}" style="font-size:0.7rem">${status}</td>
    </tr>`;
  });

  let factsHtml = facts.map(f => `<div style="padding:0.3rem 0;border-bottom:1px solid var(--border2);font-size:0.78rem">${f}</div>`).join('');

  let threshHtml = Object.entries(thresholds).map(([p, t]) =>
    `<div class="sensor-stat"><span class="sk">${p}</span><span class="sv">warn &ge;${t.warning} / viol &ge;${t.violation}</span></div>`
  ).join('');

  const feedCls = obs.reward > 0.7 ? '' : obs.reward > 0.3 ? 'warn' : 'bad';

  document.getElementById('mainPanel').innerHTML = `
    <div class="panel">
      <div class="panel-header">
        <span>Cascade Network &mdash; ${sd.network_id||''} &mdash; ${sd.location||''}</span>
        <button class="raw-toggle" onclick="toggleRaw(this)">Show Raw JSON</button>
      </div>
      <div class="panel-body">
        <div class="sensor-meta" style="margin-bottom:1rem">
          <div class="sensor-meta-item"><span class="k">Network</span><span class="v">${sd.network_id||'—'}</span></div>
          <div class="sensor-meta-item"><span class="k">Period</span><span class="v">${sd.period_days||30} days</span></div>
          <div class="sensor-meta-item"><span class="k">Sensors</span><span class="v">${sensors.length}</span></div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
          <div>
            <div class="section-label" style="margin-bottom:0.4rem">Regulatory Thresholds (EPA)</div>
            <div class="info-box">${threshHtml||'None specified'}</div>
          </div>
          <div>
            <div class="section-label" style="margin-bottom:0.4rem">Known Facts</div>
            <div class="info-box">${factsHtml||'None'}</div>
          </div>
        </div>

        <div class="section-label" style="margin-bottom:0.5rem">Dependency Graph</div>
        <table class="readings-table" style="margin-bottom:1rem">
          <thead><tr><th>Sensor</th><th>Role</th><th>Calibrates</th><th>Calibrated By</th></tr></thead>
          <tbody>${depRows}</tbody>
        </table>

        <div class="section-label" style="margin-bottom:0.5rem">30-Day Sensor Status</div>
        <table class="readings-table">
          <thead><tr><th>Sensor</th><th>Parameter</th><th>Offline</th><th>Corrupted</th><th>Data Quality</th><th>Status</th></tr></thead>
          <tbody>${sensorRows}</tbody>
        </table>
        ${obs.feedback ? '<div style="margin-top:1rem"><div class="feedback-box ' + feedCls + '">' + obs.feedback + '</div></div>' : ''}
        <pre class="raw-json">${JSON.stringify(obs, null, 2)}</pre>
      </div>
    </div>`;
}

/* ── Toggle raw JSON ── */
function toggleRaw(btn) {
  const pre = btn.closest('.panel').querySelector('.raw-json');
  if (pre.style.display === 'block') {
    pre.style.display = 'none';
    btn.textContent = 'Show Raw JSON';
  } else {
    pre.style.display = 'block';
    btn.textContent = 'Hide Raw JSON';
  }
}

/* ── Auto-refresh state bar every 5s ── */
setInterval(refreshEpisodeBar, 5000);
refreshEpisodeBar();
</script>
</body>
</html>"""
