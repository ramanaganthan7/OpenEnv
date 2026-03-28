"""
ClimateWatch — Baseline Inference Script
=========================================
MANDATORY FILE: inference.py — must be in root directory.

Runs all 3 tasks using an LLM via the HuggingFace router.
Uses OpenAI client pointed at API_BASE_URL (not openai.com).

Environment variables (all required):
  API_BASE_URL  →  LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME    →  Which model to call
  HF_TOKEN      →  HuggingFace API key  (also checked as API_KEY)
  ENV_URL       →  ClimateWatch server  (default: http://localhost:7860)

Runtime: < 20 minutes total for all 3 tasks.
Temperature: 0.0 for reproducible scores.
"""

import os
import json
import time
import requests
from openai import OpenAI

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 5
TEMPERATURE = 0.0   # deterministic → reproducible scores
TIMEOUT_S   = 30    # per HTTP request

# ── OpenAI client → HuggingFace router ───────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert environmental data engineer specialising in \
sensor fault detection, data quality assurance, and EPA regulatory compliance.

You have deep knowledge of:
- Sensor fault types: outlier, stuck, missing, drift, spike, bias
- Data cleaning methods: interpolation, recalibration, offset correction, smoothing
- Cascade failures in sensor networks (reference sensor outages propagating to dependents)
- EPA NAAQS thresholds for CH4, NOx, PM2.5, SO2, O3

Always respond with valid JSON only. No prose, no markdown, no code blocks — pure JSON."""


# ── Task-specific prompts ─────────────────────────────────────────────────────

def _make_prompt_task1(obs: dict) -> str:
    sd = obs.get("sensor_data", obs)
    return f"""You are analysing 24 hours of sensor data for fault detection.

Sensor: {sd.get('sensor_id')} — {sd.get('parameter')} ({sd.get('unit')})
Location: {sd.get('location')}
Normal range: {sd.get('normal_range')}

Readings (hour: value):
{json.dumps(sd.get('readings', []), indent=2)}

TASK: Identify ALL faulty hours. For each fault, specify:
  - hour (0–23)
  - fault: one of outlier | stuck | missing | drift | spike | bias
  - confidence: 0.0–1.0

Rules:
  outlier = single extreme value far outside normal range
  stuck   = same value repeating for multiple consecutive hours
  missing = null value (sensor dropout)
  drift   = gradual shift from baseline over many hours
  spike   = short burst (2-4 hrs) of abnormal readings
  bias    = systematic offset (all readings shifted by a constant)

If the sensor data looks completely clean, return an empty flags list.

Respond with ONLY this JSON (no prose):
{{"sensor_id": "{sd.get('sensor_id', '')}", "flags": [{{"hour": 0, "fault": "...", "confidence": 0.9}}]}}"""


def _make_prompt_task2(obs: dict) -> str:
    sd = obs.get("sensor_data", obs)
    sensors_summary = []
    for s in sd.get("sensors", []):
        sums = s.get("daily_summaries", [])
        means = [d.get("mean") for d in sums]
        sensors_summary.append({
            "sensor_id": s["sensor_id"],
            "parameter": s["parameter"],
            "normal_range": s["normal_range"],
            "stats": s.get("stats", {}),
            "daily_means": means,
        })

    return f"""You are diagnosing 7-day sensor network data.

Network: {sd.get('network_id')} — {sd.get('location')}

Sensors:
{json.dumps(sensors_summary, indent=2)}

TASK: For each sensor, diagnose the fault and recommend the fix.

Fault types: drift | missing | bias | noise | stuck | spike | valid
Severity:    none | low | medium | high | critical
Fixes:       no_action | interpolate | recalibrate | offset_correction | smooth | flag_only | replace

Key rules:
  drift  → gradual increase/decrease over days → fix: recalibrate
  bias   → constant offset vs reference → fix: offset_correction
  noise  → high std deviation → fix: smooth
  stuck  → constant value from some day onwards → fix: replace
  missing → null values / high missing% → fix: interpolate (small gap) or flag_only (large gap)
  valid  → normal stats → fix: no_action
  EXACTLY ONE sensor should be "valid".

Respond with ONLY this JSON:
{{"diagnoses": [{{"sensor_id": "S1", "fault_type": "drift", "severity": "high",
  "fix": "recalibrate", "fix_params": {{"drift_rate_per_day": 0.8}}}}]}}"""


def _make_prompt_task3(obs: dict) -> str:
    sd = obs.get("sensor_data", obs)
    # Build concise sensor status summary
    sensor_status = []
    for s in sd.get("sensors", []):
        sensor_status.append({
            "sensor_id": s["sensor_id"],
            "parameter": s["parameter"],
            "stats": s.get("stats", {}),
            "offline_days": s["stats"].get("offline_days", 0),
            "corrupted_days": s["stats"].get("corrupted_days", 0),
        })

    return f"""You are investigating a 30-day cascade sensor failure.

Network: {sd.get('network_id')} — {sd.get('location')}

Dependency graph (who calibrates whom):
{json.dumps(sd.get('dependency_graph', {}), indent=2)}

Sensor status summary:
{json.dumps(sensor_status, indent=2)}

Regulatory thresholds:
{json.dumps(sd.get('regulatory_thresholds', {}), indent=2)}

Known facts:
{json.dumps(sd.get('known_facts', []), indent=2)}

TASK: Provide cascade failure diagnosis:

1. root_cause_sensors: The sensors that ACTUALLY FAILED (reference sensors that went offline).
   IMPORTANT: Corrupted dependents are VICTIMS, not causes.

2. repair_order: The repair sequence. References MUST come before their dependents.
   E.g. if S1 calibrates S4 and S5, repair order must have S1 before S4 and S5.

3. fault_window_start / fault_window_end: Format "day_N" (e.g. "day_8", "day_21").
   Window = earliest failure start to latest recovery.

4. compliance_checks: For each regulated parameter, determine:
   - CLEAN: independent sensors confirm readings within threshold
   - POSSIBLE_VIOLATION: some sensors affected, uncertain if threshold crossed
   - CONFIRMED_VIOLATION: pre-failure data or independent sensors confirm violation
   - INSUFFICIENT_DATA: all measuring sensors were offline/corrupted

5. recommended_action: no_action | flag_for_review | file_compliance_report | emergency_shutdown

Respond with ONLY this JSON:
{{
  "root_cause_sensors": ["S1"],
  "repair_order": ["S1", "S4", "S5"],
  "fault_window_start": "day_N",
  "fault_window_end": "day_N",
  "compliance_checks": [{{"parameter": "CH4_ppm", "status": "POSSIBLE_VIOLATION",
    "confidence": 0.8, "reasoning": "..."}}],
  "recommended_action": "flag_for_review"
}}"""


def _get_prompt(obs: dict, task_id: str) -> str:
    if task_id == "task1_detect":
        return _make_prompt_task1(obs)
    if task_id == "task2_clean":
        return _make_prompt_task2(obs)
    if task_id == "task3_cascade":
        return _make_prompt_task3(obs)
    return json.dumps(obs.get("sensor_data", obs), indent=2)


def ask_llm(obs: dict, task_id: str, attempt: int = 0) -> dict:
    """Call the LLM and parse JSON response. Returns empty dict on failure."""
    prompt = _get_prompt(obs, task_id)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        try:
            raw_text = response.choices[0].message.content
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        print(f"  [WARN] JSON parse failed on attempt {attempt + 1}")
        return {}
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}")
        return {}


def run_task(task_id: str, seed: int = 42) -> dict:
    """
    Run one full episode for the given task.
    Returns {"score": float, "steps": int, "total_reward": float}.
    """
    # Reset
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    obs = resp.json()

    total_reward = 0.0
    steps        = 0
    done         = obs.get("done", False)

    while not done and steps < MAX_STEPS:
        # Ask LLM
        action = ask_llm(obs, task_id, attempt=steps)
        if not action:
            print(f"  [WARN] Empty action at step {steps + 1}, skipping.")
            break

        # Submit action
        step_resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=TIMEOUT_S,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward        = result.get("reward", 0.0)
        total_reward += reward
        done          = result.get("done", False)
        obs           = result
        steps        += 1

        ep_score = result.get("metadata", {}).get("episode_score", 0.0)
        print(f"    Step {steps}: reward={reward:.4f}  episode_score={ep_score:.4f}")
        time.sleep(0.3)   # gentle rate limiting

    # Final grade
    grade_resp = requests.post(f"{ENV_URL}/grader", timeout=TIMEOUT_S)
    final_score = 0.0
    if grade_resp.status_code == 200:
        final_score = grade_resp.json().get("final_score", 0.0)

    return {
        "score":        final_score,
        "steps":        steps,
        "total_reward": round(total_reward, 4),
    }


def main():
    print("=" * 60)
    print("ClimateWatch — Baseline Inference")
    print(f"Model:   {MODEL_NAME}")
    print(f"EnvURL:  {ENV_URL}")
    print("=" * 60)

    # Verify server is up
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        assert health.json().get("status") == "healthy"
        print("Server: HEALTHY\n")
    except Exception as e:
        print(f"[ERROR] Server not reachable at {ENV_URL}: {e}")
        raise SystemExit(1)

    tasks = [
        ("task1_detect",  "Single Sensor Anomaly Detection",     42),
        ("task2_clean",   "Multi-Sensor Data Cleaning",          42),
        ("task3_cascade", "Cascade Failure & Compliance Audit",  42),
    ]

    scores = {}
    for task_id, task_name, seed in tasks:
        print(f"Running: {task_id} — {task_name}")
        try:
            result = run_task(task_id, seed=seed)
            scores[task_id] = result["score"]
            print(f"  Final score: {result['score']:.4f}  "
                  f"(steps={result['steps']}, total_reward={result['total_reward']:.4f})")
        except Exception as e:
            print(f"  [ERROR] {task_id} failed: {e}")
            scores[task_id] = 0.0
        print()

    avg = sum(scores.values()) / len(scores)
    print("=" * 60)
    print(f"Average score: {avg:.4f}")
    print("=" * 60)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
