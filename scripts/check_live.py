"""
ClimateWatch — Live Space Health Check
Run this on a machine that can access HuggingFace.
Checks every required endpoint against the live Space.
"""

import requests
import json
import sys

BASE = "https://ramanaganthan7-climatewatch-env.hf.space"
PASS = []
FAIL = []

def check(name, result, expected=None):
    if result and (expected is None or expected in str(result)):
        PASS.append(name)
        print(f"  PASS  {name}")
        return True
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  →  got: {str(result)[:120]}")
        return False

def get(path):
    try:
        r = requests.get(f"{BASE}{path}", timeout=15)
        return r.status_code, r.json()
    except Exception as e:
        return 0, str(e)

def post(path, body):
    try:
        r = requests.post(f"{BASE}{path}", json=body,
                          headers={"Content-Type": "application/json"}, timeout=15)
        return r.status_code, r.json()
    except Exception as e:
        return 0, str(e)

print("=" * 55)
print("ClimateWatch — Live Space Check")
print(f"URL: {BASE}")
print("=" * 55)

# ── 1. Health ─────────────────────────────────────────────
print("\n[1] GET /health")
code, data = get("/health")
check("status=200",          code == 200)
check('{"status":"healthy"}', data, "healthy")

# ── 2. Tasks ──────────────────────────────────────────────
print("\n[2] GET /tasks")
code, data = get("/tasks")
check("status=200",           code == 200)
tasks = data.get("tasks", []) if isinstance(data, dict) else []
check("3 tasks returned",     len(tasks) == 3)
ids = {t.get("id") for t in tasks}
check("task1_detect present", "task1_detect"  in ids)
check("task2_clean present",  "task2_clean"   in ids)
check("task3_cascade present","task3_cascade" in ids)
diffs = {t.get("id"): t.get("difficulty") for t in tasks}
check("task1=easy",           diffs.get("task1_detect")  == "easy")
check("task2=medium",         diffs.get("task2_clean")   == "medium")
check("task3=hard",           diffs.get("task3_cascade") == "hard")
check("action_schema present",all("action_schema" in t for t in tasks))

# ── 3. Reset Task 1 ───────────────────────────────────────
print("\n[3] POST /reset  (task1_detect, seed=0)")
code, obs = post("/reset", {"task_id": "task1_detect", "seed": 0})
check("status=200",           code == 200)
check("done=false",           obs.get("done") == False)
check("reward=0.0",           obs.get("reward") == 0.0)
check("step_count=0",         obs.get("step_count") == 0)
check("sensor_data present",  "sensor_data" in obs)
check("24 readings",          len(obs.get("sensor_data", {}).get("readings", [])) == 24)
sensor_id = obs.get("sensor_data", {}).get("sensor_id", "")

# ── 4. State ──────────────────────────────────────────────
print("\n[4] GET /state")
code, state = get("/state")
check("status=200",           code == 200)
check("episode_id present",   bool(state.get("episode_id")))
check("task_id=task1_detect", state.get("task_id") == "task1_detect")
check("step_count=0",         state.get("step_count") == 0)
check("done=false",           state.get("done") == False)

# ── 5. Step (empty action) ────────────────────────────────
print("\n[5] POST /step  (empty flags)")
code, result = post("/step", {"action": {"sensor_id": sensor_id, "flags": []}})
check("status=200",           code == 200)
check("reward is float",      isinstance(result.get("reward"), float))
check("reward in [0,1]",      0.0 <= result.get("reward", -1) <= 1.0)
check("step_count=1",         result.get("step_count") == 1)
check("feedback present",     bool(result.get("feedback")))

# ── 6. Step (correct answer) ──────────────────────────────
print("\n[6] POST /reset + /step  (perfect answer for seed=0)")
_, obs2 = post("/reset", {"task_id": "task1_detect", "seed": 0})
sid = obs2.get("sensor_data", {}).get("sensor_id", "")
_, step2 = post("/step", {"action": {
    "sensor_id": sid,
    "flags": [
        {"hour": 3,  "fault": "stuck",   "confidence": 0.95},
        {"hour": 4,  "fault": "stuck",   "confidence": 0.95},
        {"hour": 5,  "fault": "stuck",   "confidence": 0.95},
        {"hour": 6,  "fault": "outlier", "confidence": 1.0},
        {"hour": 8,  "fault": "missing", "confidence": 1.0},
        {"hour": 9,  "fault": "missing", "confidence": 1.0},
    ]
}})
ep_score = step2.get("metadata", {}).get("episode_score", 0)
check("perfect score ≥ 0.8",  ep_score >= 0.8)
check("reward > 0",           step2.get("reward", 0) > 0)

# ── 7. Grader ─────────────────────────────────────────────
print("\n[7] POST /grader")
code, grade = post("/grader", {})
check("status=200",           code == 200)
check("final_score present",  "final_score" in grade)
check("score in [0,1]",       0.0 <= grade.get("final_score", -1) <= 1.0)
check("episode_id present",   bool(grade.get("episode_id")))

# ── 8. Reset Task 2 ───────────────────────────────────────
print("\n[8] POST /reset  (task2_clean)")
code, obs3 = post("/reset", {"task_id": "task2_clean", "seed": 0})
check("status=200",           code == 200)
check("5 sensors",            len(obs3.get("sensor_data", {}).get("sensors", [])) == 5)

# ── 9. Reset Task 3 ───────────────────────────────────────
print("\n[9] POST /reset  (task3_cascade)")
code, obs4 = post("/reset", {"task_id": "task3_cascade", "seed": 0})
check("status=200",           code == 200)
check("10 sensors",           len(obs4.get("sensor_data", {}).get("sensors", [])) == 10)
check("dependency_graph",     "dependency_graph" in obs4.get("sensor_data", {}))

# ── 10. Invalid task returns 400 ──────────────────────────
print("\n[10] POST /reset  (invalid task → must return 400)")
code, _ = post("/reset", {"task_id": "fake_task"})
check("400 on bad task_id",   code == 400)

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
if FAIL:
    print(f"FAILED: {len(FAIL)}")
    for f in FAIL:
        print(f"  ✗  {f}")
else:
    print("ALL CHECKS PASSED — Space is fully working!")
print("=" * 55)
