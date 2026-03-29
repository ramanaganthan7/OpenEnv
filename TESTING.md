# ClimateWatch — Live Space Testing Guide

Run these curl commands from a **non-corporate network** (mobile hotspot or personal PC not on Cognizant VPN).

**Base URL:** `https://ramanaganthan7-climatewatch-env.hf.space`

---

## 1. Health Check
```bash
curl https://ramanaganthan7-climatewatch-env.hf.space/health
```
**Expected:**
```json
{"status":"healthy"}
```

---

## 2. List All Tasks
```bash
curl https://ramanaganthan7-climatewatch-env.hf.space/tasks
```
**Expected:** 3 tasks with ids `task1_detect` (easy), `task2_clean` (medium), `task3_cascade` (hard), each with `action_schema`.

---

## 3. Reset — Task 1 (Easy)
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'
```
**Expected:**
```json
{
  "done": false,
  "reward": 0.0,
  "task_id": "task1_detect",
  "step_count": 0,
  "sensor_data": { "sensor_id": "...", "readings": [...24 items...] },
  "feedback": "Episode started...",
  "metadata": {"episode_id": "...", "max_steps": 5}
}
```

---

## 4. Get State
```bash
curl https://ramanaganthan7-climatewatch-env.hf.space/state
```
**Expected:**
```json
{
  "episode_id": "some-uuid",
  "task_id": "task1_detect",
  "step_count": 0,
  "total_reward": 0.0,
  "done": false
}
```

---

## 5. Step — Wrong Action (should score low)
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": []}}'
```
**Expected:** `reward` between 0.0 and 1.0, `step_count: 1`, `feedback` with guidance.

---

## 6. Reset Again + Perfect Answer (should score high)
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'
```

Then submit the correct answer (seed=0 scenario has stuck h3-5, outlier h6, missing h8-9):
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
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
  }'
```
**Expected:** `reward >= 0.8`, `episode_score >= 0.8`, `done: true` (early stop triggered).

---

## 7. Final Grade
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/grader
```
**Expected:**
```json
{
  "episode_id": "some-uuid",
  "task_id": "task1_detect",
  "final_score": 0.85,
  "step_count": 1,
  "breakdown": {...}
}
```

---

## 8. Reset — Task 2 (Medium)
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2_clean", "seed": 0}'
```
**Expected:** `sensor_data.sensors` has exactly 5 sensors, each with `daily_summaries`.

---

## 9. Reset — Task 3 (Hard)
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task3_cascade", "seed": 0}'
```
**Expected:** `sensor_data.sensors` has 10 sensors, `dependency_graph` key present, `known_facts` present.

---

## 10. Invalid Task → Must Return 400
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fake_task"}'
```
**Expected:** HTTP 400 with error detail.

---

## 11. Anti-Loop Penalty Check
Reset, then submit the SAME action twice:
```bash
curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'

curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": [{"hour": 3, "fault": "stuck", "confidence": 0.9}]}}'

curl -X POST https://ramanaganthan7-climatewatch-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"sensor_id": "CO2-100", "flags": [{"hour": 3, "fault": "stuck", "confidence": 0.9}]}}'
```
**Expected:** Second step reward is LOWER than first (anti-loop penalty applied).

---

## 12. Open Dashboard in Browser
```
https://ramanaganthan7-climatewatch-env.hf.space/
```
**Expected:** Green terminal-style UI with 3 task buttons.

---

## 13. Open Swagger API Docs
```
https://ramanaganthan7-climatewatch-env.hf.space/docs
```
**Expected:** Interactive Swagger UI showing all 7 endpoints.

---

## Full Checklist

| # | Check | Expected | Pass? |
|---|---|---|---|
| 1 | GET /health | `{"status":"healthy"}` | |
| 2 | GET /tasks | 3 tasks, easy/medium/hard | |
| 3 | POST /reset task1 | done=false, reward=0.0, 24 readings | |
| 4 | GET /state | episode_id, task_id, step_count, total_reward, done | |
| 5 | POST /step wrong | reward in [0.0,1.0], step_count=1 | |
| 6 | POST /step correct | reward ≥ 0.8, done=true | |
| 7 | POST /grader | final_score in [0.0,1.0] | |
| 8 | POST /reset task2 | 5 sensors | |
| 9 | POST /reset task3 | 10 sensors, dependency_graph | |
| 10 | POST /reset bad task | HTTP 400 | |
| 11 | Anti-loop | 2nd same action reward < 1st | |
| 12 | Dashboard | Green UI visible | |
| 13 | /docs | Swagger UI visible | |
