# Complete Hackathon Requirements
> Every rule, constraint, and checklist — nothing missed

---

## CRITICAL NEW RULES (Read First)

```
1. Inference script MUST be named:  inference.py
   MUST be in:  root directory of project (not inside app/)

2. Must use these 3 environment variables:
   API_BASE_URL  →  LLM endpoint  (default: https://router.huggingface.co/v1)
   MODEL_NAME    →  Which model to call
   HF_TOKEN      →  Your Hugging Face API key

3. Must use OpenAI Client with HF router:
   client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
   NOT the standard openai.com endpoint

4. Runtime limit: inference.py must complete in under 20 minutes

5. Machine constraint: must run on 2 vCPU + 8 GB RAM
   → Keep your environment lightweight
   → No heavy ML models loaded on startup
   → JSON files for data, not databases
```

---

## THE COMPLETE REQUIREMENT CHECKLIST

### A — Core OpenEnv API (Disqualified if any are missing)

```
[ ] POST /reset
    Input:  {"task_id": "task1_detect", "seed": null}
    Output: observation JSON with done=false, reward=0.0, + your data

[ ] POST /step
    Input:  {"action": { ...your action fields... }}
    Output: observation JSON with done, reward (float), + your data

[ ] GET /state
    Output: {episode_id, task_id, step_count, total_reward, done}

[ ] GET /health
    Output: {"status": "healthy"}
    HTTP:   200 OK
```

---

### B — Additional Required Endpoints

```
[ ] GET /tasks
    Output: list of all 3 tasks with:
    - task id
    - task name
    - difficulty (easy/medium/hard)
    - action_schema (what fields the agent must send in step())

[ ] POST /grader
    Output: final score for the completed episode (0.0–1.0)
    When to call: after episode ends (done=true)

[ ] POST /baseline
    Output: runs inference.py and returns scores for all 3 tasks
```

---

### C — The 3 Tasks

```
[ ] Task 1 — EASY
    - Clearly easier than task 2
    - Grader returns different scores for different agent actions
    - Score range: 0.0–1.0
    - Deterministic: same action = same score every time

[ ] Task 2 — MEDIUM
    - Clearly harder than task 1, easier than task 3
    - Grader returns different scores for different agent actions
    - Score range: 0.0–1.0
    - Deterministic

[ ] Task 3 — HARD
    - Must genuinely challenge frontier LLMs (GPT-4, Nemotron)
    - Grader returns different scores for different agent actions
    - Score range: 0.0–1.0
    - Deterministic

CRITICAL: Graders that ALWAYS return the same score = DISQUALIFIED
```

---

### D — Reward Function

```
[ ] Gives reward at EVERY step (not just at the end)
[ ] Rewards partial progress (not just 0 or 1)
[ ] Penalizes bad behavior:
    - Infinite loops (same action repeated)
    - Destructive/invalid actions
[ ] Never returns NaN or values outside 0.0–1.0
    (use: max(0.0, min(1.0, reward)))
```

---

### E — inference.py (Baseline Script)

```
[ ] File name is EXACTLY: inference.py
[ ] Located in ROOT directory of project (same level as Dockerfile)
[ ] Reads API_BASE_URL from os.environ (default: https://router.huggingface.co/v1)
[ ] Reads MODEL_NAME from os.environ
[ ] Reads HF_TOKEN from os.environ
[ ] Uses: client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
[ ] Runs all 3 tasks and prints scores
[ ] Completes in under 20 minutes total
[ ] Produces reproducible scores (use temperature=0.0)
[ ] Runs without errors
[ ] Does NOT hardcode any API keys
```

**Correct inference.py pattern:**
```python
import os, json, requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

---

### F — openenv.yaml

```
[ ] File exists in root directory
[ ] Contains: name, version, description
[ ] Contains: tasks list with id, name, difficulty for each task
[ ] Tags include "openenv"
```

**Template:**
```yaml
name: climatewatch-env
version: "1.0.0"
description: >
  Real-world environmental sensor data quality and compliance monitoring.
  AI agents detect faults, clean data, and verify regulatory compliance.
tags:
  - openenv
tasks:
  - id: task1_detect
    name: "Single Sensor Anomaly Detection"
    difficulty: easy
  - id: task2_clean
    name: "Multi-Sensor Data Cleaning"
    difficulty: medium
  - id: task3_cascade
    name: "Cascade Failure & Compliance Check"
    difficulty: hard
```

---

### G — Dockerfile

```
[ ] Dockerfile exists in root directory
[ ] docker build succeeds without errors
[ ] docker run starts the server
[ ] Server listens on port 7860
[ ] GET /health returns 200 after docker run
[ ] Image is reasonably sized (no unnecessary packages)
[ ] Does NOT require internet access at runtime
```

**Your Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

### H — Hugging Face Space

```
[ ] HF Space created with sdk: docker
[ ] README.md has the YAML header with tags: [openenv]
[ ] app_port: 7860 in README header
[ ] Space deploys without error (check build logs)
[ ] Public URL responds to GET /health → 200
[ ] Public URL responds to POST /reset → valid JSON
[ ] Space is PUBLIC (not private)
```

**README.md required header:**
```
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
```

---

### I — Documentation (README.md)

```
[ ] Environment description — what it simulates and why
[ ] Action space — what fields agent must send, what values are valid
[ ] Observation space — what fields agent receives back
[ ] Task descriptions — all 3 tasks with difficulty
[ ] Setup instructions — how to run locally
[ ] Baseline scores — what score did your inference.py get on each task
[ ] HF Space URL
```

---

### J — Performance Constraints

```
[ ] inference.py runtime < 20 minutes for all 3 tasks combined
[ ] Environment runs on: 2 vCPU, 8 GB RAM
    → No PyTorch/TensorFlow in the environment server
    → No models loaded at startup
    → Data stored as JSON, not in database
    → Responses return in < 5 seconds per step
```

---

## DISQUALIFICATION RULES

```
INSTANT DISQUALIFICATION:
  ✗ HF Space does not deploy or does not respond
  ✗ Dockerfile does not build
  ✗ inference.py does not run or errors out
  ✗ Graders always return the same score
  ✗ No inference.py in root directory
  ✗ Fewer than 3 tasks
  ✗ Plagiarized or trivially modified existing environment

LIKELY DISQUALIFICATION:
  ✗ inference.py takes more than 20 minutes
  ✗ Does not use OpenAI client with API_BASE_URL / HF_TOKEN
  ✗ Grader scores not in 0.0–1.0 range
  ✗ openenv validate fails
```

---

## SCORING WEIGHTS (What to Focus On)

```
Real-world utility       30%  ← MOST IMPORTANT
Task & grader quality    25%  ← SECOND MOST IMPORTANT
Environment design       20%  ← THIRD
Code quality/compliance  15%  ← Fourth
Creativity & novelty     10%  ← Fifth

ClimateWatch targets:
  Real-world utility  →  26–30/30  (environmental monitoring, $14.4B market)
  Task quality        →  20–25/25  (clear graders, good difficulty range)
  Environment design  →  16–20/20  (clean state, partial rewards)
  Code quality        →  12–15/15  (spec compliant, Docker works)
  Creativity          →  8–10/10   (unique domain, novel in OpenEnv)

Target total: 82–100 points
```

---

## JUDGING PHASES

### Phase 1 — Automated (Pass/Fail Gate)
```
You either pass all of these or you are OUT:
  → HF Space URL responds to reset()
  → openenv validate passes
  → docker build works
  → inference.py runs and produces scores
  → 3+ tasks with graders scoring 0.0–1.0
```

### Phase 2 — Agentic Evaluation (Scored)
```
Judges run these automatically:
  → Re-run your inference.py
  → Run Nemotron 3 Super against your environment
  → Check that different actions give different scores
  → Check score variance across episodes
```

### Phase 3 — Human Review (Top Submissions Only)
```
Meta and HF engineers manually review:
  → Real-world utility and motivation
  → Creativity and originality
  → No exploit in graders (can't get 1.0 by cheating)
  → Code quality and structure
```

---

## ENVIRONMENT VARIABLES REFERENCE

| Variable | Purpose | Default | Required |
|---|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` | Yes |
| `MODEL_NAME` | Which model to call | None | Yes |
| `HF_TOKEN` | HF API key | None | Yes |
| `API_KEY` | Fallback API key | None | Optional |
| `ENV_URL` | Your environment URL | `http://localhost:7860` | For testing |

---

## FILE STRUCTURE REQUIREMENTS

```
your-project/           ← root directory
├── inference.py        ← MANDATORY, must be here (not in subfolder)
├── openenv.yaml        ← MANDATORY, must be here
├── Dockerfile          ← MANDATORY, must be here
├── requirements.txt    ← MANDATORY
├── README.md           ← MANDATORY (with HF Spaces header)
└── app/
    ├── main.py
    ├── environment.py
    ├── models.py
    ├── reward.py
    ├── tasks/
    └── data/
```

---

*ALGORITHMIC AVENGERS — Ramana Ganthan S + Sre Sandhya K*
