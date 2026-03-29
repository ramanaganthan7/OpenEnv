# ClimateWatch — Deployment Guide
# From Local Code → HuggingFace Space → Live Public URL

---

## The Complete Flow

```
YOUR CODE (BuildVerse folder)
        │
        │  git push huggingface master
        ▼
HUGGINGFACE SPACE
huggingface.co/spaces/YOUR_USERNAME/climatewatch-env
        │
        │  HF reads Dockerfile → builds image → runs container on port 7860
        ▼
LIVE SERVER
https://YOUR_USERNAME-climatewatch-env.hf.space
        │
        ├──► Hackathon judges call /reset, /step, /grader with Nemotron
        └──► inference.py calls https://router.huggingface.co/v1 for free LLM
```

---

## Prerequisites (Other PC)

Install once:

```bash
# Git
winget install --id Git.Git

# Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
git --version
docker --version
uv --version
```

---

## Step 1 — Get Your HuggingFace Token

Go to: `huggingface.co → Profile → Settings → Access Tokens → New Token`
- Name: `climatewatch-deploy`
- Type: **Write** (needed to push code)

Keep it safe — used as the git password when pushing.

---

## Step 2 — Push Code to HuggingFace

```bash
cd path\to\BuildVerse

# Add HuggingFace as a git remote (replace YOUR_USERNAME with your HF username)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env

# Push all code
git push huggingface main
```

When prompted, enter your HuggingFace username and token.

---

## Step 3 — Set Environment Variables in Space Settings

Go to your Space → **Settings → Variables and Secrets**

Add exactly these 4:

| Name | Value | Type |
|---|---|---|
| `HF_TOKEN` | `your_hf_token_here` | **Secret** (hidden) |
| `MODEL_NAME` | `meta-llama/Llama-3.3-70B-Instruct` | Variable |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Variable |
| `ENV_URL` | `http://localhost:7860` | Variable |

Click **Save** after adding all four.

---

## Step 4 — Watch It Build

After git push, HuggingFace automatically:
1. Detects `Dockerfile` in root
2. Builds Docker image (2–5 minutes first time)
3. Starts container on port 7860
4. Status changes from **Building** → **Running**

Watch live logs:
```
Space page → Logs tab
```

---

## Step 5 — Verify Live Space

Once status shows **Running**:

```bash
# Health check
curl https://YOUR_USERNAME-climatewatch-env.hf.space/health
# → {"status":"healthy"}

# View tasks
curl https://YOUR_USERNAME-climatewatch-env.hf.space/tasks

# Start an episode
curl -X POST https://YOUR_USERNAME-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'

# Open dashboard in browser:
# https://YOUR_USERNAME-climatewatch-env.hf.space
```

---

## Step 6 — Run Baseline Against Live Space

```bash
set HF_TOKEN=your_hf_token_here
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
set ENV_URL=https://YOUR_USERNAME-climatewatch-env.hf.space

uv run task infer
```

---

## Architecture Inside HuggingFace

```
Docker Container (python:3.11-slim)
│
├── uvicorn app.main:app --host 0.0.0.0 --port 7860
│
├── GET  /              → dashboard HTML
├── GET  /health        → {"status":"healthy"}
├── GET  /state         → {episode_id, task_id, step_count, total_reward, done}
├── GET  /tasks         → 3 tasks with schemas
├── POST /reset         → new episode + sensor data
├── POST /step          → grades action → returns reward
├── POST /grader        → final score
└── POST /baseline      → runs inference.py inside container
                          calls https://router.huggingface.co/v1
                          using HF_TOKEN from environment
```

---

## What Each File Does During Deployment

| File | Role |
|---|---|
| `Dockerfile` | HuggingFace reads this to build the container |
| `requirements.txt` | Packages installed inside Docker |
| `openenv.yaml` | Hackathon validator reads this — confirms tasks + metadata |
| `README.md` | YAML header tells HF Spaces: `sdk: docker`, `app_port: 7860` |
| `inference.py` | Baseline agent that runs when `/baseline` is called |
| `app/main.py` | The FastAPI server that handles all HTTP requests |

---

## Troubleshooting

### Build fails

Check Space → Logs tab. Common causes:

```
"No module named X"
→ Add X to requirements.txt and push again

"Port already in use"
→ HF handles port 7860 automatically — not your issue

"Dockerfile syntax error"
→ Check Dockerfile for typos
```

### /health returns error after build

```bash
# Test locally first
uv run task serve
curl http://localhost:7860/health
# Must return {"status":"healthy"} locally before pushing
```

### /baseline returns empty output

```
Check: Is HF_TOKEN set as Secret in Space Settings?
Check: Is MODEL_NAME set correctly?
Check: Does your token have inference access?
       → huggingface.co → Settings → Access Tokens → verify permissions
```

### Rate limit from HF router

Free tier: ~100 requests/day per model. If hit:
```bash
# Switch to smaller model temporarily
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
```

---

## Hackathon Submission Checklist

```
Before submitting, verify:
  ✓ curl https://YOUR_USERNAME-climatewatch-env.hf.space/health  → {"status":"healthy"}
  ✓ POST /reset returns done=false, reward=0.0 with sensor data
  ✓ GET /state returns episode_id, task_id, step_count, total_reward, done
  ✓ Space is PUBLIC (not private)
  ✓ openenv.yaml has tags: [openenv]
  ✓ All 3 tasks listed with easy/medium/hard difficulty

Submit this URL:
  https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env
```

---

## Quick Reference

```bash
# Local development
uv run task serve      # start server at localhost:7860
uv run task test       # run 63 tests
uv run task kill       # free port 7860

# Push to HuggingFace
git add .
git commit -m "update"
git push huggingface master

# Test live space
curl https://YOUR_USERNAME-climatewatch-env.hf.space/health
```
