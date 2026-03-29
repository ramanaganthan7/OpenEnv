# ClimateWatch — Complete Deployment Guide
# From Your Laptop → HuggingFace Spaces → Live Public URL

---

## THE FULL PICTURE — How Everything Connects

```
YOUR CODE (this folder)
        │
        │  git push
        ▼
HUGGINGFACE SPACE  (free cloud hosting)
        │
        │  reads Dockerfile → builds Docker image → runs container
        ▼
LIVE SERVER at:
https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env
        │
        ├──► Hackathon judges call your /reset, /step, /grader
        │
        └──► inference.py inside the container calls:
             https://router.huggingface.co/v1  ← HF's free LLM API
             using YOUR HF_TOKEN
```

---

## STEP 0 — What You Need On The Other PC

Install these (all free):

```
1. Git           → https://git-scm.com/download/win
2. Docker        → https://www.docker.com/products/docker-desktop
3. uv            → powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installs:
```bash
git --version
docker --version
uv --version
```

---

## STEP 1 — Create a HuggingFace Account

Go to: https://huggingface.co/join

- Pick a username  (e.g.  algorithmic-avengers)
- Verify your email

---

## STEP 2 — Get Your HuggingFace Token

This is the key that lets your server call LLM models for free.

```
1. Log in to huggingface.co
2. Click your profile picture (top right)
3. Click Settings
4. Click Access Tokens (left sidebar)
5. Click New Token
6. Name it: climatewatch
7. Type: Read  (enough for inference)
8. Click Generate Token
9. COPY IT — you only see it once

It looks like:  hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
```

Save this somewhere safe — you need it in Step 5.

---

## STEP 3 — Create a HuggingFace Space

```
1. Go to: https://huggingface.co/new-space
2. Fill in:
   ┌─────────────────────────────────────────┐
   │ Space name:  climatewatch-env           │
   │ License:     MIT                        │
   │ SDK:         Docker          ← CRITICAL │
   │ Visibility:  Public                     │
   └─────────────────────────────────────────┘
3. Click Create Space
```

Your Space URL will be:
`https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env`

HuggingFace gives you a git repository at:
`https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env.git`

---

## STEP 4 — Push Your Code to HuggingFace

On the other PC, copy this project folder, then:

```bash
cd path\to\BuildVerse

# Connect your local folder to the HF Space repo
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env

# Push everything
git push huggingface master
```

When it asks for credentials:
```
Username: YOUR_HUGGINGFACE_USERNAME
Password: hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456   ← your token from Step 2
```

---

## STEP 5 — Set Environment Variables on HuggingFace

This is critical. Without these, inference.py cannot call the LLM.

```
1. Go to your Space: huggingface.co/spaces/YOUR_USERNAME/climatewatch-env
2. Click Settings tab
3. Scroll to Variables and Secrets section
4. Add these one by one:
```

| Variable | Value | Type |
|---|---|---|
| `HF_TOKEN` | `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456` | **Secret** (hidden) |
| `MODEL_NAME` | `meta-llama/Llama-3.3-70B-Instruct` | Variable |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Variable |
| `ENV_URL` | `http://localhost:7860` | Variable |

> **Secret vs Variable**: Use Secret for HF_TOKEN so it stays hidden.
> Variables are visible in logs — fine for MODEL_NAME.

Click **Save** after adding all four.

---

## STEP 6 — Watch It Build

After git push, HuggingFace automatically:

```
1. Detects Dockerfile in root
2. Builds Docker image (takes 2-5 minutes first time)
3. Starts the container on port 7860
4. Your Space goes from "Building..." to "Running"
```

You can watch the build logs live:
```
Go to your Space → click Logs tab → watch the output
```

If build fails, check the Logs tab — it shows exactly what went wrong.

---

## STEP 7 — Test Your Live Space

Once status shows **Running**:

```bash
# Replace YOUR_USERNAME with your actual HF username

# Health check
curl https://YOUR_USERNAME-climatewatch-env.hf.space/health

# Should return:
{"status": "healthy"}

# View all tasks
curl https://YOUR_USERNAME-climatewatch-env.hf.space/tasks

# Start an episode
curl -X POST https://YOUR_USERNAME-climatewatch-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect", "seed": 0}'

# Open the dashboard in browser:
https://YOUR_USERNAME-climatewatch-env.hf.space
```

---

## STEP 8 — Run inference.py Against Your Live Space

```bash
set HF_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
set ENV_URL=https://YOUR_USERNAME-climatewatch-env.hf.space

uv run task infer
```

This sends the sensor data to the LLM and gets scored results for all 3 tasks.

---

## THE COMPLETE ARCHITECTURE — What Runs Where

```
╔══════════════════════════════════════════════════════════════════╗
║  YOUR LAPTOP / ANOTHER PC                                        ║
║                                                                  ║
║  BuildVerse/                                                     ║
║  ├── inference.py   ← runs here, calls HF router for LLM        ║
║  ├── app/           ← same code pushed to HF Space               ║
║  └── ...                                                         ║
╚══════════════════════════════════════════════════════════════════╝
          │
          │  git push huggingface master
          ▼
╔══════════════════════════════════════════════════════════════════╗
║  HUGGINGFACE SPACES (free cloud)                                 ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  Docker Container (python:3.11-slim)                    │    ║
║  │                                                         │    ║
║  │  uvicorn app.main:app --port 7860                       │    ║
║  │                                                         │    ║
║  │  GET  /              → dashboard HTML                   │    ║
║  │  GET  /health        → {"status":"healthy"}             │    ║
║  │  GET  /tasks         → 3 task schemas                   │    ║
║  │  POST /reset         → new episode + sensor data        │    ║
║  │  POST /step          → grades action → reward           │    ║
║  │  POST /grader        → final score                      │    ║
║  │  POST /baseline      → runs inference.py inside         │    ║
║  │                                                         │    ║
║  │  Environment Variables:                                 │    ║
║  │    HF_TOKEN    = hf_...  (secret)                       │    ║
║  │    MODEL_NAME  = meta-llama/Llama-3.3-70B-Instruct      │    ║
║  │    API_BASE_URL= https://router.huggingface.co/v1       │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                    │                                             ║
║  Public URL:       │                                             ║
║  https://          │                                             ║
║  YOUR_USERNAME-    │                                             ║
║  climatewatch-     │                                             ║
║  env.hf.space      │                                             ║
╚════════════════════╪═════════════════════════════════════════════╝
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
╔══════════════════╗   ╔══════════════════════════════════╗
║  HACKATHON       ║   ║  HUGGINGFACE ROUTER              ║
║  JUDGES          ║   ║  https://router.huggingface.co   ║
║                  ║   ║                                  ║
║  Call your       ║   ║  Routes your API calls to:       ║
║  /reset          ║   ║  - Llama 3.3 70B                 ║
║  /step           ║   ║  - Nemotron                      ║
║  /grader         ║   ║  - Mistral                       ║
║  endpoints       ║   ║  - and more                      ║
╚══════════════════╝   ╚══════════════════════════════════╝
```

---

## HOW THE LLM CALL FLOWS

When `POST /baseline` is called (or you run inference.py manually):

```
inference.py
    │
    │  1. POST /reset → gets sensor data from your server
    │
    │  2. Builds a prompt with the sensor data
    │
    │  3. POST https://router.huggingface.co/v1/chat/completions
    │        headers: Authorization: Bearer hf_YOUR_TOKEN
    │        body:   {model: "meta-llama/Llama-3.3-70B-Instruct",
    │                 messages: [...sensor data as prompt...]}
    │
    │  4. LLM returns JSON analysis
    │
    │  5. POST /step → submits the analysis → gets reward score
    │
    └── Repeat for all 3 tasks, print final scores
```

**Cost:** The HuggingFace router gives **free inference** for open models
(Llama 3.3, Mistral, etc.) up to rate limits. No credit card needed.

---

## WHAT EACH FILE DOES DURING DEPLOYMENT

```
Dockerfile          → HuggingFace reads this to build the container
                      FROM python:3.11-slim
                      installs packages from requirements.txt
                      runs: uvicorn app.main:app --port 7860

requirements.txt    → packages installed inside Docker
                      (fastapi, uvicorn, pydantic, openai, etc.)

openenv.yaml        → hackathon validator reads this
                      confirms 3 tasks, correct metadata

README.md           → the YAML header at top tells HF Spaces:
                      sdk: docker
                      app_port: 7860
                      HF reads this to configure the Space

app/main.py         → the FastAPI server that runs inside Docker
app/environment.py  → episode management
app/tasks/          → the 3 tasks with scenario generation + graders
inference.py        → the baseline LLM agent (runs on /baseline call)
```

---

## TROUBLESHOOTING

### Build fails on HuggingFace

Check Logs tab. Common causes:
```
Error: package not found
Fix:  make sure requirements.txt has the correct version numbers

Error: port already in use
Fix:  Dockerfile already exposes 7860 — HF handles this automatically

Error: module not found
Fix:  make sure all app/ files were pushed (git status to check)
```

### Space shows "Error" after building

```bash
# Check your Space logs for runtime errors
# Most common: import error in app code

# Test locally first:
uv run task serve
curl http://localhost:7860/health
```

### /baseline returns empty or LLM errors

```
Check: Is HF_TOKEN set as Secret in Space settings?
Check: Is MODEL_NAME set correctly?
Check: Does your HF token have inference permissions?
       (Go to HF Settings → Tokens → check the token has Read access)
```

### Rate limit from HF router

The free tier allows ~100 requests/day per model. If you hit limits:
- Switch MODEL_NAME to a smaller model: `meta-llama/Llama-3.2-3B-Instruct`
- Or wait 24 hours for limits to reset

---

## AFTER DEPLOYMENT — SHARE WITH JUDGES

Submit this URL to the hackathon:
```
https://huggingface.co/spaces/YOUR_USERNAME/climatewatch-env
```

The judges will:
1. Visit your Space URL — see the dashboard
2. Call `/health` → verify server is running
3. Call `/tasks` → see your 3 tasks
4. Run their own agent (Nemotron/GPT-4) against your `/reset` and `/step`
5. Call `/grader` → get scores
6. Call `openenv validate` → reads your `openenv.yaml`

---

## QUICK REFERENCE CHECKLIST

```
ON THE OTHER PC:
  □ Install Git, Docker, uv
  □ Copy this BuildVerse folder

ON HUGGINGFACE:
  □ Create account at huggingface.co
  □ Get API token (Settings → Access Tokens)
  □ Create new Space (SDK: Docker, Public)
  □ Set 4 environment variables (HF_TOKEN as Secret)

PUSH CODE:
  □ git remote add huggingface https://huggingface.co/spaces/USERNAME/climatewatch-env
  □ git push huggingface master

VERIFY:
  □ Build logs show no errors
  □ Space status = Running
  □ curl /health returns {"status":"healthy"}
  □ Dashboard loads in browser
```
