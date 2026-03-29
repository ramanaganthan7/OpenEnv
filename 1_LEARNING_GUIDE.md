# OpenEnv Learning Guide — All 4 Modules
> Everything you need to understand before building ClimateWatch

---

# MODULE 1 — Why OpenEnv?

## The Big Question: Why Does OpenEnv Exist?

Before OpenEnv, training AI agents was a mess. Every team built their own environment in their own way.
Team A used one API. Team B used another. You couldn't take a training loop from one project and use it in another.

OpenEnv solves this by giving everyone the same "language" to talk to environments.

---

## The Reinforcement Learning Loop — The Core Concept

This is the heart of everything. Once you understand this loop, you understand the entire hackathon.

```
┌─────────────────────────────────────────────────┐
│                                                   │
│   ┌──────────┐    action     ┌─────────────────┐  │
│   │          │ ─────────────▶│                 │  │
│   │  AI      │               │  ENVIRONMENT    │  │
│   │  AGENT   │               │  (your project) │  │
│   │          │◀─────────────│                 │  │
│   └──────────┘  observation  └─────────────────┘  │
│                  + reward                         │
│                  + done?                          │
│                                                   │
│  Agent sees → Agent decides → Environment reacts  │
│  Agent learns from reward → Gets better over time │
└─────────────────────────────────────────────────┘
```

**Real-world translation for ClimateWatch:**
```
Agent sees    →  Sensor readings with possible faults
Agent decides →  "Hour 6 is an outlier, Hour 3-5 are stuck"
Environment   →  Checks against ground truth
Reward        →  +0.4 for correct fault type, -0.2 for false alarm
Agent learns  →  Next time it sees similar pattern, it does better
```

---

## The 3 Functions You Must Implement (The Entire OpenEnv API)

```python
# reset() — Start a fresh episode
# → Returns: first observation (what the agent sees at the start)
# → Like: "New episode. Here are 24 hours of sensor data."
obs = env.reset()

# step(action) — Agent does something, world responds
# → Takes: an action (agent's decision)
# → Returns: new observation + reward + done (is episode over?)
# → Like: "Agent said fault at hour 6. Score: +0.4. Continue."
obs, reward, done, info = env.step(action)

# state() — Read current episode status
# → Returns: episode_id, step_count, total_reward, etc.
# → Like: "Currently at step 3, total reward so far = 0.9"
current_state = env.state()
```

**That's it. The entire OpenEnv spec is these 3 functions.**

---

## Gym (Old Way) vs OpenEnv (New Way)

```
Gym (old way):                     OpenEnv (new way):
─────────────────────────────      ──────────────────────────────────
Runs on same machine as agent      Runs in a Docker container (isolated)
No standard for web deployment     Deploys to Hugging Face Spaces
No type safety                     Pydantic typed models (validated)
Only local Python                  HTTP API — any language can use it
Hard to share environments         Hub on Hugging Face — fork and use
```

**Why isolation matters:**
```
If your environment runs in Docker:
  → Judges can test it from anywhere
  → It runs the same on your laptop and on HF Spaces
  → No "it works on my machine" problems
  → Security: agent can't break out of the container
```

---

## The Architecture of an OpenEnv Environment

```
Your environment has TWO parts:

1. SERVER (inside Docker, runs on HF Spaces)
   → Your FastAPI app
   → Implements reset() / step() / state()
   → Returns JSON responses

2. CLIENT (on the agent's machine)
   → Connects to server via HTTP
   → Sends actions
   → Receives observations and rewards

You are only building the SERVER for this hackathon.
The judges bring their own client (agent) to test your server.
```

---

## Real-World Use Case — Why This Matters

```
Without OpenEnv:
  Meta's team builds a bug triage environment.
  Google's team can't use it — different API.
  Academic researcher can't benchmark against it.
  All work is siloed.

With OpenEnv:
  You build ClimateWatch with the OpenEnv API.
  Meta's agents can test against it immediately.
  Google's agents can test against it.
  Any researcher can fork it from Hugging Face.
  One standard → infinite reuse.
```

---

# MODULE 2 — Using Existing Environments

## How to Think About an Existing OpenEnv Environment

Every existing environment follows the same pattern:
1. A Docker container running a FastAPI server
2. A Pydantic model for Action (what agent sends)
3. A Pydantic model for Observation (what agent gets back)
4. A Pydantic model for State (current episode status)

---

## Type-Safe Models — The Most Important Concept in This Module

**Why type safety matters:**
```
Without types (dangerous):
  agent sends: {"fault": "VERY_BAD", "confidence": 5.0}
  → Your code silently accepts garbage
  → Grader crashes or gives wrong score
  → Hard to debug

With Pydantic (safe):
  agent sends: {"fault": "VERY_BAD", "confidence": 5.0}
  → Pydantic auto-rejects: "fault must be outlier/stuck/drift/..."
  → Pydantic auto-rejects: "confidence must be <= 1.0"
  → Agent immediately knows what's wrong
  → Grader never sees invalid data
```

**How Pydantic works:**
```python
from pydantic import BaseModel, Field
from typing import Literal

# This is a "typed model"
class FaultFlag(BaseModel):
    hour: int                                          # must be integer
    fault: Literal["outlier", "stuck", "drift",        # only these values allowed
                   "missing", "spike", "bias"]
    confidence: float = Field(ge=0.0, le=1.0)         # must be 0.0–1.0

# Valid data — Pydantic accepts it
flag = FaultFlag(hour=6, fault="outlier", confidence=0.9)
print(flag.hour)        # 6
print(flag.fault)       # "outlier"

# Invalid data — Pydantic auto-rejects with clear error
flag = FaultFlag(hour=6, fault="WRONG", confidence=5.0)
# ValidationError: fault must be one of outlier/stuck/drift/missing/spike/bias
# ValidationError: confidence must be <= 1.0
```

---

## The Echo Environment — Simplest Example to Understand

The official OpenEnv echo environment is the "Hello World" of OpenEnv.
It just echoes back whatever message you send, with reward = message_length × 0.1.

```python
# What the echo environment does:
# Action: {"message": "hello"}
# Observation: {"echoed_message": "hello", "done": false, "reward": 0.5}
# Reward: len("hello") × 0.1 = 5 × 0.1 = 0.5

# Client code to USE an existing environment:
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",  # HF router
    api_key=os.environ["HF_TOKEN"]
)

# OR use HTTP directly:
import requests
obs = requests.post("https://openenv-echo-env.hf.space/reset").json()
result = requests.post(
    "https://openenv-echo-env.hf.space/step",
    json={"action": {"message": "Hello World"}}
).json()
print(result["reward"])  # 1.1 (len=11 × 0.1)
```

---

## Policies — How Agents Interact With Environments

A "policy" is just a function that decides what action to take given an observation.
In your hackathon, the policy is an LLM (GPT-4 or Nemotron) reading sensor data.

```python
# Three types of policies:

# 1. Random policy (dumbest)
def random_policy(observation):
    return random.choice(["outlier", "stuck", "drift", "missing"])

# 2. Rule-based policy (smarter)
def rule_policy(observation):
    readings = observation["readings"]
    if readings[-1]["value"] == readings[-2]["value"] == readings[-3]["value"]:
        return {"fault": "stuck", "confidence": 0.9}
    return {"fault": "valid", "confidence": 0.8}

# 3. LLM policy (what your inference.py uses)
def llm_policy(observation, client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": str(observation)}]
    )
    return parse_response(response.choices[0].message.content)
```

---

## The Environment Hub

```
Hugging Face Hub for environments:
  → All OpenEnv environments are tagged with "openenv"
  → Fork any environment: openenv fork --repo-id openenv/echo_env
  → Deploy your own: openenv push
  → Browse at: huggingface.co/openenv

Your ClimateWatch will appear here after deployment.
Researchers worldwide can fork and use it.
```

---

# MODULE 3 — Deploying Environments

## The Deployment Journey (3 Stages)

```
Stage 1: Local Development
  Your laptop → uvicorn app.main:app --port 7860
  Test at: http://localhost:7860/docs

Stage 2: Docker
  Build image → docker build -t climatewatch .
  Run locally  → docker run -p 7860:7860 climatewatch
  Test at: http://localhost:7860/docs

Stage 3: Hugging Face Spaces
  Push code → git push
  HF builds and runs your Docker image
  Public URL: https://YOUR-USERNAME-climatewatch-env.hf.space
```

---

## Why Docker Is Required (The Core Idea)

```
Without Docker (problem):
  Your environment uses Python 3.11, numpy 1.24, fastapi 0.104
  Judge's machine has Python 3.9, different numpy
  Your environment crashes on their machine
  Disqualified.

With Docker (solution):
  Docker container has EXACTLY: Python 3.11, numpy 1.24, fastapi 0.104
  Same on your machine. Same on judge's machine. Same on HF Spaces.
  Guaranteed identical execution.
```

**Your entire Dockerfile — just 7 lines:**
```dockerfile
FROM python:3.11-slim        # Start with Python 3.11
WORKDIR /app                 # All commands run from /app
COPY requirements.txt .      # Copy requirements first (caching)
RUN pip install -r requirements.txt  # Install packages
COPY . .                     # Copy your code
EXPOSE 7860                  # HF Spaces uses port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Hugging Face Spaces — How It Works

```
1. You create a Space (free) at huggingface.co/spaces
2. HF gives you a git repo
3. You push your code to that repo (including Dockerfile)
4. HF automatically:
   → Reads your Dockerfile
   → Runs: docker build
   → Runs: docker run
   → Gives you a public URL
5. When you push updates → HF rebuilds automatically

Your public URL format:
  https://YOUR-USERNAME-climatewatch-env.hf.space

This URL must respond to:
  GET  /health  → 200 OK
  POST /reset   → observation JSON
```

**README.md header required for HF Spaces:**
```yaml
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

## The openenv CLI

```bash
# Initialize a new environment from template
openenv init climatewatch-env
# Creates: models.py, client.py, server/, openenv.yaml

# Validate your environment (run before submitting)
openenv validate
# Checks: openenv.yaml exists, typed models, Dockerfile, endpoints respond

# Push to HF Spaces
openenv push
# Builds Docker image, pushes to your HF Space
```

---

## Local Development Workflow

```bash
# Day-to-day development cycle:

# 1. Edit your code
# 2. Run locally (fast, no Docker needed)
uvicorn app.main:app --reload --port 7860

# 3. Test using /docs in browser
# Open: http://localhost:7860/docs

# 4. When ready, test with Docker
docker build -t climatewatch .
docker run -p 7860:7860 climatewatch

# 5. Test Docker version
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detect"}'

# 6. Push to HF Spaces
git add . && git commit -m "update" && git push
```

---

# MODULE 4 — Building Your Own Environment

## The 3-Component Pattern

Every OpenEnv environment you build has exactly 3 components:

```
Component 1: MODELS (models.py)
  → Define what the agent sends (Action)
  → Define what the agent receives (Observation)
  → Define the current status (State)
  → All are Pydantic BaseModel subclasses

Component 2: ENVIRONMENT LOGIC (environment.py)
  → reset() — wipe state, load new scenario, return first observation
  → step() — process action, compute reward, return new observation
  → state() — return current episode status

Component 3: SERVER (server/app.py or app/main.py)
  → FastAPI app
  → Routes that call your environment logic
  → /reset → calls env.reset()
  → /step  → calls env.step(action)
  → /state → calls env.state()
```

---

## The Scaffold → Build → Deploy Cycle

```
Step 1: Scaffold
  openenv init climatewatch-env
  → Creates empty template with all 3 components

Step 2: Define Your Models
  What does the agent see? → Design Observation
  What does the agent do?  → Design Action
  What's the episode status? → Design State

Step 3: Implement reset()
  → Load a scenario from your dataset
  → Set step_count = 0, total_reward = 0
  → Return first observation

Step 4: Implement step()
  → Validate the action (Pydantic does this automatically)
  → Compare action to ground truth
  → Compute reward (partial credit for partial correctness)
  → Return new observation + reward + done

Step 5: Implement graders
  → deterministic function: grader(action, ground_truth) → float 0.0–1.0
  → must return DIFFERENT values for DIFFERENT inputs

Step 6: Write inference.py
  → Calls reset() → gets observation
  → Sends observation to LLM
  → Sends LLM response to step()
  → Repeats until done

Step 7: Docker + HF Spaces
  → Write Dockerfile
  → Push to HF Space
  → Run openenv validate
```

---

## Reward Function Design — The Core Skill

```python
# BAD reward (binary — agent learns nothing between steps):
def bad_reward(action, ground_truth):
    if action == ground_truth:
        return 1.0
    return 0.0
# Problem: agent gets 0 for everything until perfect → no learning signal

# GOOD reward (partial credit — agent can improve gradually):
def good_reward(action, ground_truth):
    reward = 0.0

    # Layer 1: fault type (most important)
    if action.fault == ground_truth.fault:
        reward += 0.5          # full credit
    elif same_family(action.fault, ground_truth.fault):
        reward += 0.2          # partial credit (e.g. "noise" vs "spike")

    # Layer 2: correct hour identification
    if action.hour == ground_truth.hour:
        reward += 0.3

    # Layer 3: confidence calibration
    if abs(action.confidence - expected_confidence) < 0.1:
        reward += 0.1

    # Layer 4: penalties
    if action == previous_action:   # repetition = loop
        reward -= 0.3
    if action.confidence == 1.0:    # overconfident = bad signal
        reward -= 0.05

    return max(0.0, min(1.0, reward))  # always clamp to 0.0–1.0
```

---

## State Management — The Most Common Mistake

```python
class ClimateWatchEnvironment:

    def __init__(self):
        self._clear()    # Initialize with clean state

    def _clear(self):
        """WIPE EVERYTHING. Called by reset()."""
        self.episode_id   = None
        self.task_id      = None
        self.step_count   = 0
        self.total_reward = 0.0
        self.done         = False
        self.ground_truth = None
        self.scenario     = None
        self.history      = []     # track past actions (for loop detection)

    def reset(self, task_id, seed=None):
        self._clear()               # ← ALWAYS wipe first
        self.episode_id   = str(uuid.uuid4())
        self.task_id      = task_id
        self.scenario     = load_scenario(task_id, seed)
        self.ground_truth = load_ground_truth(task_id, seed)
        return build_observation(self)

    def step(self, action):
        if self.done:
            raise ValueError("Episode is over. Call reset() first.")

        reward = compute_reward(action, self.ground_truth, self.history)
        self.total_reward += reward
        self.step_count   += 1
        self.history.append(action)

        # Episode ends when task complete OR max steps reached
        self.done = task_complete(action, self.ground_truth) or self.step_count >= 5

        return build_observation(self, reward=reward)

# MOST COMMON MISTAKE: forgetting to wipe state in reset()
# If you don't wipe, episode 2 starts with episode 1's data → grader breaks
```

---

## Complete Working Example — Minimal Environment

```python
# models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class SimpleAction(BaseModel):
    fault_type: Literal["valid", "outlier", "stuck"]
    confidence: float = Field(ge=0.0, le=1.0)

class SimpleObservation(BaseModel):
    done: bool = False
    reward: float = 0.0
    sensor_value: float
    step_count: int = 0
    feedback: str = ""

class SimpleState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False
```

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from models import SimpleAction, SimpleObservation, SimpleState

app = FastAPI()

# Global environment state
state = {"episode_id": None, "step": 0, "reward": 0.0, "done": False,
         "answer": None, "value": None}

@app.post("/reset", response_model=SimpleObservation)
def reset(task_id: str = "task1"):
    state.update({"episode_id": str(uuid.uuid4()), "step": 0,
                  "reward": 0.0, "done": False,
                  "answer": "outlier", "value": 9999.0})
    return SimpleObservation(sensor_value=9999.0,
                             feedback="Is this sensor reading valid?")

@app.post("/step", response_model=SimpleObservation)
def step(action: SimpleAction):
    if not state["episode_id"]:
        raise HTTPException(400, "Call /reset first")

    reward = 0.5 if action.fault_type == state["answer"] else 0.0
    state["reward"] += reward
    state["step"] += 1
    state["done"] = True   # one step episode

    return SimpleObservation(done=True, reward=reward,
                             sensor_value=state["value"],
                             step_count=state["step"],
                             feedback="Correct!" if reward > 0 else "Wrong!")

@app.get("/state", response_model=SimpleState)
def get_state():
    return SimpleState(episode_id=state["episode_id"],
                       step_count=state["step"],
                       total_reward=state["reward"],
                       done=state["done"])

@app.get("/health")
def health():
    return {"status": "healthy"}
```

---

## The inference.py Pattern (Hackathon Requirement)

```python
# inference.py — MUST be named exactly this, in root directory
import os
import json
import requests
from openai import OpenAI

# Read from environment variables — NEVER hardcode
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

# OpenAI Client with HF router
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def run_task(task_id: str) -> float:
    obs = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}).json()
    total_reward = 0.0
    done = False

    while not done:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": json.dumps(obs)}],
            response_format={"type": "json_object"},
            temperature=0.0,        # deterministic = reproducible
            max_tokens=500
        )
        action = json.loads(response.choices[0].message.content)
        result = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
        total_reward += result.get("reward", 0.0)
        done = result.get("done", False)
        obs = result

    return total_reward

if __name__ == "__main__":
    tasks = ["task1_detect", "task2_clean", "task3_cascade"]
    for task in tasks:
        score = run_task(task)
        print(f"{task}: {score:.4f}")
```

---

## Key Things to Remember

```
1. reset() MUST wipe all state — never let state leak between episodes
2. Graders MUST return different scores for different inputs
3. Rewards MUST give partial credit — never just 0 or 1
4. Pydantic models validate inputs automatically — use Field() for constraints
5. FastAPI /docs gives you a free interactive UI — always test there first
6. Docker port MUST be 7860 for Hugging Face Spaces
7. inference.py MUST be in root directory (not in app/ folder)
8. Use HF_TOKEN and API_BASE_URL — not OPENAI_API_KEY
9. Total inference runtime MUST be under 20 minutes
10. Environment must run on 2 vCPU + 8GB RAM (keep it lean)
```

---

*ALGORITHMIC AVENGERS — Ramana Ganthan  + Sre Sandhya *
