---
name: ClimateWatch project status
description: All files built for ClimateWatch OpenEnv hackathon submission
type: project
---

Complete project built at c:\Users\2471999\Documents\BuildVerse.

**Why:** OpenEnv hackathon (Meta/HuggingFace). All requirements from 2_REQUIREMENTS.md implemented.

**How to apply:** All files are in place. Next step: push to HuggingFace Space as Docker space.

Key files built (2026-03-28):
- inference.py (root) — baseline LLM agent using HF router
- openenv.yaml — task metadata
- Dockerfile — python:3.11-slim, port 7860
- requirements.txt
- README.md — with HF Spaces YAML header
- app/main.py — all 7 endpoints (reset, step, state, health, tasks, grader, baseline)
- app/environment.py — thread-safe episode management
- app/reward.py — per-step partial rewards, anti-loop penalty
- app/models.py — Pydantic schemas
- app/tasks/task1_detect.py — 20 scenarios, F1 grader
- app/tasks/task2_clean.py — 10 scenarios, weighted fault+fix grader
- app/tasks/task3_cascade.py — 5 cascade scenarios, 4-component grader
- tests/test_graders.py + tests/test_endpoints.py

Task 3 is the hardest — requires topological graph reasoning, temporal analysis, and epistemic compliance reasoning. Designed to genuinely challenge GPT-4 and Nemotron.
