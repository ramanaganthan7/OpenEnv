---
name: ClimateWatch project status
description: All files built and tested for ClimateWatch OpenEnv hackathon submission
type: project
---

Complete project built and verified at c:\Users\2471999\Documents\BuildVerse.

**Why:** OpenEnv hackathon (Meta × HuggingFace). Submission via Docker Space on HuggingFace.

**Space URL:** https://huggingface.co/spaces/ramanaganthan7/climatewatch-env
**HF Username:** ramanaganthan7
**HF Token:** your_hf_token_here (stored, treat as secret)

**How to apply:** All files in place. Push via `git push huggingface master`. Set 4 env vars in Space Settings.

**Verified working (2026-03-29):**
- 63/63 tests pass
- All 7 endpoints return correct responses
- Reward: partial credit (0.2857 for partial answer), anti-loop penalty works (0.0 on repeat)
- Packages updated to Python 3.13 compatible versions (pydantic 2.10.4, fastapi 0.115.6)
- `scripts/kill_port.py` auto-runs before server to prevent port conflicts
- `cwd` in POST /baseline fixed to use dynamic project root (not hardcoded /app)

**Files built:**
- inference.py, openenv.yaml, Dockerfile, requirements.txt, pyproject.toml
- README.md, DEPLOYMENT_GUIDE.md
- app/main.py, app/environment.py, app/reward.py, app/models.py
- app/tasks/task1_detect.py (20 scenarios), task2_clean.py (10), task3_cascade.py (5)
- app/data/ (folder exists, scenarios generated programmatically)
- scripts/kill_port.py
- tests/test_graders.py (22 tests), tests/test_endpoints.py (41 tests)
