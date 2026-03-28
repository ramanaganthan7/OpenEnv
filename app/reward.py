"""
ClimateWatch — Reward Function
==============================
Per-step reward design principles:
  1. Reward at EVERY step (not just final)
  2. Partial rewards — continuous 0.0–1.0, not binary
  3. Penalise loops — same action repeated gets a 0.30 penalty
  4. Penalise regression — getting worse than previous step
  5. Never returns NaN or values outside [0.0, 1.0]
"""
from __future__ import annotations
import json
from typing import Any, Dict, List

from app.tasks.task1_detect  import grade_task1
from app.tasks.task2_clean   import grade_task2
from app.tasks.task3_cascade import grade_task3

_GRADERS = {
    "task1_detect":  grade_task1,
    "task2_clean":   grade_task2,
    "task3_cascade": grade_task3,
}

_LOOP_PENALTY       = 0.30   # same action as previous step
_REGRESSION_PENALTY = 0.05   # score got worse vs previous best


def compute_reward(
    action: Dict[str, Any],
    ground_truth: Dict[str, Any],
    task_id: str,
    history: List[Dict[str, Any]],
) -> float:
    """
    Return a per-step reward in [0.0, 1.0].

    Parameters
    ----------
    action       : The agent's current action dict
    ground_truth : The correct answer for this episode
    task_id      : "task1_detect" | "task2_clean" | "task3_cascade"
    history      : List of all previous actions in this episode
    """
    grader = _GRADERS.get(task_id)
    if grader is None:
        return 0.0

    base_score = grader(action, ground_truth)
    base_score = max(0.0, min(1.0, base_score))   # safety clamp

    # ── Anti-loop penalty ─────────────────────────────────────────
    if history and _actions_equal(action, history[-1]):
        base_score = max(0.0, base_score - _LOOP_PENALTY)

    # ── Regression penalty ────────────────────────────────────────
    if history:
        previous_scores = [grader(h, ground_truth) for h in history[-3:]]
        best_previous = max(previous_scores) if previous_scores else 0.0
        if base_score < best_previous - 0.05:
            base_score = max(0.0, base_score - _REGRESSION_PENALTY)

    return round(max(0.0, min(1.0, base_score)), 4)


def _actions_equal(a1: Dict, a2: Dict) -> bool:
    """JSON-based action equality (order-independent)."""
    try:
        return (json.dumps(a1, sort_keys=True, default=str) ==
                json.dumps(a2, sort_keys=True, default=str))
    except Exception:
        return False
