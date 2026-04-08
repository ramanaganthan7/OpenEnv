"""
ClimateWatch — Core Environment
================================
Thread-safe, stateful OpenEnv environment.
Manages episodes, dispatches to task loaders/graders, computes rewards.
"""
from __future__ import annotations

import uuid
import threading
from typing import Optional, Any, Dict, List

from app.models import SensorObservation, SensorState
from app.tasks  import TASK_LOADERS, TASK_GRADERS
from app.reward import compute_reward

MAX_STEPS     = 5
DONE_THRESHOLD = 0.80   # episode ends early if score reaches this


class ClimateWatchEnvironment:
    """
    Stateful environment — one active episode at a time.
    Thread-safe via a reentrant lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        self.episode_id:   Optional[str]       = None
        self.task_id:      Optional[str]       = None
        self.step_count:   int                 = 0
        self.total_reward: float               = 0.0
        self.done:         bool                = False
        self.scenario:     Optional[Any]       = None
        self.ground_truth: Optional[Any]       = None
        self.history:      List[Dict]          = []
        self.last_action:  Optional[Dict]      = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self,
              task_id: str = "task1_detect",
              seed: Optional[int] = None) -> SensorObservation:
        """Start a fresh episode."""
        with self._lock:
            if task_id not in TASK_LOADERS:
                raise ValueError(f"Unknown task_id: {task_id!r}. "
                                 f"Valid: {list(TASK_LOADERS.keys())}")
            self._clear()
            self.episode_id  = str(uuid.uuid4())
            self.task_id     = task_id
            self.scenario, self.ground_truth = TASK_LOADERS[task_id](seed=seed)

            return SensorObservation(
                done=False,
                reward=0.0,
                task_id=task_id,
                step_count=0,
                sensor_data=self.scenario,
                feedback=self._welcome(task_id),
                metadata={
                    "episode_id": self.episode_id,
                    "max_steps":  MAX_STEPS,
                },
            )

    def step(self, raw_action: Dict) -> SensorObservation:
        """Submit an action and receive reward + observation."""
        with self._lock:
            if self.episode_id is None:
                raise RuntimeError("Call POST /reset before POST /step.")
            if self.done:
                raise RuntimeError("Episode is finished. Call POST /reset to start a new one.")

            reward = compute_reward(
                raw_action, self.ground_truth,
                self.task_id, self.history
            )
            self.total_reward += reward
            self.step_count   += 1
            self.history.append(raw_action)
            self.last_action  = raw_action

            # Grade to check early-stop
            grader        = TASK_GRADERS[self.task_id]
            episode_score = grader(raw_action, self.ground_truth)

            self.done = (
                self.step_count >= MAX_STEPS or
                episode_score >= DONE_THRESHOLD
            )

            feedback = self._feedback(reward, episode_score)

            return SensorObservation(
                done=self.done,
                reward=round(reward, 4),
                task_id=self.task_id,
                step_count=self.step_count,
                sensor_data=self.scenario,
                feedback=feedback,
                metadata={
                    "episode_id":    self.episode_id,
                    "episode_score": round(episode_score, 4),
                    "total_reward":  round(self.total_reward, 4),
                    "steps_left":    max(0, MAX_STEPS - self.step_count),
                },
            )

    def state(self) -> SensorState:
        with self._lock:
            return SensorState(
                episode_id=self.episode_id,
                task_id=self.task_id,
                step_count=self.step_count,
                total_reward=round(self.total_reward, 4),
                done=self.done,
            )

    def final_grade(self) -> float:
        """Return final grader score for the last action submitted (strictly in (0.01, 0.99))."""
        with self._lock:
            if self.last_action is None or self.ground_truth is None:
                return 0.01
            raw = TASK_GRADERS[self.task_id](self.last_action, self.ground_truth)
            return round(max(0.01, min(0.99, raw)), 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _welcome(task_id: str) -> str:
        msgs = {
            "task1_detect": (
                "Episode started. You have 5 steps to analyse 24 hours of sensor data "
                "and flag all faulty readings by hour and fault type."
            ),
            "task2_clean": (
                "Episode started. You have 5 steps to diagnose all 5 sensors in this "
                "7-day multi-sensor dataset and recommend the correct fix for each."
            ),
            "task3_cascade": (
                "Episode started. Analyse this 30-day 10-sensor network. Identify root "
                "cause sensors, repair order, fault window, and compliance status."
            ),
        }
        return msgs.get(task_id, "Episode started. Analyse the sensor data.")

    @staticmethod
    def _feedback(reward: float, episode_score: float) -> str:
        if episode_score >= 0.90:
            return (f"Excellent analysis — score {episode_score:.2f}. "
                    "Near-perfect fault detection and diagnosis.")
        if episode_score >= 0.70:
            return (f"Good analysis — score {episode_score:.2f}. "
                    "Most faults correctly identified. Review missed detections.")
        if episode_score >= 0.45:
            return (f"Partial analysis — score {episode_score:.2f}. "
                    "Several faults missed or misclassified. "
                    "Re-examine sensor statistics and time patterns.")
        if episode_score >= 0.20:
            return (f"Low accuracy — score {episode_score:.2f}. "
                    "Reconsider your fault taxonomy: "
                    "outlier=single spike, stuck=repeated value, "
                    "drift=gradual shift, bias=constant offset.")
        return (f"Very low accuracy — score {episode_score:.2f}. "
                "Start fresh: compare each reading to the stated normal_range "
                "and look for statistical anomalies.")
