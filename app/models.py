"""
ClimateWatch — Pydantic Models
All request/response schemas for the API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any


# ── Sensor primitives ─────────────────────────────────────────────────────────

class DailyReading(BaseModel):
    day: int
    value: Optional[float] = None
    status: str = "OK"   # OK | SENSOR_OFFLINE | CORRUPTED | NOISY


class HourlyReading(BaseModel):
    hour: int
    value: Optional[float] = None


# ── Task 1: Single Sensor Anomaly Detection ───────────────────────────────────

class FaultFlag(BaseModel):
    hour: int = Field(ge=0, le=23)
    fault: Literal["outlier", "missing", "stuck", "drift", "spike", "bias", "valid"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DetectAction(BaseModel):
    sensor_id: str
    flags: List[FaultFlag] = []


# ── Task 2: Multi-Sensor Data Cleaning ───────────────────────────────────────

class SensorDiagnosis(BaseModel):
    sensor_id: str
    fault_type: Literal["drift", "missing", "bias", "noise", "stuck", "spike", "valid"]
    severity: Literal["none", "low", "medium", "high", "critical"]
    fix: Literal[
        "no_action", "interpolate", "recalibrate",
        "offset_correction", "smooth", "flag_only", "replace"
    ]
    fix_params: Dict[str, Any] = {}


class CleanAction(BaseModel):
    diagnoses: List[SensorDiagnosis]


# ── Task 3: Cascade Failure & Compliance ──────────────────────────────────────

class ComplianceCheck(BaseModel):
    parameter: str
    status: Literal[
        "CLEAN", "POSSIBLE_VIOLATION",
        "CONFIRMED_VIOLATION", "INSUFFICIENT_DATA"
    ]
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = ""


class CascadeAction(BaseModel):
    root_cause_sensors: List[str]
    repair_order: List[str]
    fault_window_start: str   # e.g. "day_8"
    fault_window_end: str     # e.g. "day_21"
    compliance_checks: List[ComplianceCheck] = []
    recommended_action: Literal[
        "no_action", "flag_for_review",
        "file_compliance_report", "emergency_shutdown"
    ] = "flag_for_review"


# ── API Request models ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_detect"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ── Observation & State ───────────────────────────────────────────────────────

class SensorObservation(BaseModel):
    done: bool = False
    reward: float = 0.0
    task_id: str = ""
    step_count: int = 0
    sensor_data: Any = None
    feedback: str = ""
    metadata: Dict[str, Any] = {}


class SensorState(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False


# ── Grader / Baseline responses ───────────────────────────────────────────────

class GraderResponse(BaseModel):
    episode_id: Optional[str]
    task_id: Optional[str]
    final_score: float
    step_count: int
    breakdown: Dict[str, Any] = {}


class BaselineResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int
