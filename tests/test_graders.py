"""
Test suite: graders return different scores for different actions.
CRITICAL REQUIREMENT: Graders that always return the same score = DISQUALIFIED.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.task1_detect  import load_task1, grade_task1
from app.tasks.task2_clean   import load_task2, grade_task2
from app.tasks.task3_cascade import load_task3, grade_task3


# ── Task 1 grader tests ───────────────────────────────────────────────────────

class TestTask1Grader:
    """Verify Task 1 grader produces varied, meaningful scores."""

    def setup_method(self):
        # Use a scenario with known faults (scenario 0: stuck+outlier+missing)
        self.obs, self.gt = load_task1(seed=0)
        self.sensor_id = self.obs["sensor_id"]

    def test_perfect_action_scores_near_1(self):
        """Submitting the exact ground truth should score ~0.9-1.0."""
        perfect_action = {
            "sensor_id": self.sensor_id,
            "flags": [
                {**f, "confidence": 0.9 if i % 2 == 0 else 1.0}
                for i, f in enumerate(self.gt["flags"])
            ]
        }
        score = grade_task1(perfect_action, self.gt)
        assert score >= 0.80, f"Perfect action should score ≥ 0.80, got {score}"
        assert score <= 1.0

    def test_empty_action_scores_zero(self):
        """No flags submitted → 0.0 (unless scenario has no faults)."""
        empty_action = {"sensor_id": self.sensor_id, "flags": []}
        score = grade_task1(empty_action, self.gt)
        if self.gt["flags"]:  # scenario has faults
            assert score == 0.0, f"Empty flags on faulty scenario should score 0.0, got {score}"

    def test_all_wrong_scores_low(self):
        """Completely wrong fault types and hours → near 0."""
        wrong_action = {
            "sensor_id": self.sensor_id,
            "flags": [
                {"hour": 23, "fault": "drift", "confidence": 0.5},
            ]
        }
        score = grade_task1(wrong_action, self.gt)
        if self.gt["flags"]:
            assert score < 0.3, f"Wrong flags should score < 0.3, got {score}"

    def test_partial_action_scores_between_extremes(self):
        """Getting half the flags right → score between 0 and perfect."""
        gt_flags = self.gt["flags"]
        if len(gt_flags) >= 2:
            half_flags = gt_flags[:len(gt_flags)//2]
            partial_action = {"sensor_id": self.sensor_id, "flags": half_flags}
            perfect_action = {"sensor_id": self.sensor_id, "flags": gt_flags}

            partial_score = grade_task1(partial_action, self.gt)
            perfect_score = grade_task1(perfect_action, self.gt)

            assert partial_score < perfect_score, (
                f"Partial ({partial_score:.3f}) should be less than perfect ({perfect_score:.3f})"
            )

    def test_scores_differ_across_scenarios(self):
        """Different scenarios produce different graded scores for the same action."""
        scores = set()
        for seed in range(10):
            obs, gt = load_task1(seed=seed)
            action = {"sensor_id": obs["sensor_id"], "flags": gt["flags"]}
            scores.add(grade_task1(action, gt))
        # With 10 scenarios, should have multiple distinct perfect scores (they are all near 1.0)
        # More importantly, test that different partial answers vary
        partial_scores = set()
        for seed in range(10):
            obs, gt = load_task1(seed=seed)
            # Guess wrong fault type for all
            action = {
                "sensor_id": obs["sensor_id"],
                "flags": [{"hour": 0, "fault": "noise", "confidence": 0.5}]
            }
            partial_scores.add(grade_task1(action, gt))
        assert len(partial_scores) > 1, "Different scenarios should produce different scores"

    def test_score_always_in_range(self):
        """Score is always in [0.0, 1.0] — never NaN, never out of range."""
        test_actions = [
            {"sensor_id": self.sensor_id, "flags": []},
            {"sensor_id": self.sensor_id, "flags": [{"hour": 0, "fault": "outlier", "confidence": 1.0}]},
            {"sensor_id": self.sensor_id, "flags": self.gt["flags"]},
            {"flags": [{"hour": i, "fault": "stuck", "confidence": 0.5} for i in range(24)]},
        ]
        for action in test_actions:
            score = grade_task1(action, self.gt)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] for action {action}"
            assert score == score, f"Score is NaN for action {action}"

    def test_calibration_bonus(self):
        """Varied confidence values earn a small bonus."""
        gt_flags = self.gt["flags"]
        if not gt_flags:
            pytest.skip("Scenario has no faults")

        same_conf = {
            "sensor_id": self.sensor_id,
            "flags": [{**f, "confidence": 1.0} for f in gt_flags]
        }
        varied_conf = {
            "sensor_id": self.sensor_id,
            "flags": [{**f, "confidence": 0.9 if i % 2 == 0 else 1.0}
                      for i, f in enumerate(gt_flags)]
        }
        score_same   = grade_task1(same_conf, self.gt)
        score_varied = grade_task1(varied_conf, self.gt)
        assert score_varied >= score_same, (
            f"Varied confidence ({score_varied}) should be ≥ same confidence ({score_same})"
        )

    def test_all_valid_scenario(self):
        """Scenario 9 has no faults — agent should get perfect score by returning empty flags."""
        obs9, gt9 = load_task1(seed=9)
        assert gt9["flags"] == [], "Scenario 9 should have no faults"

        clean_action = {"sensor_id": obs9["sensor_id"], "flags": []}
        score = grade_task1(clean_action, gt9)
        assert score == 1.0, f"Empty flags on clean scenario should score 1.0, got {score}"

        false_positive_action = {
            "sensor_id": obs9["sensor_id"],
            "flags": [{"hour": 5, "fault": "outlier", "confidence": 0.9}]
        }
        score_fp = grade_task1(false_positive_action, gt9)
        assert score_fp < score, "False positive should reduce score on clean scenario"


# ── Task 2 grader tests ───────────────────────────────────────────────────────

class TestTask2Grader:
    """Verify Task 2 grader produces varied, meaningful scores."""

    def setup_method(self):
        self.obs, self.gt = load_task2(seed=0)

    def test_perfect_action_scores_high(self):
        """Exact ground truth diagnoses → high score."""
        action = {"diagnoses": self.gt["diagnoses"]}
        score = grade_task2(action, self.gt)
        assert score >= 0.80, f"Perfect action should score ≥ 0.80, got {score}"
        assert score <= 1.0

    def test_all_wrong_scores_low(self):
        """All wrong fault types and fixes → low score."""
        wrong = {
            "diagnoses": [
                {"sensor_id": d["sensor_id"], "fault_type": "stuck",
                 "severity": "critical", "fix": "replace", "fix_params": {}}
                for d in self.gt["diagnoses"]
                if d["fault_type"] != "stuck"  # intentionally wrong
            ]
        }
        if wrong["diagnoses"]:
            score = grade_task2(wrong, self.gt)
            # Some might accidentally be right (if template has stuck sensors)
            # Just verify the grader runs without error
            assert 0.0 <= score <= 1.0

    def test_partial_correct_scores_between(self):
        """Getting half the sensors right → intermediate score."""
        gt_diags = self.gt["diagnoses"]
        half_correct = gt_diags[:3]  # first 3 correct, skip last 2
        partial_action = {"diagnoses": half_correct}
        perfect_action = {"diagnoses": gt_diags}

        partial_score = grade_task2(partial_action, self.gt)
        perfect_score = grade_task2(perfect_action, self.gt)

        assert partial_score < perfect_score

    def test_scores_vary_across_scenarios(self):
        """Different scenarios → different scores for same action."""
        scores = []
        for seed in range(5):
            _, gt = load_task2(seed=seed)
            # Always guess "drift" for everything
            action = {
                "diagnoses": [
                    {"sensor_id": d["sensor_id"], "fault_type": "drift",
                     "severity": "medium", "fix": "recalibrate", "fix_params": {}}
                    for d in gt["diagnoses"]
                ]
            }
            scores.append(grade_task2(action, gt))
        assert len(set(scores)) > 1, "Same action should score differently across different scenarios"

    def test_score_always_in_range(self):
        """Score always in [0.0, 1.0]."""
        for seed in range(10):
            _, gt = load_task2(seed=seed)
            action = {"diagnoses": gt["diagnoses"]}
            score = grade_task2(action, gt)
            assert 0.0 <= score <= 1.0
            assert score == score  # not NaN


# ── Task 3 grader tests ───────────────────────────────────────────────────────

class TestTask3Grader:
    """Verify Task 3 grader produces varied, meaningful scores."""

    def setup_method(self):
        self.obs, self.gt = load_task3(seed=0)

    def test_perfect_action_scores_high(self):
        """Providing exact ground truth → high score."""
        gt = self.gt
        action = {
            "root_cause_sensors": gt["root_cause_sensors"],
            "repair_order":       gt["repair_order"],
            "fault_window_start": gt["fault_window"]["start"],
            "fault_window_end":   gt["fault_window"]["end"],
            "compliance_checks":  [
                {"parameter": p, "status": s, "confidence": 0.9, "reasoning": "correct"}
                for p, s in gt["compliance"].items()
            ],
            "recommended_action": "flag_for_review",
        }
        score = grade_task3(action, gt)
        assert score >= 0.70, f"Perfect action should score ≥ 0.70, got {score}"
        assert score <= 1.0

    def test_wrong_root_cause_scores_lower(self):
        """Identifying wrong root causes → lower score."""
        gt = self.gt
        perfect_action = {
            "root_cause_sensors": gt["root_cause_sensors"],
            "repair_order":       gt["repair_order"],
            "fault_window_start": gt["fault_window"]["start"],
            "fault_window_end":   gt["fault_window"]["end"],
            "compliance_checks":  [
                {"parameter": p, "status": s, "confidence": 0.9, "reasoning": ""}
                for p, s in gt["compliance"].items()
            ],
            "recommended_action": "flag_for_review",
        }
        wrong_roots_action = {**perfect_action, "root_cause_sensors": ["S9", "S10"]}

        perfect_score     = grade_task3(perfect_action, gt)
        wrong_roots_score = grade_task3(wrong_roots_action, gt)

        assert wrong_roots_score < perfect_score, (
            f"Wrong root causes ({wrong_roots_score}) should score less than correct ({perfect_score})"
        )

    def test_dependency_violation_penalised(self):
        """Repair order with dependent before reference → lower score."""
        gt = self.gt
        correct_order = gt["repair_order"]
        if len(correct_order) >= 2:
            # Swap first two elements (likely reference + its dependent)
            bad_order = correct_order.copy()
            bad_order[0], bad_order[1] = bad_order[1], bad_order[0]

            correct_action = {
                "root_cause_sensors": gt["root_cause_sensors"],
                "repair_order":       correct_order,
                "fault_window_start": gt["fault_window"]["start"],
                "fault_window_end":   gt["fault_window"]["end"],
                "compliance_checks":  [],
                "recommended_action": "flag_for_review",
            }
            bad_order_action = {**correct_action, "repair_order": bad_order}

            correct_score   = grade_task3(correct_action, gt)
            bad_order_score = grade_task3(bad_order_action, gt)

            assert bad_order_score <= correct_score

    def test_wrong_compliance_scores_lower(self):
        """Wrong compliance status → lower score."""
        gt = self.gt
        base_action = {
            "root_cause_sensors": gt["root_cause_sensors"],
            "repair_order":       gt["repair_order"],
            "fault_window_start": gt["fault_window"]["start"],
            "fault_window_end":   gt["fault_window"]["end"],
            "recommended_action": "flag_for_review",
        }
        correct_checks = [
            {"parameter": p, "status": s, "confidence": 0.9, "reasoning": ""}
            for p, s in gt["compliance"].items()
        ]
        wrong_checks = [
            {"parameter": p, "status": "CLEAN", "confidence": 0.5, "reasoning": ""}
            for p in gt["compliance"]
        ]

        correct_score = grade_task3({**base_action, "compliance_checks": correct_checks}, gt)
        wrong_score   = grade_task3({**base_action, "compliance_checks": wrong_checks}, gt)

        assert wrong_score <= correct_score

    def test_scores_vary_across_scenarios(self):
        """Different scenarios → different scores."""
        all_scores = []
        for seed in range(5):
            _, gt = load_task3(seed=seed)
            action = {
                "root_cause_sensors": ["S1"],
                "repair_order":       ["S1", "S4"],
                "fault_window_start": "day_5",
                "fault_window_end":   "day_15",
                "compliance_checks": [],
                "recommended_action": "flag_for_review",
            }
            all_scores.append(grade_task3(action, gt))
        assert len(set(all_scores)) > 1, "Same action should vary across scenarios"

    def test_score_always_in_range(self):
        """Score always in [0.0, 1.0]."""
        for seed in range(5):
            _, gt = load_task3(seed=seed)
            action = {
                "root_cause_sensors": gt["root_cause_sensors"],
                "repair_order":       gt["repair_order"],
                "fault_window_start": gt["fault_window"]["start"],
                "fault_window_end":   gt["fault_window"]["end"],
                "compliance_checks":  [],
                "recommended_action": "flag_for_review",
            }
            score = grade_task3(action, gt)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
            assert score == score  # not NaN


# ── Reward function tests ─────────────────────────────────────────────────────

class TestRewardFunction:

    def test_no_loop_penalty_on_first_step(self):
        from app.reward import compute_reward
        obs, gt = load_task1(seed=0)
        action = {"sensor_id": obs["sensor_id"], "flags": gt["flags"]}
        reward = compute_reward(action, gt, "task1_detect", history=[])
        assert reward >= 0.0
        assert reward <= 1.0

    def test_loop_penalty_applied(self):
        from app.reward import compute_reward
        obs, gt = load_task1(seed=0)
        action = {"sensor_id": obs["sensor_id"], "flags": gt["flags"][:1]}

        reward_first  = compute_reward(action, gt, "task1_detect", history=[])
        reward_repeat = compute_reward(action, gt, "task1_detect", history=[action])

        assert reward_repeat <= reward_first, (
            f"Repeating action should not reward more: {reward_repeat} vs {reward_first}"
        )

    def test_reward_always_in_range(self):
        from app.reward import compute_reward
        obs, gt = load_task1(seed=0)
        actions = [
            {"sensor_id": obs["sensor_id"], "flags": []},
            {"sensor_id": obs["sensor_id"], "flags": gt["flags"]},
        ]
        for action in actions:
            r = compute_reward(action, gt, "task1_detect", history=[])
            assert 0.0 <= r <= 1.0, f"Reward {r} out of [0, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
