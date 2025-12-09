"""Tests for evaluation metrics."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    compute_accuracy,
    compute_step_validity,
    compute_trace_validity,
    compute_brier_score,
    compute_ece,
    compute_calibration_curve,
)
from src.data.structures import (
    LogicalState,
    ProofStep,
    OptionType,
    OptionizedTrace,
)


class TestAccuracy:
    """Tests for accuracy computation."""
    
    def test_perfect_accuracy(self):
        """Should compute 100% accuracy for perfect predictions."""
        predictions = ["TRUE", "FALSE", "UNKNOWN"]
        labels = ["TRUE", "FALSE", "UNKNOWN"]
        assert compute_accuracy(predictions, labels) == 1.0
    
    def test_zero_accuracy(self):
        """Should compute 0% accuracy for all wrong predictions."""
        predictions = ["FALSE", "TRUE", "TRUE"]
        labels = ["TRUE", "FALSE", "UNKNOWN"]
        assert compute_accuracy(predictions, labels) == 0.0
    
    def test_partial_accuracy(self):
        """Should compute partial accuracy correctly."""
        predictions = ["TRUE", "TRUE", "FALSE", "FALSE"]
        labels = ["TRUE", "FALSE", "FALSE", "TRUE"]
        assert compute_accuracy(predictions, labels) == 0.5
    
    def test_case_insensitive(self):
        """Should be case insensitive."""
        predictions = ["true", "FALSE", "Unknown"]
        labels = ["TRUE", "false", "UNKNOWN"]
        assert compute_accuracy(predictions, labels) == 1.0
    
    def test_empty_lists(self):
        """Should handle empty lists."""
        assert compute_accuracy([], []) == 0.0


class TestStepValidity:
    """Tests for step validity metrics."""
    
    def create_trace_with_validity(self, validities: list[bool]) -> OptionizedTrace:
        """Helper to create trace with specific step validities."""
        state = LogicalState("test", [], [])
        steps = []
        for i, valid in enumerate(validities):
            step = ProofStep(i, f"Step {i}", OptionType.MODUS_PONENS, [0, 1])
            step.solver_valid = valid
            steps.append(step)
        return OptionizedTrace("test", state, steps, "TRUE")
    
    def test_all_valid_steps(self):
        """Should compute 100% validity for all valid steps."""
        trace = self.create_trace_with_validity([True, True, True])
        result = compute_step_validity([trace])
        assert result["overall"] == 1.0
    
    def test_no_valid_steps(self):
        """Should compute 0% validity for no valid steps."""
        trace = self.create_trace_with_validity([False, False, False])
        result = compute_step_validity([trace])
        assert result["overall"] == 0.0
    
    def test_mixed_validity(self):
        """Should compute correct rate for mixed validity."""
        trace = self.create_trace_with_validity([True, False, True, False])
        result = compute_step_validity([trace])
        assert result["overall"] == 0.5
    
    def test_multiple_traces(self):
        """Should aggregate across multiple traces."""
        trace1 = self.create_trace_with_validity([True, True])  # 2/2
        trace2 = self.create_trace_with_validity([False, False])  # 0/2
        result = compute_step_validity([trace1, trace2])
        assert result["overall"] == 0.5
        assert result["total_steps"] == 4
        assert result["valid_steps"] == 2


class TestTraceValidity:
    """Tests for trace validity metrics."""
    
    def create_trace(self, step_validities: list[bool], answer: str) -> OptionizedTrace:
        """Helper to create trace."""
        state = LogicalState("test", [], [])
        steps = []
        for i, valid in enumerate(step_validities):
            step = ProofStep(i, f"Step {i}", OptionType.MODUS_PONENS, [0, 1])
            step.solver_valid = valid
            steps.append(step)
        return OptionizedTrace("test", state, steps, answer)
    
    def test_valid_trace(self):
        """Should identify fully valid trace."""
        trace = self.create_trace([True, True], "TRUE")
        result = compute_trace_validity([trace], ["TRUE"])
        assert result == 1.0
    
    def test_invalid_step_makes_trace_invalid(self):
        """One invalid step should make trace invalid."""
        trace = self.create_trace([True, False, True], "TRUE")
        result = compute_trace_validity([trace], ["TRUE"])
        assert result == 0.0
    
    def test_wrong_answer_makes_trace_invalid(self):
        """Wrong answer should make trace invalid."""
        trace = self.create_trace([True, True], "FALSE")
        result = compute_trace_validity([trace], ["TRUE"])
        assert result == 0.0
    
    def test_multiple_traces(self):
        """Should compute rate across multiple traces."""
        trace1 = self.create_trace([True, True], "TRUE")
        trace2 = self.create_trace([True, False], "TRUE")
        trace3 = self.create_trace([True, True], "FALSE")
        trace4 = self.create_trace([True, True], "TRUE")
        
        result = compute_trace_validity(
            [trace1, trace2, trace3, trace4],
            ["TRUE", "TRUE", "TRUE", "TRUE"]
        )
        assert result == 0.5  # trace1 and trace4 are valid


class TestBrierScore:
    """Tests for Brier score computation."""
    
    def test_perfect_predictions(self):
        """Should be 0 for perfect predictions."""
        predictions = [1.0, 0.0, 1.0]
        labels = [1, 0, 1]
        assert compute_brier_score(predictions, labels) == 0.0
    
    def test_worst_predictions(self):
        """Should be 1 for worst predictions."""
        predictions = [0.0, 1.0, 0.0]
        labels = [1, 0, 1]
        assert compute_brier_score(predictions, labels) == 1.0
    
    def test_intermediate_predictions(self):
        """Should compute intermediate scores."""
        predictions = [0.5, 0.5]
        labels = [1, 0]
        # (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5 / 2 = 0.25
        assert compute_brier_score(predictions, labels) == pytest.approx(0.25)
    
    def test_empty_lists(self):
        """Should return 1.0 for empty lists."""
        assert compute_brier_score([], []) == 1.0


class TestECE:
    """Tests for Expected Calibration Error computation."""
    
    def test_perfect_calibration(self):
        """Should be 0 for perfectly calibrated predictions."""
        # All predictions of 0.8 with 80% positive
        predictions = [0.8] * 10
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        ece = compute_ece(predictions, labels)
        assert ece == pytest.approx(0.0, abs=0.01)
    
    def test_empty_lists(self):
        """Should return 1.0 for empty lists."""
        assert compute_ece([], []) == 1.0
    
    def test_ece_range(self):
        """ECE should be between 0 and 1."""
        import random
        predictions = [random.random() for _ in range(100)]
        labels = [random.randint(0, 1) for _ in range(100)]
        ece = compute_ece(predictions, labels)
        assert 0 <= ece <= 1


class TestCalibrationCurve:
    """Tests for calibration curve computation."""
    
    def test_calibration_curve_structure(self):
        """Should return proper structure."""
        predictions = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        result = compute_calibration_curve(predictions, labels, n_bins=5)
        
        assert "mean_predicted" in result
        assert "fraction_positive" in result
        assert "bin_counts" in result
        assert "bin_boundaries" in result
        assert len(result["bin_boundaries"]) == 6  # n_bins + 1
    
    def test_empty_bins_handled(self):
        """Should handle empty bins gracefully."""
        predictions = [0.1, 0.1, 0.9, 0.9]
        labels = [0, 0, 1, 1]
        result = compute_calibration_curve(predictions, labels, n_bins=10)
        
        # Some bins should be None (empty)
        assert None in result["mean_predicted"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

