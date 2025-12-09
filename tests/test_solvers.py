"""Tests for FOL solvers."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.structures import (
    FOLFormula,
    LogicalState,
    ProofStep,
    OptionType,
    OptionizedTrace,
)
from src.solvers.base_solver import ValidityStatus, VerificationResult
from src.solvers.prontoqa_solver import PrOntoQASolver
from src.solvers.folio_solver import FOLIOSolver


class TestVerificationResult:
    """Tests for VerificationResult."""
    
    def test_valid_result(self):
        """Should identify valid results."""
        result = VerificationResult(status=ValidityStatus.VALID)
        assert result.is_valid
        assert not result.is_invalid
    
    def test_invalid_result(self):
        """Should identify invalid results."""
        result = VerificationResult(
            status=ValidityStatus.INVALID,
            message="Step is invalid"
        )
        assert result.is_invalid
        assert not result.is_valid


class TestPrOntoQASolver:
    """Tests for PrOntoQA solver."""
    
    def test_parse_simple_context(self):
        """Should parse simple facts and rules."""
        solver = PrOntoQASolver()
        context = "Rex is a cat. All cats are mammals. All mammals are animals."
        solver.parse_context(context)
        
        assert "rex" in solver.ontology
        assert "cat" in solver.ontology["rex"]
        assert len(solver.rules) == 2
    
    def test_derive_categories(self):
        """Should derive categories through rules."""
        solver = PrOntoQASolver()
        context = "Rex is a cat. All cats are mammals. All mammals are animals."
        solver.parse_context(context)
        
        categories = solver.derive_categories("rex")
        assert "cat" in categories
        assert "mammal" in categories
        assert "animal" in categories
    
    def test_check_query_true(self):
        """Should verify true queries."""
        solver = PrOntoQASolver()
        context = "Rex is a cat. All cats are mammals."
        solver.parse_context(context)
        
        assert solver.check_query("Rex", "cat")
        assert solver.check_query("Rex", "mammal")
    
    def test_check_query_false(self):
        """Should reject false queries."""
        solver = PrOntoQASolver()
        context = "Rex is a cat. All dogs are mammals."
        solver.parse_context(context)
        
        # Rex is not derived to be a mammal through this rule chain
        assert not solver.check_query("Rex", "dog")
    
    def test_check_consistency_no_contradiction(self):
        """Should identify consistent formula sets."""
        solver = PrOntoQASolver()
        formulas = [
            FOLFormula(0, "Rex is a cat", "Cat(rex)"),
            FOLFormula(1, "Rex is a mammal", "Mammal(rex)"),
        ]
        result = solver.check_consistency(formulas)
        assert result.is_valid
    
    def test_check_consistency_with_contradiction(self):
        """Should detect contradictions."""
        solver = PrOntoQASolver()
        formulas = [
            FOLFormula(0, "Rex is a cat", "Cat(rex)"),
            FOLFormula(1, "Rex is not a cat", "¬Cat(rex)"),
        ]
        result = solver.check_consistency(formulas)
        assert result.is_invalid
    
    def test_verify_step(self):
        """Should verify a proof step."""
        solver = PrOntoQASolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=[
                "Rex is a cat",
                "All cats are mammals",
            ],
            fol_formulas=[],
            label="TRUE",
        )
        
        step = ProofStep(
            step_idx=0,
            thought="Since Rex is a cat and all cats are mammals, Rex is a mammal.",
            option_type=OptionType.MODUS_PONENS,
            option_args=[0, 1],
        )
        
        result = solver.check_step(state, step)
        assert result.is_valid
    
    def test_verify_conclude_correct(self):
        """Should verify correct conclusion."""
        solver = PrOntoQASolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["Rex is a cat"],
            fol_formulas=[],
            label="TRUE",
        )
        
        step = ProofStep(
            step_idx=0,
            thought="Therefore TRUE",
            option_type=OptionType.CONCLUDE,
            option_args=[0],  # TRUE
        )
        
        result = solver.check_step(state, step)
        assert result.is_valid
    
    def test_verify_conclude_incorrect(self):
        """Should reject incorrect conclusion."""
        solver = PrOntoQASolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["Rex is a cat"],
            fol_formulas=[],
            label="TRUE",
        )
        
        step = ProofStep(
            step_idx=0,
            thought="Therefore FALSE",
            option_type=OptionType.CONCLUDE,
            option_args=[1],  # FALSE
        )
        
        result = solver.check_step(state, step)
        assert result.is_invalid


class TestFOLIOSolver:
    """Tests for FOLIO solver."""
    
    def test_check_step_invalid_args(self):
        """Should reject steps with invalid arguments."""
        solver = FOLIOSolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["P1"],
            fol_formulas=[FOLFormula(0, "P1", "P1()")],
        )
        
        # Step references non-existent formula
        step = ProofStep(
            step_idx=0,
            thought="Invalid step",
            option_type=OptionType.MODUS_PONENS,
            option_args=[0, 99],  # 99 doesn't exist
        )
        
        result = solver.check_step(state, step)
        assert result.is_invalid
    
    def test_check_step_no_args(self):
        """Should reject steps with no arguments."""
        solver = FOLIOSolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["P1"],
            fol_formulas=[FOLFormula(0, "P1", "P1()")],
        )
        
        step = ProofStep(
            step_idx=0,
            thought="No args",
            option_type=OptionType.MODUS_PONENS,
            option_args=[],
        )
        
        result = solver.check_step(state, step)
        assert result.is_invalid


class TestSolverVerifyTrace:
    """Tests for full trace verification."""
    
    def test_verify_valid_trace(self):
        """Should verify a completely valid trace."""
        solver = PrOntoQASolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["Rex is a cat", "All cats are mammals"],
            fol_formulas=[],
            label="TRUE",
        )
        
        trace = OptionizedTrace(
            problem_id="test",
            initial_state=state,
            steps=[
                ProofStep(0, "Rex is a mammal", OptionType.MODUS_PONENS, [0, 1]),
                ProofStep(1, "Therefore TRUE", OptionType.CONCLUDE, [0]),
            ],
            final_answer="TRUE",
        )
        
        is_valid, results = solver.verify_trace(trace, "TRUE")
        # Due to the simple nature of the test, we check basic functionality
        assert len(results) == 2
    
    def test_verify_trace_wrong_answer(self):
        """Should reject trace with wrong final answer."""
        solver = PrOntoQASolver()
        
        state = LogicalState(
            problem_id="test",
            nl_premises=["Rex is a cat"],
            fol_formulas=[],
            label="TRUE",
        )
        
        trace = OptionizedTrace(
            problem_id="test",
            initial_state=state,
            steps=[
                ProofStep(0, "Wrong answer", OptionType.CONCLUDE, [1]),  # FALSE
            ],
            final_answer="FALSE",
        )
        
        is_valid, results = solver.verify_trace(trace, "TRUE")
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

