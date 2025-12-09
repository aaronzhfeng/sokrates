"""Tests for core data structures."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.structures import (
    OptionType,
    FOLFormula,
    ProofStep,
    LogicalState,
    OptionizedTrace,
    PreferencePair,
    OPTION_VOCABULARY,
    OPTION_ARG_COUNTS,
)


class TestOptionType:
    """Tests for OptionType enum."""
    
    def test_option_vocabulary_not_empty(self):
        """Vocabulary should contain options."""
        assert len(OPTION_VOCABULARY) > 0
    
    def test_all_options_have_arg_counts(self):
        """All options should have defined argument counts."""
        for opt in OPTION_VOCABULARY:
            assert opt in OPTION_ARG_COUNTS
    
    def test_modus_ponens_takes_two_args(self):
        """Modus ponens should take 2 arguments."""
        assert OPTION_ARG_COUNTS[OptionType.MODUS_PONENS] == 2
    
    def test_conclude_takes_one_arg(self):
        """Conclude should take 1 argument."""
        assert OPTION_ARG_COUNTS[OptionType.CONCLUDE] == 1


class TestFOLFormula:
    """Tests for FOLFormula."""
    
    def test_create_formula(self):
        """Should create a formula with required fields."""
        formula = FOLFormula(
            id=0,
            nl_text="Socrates is a man",
            fol_string="Man(socrates)",
        )
        assert formula.id == 0
        assert formula.nl_text == "Socrates is a man"
        assert formula.fol_string == "Man(socrates)"
        assert formula.source == "premise"
    
    def test_formula_string_representation(self):
        """Should have readable string representation."""
        formula = FOLFormula(
            id=1,
            nl_text="All men are mortal",
            fol_string="∀x.(Man(x) → Mortal(x))",
        )
        str_repr = str(formula)
        assert "[1]" in str_repr
        assert "All men are mortal" in str_repr


class TestProofStep:
    """Tests for ProofStep."""
    
    def test_create_step(self):
        """Should create a proof step."""
        step = ProofStep(
            step_idx=0,
            thought="Applying modus ponens",
            option_type=OptionType.MODUS_PONENS,
            option_args=[0, 1],
        )
        assert step.step_idx == 0
        assert step.option_type == OptionType.MODUS_PONENS
        assert step.option_args == [0, 1]
    
    def test_action_string_format(self):
        """Should generate correct action string."""
        step = ProofStep(
            step_idx=0,
            thought="Test",
            option_type=OptionType.MODUS_PONENS,
            option_args=[0, 1],
        )
        action_str = step.to_action_string()
        assert '<Option type="MODUS_PONENS"' in action_str
        assert 'args="[0, 1]"' in action_str
        assert '/>' in action_str
    
    def test_parse_action_string(self):
        """Should parse action string back to ProofStep."""
        action_str = '<Option type="AND_ELIM" args="[2, 0]" />'
        step = ProofStep.from_action_string(action_str, step_idx=5, thought="Test")
        
        assert step.step_idx == 5
        assert step.option_type == OptionType.AND_ELIM
        assert step.option_args == [2, 0]
    
    def test_full_string_format(self):
        """Should generate full Thought/Action format."""
        step = ProofStep(
            step_idx=0,
            thought="Since we have P and P→Q...",
            option_type=OptionType.MODUS_PONENS,
            option_args=[0, 1],
        )
        full_str = step.to_full_string()
        assert "Thought:" in full_str
        assert "Action:" in full_str
        assert "Since we have P" in full_str


class TestLogicalState:
    """Tests for LogicalState."""
    
    def test_create_state(self):
        """Should create a logical state."""
        state = LogicalState(
            problem_id="test_001",
            nl_premises=["Socrates is a man", "All men are mortal"],
            fol_formulas=[
                FOLFormula(0, "Socrates is a man", "Man(socrates)"),
                FOLFormula(1, "All men are mortal", "∀x.(Man(x)→Mortal(x))"),
            ],
            target_conclusion="Socrates is mortal",
            label="TRUE",
        )
        assert state.problem_id == "test_001"
        assert state.num_formulas == 2
        assert state.num_steps == 0
    
    def test_add_formula(self):
        """Should add formula to state."""
        state = LogicalState(
            problem_id="test",
            nl_premises=[],
            fol_formulas=[],
        )
        formula = FOLFormula(0, "Test", "Test()")
        state.add_formula(formula)
        assert state.num_formulas == 1
    
    def test_get_formula_by_id(self):
        """Should retrieve formula by index."""
        state = LogicalState(
            problem_id="test",
            nl_premises=["A", "B"],
            fol_formulas=[
                FOLFormula(0, "A", "A()"),
                FOLFormula(1, "B", "B()"),
            ],
        )
        formula = state.get_formula_by_id(1)
        assert formula.nl_text == "B"
        
        # Out of range should return None
        assert state.get_formula_by_id(99) is None
    
    def test_to_prompt(self):
        """Should generate prompt string."""
        state = LogicalState(
            problem_id="test",
            nl_premises=["Premise 1", "Premise 2"],
            fol_formulas=[],
            target_conclusion="Conclusion",
        )
        prompt = state.to_prompt()
        assert "Premise 1" in prompt
        assert "Premise 2" in prompt
        assert "Conclusion" in prompt


class TestOptionizedTrace:
    """Tests for OptionizedTrace."""
    
    def test_create_trace(self):
        """Should create an optionized trace."""
        state = LogicalState(
            problem_id="test",
            nl_premises=["P"],
            fol_formulas=[],
        )
        trace = OptionizedTrace(
            problem_id="test",
            initial_state=state,
            steps=[
                ProofStep(0, "Step 1", OptionType.MODUS_PONENS, [0, 1]),
                ProofStep(1, "Done", OptionType.CONCLUDE, [0]),
            ],
            final_answer="TRUE",
        )
        assert trace.num_steps == 2
        assert trace.final_answer == "TRUE"
    
    def test_step_validity_rate(self):
        """Should compute step validity rate."""
        state = LogicalState("test", [], [])
        steps = [
            ProofStep(0, "S1", OptionType.MODUS_PONENS, [0, 1]),
            ProofStep(1, "S2", OptionType.AND_ELIM, [0, 0]),
            ProofStep(2, "S3", OptionType.CONCLUDE, [0]),
        ]
        steps[0].solver_valid = True
        steps[1].solver_valid = False
        steps[2].solver_valid = True
        
        trace = OptionizedTrace("test", state, steps, "TRUE")
        assert trace.step_validity_rate == pytest.approx(2/3)
    
    def test_to_training_string(self):
        """Should generate training format string."""
        state = LogicalState(
            problem_id="test",
            nl_premises=["P1"],
            fol_formulas=[],
            target_conclusion="C",
        )
        trace = OptionizedTrace(
            problem_id="test",
            initial_state=state,
            steps=[
                ProofStep(0, "Reasoning", OptionType.CONCLUDE, [0]),
            ],
            final_answer="TRUE",
        )
        training_str = trace.to_training_string()
        assert "Reasoning" in training_str
        assert "Final Answer: TRUE" in training_str


class TestPreferencePair:
    """Tests for PreferencePair."""
    
    def test_create_preference_pair(self):
        """Should create a preference pair."""
        state = LogicalState("test", ["P"], [])
        winner = OptionizedTrace("test", state, [], "TRUE")
        loser = OptionizedTrace("test", state, [], "FALSE")
        
        pair = PreferencePair(
            problem_id="test",
            prompt="Test prompt",
            winner=winner,
            loser=loser,
        )
        assert pair.winner.final_answer == "TRUE"
        assert pair.loser.final_answer == "FALSE"
    
    def test_to_dpo_format(self):
        """Should convert to DPO training format."""
        state = LogicalState("test", ["P"], [], target_conclusion="C")
        winner = OptionizedTrace("test", state, [], "TRUE")
        loser = OptionizedTrace("test", state, [], "FALSE")
        
        pair = PreferencePair("test", state.to_prompt(), winner, loser)
        dpo_format = pair.to_dpo_format()
        
        assert "prompt" in dpo_format
        assert "chosen" in dpo_format
        assert "rejected" in dpo_format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

