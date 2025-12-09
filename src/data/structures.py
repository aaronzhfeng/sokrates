"""
Core data structures for SOKRATES.

Defines the option vocabulary and all data classes used throughout the system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """
    Discrete inference rule options derived from P-FOLIO taxonomy.
    Each option is a reusable reasoning macro.
    """

    # Basic inference rules
    MODUS_PONENS = "MP"  # From P and P→Q, derive Q
    MODUS_TOLLENS = "MT"  # From ¬Q and P→Q, derive ¬P
    HYPOTHETICAL_SYLLOGISM = "HS"  # From P→Q and Q→R, derive P→R
    DISJUNCTIVE_SYLLOGISM = "DS"  # From P∨Q and ¬P, derive Q

    # Quantifier rules
    UNIV_INSTANTIATION = "UI"  # From ∀x.P(x), derive P(c)
    UNIV_GENERALIZATION = "UG"  # From P(c) for arbitrary c, derive ∀x.P(x)
    EXIST_INSTANTIATION = "EI"  # From ∃x.P(x), derive P(sk)
    EXIST_GENERALIZATION = "EG"  # From P(c), derive ∃x.P(x)

    # Conjunction rules
    AND_INTRO = "AI"  # From P and Q, derive P∧Q
    AND_ELIM = "AE"  # From P∧Q, derive P or Q

    # Disjunction rules
    OR_INTRO = "OI"  # From P, derive P∨Q
    CASE_SPLIT = "CS"  # Split into cases for P∨Q

    # Biconditional rules
    BICONDITIONAL_INTRO = "BI"  # From P→Q and Q→P, derive P↔Q
    BICONDITIONAL_ELIM = "BE"  # From P↔Q, derive P→Q or Q→P

    # Negation rules
    CONTRADICTION = "CN"  # From P and ¬P, derive ⊥
    DOUBLE_NEGATION = "DN"  # From ¬¬P, derive P

    # Proof structure
    CONDITIONAL_PROOF = "CP"  # Assuming P, derived Q → conclude P→Q

    # Terminal
    CONCLUDE = "DONE"  # Terminal: output TRUE/FALSE/UNKNOWN


# Mapping from option type to expected number of arguments
OPTION_ARG_COUNTS = {
    OptionType.MODUS_PONENS: 2,  # (premise_idx, implication_idx)
    OptionType.MODUS_TOLLENS: 2,  # (negated_consequent_idx, implication_idx)
    OptionType.HYPOTHETICAL_SYLLOGISM: 2,  # (impl_1_idx, impl_2_idx)
    OptionType.DISJUNCTIVE_SYLLOGISM: 2,  # (disjunction_idx, negated_disjunct_idx)
    OptionType.UNIV_INSTANTIATION: 2,  # (formula_idx, constant)
    OptionType.UNIV_GENERALIZATION: 2,  # (formula_idx, variable)
    OptionType.EXIST_INSTANTIATION: 2,  # (formula_idx, skolem_constant)
    OptionType.EXIST_GENERALIZATION: 2,  # (formula_idx, variable)
    OptionType.AND_INTRO: 2,  # (formula_idx_1, formula_idx_2)
    OptionType.AND_ELIM: 2,  # (conjunction_idx, side: 0=left, 1=right)
    OptionType.OR_INTRO: 2,  # (formula_idx, new_disjunct_idx or -1 for fresh)
    OptionType.CASE_SPLIT: 1,  # (disjunction_idx,)
    OptionType.BICONDITIONAL_INTRO: 2,  # (impl_1_idx, impl_2_idx)
    OptionType.BICONDITIONAL_ELIM: 2,  # (biconditional_idx, direction: 0=left, 1=right)
    OptionType.CONTRADICTION: 2,  # (formula_idx, negation_idx)
    OptionType.DOUBLE_NEGATION: 1,  # (formula_idx,)
    OptionType.CONDITIONAL_PROOF: 2,  # (assumption_idx, conclusion_idx)
    OptionType.CONCLUDE: 1,  # (conclusion_type: 0=TRUE, 1=FALSE, 2=UNKNOWN)
}

# Human-readable option descriptions
OPTION_DESCRIPTIONS = {
    OptionType.MODUS_PONENS: "From P and P→Q, derive Q",
    OptionType.MODUS_TOLLENS: "From ¬Q and P→Q, derive ¬P",
    OptionType.HYPOTHETICAL_SYLLOGISM: "From P→Q and Q→R, derive P→R",
    OptionType.DISJUNCTIVE_SYLLOGISM: "From P∨Q and ¬P, derive Q",
    OptionType.UNIV_INSTANTIATION: "From ∀x.P(x), derive P(c) for constant c",
    OptionType.UNIV_GENERALIZATION: "From P(c) for arbitrary c, derive ∀x.P(x)",
    OptionType.EXIST_INSTANTIATION: "From ∃x.P(x), derive P(sk) for Skolem constant",
    OptionType.EXIST_GENERALIZATION: "From P(c), derive ∃x.P(x)",
    OptionType.AND_INTRO: "From P and Q, derive P∧Q",
    OptionType.AND_ELIM: "From P∧Q, derive P or Q",
    OptionType.OR_INTRO: "From P, derive P∨Q",
    OptionType.CASE_SPLIT: "Split proof into cases for P∨Q",
    OptionType.BICONDITIONAL_INTRO: "From P→Q and Q→P, derive P↔Q",
    OptionType.BICONDITIONAL_ELIM: "From P↔Q, derive P→Q or Q→P",
    OptionType.CONTRADICTION: "From P and ¬P, derive contradiction",
    OptionType.DOUBLE_NEGATION: "From ¬¬P, derive P",
    OptionType.CONDITIONAL_PROOF: "From assumption P leading to Q, derive P→Q",
    OptionType.CONCLUDE: "Conclude with final answer",
}

# Export vocabulary for easy access
OPTION_VOCABULARY = list(OptionType)


@dataclass
class FOLFormula:
    """
    A first-order logic formula with both natural language and formal representations.
    """

    id: int
    nl_text: str  # Natural language gloss
    fol_string: str  # FOL syntax string (e.g., "∀x.(Man(x) → Mortal(x))")
    source: str = "premise"  # "premise" | "derived" | "assumption"
    derived_by: Optional[str] = None  # Option that derived this formula
    derived_from: list[int] = field(default_factory=list)  # Indices of parent formulas

    def __str__(self) -> str:
        return f"[{self.id}] {self.nl_text} | {self.fol_string}"

    def __repr__(self) -> str:
        return f"FOLFormula(id={self.id}, fol='{self.fol_string}')"


@dataclass
class ProofStep:
    """
    A single reasoning step in an optionized proof trace.
    
    Format: Thought: <NL justification> Action: <Option type="..." args="[...]" />
    """

    step_idx: int
    thought: str  # Natural language justification
    option_type: OptionType  # The inference rule being applied
    option_args: list[int]  # Arguments (indices into formula list or special values)
    result_formula: Optional[FOLFormula] = None  # Formula produced by this step
    solver_valid: Optional[bool] = None  # Ground truth from solver
    predicted_valid: Optional[float] = None  # q̂_φ prediction

    def to_action_string(self) -> str:
        """Convert to the Action format string."""
        args_str = str(self.option_args)
        return f'<Option type="{self.option_type.name}" args="{args_str}" />'

    def to_full_string(self) -> str:
        """Convert to full Thought/Action format."""
        return f"Thought: {self.thought}\nAction: {self.to_action_string()}"

    @classmethod
    def from_action_string(cls, action_str: str, step_idx: int, thought: str = "") -> "ProofStep":
        """Parse an Action format string into a ProofStep."""
        import re

        pattern = r'<Option type="(\w+)" args="\[([\d,\s]*)\]" />'
        match = re.match(pattern, action_str.strip())
        if not match:
            raise ValueError(f"Invalid action string: {action_str}")

        option_name = match.group(1)
        args_str = match.group(2)
        args = [int(x.strip()) for x in args_str.split(",") if x.strip()]

        return cls(
            step_idx=step_idx,
            thought=thought,
            option_type=OptionType[option_name],
            option_args=args,
        )


@dataclass
class LogicalState:
    """
    The current state of a proof, including all derived formulas.
    """

    problem_id: str
    nl_premises: list[str]  # Original natural language premises
    fol_formulas: list[FOLFormula]  # Current formula set (premises + derived)
    derived_steps: list[ProofStep] = field(default_factory=list)  # History
    target_conclusion: str = ""  # NL conclusion to prove
    target_fol: Optional[FOLFormula] = None  # FOL form of conclusion
    label: Optional[str] = None  # Ground truth: "TRUE" | "FALSE" | "UNKNOWN"

    @property
    def num_formulas(self) -> int:
        return len(self.fol_formulas)

    @property
    def num_steps(self) -> int:
        return len(self.derived_steps)

    def add_formula(self, formula: FOLFormula) -> None:
        """Add a new derived formula to the state."""
        self.fol_formulas.append(formula)

    def add_step(self, step: ProofStep) -> None:
        """Add a proof step and its result formula to the state."""
        self.derived_steps.append(step)
        if step.result_formula:
            self.add_formula(step.result_formula)

    def get_formula_by_id(self, idx: int) -> Optional[FOLFormula]:
        """Get formula by index."""
        if 0 <= idx < len(self.fol_formulas):
            return self.fol_formulas[idx]
        return None

    def to_prompt(self) -> str:
        """Convert state to a prompt string for the LLM."""
        lines = ["Premises:"]
        for i, premise in enumerate(self.nl_premises):
            lines.append(f"  [{i}] {premise}")

        if self.derived_steps:
            lines.append("\nDerived so far:")
            for step in self.derived_steps:
                if step.result_formula:
                    lines.append(f"  [{step.result_formula.id}] {step.result_formula.nl_text}")

        lines.append(f"\nConclusion to evaluate: {self.target_conclusion}")
        lines.append("\nDetermine if the conclusion is TRUE, FALSE, or UNKNOWN.")

        return "\n".join(lines)


@dataclass
class OptionizedTrace:
    """
    A complete proof trace consisting of optionized steps.
    """

    problem_id: str
    initial_state: LogicalState
    steps: list[ProofStep]
    final_answer: str  # "TRUE" | "FALSE" | "UNKNOWN"
    trace_valid: Optional[bool] = None  # All steps valid AND answer correct

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def step_validity_rate(self) -> float:
        """Fraction of steps that are solver-valid."""
        valid_steps = [s for s in self.steps if s.solver_valid is not None]
        if not valid_steps:
            return 0.0
        return sum(1 for s in valid_steps if s.solver_valid) / len(valid_steps)

    def to_training_string(self) -> str:
        """Convert trace to training format string."""
        lines = [self.initial_state.to_prompt(), "\nReasoning:"]
        for step in self.steps:
            lines.append(step.to_full_string())
        lines.append(f"\nFinal Answer: {self.final_answer}")
        return "\n".join(lines)


@dataclass
class PreferencePair:
    """
    A preference pair for DPO training.
    
    The winner trace is solver-valid; the loser contains invalid steps.
    """

    problem_id: str
    prompt: str  # Shared context (initial state as prompt)
    winner: OptionizedTrace  # Solver-valid trace (preferred)
    loser: OptionizedTrace  # Invalid trace (dispreferred)

    def to_dpo_format(self) -> dict:
        """Convert to format expected by TRL's DPOTrainer."""
        return {
            "prompt": self.prompt,
            "chosen": self.winner.to_training_string().replace(self.prompt, "").strip(),
            "rejected": self.loser.to_training_string().replace(self.prompt, "").strip(),
        }


# Conclusion type mapping for CONCLUDE option
class ConclusionType(Enum):
    TRUE = 0
    FALSE = 1
    UNKNOWN = 2


def conclusion_from_int(val: int) -> str:
    """Convert conclusion integer to string."""
    return ConclusionType(val).name


def conclusion_to_int(val: str) -> int:
    """Convert conclusion string to integer."""
    return ConclusionType[val.upper()].value

