"""
Abstract base class for FOL solvers.

Defines the interface that all solver implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.data.structures import FOLFormula, LogicalState, ProofStep


class ValidityStatus(Enum):
    """Status of a verification check."""
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"  # Solver couldn't determine
    ERROR = "error"  # Solver encountered an error


@dataclass
class VerificationResult:
    """Result of verifying a proof step or entailment."""
    
    status: ValidityStatus
    new_formula: Optional[FOLFormula] = None  # If valid, the derived formula
    message: str = ""  # Human-readable explanation
    details: dict = None  # Additional solver-specific information
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def is_valid(self) -> bool:
        return self.status == ValidityStatus.VALID
    
    @property
    def is_invalid(self) -> bool:
        return self.status == ValidityStatus.INVALID


class FOLSolver(ABC):
    """
    Abstract interface for first-order logic verification.
    
    Implementations provide ground-truth "knowledge" about whether
    proof steps are logically valid.
    """
    
    @abstractmethod
    def check_step(
        self,
        state: LogicalState,
        step: ProofStep,
    ) -> VerificationResult:
        """
        Verify a single proof step.
        
        Checks whether applying the given option (inference rule) to the
        current state produces a logically valid result.
        
        Args:
            state: Current logical state (formulas derived so far)
            step: The proof step to verify
            
        Returns:
            VerificationResult with validity status and any derived formula
        """
        pass
    
    @abstractmethod
    def check_entailment(
        self,
        premises: list[FOLFormula],
        conclusion: FOLFormula,
    ) -> VerificationResult:
        """
        Check if premises logically entail conclusion.
        
        Args:
            premises: List of premise formulas
            conclusion: The conclusion to check
            
        Returns:
            VerificationResult indicating whether entailment holds
        """
        pass
    
    @abstractmethod
    def check_consistency(
        self,
        formulas: list[FOLFormula],
    ) -> VerificationResult:
        """
        Check if a set of formulas is consistent (no contradictions).
        
        Args:
            formulas: List of formulas to check
            
        Returns:
            VerificationResult indicating consistency status
        """
        pass
    
    def verify_trace(
        self,
        trace,  # OptionizedTrace - avoiding circular import
        ground_truth_label: Optional[str] = None,
    ) -> tuple[bool, list[VerificationResult]]:
        """
        Verify an entire proof trace.
        
        Checks each step sequentially and determines overall validity.
        A trace is valid if:
        1. All steps are individually valid
        2. The final answer matches ground truth (if provided)
        
        Args:
            trace: The OptionizedTrace to verify
            ground_truth_label: Optional ground truth for the problem
            
        Returns:
            Tuple of (overall_valid, list of per-step results)
        """
        # Start with initial state
        current_state = trace.initial_state
        step_results = []
        all_valid = True
        
        for step in trace.steps:
            result = self.check_step(current_state, step)
            step_results.append(result)
            
            # Update step with verification result
            step.solver_valid = result.is_valid
            
            if not result.is_valid:
                all_valid = False
            
            # Update state if step was valid and produced a formula
            if result.is_valid and result.new_formula:
                current_state.add_formula(result.new_formula)
        
        # Check final answer if ground truth provided
        if ground_truth_label and trace.final_answer != ground_truth_label:
            all_valid = False
        
        # Update trace validity
        trace.trace_valid = all_valid
        
        return all_valid, step_results
    
    def get_solver_name(self) -> str:
        """Return the name of this solver implementation."""
        return self.__class__.__name__

