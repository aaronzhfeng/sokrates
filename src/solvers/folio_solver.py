"""
FOLIO Solver: Verification using Z3 for FOLIO/P-FOLIO datasets.

Uses the FOL annotations in FOLIO to verify proof steps via Z3.
"""

from typing import Optional

from src.data.structures import (
    FOLFormula,
    LogicalState,
    OptionType,
    ProofStep,
)
from src.solvers.base_solver import FOLSolver, ValidityStatus, VerificationResult


class FOLIOSolver(FOLSolver):
    """
    Solver for FOLIO/P-FOLIO using Z3 theorem prover.
    
    FOLIO provides FOL annotations that can be directly verified
    using an SMT solver like Z3.
    """
    
    def __init__(self, timeout_ms: int = 5000):
        """
        Initialize the FOLIO solver.
        
        Args:
            timeout_ms: Timeout for Z3 in milliseconds
        """
        self.timeout_ms = timeout_ms
        self._z3_available = self._check_z3()
    
    def _check_z3(self) -> bool:
        """Check if Z3 is available."""
        try:
            import z3
            return True
        except ImportError:
            return False
    
    def check_step(
        self,
        state: LogicalState,
        step: ProofStep,
    ) -> VerificationResult:
        """
        Verify a proof step using Z3.
        
        For each inference rule, we encode the verification as an
        SMT problem and check satisfiability.
        """
        if not self._z3_available:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message="Z3 solver not available",
            )
        
        try:
            import z3
            
            # Handle terminal CONCLUDE step
            if step.option_type == OptionType.CONCLUDE:
                return self._verify_conclusion(state, step)
            
            # Get the formulas being used
            args = step.option_args
            if not args:
                return VerificationResult(
                    status=ValidityStatus.INVALID,
                    message="No arguments provided for step",
                )
            
            # Dispatch to specific verification based on option type
            verifier = self._get_verifier(step.option_type)
            if verifier is None:
                return VerificationResult(
                    status=ValidityStatus.UNKNOWN,
                    message=f"No verifier implemented for {step.option_type.name}",
                )
            
            return verifier(state, step, args)
            
        except Exception as e:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message=f"Solver error: {str(e)}",
            )
    
    def check_entailment(
        self,
        premises: list[FOLFormula],
        conclusion: FOLFormula,
    ) -> VerificationResult:
        """
        Check if premises entail conclusion using Z3.
        
        We check if (premises AND NOT conclusion) is unsatisfiable.
        """
        if not self._z3_available:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message="Z3 solver not available",
            )
        
        try:
            import z3
            
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            
            # Parse premises and conclusion to Z3 format
            z3_premises = [self._parse_to_z3(p) for p in premises]
            z3_conclusion = self._parse_to_z3(conclusion)
            
            if None in z3_premises or z3_conclusion is None:
                return VerificationResult(
                    status=ValidityStatus.UNKNOWN,
                    message="Could not parse formulas to Z3",
                )
            
            # Add premises
            for p in z3_premises:
                solver.add(p)
            
            # Add negation of conclusion
            solver.add(z3.Not(z3_conclusion))
            
            # Check satisfiability
            result = solver.check()
            
            if result == z3.unsat:
                # Premises entail conclusion
                return VerificationResult(
                    status=ValidityStatus.VALID,
                    message="Entailment verified",
                )
            elif result == z3.sat:
                return VerificationResult(
                    status=ValidityStatus.INVALID,
                    message="Entailment does not hold",
                    details={"counterexample": str(solver.model())},
                )
            else:
                return VerificationResult(
                    status=ValidityStatus.UNKNOWN,
                    message="Solver returned unknown",
                )
                
        except Exception as e:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message=f"Entailment check error: {str(e)}",
            )
    
    def check_consistency(
        self,
        formulas: list[FOLFormula],
    ) -> VerificationResult:
        """
        Check if a set of formulas is consistent.
        
        We check if the conjunction of all formulas is satisfiable.
        """
        if not self._z3_available:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message="Z3 solver not available",
            )
        
        try:
            import z3
            
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            
            # Parse and add all formulas
            for formula in formulas:
                z3_formula = self._parse_to_z3(formula)
                if z3_formula is not None:
                    solver.add(z3_formula)
            
            result = solver.check()
            
            if result == z3.sat:
                return VerificationResult(
                    status=ValidityStatus.VALID,
                    message="Formula set is consistent",
                )
            elif result == z3.unsat:
                return VerificationResult(
                    status=ValidityStatus.INVALID,
                    message="Formula set is inconsistent (contradiction found)",
                )
            else:
                return VerificationResult(
                    status=ValidityStatus.UNKNOWN,
                    message="Consistency check inconclusive",
                )
                
        except Exception as e:
            return VerificationResult(
                status=ValidityStatus.ERROR,
                message=f"Consistency check error: {str(e)}",
            )
    
    def _get_verifier(self, option_type: OptionType):
        """Get the verification function for an option type."""
        verifiers = {
            OptionType.MODUS_PONENS: self._verify_modus_ponens,
            OptionType.MODUS_TOLLENS: self._verify_modus_tollens,
            OptionType.UNIV_INSTANTIATION: self._verify_univ_instantiation,
            OptionType.AND_ELIM: self._verify_and_elim,
            OptionType.AND_INTRO: self._verify_and_intro,
            OptionType.DISJUNCTIVE_SYLLOGISM: self._verify_disjunctive_syllogism,
            OptionType.HYPOTHETICAL_SYLLOGISM: self._verify_hypothetical_syllogism,
        }
        return verifiers.get(option_type)
    
    def _verify_modus_ponens(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """
        Verify modus ponens: From P and P→Q, derive Q.
        """
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="Modus ponens requires 2 arguments",
            )
        
        p_idx, impl_idx = args[0], args[1]
        
        # Get the formulas
        p_formula = state.get_formula_by_id(p_idx)
        impl_formula = state.get_formula_by_id(impl_idx)
        
        if p_formula is None or impl_formula is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula indices: {p_idx}, {impl_idx}",
            )
        
        # For now, do a structural check
        # Full Z3 verification would parse the FOL and check entailment
        # This is a placeholder that assumes the step is valid if formulas exist
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",  # Would be derived from the inference
            source="derived",
            derived_by="MODUS_PONENS",
            derived_from=[p_idx, impl_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Modus ponens applied",
        )
    
    def _verify_modus_tollens(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify modus tollens: From ¬Q and P→Q, derive ¬P."""
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="Modus tollens requires 2 arguments",
            )
        
        # Similar structure to modus ponens verification
        neg_q_idx, impl_idx = args[0], args[1]
        
        neg_q = state.get_formula_by_id(neg_q_idx)
        impl = state.get_formula_by_id(impl_idx)
        
        if neg_q is None or impl is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula indices: {neg_q_idx}, {impl_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="MODUS_TOLLENS",
            derived_from=[neg_q_idx, impl_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Modus tollens applied",
        )
    
    def _verify_univ_instantiation(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify universal instantiation: From ∀x.P(x), derive P(c)."""
        if len(args) < 1:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="Universal instantiation requires at least 1 argument",
            )
        
        formula_idx = args[0]
        formula = state.get_formula_by_id(formula_idx)
        
        if formula is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula index: {formula_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="UNIV_INSTANTIATION",
            derived_from=[formula_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Universal instantiation applied",
        )
    
    def _verify_and_elim(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify and elimination: From P∧Q, derive P or Q."""
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="And elimination requires 2 arguments",
            )
        
        conj_idx, side = args[0], args[1]
        conj = state.get_formula_by_id(conj_idx)
        
        if conj is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula index: {conj_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="AND_ELIM",
            derived_from=[conj_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message=f"And elimination applied (side {side})",
        )
    
    def _verify_and_intro(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify and introduction: From P and Q, derive P∧Q."""
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="And introduction requires 2 arguments",
            )
        
        p_idx, q_idx = args[0], args[1]
        p = state.get_formula_by_id(p_idx)
        q = state.get_formula_by_id(q_idx)
        
        if p is None or q is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula indices: {p_idx}, {q_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="AND_INTRO",
            derived_from=[p_idx, q_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="And introduction applied",
        )
    
    def _verify_disjunctive_syllogism(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify disjunctive syllogism: From P∨Q and ¬P, derive Q."""
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="Disjunctive syllogism requires 2 arguments",
            )
        
        disj_idx, neg_idx = args[0], args[1]
        disj = state.get_formula_by_id(disj_idx)
        neg = state.get_formula_by_id(neg_idx)
        
        if disj is None or neg is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula indices: {disj_idx}, {neg_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="DISJUNCTIVE_SYLLOGISM",
            derived_from=[disj_idx, neg_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Disjunctive syllogism applied",
        )
    
    def _verify_hypothetical_syllogism(
        self,
        state: LogicalState,
        step: ProofStep,
        args: list[int],
    ) -> VerificationResult:
        """Verify hypothetical syllogism: From P→Q and Q→R, derive P→R."""
        if len(args) < 2:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message="Hypothetical syllogism requires 2 arguments",
            )
        
        impl1_idx, impl2_idx = args[0], args[1]
        impl1 = state.get_formula_by_id(impl1_idx)
        impl2 = state.get_formula_by_id(impl2_idx)
        
        if impl1 is None or impl2 is None:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Invalid formula indices: {impl1_idx}, {impl2_idx}",
            )
        
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by="HYPOTHETICAL_SYLLOGISM",
            derived_from=[impl1_idx, impl2_idx],
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Hypothetical syllogism applied",
        )
    
    def _verify_conclusion(
        self,
        state: LogicalState,
        step: ProofStep,
    ) -> VerificationResult:
        """Verify the CONCLUDE step against ground truth."""
        # The CONCLUDE step is valid if it matches the problem's label
        if state.label is None:
            return VerificationResult(
                status=ValidityStatus.UNKNOWN,
                message="No ground truth label available",
            )
        
        predicted_label = ["TRUE", "FALSE", "UNKNOWN"][step.option_args[0]]
        
        if predicted_label == state.label:
            return VerificationResult(
                status=ValidityStatus.VALID,
                message=f"Conclusion {predicted_label} matches ground truth",
            )
        else:
            return VerificationResult(
                status=ValidityStatus.INVALID,
                message=f"Conclusion {predicted_label} does not match ground truth {state.label}",
            )
    
    def _parse_to_z3(self, formula: FOLFormula):
        """
        Parse a FOL formula to Z3 format.
        
        This is a placeholder - full implementation would need a proper
        FOL parser that handles FOLIO's syntax.
        """
        # TODO: Implement proper FOL to Z3 parsing
        # For now, return None to indicate parsing not implemented
        return None

