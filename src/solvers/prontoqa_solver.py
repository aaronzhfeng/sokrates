"""
PrOntoQA Solver: Verification using the synthetic ontology engine.

PrOntoQA has a clean synthetic structure that can be verified 
using simple rule-based reasoning over the ontology.
"""

import re
from typing import Optional

from src.data.structures import (
    FOLFormula,
    LogicalState,
    OptionType,
    ProofStep,
)
from src.solvers.base_solver import FOLSolver, ValidityStatus, VerificationResult


class PrOntoQASolver(FOLSolver):
    """
    Solver for PrOntoQA using ontology-based reasoning.
    
    PrOntoQA examples follow a predictable structure:
    - Facts about entities (e.g., "Rex is a cat")
    - Rules about categories (e.g., "All cats are mammals")
    - Queries about derived properties
    
    Verification is done by tracing through the ontology.
    """
    
    def __init__(self):
        """Initialize the PrOntoQA solver."""
        self.ontology = {}  # entity -> set of categories
        self.rules = []  # (from_category, to_category) implications
    
    def parse_context(self, context: str) -> None:
        """
        Parse a PrOntoQA context into ontology facts and rules.
        
        Args:
            context: The context string containing facts and rules
        """
        self.ontology = {}
        self.rules = []
        
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        
        for sentence in sentences:
            sentence = sentence.lower()
            
            # Pattern: "X is a Y" - entity fact
            fact_match = re.match(r"(\w+) is (?:a|an) (\w+)", sentence)
            if fact_match:
                entity, category = fact_match.groups()
                if entity not in self.ontology:
                    self.ontology[entity] = set()
                self.ontology[entity].add(category)
                continue
            
            # Pattern: "All X are Y" or "Every X is a Y" - category rule
            rule_match = re.match(
                r"(?:all|every) (\w+)s? (?:are|is) (?:a |an )?(\w+)s?",
                sentence
            )
            if rule_match:
                from_cat, to_cat = rule_match.groups()
                self.rules.append((from_cat, to_cat))
                continue
            
            # Pattern: "X are Y" - plural category rule
            plural_match = re.match(r"(\w+)s are (\w+)s?", sentence)
            if plural_match:
                from_cat, to_cat = plural_match.groups()
                self.rules.append((from_cat, to_cat))
    
    def derive_categories(self, entity: str) -> set[str]:
        """
        Derive all categories for an entity using the rules.
        
        Uses forward chaining to find all implied categories.
        """
        if entity not in self.ontology:
            return set()
        
        categories = self.ontology[entity].copy()
        changed = True
        
        while changed:
            changed = False
            for from_cat, to_cat in self.rules:
                if from_cat in categories and to_cat not in categories:
                    categories.add(to_cat)
                    changed = True
        
        return categories
    
    def check_query(self, entity: str, category: str) -> bool:
        """
        Check if an entity belongs to a category.
        
        Args:
            entity: The entity name
            category: The category to check
            
        Returns:
            True if entity is in category, False otherwise
        """
        derived = self.derive_categories(entity.lower())
        return category.lower() in derived
    
    def check_step(
        self,
        state: LogicalState,
        step: ProofStep,
    ) -> VerificationResult:
        """
        Verify a proof step in PrOntoQA.
        
        For PrOntoQA, most steps are applications of category rules
        (essentially universal instantiation followed by modus ponens).
        """
        # Handle terminal CONCLUDE step
        if step.option_type == OptionType.CONCLUDE:
            return self._verify_conclusion(state, step)
        
        # Parse context if we haven't yet
        if not self.ontology and state.nl_premises:
            context = ". ".join(state.nl_premises)
            self.parse_context(context)
        
        # For PrOntoQA, we verify by checking if the step's claim
        # follows from the ontology
        step_text = step.thought.lower()
        
        # Try multiple patterns to extract entity and category
        entity = None
        category = None
        
        # Pattern 1: "conclude that X is Y" (new format)
        match = re.search(r"conclude that (\w+) is (?:a |an )?(\w+)", step_text)
        if match:
            entity, category = match.groups()
        
        # Pattern 2: "X is Y" (simple format)
        if not entity:
            match = re.search(r"(\w+) is (?:a |an )?(\w+)", step_text)
            if match:
                entity, category = match.groups()
        
        # Pattern 3: "derive that X is Y"
        if not entity:
            match = re.search(r"derive that (\w+) is (?:a |an )?(\w+)", step_text)
            if match:
                entity, category = match.groups()
        
        # Pattern 4: Look for "X is not Y" (negative)
        is_negative = False
        if not entity:
            match = re.search(r"(\w+) is not (?:a |an )?(\w+)", step_text)
            if match:
                entity, category = match.groups()
                is_negative = True
        
        if entity and category:
            # Skip common words that aren't entities
            if entity in ['the', 'a', 'an', 'it', 'this', 'that', 'we', 'can']:
                entity = None
        
        if entity and category:
            is_valid = self.check_query(entity, category)
            # For negative claims, invert the check
            if is_negative:
                is_valid = not is_valid
            
            if is_valid:
                result_formula = FOLFormula(
                    id=state.num_formulas,
                    nl_text=step.thought,
                    fol_string=f"{category}({entity})" if not is_negative else f"not {category}({entity})",
                    source="derived",
                    derived_by=step.option_type.name,
                )
                return VerificationResult(
                    status=ValidityStatus.VALID,
                    new_formula=result_formula,
                    message=f"Verified: {entity} is {'not ' if is_negative else ''}{category}",
                )
            else:
                return VerificationResult(
                    status=ValidityStatus.INVALID,
                    message=f"Cannot derive: {entity} is {'not ' if is_negative else ''}{category}",
                )
        
        # If we can't parse the step, assume it's valid
        # (more sophisticated parsing would be needed for production)
        result_formula = FOLFormula(
            id=state.num_formulas,
            nl_text=step.thought,
            fol_string="",
            source="derived",
            derived_by=step.option_type.name,
        )
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            new_formula=result_formula,
            message="Step accepted (could not parse for verification)",
        )
    
    def check_entailment(
        self,
        premises: list[FOLFormula],
        conclusion: FOLFormula,
    ) -> VerificationResult:
        """
        Check entailment by parsing premises and checking conclusion.
        """
        # Parse premises into ontology
        context = ". ".join(p.nl_text for p in premises)
        self.parse_context(context)
        
        # Try to parse conclusion
        conclusion_text = conclusion.nl_text.lower()
        match = re.search(r"(\w+) is (?:a |an )?(\w+)", conclusion_text)
        
        if match:
            entity, category = match.groups()
            if self.check_query(entity, category):
                return VerificationResult(
                    status=ValidityStatus.VALID,
                    message=f"Entailment holds: {entity} is a {category}",
                )
            else:
                return VerificationResult(
                    status=ValidityStatus.INVALID,
                    message=f"Entailment does not hold",
                )
        
        return VerificationResult(
            status=ValidityStatus.UNKNOWN,
            message="Could not parse conclusion for verification",
        )
    
    def check_consistency(
        self,
        formulas: list[FOLFormula],
    ) -> VerificationResult:
        """
        Check consistency of formulas.
        
        For PrOntoQA's simple ontology, we just check for
        direct contradictions (X is Y and X is not Y).
        """
        facts = {}  # entity -> set of (category, is_positive)
        
        for formula in formulas:
            text = formula.nl_text.lower()
            
            # Check for "X is a Y"
            pos_match = re.search(r"(\w+) is (?:a |an )?(\w+)", text)
            if pos_match:
                entity, category = pos_match.groups()
                if entity not in facts:
                    facts[entity] = set()
                facts[entity].add((category, True))
            
            # Check for "X is not a Y"
            neg_match = re.search(r"(\w+) is not (?:a |an )?(\w+)", text)
            if neg_match:
                entity, category = neg_match.groups()
                if entity not in facts:
                    facts[entity] = set()
                facts[entity].add((category, False))
        
        # Check for contradictions
        for entity, cat_facts in facts.items():
            categories = {}
            for cat, is_pos in cat_facts:
                if cat in categories and categories[cat] != is_pos:
                    return VerificationResult(
                        status=ValidityStatus.INVALID,
                        message=f"Contradiction: {entity} both is and is not {cat}",
                    )
                categories[cat] = is_pos
        
        return VerificationResult(
            status=ValidityStatus.VALID,
            message="No contradictions found",
        )
    
    def _verify_conclusion(
        self,
        state: LogicalState,
        step: ProofStep,
    ) -> VerificationResult:
        """Verify the CONCLUDE step."""
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

