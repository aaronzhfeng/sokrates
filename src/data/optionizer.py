"""
Optionizer: Convert raw proofs into optionized Thought/Action format.

This module handles the conversion of P-FOLIO proof annotations and 
PrOntoQA traces into the structured option format used by SOKRATES.
"""

import re
from typing import Optional

from src.data.structures import (
    FOLFormula,
    LogicalState,
    OptionType,
    OptionizedTrace,
    ProofStep,
)


class Optionizer:
    """
    Converts natural language proofs with inference rule annotations
    into optionized traces.
    """

    # Mapping from P-FOLIO rule names to OptionType
    PFOLIO_RULE_MAP = {
        "modus ponens": OptionType.MODUS_PONENS,
        "modus tollens": OptionType.MODUS_TOLLENS,
        "hypothetical syllogism": OptionType.HYPOTHETICAL_SYLLOGISM,
        "disjunctive syllogism": OptionType.DISJUNCTIVE_SYLLOGISM,
        "universal instantiation": OptionType.UNIV_INSTANTIATION,
        "universal generalization": OptionType.UNIV_GENERALIZATION,
        "existential instantiation": OptionType.EXIST_INSTANTIATION,
        "existential generalization": OptionType.EXIST_GENERALIZATION,
        "conjunction introduction": OptionType.AND_INTRO,
        "conjunction elimination": OptionType.AND_ELIM,
        "and introduction": OptionType.AND_INTRO,
        "and elimination": OptionType.AND_ELIM,
        "disjunction introduction": OptionType.OR_INTRO,
        "or introduction": OptionType.OR_INTRO,
        "case split": OptionType.CASE_SPLIT,
        "proof by cases": OptionType.CASE_SPLIT,
        "biconditional introduction": OptionType.BICONDITIONAL_INTRO,
        "biconditional elimination": OptionType.BICONDITIONAL_ELIM,
        "contradiction": OptionType.CONTRADICTION,
        "double negation": OptionType.DOUBLE_NEGATION,
        "double negation elimination": OptionType.DOUBLE_NEGATION,
        "conditional proof": OptionType.CONDITIONAL_PROOF,
        "implication introduction": OptionType.CONDITIONAL_PROOF,
    }

    def __init__(self):
        pass

    def optionize_pfolio_example(
        self,
        problem_id: str,
        premises: list[str],
        fol_premises: list[str],
        conclusion: str,
        fol_conclusion: str,
        proof_steps: list[dict],
        label: str,
    ) -> OptionizedTrace:
        """
        Convert a P-FOLIO example with proof annotations into an optionized trace.

        Args:
            problem_id: Unique identifier for the problem
            premises: List of NL premise strings
            fol_premises: List of FOL premise strings
            conclusion: NL conclusion string
            fol_conclusion: FOL conclusion string
            proof_steps: List of dicts with keys:
                - 'step': NL description of the step
                - 'rule': Name of inference rule used
                - 'from': List of premise/step indices used
                - 'result': The derived statement (optional)
            label: Ground truth label (TRUE/FALSE/UNKNOWN)

        Returns:
            OptionizedTrace with structured proof steps
        """
        # Build initial state with premises
        formulas = []
        for i, (nl, fol) in enumerate(zip(premises, fol_premises)):
            formulas.append(
                FOLFormula(
                    id=i,
                    nl_text=nl,
                    fol_string=fol,
                    source="premise",
                )
            )

        target_formula = FOLFormula(
            id=-1,  # Special ID for target
            nl_text=conclusion,
            fol_string=fol_conclusion,
            source="target",
        )

        initial_state = LogicalState(
            problem_id=problem_id,
            nl_premises=premises,
            fol_formulas=formulas.copy(),
            target_conclusion=conclusion,
            target_fol=target_formula,
            label=label,
        )

        # Convert proof steps
        optionized_steps = []
        next_formula_id = len(formulas)

        for step_idx, step_data in enumerate(proof_steps):
            step_nl = step_data.get("step", "")
            rule_name = step_data.get("rule", "").lower().strip()
            from_indices = step_data.get("from", [])
            result_nl = step_data.get("result", "")
            result_fol = step_data.get("result_fol", "")

            # Map rule name to option type
            option_type = self._map_rule_to_option(rule_name)
            if option_type is None:
                # Default to a generic inference if rule not recognized
                option_type = OptionType.MODUS_PONENS

            # Build option arguments
            option_args = self._build_option_args(option_type, from_indices)

            # Create result formula if this step derives something
            result_formula = None
            if result_nl or result_fol:
                result_formula = FOLFormula(
                    id=next_formula_id,
                    nl_text=result_nl or step_nl,
                    fol_string=result_fol or "",
                    source="derived",
                    derived_by=option_type.name,
                    derived_from=from_indices,
                )
                next_formula_id += 1

            proof_step = ProofStep(
                step_idx=step_idx,
                thought=step_nl,
                option_type=option_type,
                option_args=option_args,
                result_formula=result_formula,
            )
            optionized_steps.append(proof_step)

        # Add terminal CONCLUDE step
        conclude_step = ProofStep(
            step_idx=len(optionized_steps),
            thought=f"Therefore, the conclusion is {label}.",
            option_type=OptionType.CONCLUDE,
            option_args=[self._label_to_int(label)],
        )
        optionized_steps.append(conclude_step)

        return OptionizedTrace(
            problem_id=problem_id,
            initial_state=initial_state,
            steps=optionized_steps,
            final_answer=label,
        )

    def optionize_prontoqa_example(
        self,
        problem_id: str,
        context: str,
        query: str,
        chain: list[str],
        label: bool,
    ) -> OptionizedTrace:
        """
        Convert a PrOntoQA example into an optionized trace.

        PrOntoQA has simpler structure: a context, query, and reasoning chain.

        Args:
            problem_id: Unique identifier
            context: The ontology context (facts and rules)
            query: The question to answer
            chain: List of reasoning steps (predicate format like "Nervous('Wren', True)")
            label: True/False answer

        Returns:
            OptionizedTrace
        """
        # Parse context into premises
        premises = [s.strip() for s in context.split(".") if s.strip()]
        fol_premises = [""] * len(premises)  # PrOntoQA doesn't provide FOL

        formulas = []
        for i, premise in enumerate(premises):
            formulas.append(
                FOLFormula(
                    id=i,
                    nl_text=premise,
                    fol_string="",  # Will be filled by solver
                    source="premise",
                )
            )

        initial_state = LogicalState(
            problem_id=problem_id,
            nl_premises=premises,
            fol_formulas=formulas.copy(),
            target_conclusion=query,
            label="TRUE" if label else "FALSE",
        )

        # Convert chain to proof steps with proper reasoning
        optionized_steps = []
        next_formula_id = len(formulas)
        
        # Track derived facts for premise matching
        derived_facts = []  # List of (entity, property, value) tuples

        for step_idx, step_text in enumerate(chain):
            # Parse the predicate format: Property('Entity', True/False)
            entity, prop, value = self._parse_predicate(step_text)
            
            # Find relevant premises for this derivation
            option_type, option_args, thought = self._build_step_with_reasoning(
                entity, prop, value, premises, formulas, derived_facts, step_idx
            )
            
            # Track this derived fact
            if entity and prop:
                derived_facts.append((entity, prop, value))

            result_formula = FOLFormula(
                id=next_formula_id,
                nl_text=self._predicate_to_natural(entity, prop, value),
                fol_string="",
                source="derived",
                derived_by=option_type.name,
            )
            next_formula_id += 1
            formulas.append(result_formula)

            proof_step = ProofStep(
                step_idx=step_idx,
                thought=thought,
                option_type=option_type,
                option_args=option_args,
                result_formula=result_formula,
            )
            optionized_steps.append(proof_step)

        # Add terminal step with natural language
        final_label = "TRUE" if label else "FALSE"
        conclude_thought = f"Based on the reasoning above, the conclusion '{query}' is {final_label}."
        conclude_step = ProofStep(
            step_idx=len(optionized_steps),
            thought=conclude_thought,
            option_type=OptionType.CONCLUDE,
            option_args=[0 if label else 1],
        )
        optionized_steps.append(conclude_step)

        return OptionizedTrace(
            problem_id=problem_id,
            initial_state=initial_state,
            steps=optionized_steps,
            final_answer=final_label,
        )
    
    def _parse_predicate(self, pred_str: str) -> tuple[str, str, bool]:
        """
        Parse predicate format like "Nervous('Wren', True)" into (entity, property, value).
        """
        import re
        # Match: Property('Entity', True/False)
        match = re.match(r"(\w+)\s*\(\s*['\"]?(\w+)['\"]?\s*,\s*(True|False)\s*\)", pred_str.strip())
        if match:
            prop = match.group(1)
            entity = match.group(2)
            value = match.group(3) == "True"
            return entity, prop, value
        return "", "", True
    
    def _predicate_to_natural(self, entity: str, prop: str, value: bool) -> str:
        """Convert predicate to natural language."""
        if not entity or not prop:
            return ""
        prop_lower = prop.lower()
        if value:
            return f"{entity} is {prop_lower}"
        else:
            return f"{entity} is not {prop_lower}"
    
    def _build_step_with_reasoning(
        self,
        entity: str,
        prop: str,
        value: bool,
        premises: list[str],
        formulas: list[FOLFormula],
        derived_facts: list[tuple],
        step_idx: int,
    ) -> tuple[OptionType, list[int], str]:
        """
        Build a reasoning step with proper premise indices and natural language thought.
        
        Returns: (option_type, option_args, thought)
        """
        entity_lower = entity.lower() if entity else ""
        prop_lower = prop.lower() if prop else ""
        
        # Find premises that mention this entity or property
        entity_premise_idx = -1
        rule_premise_idx = -1
        
        for i, premise in enumerate(premises):
            premise_lower = premise.lower()
            # Check if premise mentions the entity directly
            if entity_lower and entity_lower in premise_lower:
                if "is a" in premise_lower or "is " in premise_lower:
                    entity_premise_idx = i
            # Check if premise is a rule about this property
            if prop_lower in premise_lower:
                if "every" in premise_lower or "each" in premise_lower or "all" in premise_lower:
                    rule_premise_idx = i
        
        # Also check derived formulas
        for i, formula in enumerate(formulas):
            if i >= len(premises):  # This is a derived formula
                formula_lower = formula.nl_text.lower()
                if entity_lower and entity_lower in formula_lower:
                    entity_premise_idx = i
        
        # Determine option type and build thought
        if rule_premise_idx >= 0 and entity_premise_idx >= 0:
            # We have both a rule and a fact - use MODUS_PONENS
            rule_text = premises[rule_premise_idx] if rule_premise_idx < len(premises) else formulas[rule_premise_idx].nl_text
            entity_text = premises[entity_premise_idx] if entity_premise_idx < len(premises) else formulas[entity_premise_idx].nl_text
            
            if value:
                thought = f"Since {entity_text.lower()} (premise {entity_premise_idx}) and {rule_text.lower()} (premise {rule_premise_idx}), we can conclude that {entity} is {prop_lower}."
            else:
                thought = f"Since {entity_text.lower()} (premise {entity_premise_idx}) and {rule_text.lower()} (premise {rule_premise_idx}), we can conclude that {entity} is not {prop_lower}."
            
            return OptionType.MODUS_PONENS, [entity_premise_idx, rule_premise_idx], thought
        
        elif rule_premise_idx >= 0:
            # Only have a rule - use UNIV_INSTANTIATION
            rule_text = premises[rule_premise_idx] if rule_premise_idx < len(premises) else ""
            thought = f"Applying the rule '{rule_text}' to {entity}, we derive that {entity} is {prop_lower}."
            return OptionType.UNIV_INSTANTIATION, [rule_premise_idx, step_idx], thought
        
        else:
            # Fallback - try to find any relevant premise
            best_idx = 0
            for i, premise in enumerate(premises):
                if entity_lower in premise.lower() or prop_lower in premise.lower():
                    best_idx = i
                    break
            
            thought = f"From premise {best_idx}, we can derive that {entity} is {prop_lower}." if value else f"From premise {best_idx}, we can derive that {entity} is not {prop_lower}."
            return OptionType.MODUS_PONENS, [best_idx, min(best_idx + 1, len(premises) - 1)], thought

    def _map_rule_to_option(self, rule_name: str) -> Optional[OptionType]:
        """Map a P-FOLIO rule name to an OptionType."""
        rule_name = rule_name.lower().strip()
        return self.PFOLIO_RULE_MAP.get(rule_name)

    def _build_option_args(
        self, option_type: OptionType, from_indices: list[int]
    ) -> list[int]:
        """Build option arguments from source indices."""
        # Ensure we have the right number of args
        if len(from_indices) >= 2:
            return from_indices[:2]
        elif len(from_indices) == 1:
            return from_indices + [0]  # Pad with default
        else:
            return [0, 0]  # Default args

    def _infer_option_from_text(
        self, text: str, formulas: list[FOLFormula]
    ) -> tuple[OptionType, list[int]]:
        """
        Heuristically infer option type and args from step text.
        
        This is used for PrOntoQA where explicit rule labels aren't provided.
        """
        text_lower = text.lower()

        # Check for common patterns
        if "all" in text_lower or "every" in text_lower:
            # Likely universal instantiation
            return OptionType.UNIV_INSTANTIATION, [0, 0]
        elif "therefore" in text_lower or "so" in text_lower:
            # Likely modus ponens
            return OptionType.MODUS_PONENS, [0, 1]
        elif "not" in text_lower and "because" in text_lower:
            # Might be modus tollens
            return OptionType.MODUS_TOLLENS, [0, 1]
        else:
            # Default to modus ponens for simple chains
            return OptionType.MODUS_PONENS, [0, 1]

    def _label_to_int(self, label: str) -> int:
        """Convert label string to integer for CONCLUDE option."""
        label = label.upper().strip()
        if label == "TRUE":
            return 0
        elif label == "FALSE":
            return 1
        else:
            return 2  # UNKNOWN

    def parse_trace_string(self, trace_str: str, problem_id: str = "") -> OptionizedTrace:
        """
        Parse a trace string (model output) back into an OptionizedTrace.
        
        Expected format:
        Thought: <explanation>
        Action: <Option type="..." args="[...]" />
        ...
        Final Answer: TRUE/FALSE/UNKNOWN
        """
        lines = trace_str.strip().split("\n")
        steps = []
        current_thought = ""
        step_idx = 0
        final_answer = "UNKNOWN"

        for line in lines:
            line = line.strip()

            if line.startswith("Thought:"):
                current_thought = line[8:].strip()

            elif line.startswith("Action:"):
                action_str = line[7:].strip()
                try:
                    step = ProofStep.from_action_string(
                        action_str, step_idx, current_thought
                    )
                    steps.append(step)
                    step_idx += 1
                    current_thought = ""
                except ValueError:
                    continue

            elif line.startswith("Final Answer:"):
                final_answer = line[13:].strip().upper()

        # Create minimal initial state (actual state would need to be provided)
        initial_state = LogicalState(
            problem_id=problem_id,
            nl_premises=[],
            fol_formulas=[],
        )

        return OptionizedTrace(
            problem_id=problem_id,
            initial_state=initial_state,
            steps=steps,
            final_answer=final_answer,
        )

