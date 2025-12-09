"""
Trace generation utilities for SOKRATES.

Generates optionized proof traces from problems using the trained model.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from src.data.structures import (
    LogicalState,
    ProofStep,
    OptionizedTrace,
    OptionType,
    FOLFormula,
)
from src.inference.constrained_decode import (
    OptionConstrainer,
    parse_thought_action,
    OPTION_PATTERN,
)


@dataclass
class GenerationConfig:
    """Configuration for trace generation."""
    
    max_steps: int = 15  # Maximum proof steps
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    max_thought_tokens: int = 150
    max_action_tokens: int = 50
    use_constrained_decoding: bool = True
    validate_steps: bool = True


class TraceGenerator:
    """
    Generates optionized proof traces from a trained model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GenerationConfig] = None,
        constrainer: Optional[OptionConstrainer] = None,
        device: str = "cuda",
    ):
        """
        Initialize the trace generator.
        
        Args:
            model: The fine-tuned language model
            tokenizer: The tokenizer
            config: Generation configuration
            constrainer: Option constrainer for validation
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.constrainer = constrainer or OptionConstrainer()
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def generate_trace(
        self,
        state: LogicalState,
        num_samples: int = 1,
    ) -> list[OptionizedTrace]:
        """
        Generate one or more proof traces for a problem.
        
        Args:
            state: The initial logical state
            num_samples: Number of traces to generate
            
        Returns:
            List of OptionizedTrace objects
        """
        traces = []
        
        for _ in range(num_samples):
            trace = self._generate_single_trace(state)
            traces.append(trace)
        
        return traces
    
    def _generate_single_trace(self, initial_state: LogicalState) -> OptionizedTrace:
        """Generate a single proof trace."""
        # Build initial prompt
        prompt = self._build_prompt(initial_state)
        
        steps = []
        current_state = initial_state
        generation_history = prompt
        
        for step_idx in range(self.config.max_steps):
            # Generate next step
            thought, action = self._generate_step(
                generation_history,
                current_state.num_formulas,
            )
            
            if thought is None or action is None:
                break
            
            # Parse action into ProofStep
            step = self._parse_step(step_idx, thought, action)
            if step is None:
                break
            
            steps.append(step)
            
            # Update generation history
            generation_history += f"\nThought: {thought}\nAction: {action}"
            
            # Check for terminal step
            if step.option_type == OptionType.CONCLUDE:
                break
            
            # Update state with derived formula (simplified - actual impl would use solver)
            if step.result_formula:
                current_state.add_formula(step.result_formula)
        
        # Determine final answer
        final_answer = "UNKNOWN"
        if steps and steps[-1].option_type == OptionType.CONCLUDE:
            conclude_arg = steps[-1].option_args[0] if steps[-1].option_args else 2
            final_answer = ["TRUE", "FALSE", "UNKNOWN"][min(conclude_arg, 2)]
        
        return OptionizedTrace(
            problem_id=initial_state.problem_id,
            initial_state=initial_state,
            steps=steps,
            final_answer=final_answer,
        )
    
    def _build_prompt(self, state: LogicalState) -> str:
        """Build the prompt for generation."""
        lines = [
            "You are a logical reasoning assistant. Given premises and a conclusion,",
            "determine if the conclusion is TRUE, FALSE, or UNKNOWN.",
            "Reason step by step using formal inference rules.",
            "",
            "For each step, provide:",
            "Thought: Your reasoning in natural language",
            "Action: <Option type=\"RULE_NAME\" args=\"[indices]\" />",
            "",
            "Available rules: MODUS_PONENS, MODUS_TOLLENS, UNIV_INSTANTIATION,",
            "AND_INTRO, AND_ELIM, OR_INTRO, DISJUNCTIVE_SYLLOGISM, etc.",
            "End with: <Option type=\"CONCLUDE\" args=\"[0/1/2]\" />",
            "(0=TRUE, 1=FALSE, 2=UNKNOWN)",
            "",
            "---",
            "",
            "Premises:",
        ]
        
        for i, premise in enumerate(state.nl_premises):
            lines.append(f"  [{i}] {premise}")
        
        lines.append(f"\nConclusion to evaluate: {state.target_conclusion}")
        lines.append("\nReasoning:")
        
        return "\n".join(lines)
    
    def _generate_step(
        self,
        prompt: str,
        num_formulas: int,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Generate a single Thought/Action step.
        
        Returns:
            Tuple of (thought, action) or (None, None) on failure
        """
        # Prepare input
        full_prompt = prompt + "\nThought:"
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_thought_tokens + self.config.max_action_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                top_p=self.config.top_p if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse thought and action
        thought, action = parse_thought_action("Thought: " + generated)
        
        # Validate and fix action if needed
        if action and self.config.validate_steps:
            is_valid, error = self.constrainer.validate_action(action)
            if not is_valid:
                action = self.constrainer.fix_action(action, num_formulas)
        
        return thought, action
    
    def _parse_step(
        self,
        step_idx: int,
        thought: str,
        action: str,
    ) -> Optional[ProofStep]:
        """Parse generated text into a ProofStep."""
        try:
            step = ProofStep.from_action_string(action, step_idx, thought)
            
            # Create a placeholder result formula
            if step.option_type != OptionType.CONCLUDE:
                step.result_formula = FOLFormula(
                    id=-1,  # Will be assigned later
                    nl_text=thought,
                    fol_string="",
                    source="derived",
                    derived_by=step.option_type.name,
                )
            
            return step
            
        except (ValueError, KeyError) as e:
            return None
    
    def generate_batch(
        self,
        states: list[LogicalState],
        num_samples_per_problem: int = 1,
    ) -> dict[str, list[OptionizedTrace]]:
        """
        Generate traces for multiple problems.
        
        Args:
            states: List of LogicalState objects
            num_samples_per_problem: Number of traces per problem
            
        Returns:
            Dict mapping problem_id to list of traces
        """
        results = {}
        
        for state in states:
            traces = self.generate_trace(state, num_samples_per_problem)
            results[state.problem_id] = traces
        
        return results


def sample_traces_for_dpo(
    generator: TraceGenerator,
    problems: list[LogicalState],
    solver,
    num_samples: int = 8,
) -> list[tuple[LogicalState, list[OptionizedTrace]]]:
    """
    Sample traces and verify them for DPO training.
    
    Args:
        generator: The trace generator
        problems: List of problems
        solver: FOL solver for verification
        num_samples: Number of traces to sample per problem
        
    Returns:
        List of (problem, traces) tuples with solver verification
    """
    results = []
    
    for problem in problems:
        # Generate traces
        traces = generator.generate_trace(problem, num_samples)
        
        # Verify each trace
        for trace in traces:
            solver.verify_trace(trace, problem.label)
        
        results.append((problem, traces))
    
    return results

