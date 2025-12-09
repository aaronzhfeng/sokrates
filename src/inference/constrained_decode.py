"""
Constrained decoding for option actions.

Ensures that generated option actions are always syntactically valid
by constraining the generation to follow the option grammar.
"""

import re
from typing import Optional, Callable
from dataclasses import dataclass

from src.data.structures import OptionType, OPTION_VOCABULARY, OPTION_ARG_COUNTS


# Grammar specification for option actions
OPTION_GRAMMAR = r"""
start: "<Option type=\"" OPTION_TYPE "\" args=\"[" ARGS "]\" />"
OPTION_TYPE: "MODUS_PONENS" | "MODUS_TOLLENS" | "HYPOTHETICAL_SYLLOGISM" 
           | "DISJUNCTIVE_SYLLOGISM" | "UNIV_INSTANTIATION" | "UNIV_GENERALIZATION"
           | "EXIST_INSTANTIATION" | "EXIST_GENERALIZATION" | "AND_INTRO" | "AND_ELIM"
           | "OR_INTRO" | "CASE_SPLIT" | "BICONDITIONAL_INTRO" | "BICONDITIONAL_ELIM"
           | "CONTRADICTION" | "DOUBLE_NEGATION" | "CONDITIONAL_PROOF" | "CONCLUDE"
ARGS: INT ("," INT)*
INT: /[0-9]+/
"""

# Regex pattern for validating option strings
OPTION_PATTERN = re.compile(
    r'<Option type="(\w+)" args="\[([\d,\s]*)\]" />'
)


@dataclass
class ConstrainedDecodingConfig:
    """Configuration for constrained decoding."""
    
    max_formula_idx: int = 50  # Maximum valid formula index
    allow_empty_args: bool = False  # Allow empty argument lists
    strict_arg_count: bool = True  # Enforce correct arg count per option
    

class OptionConstrainer:
    """
    Constrains LLM generation to valid option syntax.
    
    Can be used with various constrained decoding libraries (Outlines, etc.)
    or as a post-processing validator.
    """
    
    def __init__(self, config: Optional[ConstrainedDecodingConfig] = None):
        """
        Initialize the option constrainer.
        
        Args:
            config: Configuration for constraints
        """
        self.config = config or ConstrainedDecodingConfig()
        
        # Build valid option type names
        self.valid_option_names = {opt.name for opt in OPTION_VOCABULARY}
    
    def validate_action(self, action_str: str) -> tuple[bool, str]:
        """
        Validate an action string.
        
        Args:
            action_str: The action string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        action_str = action_str.strip()
        
        # Check basic format
        match = OPTION_PATTERN.match(action_str)
        if not match:
            return False, f"Invalid action format: {action_str}"
        
        option_name = match.group(1)
        args_str = match.group(2)
        
        # Check option type is valid
        if option_name not in self.valid_option_names:
            return False, f"Unknown option type: {option_name}"
        
        # Parse arguments
        args = []
        if args_str.strip():
            try:
                args = [int(x.strip()) for x in args_str.split(",") if x.strip()]
            except ValueError:
                return False, f"Invalid argument format: {args_str}"
        
        # Check argument validity
        if not self.config.allow_empty_args and not args:
            option_type = OptionType[option_name]
            expected_count = OPTION_ARG_COUNTS.get(option_type, 0)
            if expected_count > 0:
                return False, f"Empty args for {option_name}, expected {expected_count}"
        
        # Check argument count
        if self.config.strict_arg_count:
            option_type = OptionType[option_name]
            expected_count = OPTION_ARG_COUNTS.get(option_type, 0)
            if len(args) != expected_count:
                return False, f"Wrong arg count for {option_name}: got {len(args)}, expected {expected_count}"
        
        # Check argument range
        for arg in args:
            if arg < 0 or arg > self.config.max_formula_idx:
                return False, f"Argument out of range: {arg}"
        
        return True, ""
    
    def fix_action(self, action_str: str, num_formulas: int = 10) -> str:
        """
        Attempt to fix an invalid action string.
        
        Args:
            action_str: The action string to fix
            num_formulas: Current number of formulas (for valid arg range)
            
        Returns:
            Fixed action string (or original if unfixable)
        """
        action_str = action_str.strip()
        
        # Try to extract components
        match = OPTION_PATTERN.match(action_str)
        if not match:
            # Try partial matching
            type_match = re.search(r'type="(\w+)"', action_str)
            args_match = re.search(r'args="\[([\d,\s]*)\]"', action_str)
            
            if type_match:
                option_name = type_match.group(1)
                args_str = args_match.group(1) if args_match else ""
            else:
                # Can't fix, return default
                return '<Option type="MODUS_PONENS" args="[0, 1]" />'
        else:
            option_name = match.group(1)
            args_str = match.group(2)
        
        # Validate/fix option name
        if option_name not in self.valid_option_names:
            # Try to find closest match
            option_name = "MODUS_PONENS"  # Default
        
        # Parse and fix arguments
        try:
            args = [int(x.strip()) for x in args_str.split(",") if x.strip()]
        except ValueError:
            args = []
        
        # Ensure correct arg count
        option_type = OptionType[option_name]
        expected_count = OPTION_ARG_COUNTS.get(option_type, 2)
        
        # Pad or truncate args
        while len(args) < expected_count:
            args.append(min(len(args), num_formulas - 1))
        args = args[:expected_count]
        
        # Ensure args are in valid range
        args = [min(max(0, a), num_formulas - 1) for a in args]
        
        # Reconstruct
        args_str = ", ".join(str(a) for a in args)
        return f'<Option type="{option_name}" args="[{args_str}]" />'
    
    def get_valid_next_tokens(
        self,
        partial: str,
        tokenizer,
    ) -> list[int]:
        """
        Get valid next token IDs given a partial generation.
        
        This is used for integration with constrained decoding libraries.
        
        Args:
            partial: The partial action string generated so far
            tokenizer: The tokenizer to use
            
        Returns:
            List of valid next token IDs
        """
        # Determine what we're expecting next based on partial
        partial = partial.strip()
        
        if not partial:
            # Must start with '<Option type="'
            prefix = '<Option type="'
            return [tokenizer.encode(prefix, add_special_tokens=False)[0]]
        
        if partial.endswith('type="'):
            # Must be one of the option type names
            valid_tokens = []
            for opt_name in self.valid_option_names:
                tokens = tokenizer.encode(opt_name, add_special_tokens=False)
                if tokens:
                    valid_tokens.append(tokens[0])
            return valid_tokens
        
        # Check if we're in args section
        if 'args="[' in partial and not partial.endswith(']" />'):
            # Allow digits, commas, spaces, or closing
            valid_chars = "0123456789, ]"
            valid_tokens = []
            for char in valid_chars:
                tokens = tokenizer.encode(char, add_special_tokens=False)
                if tokens:
                    valid_tokens.extend(tokens)
            return list(set(valid_tokens))
        
        # Default: allow all tokens
        return list(range(tokenizer.vocab_size))


def create_outlines_generator(model, tokenizer):
    """
    Create an Outlines-based constrained generator for options.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        
    Returns:
        Outlines generator function
    """
    try:
        from outlines import generate, models
        
        # Create regex pattern for options
        option_types = "|".join(opt.name for opt in OPTION_VOCABULARY)
        pattern = f'<Option type="({option_types})" args="\\[[0-9]+(, [0-9]+)*\\]" />'
        
        # Create generator
        generator = generate.regex(model, pattern)
        
        return generator
        
    except ImportError:
        raise ImportError(
            "Outlines library not installed. "
            "Install with: pip install outlines"
        )


class ThoughtActionGenerator:
    """
    Generates complete Thought/Action sequences with constrained Actions.
    
    The Thought is generated freely, but the Action is constrained to
    valid option syntax.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        constrainer: Optional[OptionConstrainer] = None,
        max_thought_tokens: int = 200,
        max_action_tokens: int = 50,
    ):
        """
        Initialize the generator.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            constrainer: Optional OptionConstrainer for validation
            max_thought_tokens: Maximum tokens for Thought generation
            max_action_tokens: Maximum tokens for Action generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constrainer = constrainer or OptionConstrainer()
        self.max_thought_tokens = max_thought_tokens
        self.max_action_tokens = max_action_tokens
    
    def generate_step(
        self,
        prompt: str,
        num_formulas: int = 10,
        temperature: float = 0.7,
    ) -> tuple[str, str]:
        """
        Generate a single Thought/Action step.
        
        Args:
            prompt: The prompt including problem context
            num_formulas: Number of formulas for arg validation
            temperature: Sampling temperature
            
        Returns:
            Tuple of (thought, action)
        """
        # Generate thought
        thought_prompt = prompt + "\nThought:"
        thought_output = self._generate_until(
            thought_prompt,
            stop_strings=["Action:", "\n\n"],
            max_tokens=self.max_thought_tokens,
            temperature=temperature,
        )
        thought = thought_output.strip()
        
        # Generate action with constraints
        action_prompt = thought_prompt + " " + thought + "\nAction: "
        action_output = self._generate_until(
            action_prompt,
            stop_strings=["/>", "\n"],
            max_tokens=self.max_action_tokens,
            temperature=0.1,  # Lower temperature for structured output
        )
        
        # Ensure action ends properly
        action = action_output.strip()
        if not action.endswith("/>"):
            action = action + "/>"
        
        # Validate and potentially fix action
        is_valid, error = self.constrainer.validate_action(action)
        if not is_valid:
            action = self.constrainer.fix_action(action, num_formulas)
        
        return thought, action
    
    def _generate_until(
        self,
        prompt: str,
        stop_strings: list[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate text until stop string or max tokens.
        
        This is a placeholder - actual implementation depends on
        the specific model/inference framework being used.
        """
        # This would be implemented based on the actual model
        # For now, return a placeholder
        raise NotImplementedError(
            "Implement based on your model/inference framework"
        )


def parse_thought_action(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a Thought/Action block from text.
    
    Args:
        text: Text containing Thought: ... Action: ... format
        
    Returns:
        Tuple of (thought, action) or (None, None) if not found
    """
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL)
    action_match = re.search(r'Action:\s*(<Option[^>]+/>)', text)
    
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    
    return thought, action


def extract_all_steps(text: str) -> list[tuple[str, str]]:
    """
    Extract all Thought/Action pairs from a multi-step trace.
    
    Args:
        text: Full trace text
        
    Returns:
        List of (thought, action) tuples
    """
    steps = []
    
    # Split by "Thought:" to find each step
    parts = re.split(r'(?=Thought:)', text)
    
    for part in parts:
        if not part.strip():
            continue
        
        thought, action = parse_thought_action(part)
        if thought and action:
            steps.append((thought, action))
    
    return steps

