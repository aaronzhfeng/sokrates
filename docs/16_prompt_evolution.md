# Prompt Evolution for Trace Generation

This document tracks the evolution of the inference prompt used in `src/inference/generate_trace.py` for generating optionized reasoning traces.

## Context

During trace generation, we need to prompt the SFT-fine-tuned model to generate Thought/Action steps. The prompt design affects:
1. **Task understanding** - Does the model know what to output?
2. **Format consistency** - Does the output match expected Thought/Action format?
3. **Learned behavior trigger** - Does the prompt activate the fine-tuned weights?

---

## Version 1: Original (System Instructions Only)

**Location**: `_build_prompt()` in `src/inference/generate_trace.py`

**Issues**: 
- Problem section format differed from SFT training data
- Missing "Determine if the conclusion is TRUE, FALSE, or UNKNOWN." line
- No clear separation between instructions and problem

```python
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
```

**Output Example**:
```
You are a logical reasoning assistant. Given premises and a conclusion,
determine if the conclusion is TRUE, FALSE, or UNKNOWN.
Reason step by step using formal inference rules.

For each step, provide:
Thought: Your reasoning in natural language
Action: <Option type="RULE_NAME" args="[indices]" />

Available rules: MODUS_PONENS, MODUS_TOLLENS, UNIV_INSTANTIATION,
AND_INTRO, AND_ELIM, OR_INTRO, DISJUNCTIVE_SYLLOGISM, etc.
End with: <Option type="CONCLUDE" args="[0/1/2]" />
(0=TRUE, 1=FALSE, 2=UNKNOWN)

---

Premises:
  [0] Wren is a jompus
  [1] Each jompus is nervous

Conclusion to evaluate: Wren is nervous.

Reasoning:
```

---

## Version 2: Intermediate (Training Format Only - Too Simple)

**Rationale**: Hypothesis that prompt must exactly match SFT training format to trigger learned behavior.

**Issues**: 
- No instructions to guide the model
- Model may not understand what Thought/Action format means
- Less robust for base model's instruction-following capabilities

```python
def _build_prompt(self, state: LogicalState) -> str:
    """Build the prompt for generation.
    
    CRITICAL: This format MUST match the SFT training data format exactly!
    The model learns to continue from this specific format.
    """
    # Format must match LogicalState.to_prompt() used in training data
    lines = ["Premises:"]
    
    for i, premise in enumerate(state.nl_premises):
        lines.append(f"  [{i}] {premise}")
    
    lines.append(f"\nConclusion to evaluate: {state.target_conclusion}")
    lines.append("\nDetermine if the conclusion is TRUE, FALSE, or UNKNOWN.")
    lines.append("\nReasoning:")
    
    return "\n".join(lines)
```

**Output Example**:
```
Premises:
  [0] Wren is a jompus
  [1] Each jompus is nervous

Conclusion to evaluate: Wren is nervous.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Reasoning:
```

---

## Version 3: Final (Hybrid - Instructions + Training Format)

**Rationale**: Best of both worlds:
1. System instructions help guide the base model (Qwen3-8B is instruction-tuned)
2. Problem section in exact training format triggers fine-tuned LoRA weights

```python
def _build_prompt(self, state: LogicalState) -> str:
    """Build the prompt for generation.
    
    Structure:
    1. System instructions (helps guide the model)
    2. Problem in EXACT training format (triggers learned behavior)
    """
    # System instructions to guide the model
    instructions = """You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN.

For each reasoning step, output:
Thought: <explain which premises you're using and why>
Action: <Option type="RULE" args="[premise_indices]" />

Available rules: MODUS_PONENS, MODUS_TOLLENS, UNIV_INSTANTIATION
End with: <Option type="CONCLUDE" args="[0]" /> for TRUE, [1] for FALSE, [2] for UNKNOWN

---

"""
    # Problem section - MUST match SFT training format exactly
    problem_lines = ["Premises:"]
    for i, premise in enumerate(state.nl_premises):
        problem_lines.append(f"  [{i}] {premise}")
    
    problem_lines.append(f"\nConclusion to evaluate: {state.target_conclusion}")
    problem_lines.append("\nDetermine if the conclusion is TRUE, FALSE, or UNKNOWN.")
    problem_lines.append("\nReasoning:")
    
    return instructions + "\n".join(problem_lines)
```

**Output Example**:
```
You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN.

For each reasoning step, output:
Thought: <explain which premises you're using and why>
Action: <Option type="RULE" args="[premise_indices]" />

Available rules: MODUS_PONENS, MODUS_TOLLENS, UNIV_INSTANTIATION
End with: <Option type="CONCLUDE" args="[0]" /> for TRUE, [1] for FALSE, [2] for UNKNOWN

---

Premises:
  [0] Wren is a jompus
  [1] Each jompus is nervous

Conclusion to evaluate: Wren is nervous.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Reasoning:
```

---

## SFT Training Data Format (Reference)

For comparison, here's what the model sees during SFT training:

```
Premises:
  [0] Wren is a jompus
  [1] Rompuses are not spicy
  [2] Each rompus is a dumpus
  ...
  [11] Each yumpus is a zumpus

Conclusion to evaluate: Wren is not Jompus.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Reasoning:
Thought: Since wren is a jompus (premise 0) and each jompus is nervous (premise 8), we can conclude that Wren is nervous.
Action: <Option type="MODUS_PONENS" args="[0, 8]" />
Thought: Since wren is nervous (premise 12) and each yumpus is a zumpus (premise 11), we can conclude that Wren is yumpus.
Action: <Option type="MODUS_PONENS" args="[12, 11]" />
...
Thought: Based on the reasoning above, the conclusion 'Wren is not Jompus.' is FALSE.
Action: <Option type="CONCLUDE" args="[1]" />

Final Answer: FALSE
```

---

## Key Design Decisions

| Aspect | V1 (Original) | V2 (Simple) | V3 (Final) |
|--------|--------------|-------------|------------|
| Instructions | ✅ Detailed | ❌ None | ✅ Concise |
| Training format match | ❌ Missing line | ✅ Exact | ✅ Exact |
| Separator | `---` embedded | None | `---` before problem |
| Guidance for model | ✅ Good | ❌ Poor | ✅ Good |

---

## Date

Created: December 10, 2025

