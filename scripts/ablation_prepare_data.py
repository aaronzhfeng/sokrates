#!/usr/bin/env python3
"""
Prepare ablation training data formats:
1. Raw CoT (no Option tags) - natural language reasoning
2. Action-only (no Thought lines) - just Option tags
"""

import json
import re
import argparse
from pathlib import Path


def convert_to_raw_cot(example: dict) -> dict:
    """
    Convert optionized format to raw CoT format.
    
    From:
        Thought: Since X and Y, we can conclude Z.
        Action: <Option type="MODUS_PONENS" args="[0, 1]" />
        ...
        Action: <Option type="CONCLUDE" args="[0]" />
        Final Answer: TRUE
    
    To:
        Step 1: Since X and Y, we can conclude Z.
        Step 2: ...
        Therefore, the answer is TRUE.
    """
    training_text = example.get("training_text", "")
    
    # Split into problem and reasoning parts
    if "Reasoning:" in training_text:
        problem_part, reasoning_part = training_text.split("Reasoning:", 1)
    else:
        return None  # Skip if no reasoning section
    
    # Extract thoughts from the reasoning
    thoughts = []
    for line in reasoning_part.split("\n"):
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line[8:].strip()
            thoughts.append(thought)
    
    # Extract final answer
    label = example.get("label", "UNKNOWN")
    
    # Build raw CoT format
    steps = []
    for i, thought in enumerate(thoughts[:-1], 1):  # Exclude the final "Based on..." thought
        steps.append(f"Step {i}: {thought}")
    
    # Add conclusion
    steps.append(f"Therefore, the answer is {label}.")
    
    # Build new training text
    new_training_text = problem_part.strip() + "\n\nReasoning:\n" + "\n".join(steps)
    
    new_example = example.copy()
    new_example["training_text"] = new_training_text
    new_example["format"] = "raw_cot"
    
    return new_example


def convert_to_action_only(example: dict) -> dict:
    """
    Convert optionized format to action-only format.
    
    From:
        Thought: Since X and Y, we can conclude Z.
        Action: <Option type="MODUS_PONENS" args="[0, 1]" />
    
    To:
        Action: <Option type="MODUS_PONENS" args="[0, 1]" />
    """
    training_text = example.get("training_text", "")
    
    # Split into problem and reasoning parts
    if "Reasoning:" in training_text:
        problem_part, reasoning_part = training_text.split("Reasoning:", 1)
    else:
        return None
    
    # Extract only Action lines
    actions = []
    for line in reasoning_part.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            actions.append(line)
    
    # Extract final answer
    label = example.get("label", "UNKNOWN")
    
    # Build action-only format
    new_training_text = problem_part.strip() + "\n\nReasoning:\n" + "\n".join(actions) + f"\n\nFinal Answer: {label}"
    
    new_example = example.copy()
    new_example["training_text"] = new_training_text
    new_example["format"] = "action_only"
    
    return new_example


def main():
    parser = argparse.ArgumentParser(description="Prepare ablation data formats")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--format", type=str, required=True, choices=["raw_cot", "action_only"],
                        help="Target format")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converter = convert_to_raw_cot if args.format == "raw_cot" else convert_to_action_only
    
    converted = []
    skipped = 0
    
    with open(input_path) as f:
        for line in f:
            example = json.loads(line.strip())
            new_example = converter(example)
            if new_example:
                converted.append(new_example)
            else:
                skipped += 1
    
    with open(output_path, "w") as f:
        for example in converted:
            f.write(json.dumps(example) + "\n")
    
    print(f"Converted {len(converted)} examples to {args.format} format")
    print(f"Skipped {skipped} examples")
    print(f"Saved to {output_path}")
    
    # Show sample
    if converted:
        print("\n--- Sample output ---")
        print(converted[0]["training_text"][:1000])


if __name__ == "__main__":
    main()

