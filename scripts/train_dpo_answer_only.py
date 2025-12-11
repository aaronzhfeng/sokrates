#!/usr/bin/env python3
"""
Train DPO with ANSWER-ONLY preferences (no step validity).

This is for the "w/o solver verification" ablation.
Preference pairs are created based only on whether the final answer is correct,
ignoring step validity entirely.

Usage:
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes=6 --mixed_precision=bf16 \
        scripts/train_dpo_answer_only.py \
        --traces outputs/traces/run1/traces.jsonl \
        --model outputs/sft/latest/final \
        --output outputs/dpo/ablation_answer_only
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
from datasets import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_traces(traces_path: str) -> list[dict]:
    """Load traces from JSONL file."""
    traces = []
    with open(traces_path) as f:
        for line in f:
            traces.append(json.loads(line.strip()))
    return traces


def build_trace_text(trace: dict) -> str:
    """Build the trace text (response) from steps."""
    steps = trace.get("steps", [])
    final_answer = trace.get("final_answer", "UNKNOWN")
    
    lines = []
    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action", "")
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
    
    # Add final answer
    lines.append(f"Final Answer: {final_answer}")
    
    return "\n".join(lines)


def build_preference_pairs_answer_only(traces: list[dict]) -> list[dict]:
    """
    Build DPO preference pairs based ONLY on answer correctness.
    
    This ignores step validity entirely - only the final answer matters.
    This simulates "w/o solver verification" ablation.
    """
    # Group traces by problem
    by_problem = defaultdict(list)
    for trace in traces:
        by_problem[trace["problem_id"]].append(trace)
    
    pairs = []
    
    for problem_id, problem_traces in by_problem.items():
        if len(problem_traces) < 2:
            continue
        
        # Split into correct and incorrect
        correct_traces = [t for t in problem_traces if t["correct"]]
        incorrect_traces = [t for t in problem_traces if not t["correct"]]
        
        # Need at least one of each
        if not correct_traces or not incorrect_traces:
            continue
        
        # Create pairs: correct vs incorrect
        for correct_trace in correct_traces[:2]:  # Limit pairs per problem
            for incorrect_trace in incorrect_traces[:2]:
                prompt = build_prompt(correct_trace)
                
                pairs.append({
                    "prompt": prompt,
                    "chosen": build_trace_text(correct_trace),
                    "rejected": build_trace_text(incorrect_trace),
                    "problem_id": problem_id,
                    "chosen_correct": True,
                    "rejected_correct": False,
                    # Note: we ignore step validity here
                })
    
    return pairs


def build_prompt(trace: dict) -> str:
    """Build the input prompt from a trace."""
    premises = trace.get("premises", [])
    conclusion = trace.get("conclusion", "")
    
    lines = ["Premises:"]
    for i, p in enumerate(premises):
        lines.append(f"  [{i}] {p}")
    lines.append(f"\nConclusion to evaluate: {conclusion}")
    lines.append("\nDetermine if the conclusion is TRUE, FALSE, or UNKNOWN.")
    lines.append("\nReasoning:")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train DPO with answer-only preferences")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSONL")
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load traces
    print(f"Loading traces from {args.traces}")
    traces = load_traces(args.traces)
    print(f"Loaded {len(traces)} traces")
    
    # Build preference pairs (answer-only)
    print("Building answer-only preference pairs...")
    pairs = build_preference_pairs_answer_only(traces)
    print(f"Created {len(pairs)} preference pairs (answer-only)")
    
    if len(pairs) < 10:
        print("WARNING: Very few preference pairs. Check trace diversity.")
    
    # Save pairs for reference
    pairs_file = output_path / "preference_pairs.jsonl"
    with open(pairs_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved pairs to {pairs_file}")
    
    # Save config
    config = {
        "traces_path": args.traces,
        "model_path": args.model,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "num_pairs": len(pairs),
        "ablation": "answer_only",  # Mark as ablation
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Import TRL for DPO training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig
    
    # Load model and tokenizer
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use PyTorch's scaled dot product attention
    )
    
    # Create dataset
    dataset = Dataset.from_list(pairs)
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # DPO config
    training_args = DPOConfig(
        output_dir=str(output_path),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=1024,
    )
    
    # Train
    print("Starting DPO training (answer-only preferences)...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    trainer.train()
    
    # Save final model
    final_path = output_path / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved model to {final_path}")
    
    # Save training summary
    summary = {
        "total_pairs": len(pairs),
        "epochs": args.num_epochs,
        "ablation": "answer_only",
        "description": "DPO trained with answer-only preferences (no step validity)",
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()

