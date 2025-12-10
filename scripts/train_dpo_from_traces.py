#!/usr/bin/env python3
"""
Stage 2: Train DPO from pre-generated traces.

Reads traces from Stage 1 and trains DPO on preference pairs.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_from_traces.py \
        --traces outputs/traces/run1/traces.jsonl \
        --model outputs/sft/latest/final \
        --output outputs/dpo/run1

    # Multi-GPU with accelerate
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 \
        scripts/train_dpo_from_traces.py \
        --traces outputs/traces/run1/traces.jsonl \
        --model outputs/sft/latest/final \
        --output outputs/dpo/run1
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

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_traces(traces_path: str) -> list[dict]:
    """Load traces from JSONL file."""
    traces = []
    with open(traces_path) as f:
        for line in f:
            traces.append(json.loads(line.strip()))
    return traces


def build_preference_pairs(
    traces: list[dict],
    require_valid_winner: bool = True,
    min_validity_gap: float = 0.1,
) -> list[dict]:
    """
    Build DPO preference pairs from traces.
    
    For each problem, compare traces and create pairs where:
    - chosen: trace with more valid steps (or correct answer)
    - rejected: trace with fewer valid steps (or wrong answer)
    """
    # Group traces by problem
    by_problem = defaultdict(list)
    for trace in traces:
        by_problem[trace["problem_id"]].append(trace)
    
    pairs = []
    
    for problem_id, problem_traces in by_problem.items():
        if len(problem_traces) < 2:
            continue
        
        # Score each trace
        scored_traces = []
        for trace in problem_traces:
            total = trace["total_step_count"]
            valid = trace["valid_step_count"]
            validity_rate = valid / total if total > 0 else 0
            
            # Higher score = better
            score = validity_rate
            if trace["correct"]:
                score += 1.0  # Bonus for correct answer
            if trace["all_steps_valid"]:
                score += 0.5  # Bonus for fully valid trace
            
            scored_traces.append((score, trace))
        
        # Sort by score (descending)
        scored_traces.sort(key=lambda x: x[0], reverse=True)
        
        # Create pairs from best vs rest
        best_score, best_trace = scored_traces[0]
        
        for other_score, other_trace in scored_traces[1:]:
            # Skip if scores too similar
            if require_valid_winner and best_score - other_score < min_validity_gap:
                continue
            
            # Build prompt from problem
            prompt = build_prompt(best_trace)
            
            # Build chosen/rejected from traces
            chosen = trace_to_completion(best_trace)
            rejected = trace_to_completion(other_trace)
            
            if chosen and rejected and chosen != rejected:
                pairs.append({
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_score": best_score,
                    "rejected_score": other_score,
                })
    
    return pairs


def build_prompt(trace: dict) -> str:
    """Build the prompt from trace (problem setup).
    
    Reconstructs the full problem context from trace metadata.
    """
    # Check if we have stored prompt context
    if "prompt" in trace:
        return trace["prompt"]
    
    # Check if we have premises in the trace
    if "premises" in trace and trace["premises"]:
        lines = ["Premises:"]
        for i, premise in enumerate(trace["premises"]):
            lines.append(f"  [{i}] {premise}")
        
        if "conclusion" in trace:
            lines.append(f"\nConclusion to evaluate: {trace['conclusion']}")
        
        lines.append("\nDetermine if the conclusion is TRUE, FALSE, or UNKNOWN.")
        lines.append("\nReasoning:")
        return "\n".join(lines)
    
    # Fallback: construct minimal prompt from problem_id
    # This is suboptimal but better than nothing
    return f"""You are a logical reasoning assistant. Given a problem, determine if the conclusion is TRUE, FALSE, or UNKNOWN.
Reason step by step using the Thought/Action format.

Problem ID: {trace['problem_id']}

Reasoning:"""


def trace_to_completion(trace: dict) -> str:
    """Convert trace steps to completion string."""
    if not trace["steps"]:
        return ""
    
    lines = []
    for step in trace["steps"]:
        if step["thought"]:
            lines.append(f"Thought: {step['thought']}")
        if step["action"]:
            lines.append(f"Action: {step['action']}")
    
    return "\n".join(lines)


def load_model_for_dpo(model_path: str):
    """Load model for DPO training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model, LoraConfig
    
    model_path = Path(model_path)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        if (model_path / "adapter_config.json").exists():
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_config.get("base_model_name_or_path")
            )
        else:
            raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Check if distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Load base model
    if (model_path / "adapter_config.json").exists():
        with open(model_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        if is_distributed:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Make trainable for DPO
        model = model.merge_and_unload()
        
        # Add new LoRA for DPO training
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Added new LoRA adapter for DPO training")
    else:
        if is_distributed:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    
    return model, tokenizer


def train_dpo(
    model,
    tokenizer,
    pairs: list[dict],
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 4,
    logging_steps: int = 10,
    save_steps: int = 100,
    warmup_ratio: float = 0.1,
    use_wandb: bool = False,
):
    """Train DPO on preference pairs."""
    from trl import DPOConfig, DPOTrainer
    
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])
    
    print(f"Created dataset with {len(hf_dataset)} preference pairs")
    
    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will use implicit reference
        args=dpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nStarting DPO training...")
    trainer.train()
    
    # Save
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved model to {final_path}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train DPO from traces")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces.jsonl")
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    # DPO hyperparameters
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    
    # Pair construction
    parser.add_argument("--require-valid-winner", action="store_true", default=True)
    parser.add_argument("--min-validity-gap", type=float, default=0.1)
    
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load traces
    print(f"Loading traces from {args.traces}")
    traces = load_traces(args.traces)
    print(f"Loaded {len(traces)} traces")
    
    # Build preference pairs
    print("\nBuilding preference pairs...")
    pairs = build_preference_pairs(
        traces,
        require_valid_winner=args.require_valid_winner,
        min_validity_gap=args.min_validity_gap,
    )
    print(f"Created {len(pairs)} preference pairs")
    
    if not pairs:
        print("ERROR: No preference pairs created. Check trace validity rates.")
        return
    
    # Save pairs for inspection
    pairs_path = output_dir / "preference_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved pairs to {pairs_path}")
    
    # Save config
    config = {
        "traces_path": args.traces,
        "model_path": args.model,
        "num_traces": len(traces),
        "num_pairs": len(pairs),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "require_valid_winner": args.require_valid_winner,
        "min_validity_gap": args.min_validity_gap,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load model
    print(f"\nLoading model from {args.model}")
    model, tokenizer = load_model_for_dpo(args.model)
    
    # Train
    print(f"\n{'='*60}")
    print(f"DPO Training")
    print(f"{'='*60}")
    print(f"Pairs: {len(pairs)}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    trainer = train_dpo(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        output_dir=str(output_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        use_wandb=args.wandb,
    )
    
    # Save training summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "final_model": str(output_dir / "final"),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model saved to: {output_dir / 'final'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

