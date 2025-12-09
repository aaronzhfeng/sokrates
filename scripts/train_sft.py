#!/usr/bin/env python3
"""
Run Supervised Fine-Tuning (SFT) for SOKRATES.

Trains the base model on optionized proof traces to learn
the Thought/Action format.
"""

import argparse
import json
import os
from pathlib import Path

import yaml
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.sft import SFTConfig, run_sft_pipeline
from src.data.structures import OptionizedTrace, LogicalState, ProofStep, OptionType


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str) -> list[OptionizedTrace]:
    """Load optionized training data."""
    traces = []
    
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_file) as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Reconstruct trace from saved data
            # In practice, would have full trace serialization
            initial_state = LogicalState(
                problem_id=item["problem_id"],
                nl_premises=item.get("premises", []),
                fol_formulas=[],
                target_conclusion=item.get("conclusion", ""),
                label=item.get("label", "UNKNOWN"),
            )
            
            # Parse steps if available, otherwise create minimal trace
            steps = []
            if "steps" in item:
                for i, step_data in enumerate(item["steps"]):
                    step = ProofStep(
                        step_idx=i,
                        thought=step_data.get("thought", ""),
                        option_type=OptionType[step_data.get("option_type", "MODUS_PONENS")],
                        option_args=step_data.get("args", [0, 1]),
                    )
                    steps.append(step)
            else:
                # Create a minimal conclude step
                steps = [ProofStep(
                    step_idx=0,
                    thought=f"Based on the premises, the conclusion is {item.get('label', 'UNKNOWN')}.",
                    option_type=OptionType.CONCLUDE,
                    option_args=[{"TRUE": 0, "FALSE": 1, "UNKNOWN": 2}.get(item.get("label", "UNKNOWN"), 2)],
                )]
            
            trace = OptionizedTrace(
                problem_id=item["problem_id"],
                initial_state=initial_state,
                steps=steps,
                final_answer=item.get("label", "UNKNOWN"),
            )
            traces.append(trace)
    
    return traces


def main():
    parser = argparse.ArgumentParser(description="Run SOKRATES SFT training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/prontoqa_train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    sft_config = config.get("sft", {})
    
    # Create SFTConfig
    training_config = SFTConfig(
        model_name=args.model or config.get("model", {}).get("name", "meta-llama/Llama-3.1-8B-Instruct"),
        output_dir=args.output_dir or sft_config.get("output_dir", "outputs/sft"),
        num_epochs=args.epochs or sft_config.get("num_epochs", 3),
        batch_size=args.batch_size or sft_config.get("batch_size", 4),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 8),
        learning_rate=sft_config.get("learning_rate", 2e-5),
        warmup_ratio=sft_config.get("warmup_ratio", 0.1),
        max_seq_length=sft_config.get("max_seq_length", 2048),
        logging_steps=sft_config.get("logging_steps", 10),
        save_steps=sft_config.get("save_steps", 500),
        eval_steps=sft_config.get("eval_steps", 500),
        bf16=sft_config.get("bf16", True),
        gradient_checkpointing=sft_config.get("gradient_checkpointing", True),
    )
    
    # Apply PEFT config
    peft_config = config.get("peft", {})
    if peft_config.get("enabled", True):
        training_config.use_peft = True
        training_config.lora_r = peft_config.get("r", 64)
        training_config.lora_alpha = peft_config.get("lora_alpha", 128)
        training_config.lora_dropout = peft_config.get("lora_dropout", 0.05)
        training_config.lora_target_modules = peft_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
    
    # Load training data
    print(f"Loading training data from {args.data}")
    traces = load_training_data(args.data)
    print(f"Loaded {len(traces)} training traces")
    
    # Set up wandb
    if args.wandb:
        import wandb
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "sokrates"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []),
            config={
                "training_config": training_config.__dict__,
                "num_traces": len(traces),
            }
        )
    
    # Run SFT
    print("\nStarting SFT training...")
    print(f"  Model: {training_config.model_name}")
    print(f"  Output: {training_config.output_dir}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    
    model, tokenizer, trainer = run_sft_pipeline(traces, training_config)
    
    print("\nSFT training complete!")
    print(f"Model saved to: {training_config.output_dir}/final")


if __name__ == "__main__":
    main()

