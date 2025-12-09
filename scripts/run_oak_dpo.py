#!/usr/bin/env python3
"""
Run the OaK-DPO training loop for SOKRATES.

This is the main training script that implements the iterative
Options and Knowledge cycle with DPO alignment.
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

from src.training.oak_loop import OaKLoopConfig, run_oak_pipeline
from src.training.dpo import DPOConfig
from src.data.structures import LogicalState, FOLFormula
from src.solvers.folio_solver import FOLIOSolver
from src.solvers.prontoqa_solver import PrOntoQASolver
from src.utils.logging import ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, config: dict):
    """Load the SFT model for DPO training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    model_config = config.get("model", {})
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model_name = model_config.get("name", "meta-llama/Llama-3.1-8B-Instruct")
    
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if model_config.get("torch_dtype") == "bfloat16" else torch.float32,
        device_map="auto",
    )
    
    # Load LoRA adapter if it exists
    adapter_path = Path(model_path)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer


def load_problems(data_path: str) -> list[LogicalState]:
    """Load problems from processed data file."""
    problems = []
    
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Create formulas
            formulas = []
            for i, (nl, fol) in enumerate(zip(
                item.get("premises", []),
                item.get("premises_fol", [])
            )):
                formulas.append(FOLFormula(
                    id=i,
                    nl_text=nl,
                    fol_string=fol if fol else "",
                    source="premise",
                ))
            
            state = LogicalState(
                problem_id=item["problem_id"],
                nl_premises=item.get("premises", []),
                fol_formulas=formulas,
                target_conclusion=item.get("conclusion", item.get("query", "")),
                label=item.get("label", "UNKNOWN").upper(),
            )
            problems.append(state)
    
    return problems


def get_solver(dataset_type: str):
    """Get the appropriate solver for the dataset."""
    if dataset_type == "prontoqa":
        return PrOntoQASolver()
    else:
        return FOLIOSolver()


def main():
    parser = argparse.ArgumentParser(description="Run SOKRATES OaK-DPO training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default="outputs/sft/final",
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/prontoqa_train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["folio", "prontoqa"],
        default="prontoqa",
        help="Dataset type (for solver selection)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of OaK iterations",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override samples per problem",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up experiment logging
    oak_config_dict = config.get("oak_loop", {})
    dpo_config_dict = config.get("dpo", {})
    base_output_dir = args.output_dir or oak_config_dict.get("output_dir", "outputs/oak_dpo")
    
    exp = ExperimentLogger("oak_dpo", base_dir=os.path.dirname(base_output_dir) or "outputs")
    logger = exp.logger
    
    logger.info(f"Loading config from {args.config}")
    
    # Build DPO config - use experiment subdir
    dpo_config = DPOConfig(
        beta=dpo_config_dict.get("beta", 0.1),
        loss_type=dpo_config_dict.get("loss_type", "sigmoid"),
        output_dir=str(exp.exp_dir / "dpo"),
        num_epochs=dpo_config_dict.get("num_epochs", 1),
        batch_size=dpo_config_dict.get("batch_size", 2),
        gradient_accumulation_steps=dpo_config_dict.get("gradient_accumulation_steps", 16),
        learning_rate=dpo_config_dict.get("learning_rate", 5e-6),
        warmup_ratio=dpo_config_dict.get("warmup_ratio", 0.1),
        max_length=dpo_config_dict.get("max_length", 2048),
        max_prompt_length=dpo_config_dict.get("max_prompt_length", 1024),
        bf16=dpo_config_dict.get("bf16", True),
    )
    
    # Build OaK config - use experiment dir
    oak_config = OaKLoopConfig(
        num_iterations=args.iterations or oak_config_dict.get("num_iterations", 3),
        samples_per_problem=args.samples or oak_config_dict.get("samples_per_problem", 8),
        output_dir=str(exp.exp_dir),
        checkpoint_dir=str(exp.exp_dir / "checkpoints"),
        dpo_config=dpo_config,
        train_option_head=oak_config_dict.get("train_option_head", True),
        option_head_lr=oak_config_dict.get("option_head_lr", 1e-4),
        option_head_epochs=oak_config_dict.get("option_head_epochs", 3),
        log_calibration=oak_config_dict.get("log_calibration", True),
        save_traces=oak_config_dict.get("save_traces", True),
    )
    
    # Log experiment config
    exp.log_config({
        "model_name": args.sft_model,
        "dataset": args.train_data,
        "oak_iterations": oak_config.num_iterations,
        "samples_per_problem": oak_config.samples_per_problem,
        "dpo_beta": dpo_config.beta,
        "learning_rate": dpo_config.learning_rate,
        "extra": {
            "dataset_type": args.dataset_type,
            "val_data": args.val_data,
            "train_option_head": oak_config.train_option_head,
        }
    })
    
    # Load model
    logger.info(f"Loading SFT model from {args.sft_model}")
    model, tokenizer = load_model_and_tokenizer(args.sft_model, config)
    
    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_problems = load_problems(args.train_data)
    logger.info(f"Loaded {len(train_problems)} training problems")
    
    val_problems = None
    if args.val_data:
        logger.info(f"Loading validation data from {args.val_data}")
        val_problems = load_problems(args.val_data)
        logger.info(f"Loaded {len(val_problems)} validation problems")
    
    # Get solver
    solver = get_solver(args.dataset_type)
    logger.info(f"Using solver: {solver.get_solver_name()}")
    
    # Set up wandb
    if args.wandb:
        import wandb
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "sokrates"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []) + ["oak-loop"],
            name=f"oak_dpo_{exp.timestamp}",
            config={
                "oak_config": oak_config.__dict__,
                "dpo_config": dpo_config.__dict__,
                "num_train_problems": len(train_problems),
                "num_val_problems": len(val_problems) if val_problems else 0,
            }
        )
    
    # Log training start
    logger.info("=" * 60)
    logger.info("Starting OaK-DPO training loop")
    logger.info("=" * 60)
    logger.info(f"  SFT Model: {args.sft_model}")
    logger.info(f"  Iterations: {oak_config.num_iterations}")
    logger.info(f"  Samples per problem: {oak_config.samples_per_problem}")
    logger.info(f"  DPO beta: {dpo_config.beta}")
    logger.info(f"  Training problems: {len(train_problems)}")
    logger.info(f"  Output: {oak_config.output_dir}")
    
    # Run OaK loop
    summary = run_oak_pipeline(
        model=model,
        tokenizer=tokenizer,
        solver=solver,
        train_problems=train_problems,
        val_problems=val_problems,
        config=oak_config,
    )
    
    # Log iteration metrics
    for i, hist in enumerate(summary.get("history", [])):
        exp.metrics.log_iteration(i, hist)
    
    # Finish experiment with final metrics
    final_metrics = summary.get("final_calibration", {})
    if summary.get("history"):
        last_iter = summary["history"][-1]
        final_metrics.update({
            "final_valid_traces": last_iter.get("valid_traces", 0),
            "final_valid_steps": last_iter.get("valid_steps", 0),
        })
        if "val_metrics" in last_iter:
            final_metrics["val_accuracy"] = last_iter["val_metrics"].get("accuracy")
    
    exp.log_artifact("training_summary", summary)
    exp.finish(final_metrics)
    
    logger.info(f"Final model checkpoint: {oak_config.checkpoint_dir}/iter_{oak_config.num_iterations - 1}")


if __name__ == "__main__":
    main()

