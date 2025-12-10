#!/usr/bin/env python3
"""
Evaluate a trained SOKRATES model.

Computes accuracy, step validity, trace validity, and calibration metrics.

Usage:
    # Evaluate SFT model
    python scripts/evaluate.py \
        --model outputs/sft/latest/final \
        --data data/processed/prontoqa_test.jsonl \
        --dataset-type prontoqa \
        --output-dir outputs/evaluation/sft

    # Evaluate DPO model
    python scripts/evaluate.py \
        --model outputs/oak_dpo/latest/checkpoints/iter_1/model \
        --data data/processed/prontoqa_test.jsonl \
        --dataset-type prontoqa \
        --output-dir outputs/evaluation/dpo_iter1
        
    # Quick evaluation (subset)
    python scripts/evaluate.py \
        --model outputs/sft/latest/final \
        --data data/processed/prontoqa_test.jsonl \
        --max-samples 100 \
        --dataset-name quick_test
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_all_metrics, format_metrics_report
from src.evaluation.calibration import CalibrationAnalyzer, format_calibration_report
from src.inference.generate_trace import TraceGenerator, GenerationConfig
from src.data.structures import LogicalState, FOLFormula
from src.solvers.folio_solver import FOLIOSolver
from src.solvers.prontoqa_solver import PrOntoQASolver


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, merge_adapter: bool = False):
    """
    Load model and tokenizer for evaluation.
    
    Args:
        model_path: Path to model checkpoint (PEFT adapter or full model)
        merge_adapter: If True, merge LoRA adapter into base model for faster inference
        
    Returns:
        model, tokenizer tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Check if this is a PEFT model
    model_path = Path(model_path)
    
    # Try to load tokenizer from model path first, then from base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Could not load tokenizer from {model_path}, trying base model...")
        if (model_path / "adapter_config.json").exists():
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Required for decoder-only batched generation
    
    # Try to load as PEFT model first
    if (model_path / "adapter_config.json").exists():
        # Load adapter config to get base model
        with open(model_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Optionally merge adapter for faster inference
        if merge_adapter:
            print("Merging adapter into base model...")
            model = model.merge_and_unload()
    else:
        # Load as regular model
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def load_problems(data_path: str, max_samples: int = None) -> list[LogicalState]:
    """Load test problems."""
    problems = []
    
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            item = json.loads(line.strip())
            
            formulas = []
            for j, (nl, fol) in enumerate(zip(
                item.get("premises", []),
                item.get("premises_fol", [])
            )):
                formulas.append(FOLFormula(
                    id=j,
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


def evaluate_model(
    model,
    tokenizer,
    problems: list[LogicalState],
    solver,
    config: dict,
) -> dict:
    """Run evaluation on a set of problems."""
    eval_config = config.get("evaluation", {})
    
    # Create generator
    gen_config = GenerationConfig(
        max_steps=eval_config.get("max_steps", 15),
        temperature=eval_config.get("temperature", 0.1),
        do_sample=not eval_config.get("greedy", True),
    )
    generator = TraceGenerator(model, tokenizer, gen_config)
    
    # Generate and verify traces
    traces = []
    labels = []
    calibration_data = []
    
    print("Generating and verifying traces...")
    for problem in tqdm(problems):
        # Generate trace
        trace = generator.generate_trace(problem, num_samples=1)[0]
        
        # Verify with solver
        solver.verify_trace(trace, problem.label)
        
        traces.append(trace)
        labels.append(problem.label)
        
        # Collect calibration data
        for step in trace.steps:
            if step.predicted_valid is not None and step.solver_valid is not None:
                calibration_data.append((step.predicted_valid, int(step.solver_valid)))
    
    # Compute metrics
    metrics = compute_all_metrics(traces, labels, calibration_data if calibration_data else None)
    
    return {
        "metrics": metrics,
        "traces": traces,
        "calibration_data": calibration_data,
    }


def save_results(results: dict, output_dir: str, dataset_name: str, save_traces: bool = False):
    """
    Save evaluation results.
    
    Args:
        results: Dict with metrics, traces, and calibration_data
        output_dir: Directory to save results
        dataset_name: Name prefix for output files
        save_traces: Whether to save individual trace predictions (can be large)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    with open(metrics_path, "w") as f:
        # Convert non-serializable items
        metrics_copy = results["metrics"].copy()
        json.dump(metrics_copy, f, indent=2, default=str)
    
    # Save human-readable report
    report_path = os.path.join(output_dir, f"{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(format_metrics_report(results["metrics"]))
        f.write(f"\n\nGenerated: {datetime.now().isoformat()}\n")
    
    # Save trace predictions if requested
    if save_traces and "traces" in results:
        traces_path = os.path.join(output_dir, f"{dataset_name}_traces.jsonl")
        with open(traces_path, "w") as f:
            for trace in results["traces"]:
                trace_dict = {
                    "problem_id": trace.problem_id,
                    "final_answer": trace.final_answer,
                    "num_steps": len(trace.steps),
                    "steps": [
                        {
                            "option_type": step.option_type.name if step.option_type else None,
                            "option_args": step.option_args,
                            "solver_valid": step.solver_valid,
                        }
                        for step in trace.steps
                    ],
                }
                f.write(json.dumps(trace_dict) + "\n")
        print(f"Saved {len(results['traces'])} traces to {traces_path}")
    
    # Save calibration data if available
    if results.get("calibration_data"):
        cal_path = os.path.join(output_dir, f"{dataset_name}_calibration.json")
        with open(cal_path, "w") as f:
            json.dump({
                "predictions": [p for p, _ in results["calibration_data"]],
                "labels": [l for _, l in results["calibration_data"]],
            }, f)
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOKRATES model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SFT model on test set
  python scripts/evaluate.py \\
      --model outputs/sft/latest/final \\
      --data data/processed/prontoqa_test.jsonl

  # Quick evaluation with subset
  python scripts/evaluate.py \\
      --model outputs/sft/latest/final \\
      --data data/processed/prontoqa_test.jsonl \\
      --max-samples 100 \\
      --dataset-name quick_test

  # Evaluate with merged adapter (faster inference)
  python scripts/evaluate.py \\
      --model outputs/oak_dpo/latest/checkpoints/iter_1/model \\
      --data data/processed/prontoqa_test.jsonl \\
      --merge-adapter
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.yaml",
        help="Path to evaluation config",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["folio", "prontoqa"],
        default="prontoqa",
        help="Dataset type (for solver selection)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="test",
        help="Name for output files",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA adapter for faster inference",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save individual trace predictions",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=True,
        help="Use greedy decoding (default: True)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    args = parser.parse_args()
    
    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    
    # Override config with CLI args
    if "evaluation" not in config:
        config["evaluation"] = {}
    if args.greedy or args.temperature == 0:
        config["evaluation"]["greedy"] = True
        config["evaluation"]["temperature"] = 0.0
    else:
        config["evaluation"]["greedy"] = False
        config["evaluation"]["temperature"] = args.temperature
    
    # Load model
    print(f"\n{'='*60}")
    print(f"SOKRATES Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"{'='*60}\n")
    
    model, tokenizer = load_model_and_tokenizer(args.model, merge_adapter=args.merge_adapter)
    
    # Load test data
    print(f"\nLoading test data from {args.data}")
    problems = load_problems(args.data, args.max_samples)
    print(f"Loaded {len(problems)} test problems")
    
    # Get solver
    if args.dataset_type == "prontoqa":
        solver = PrOntoQASolver()
    else:
        solver = FOLIOSolver()
    print(f"Using solver: {solver.get_solver_name()}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    start_time = datetime.now()
    results = evaluate_model(model, tokenizer, problems, solver, config)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Print results
    print("\n" + "=" * 60)
    print(format_metrics_report(results["metrics"]))
    print(f"\nEvaluation completed in {elapsed:.1f}s ({len(problems)/elapsed:.2f} problems/sec)")
    
    # Save results
    save_results(results, args.output_dir, args.dataset_name, save_traces=args.save_traces)


if __name__ == "__main__":
    main()

