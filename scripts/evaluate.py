#!/usr/bin/env python3
"""
Evaluate a trained SOKRATES model.

Computes accuracy, step validity, trace validity, and calibration metrics.
"""

import argparse
import json
import os
from pathlib import Path

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


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Check if this is a PEFT model
    model_path = Path(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
        )
        
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load as regular model
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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


def save_results(results: dict, output_dir: str, dataset_name: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    with open(metrics_path, "w") as f:
        # Convert non-serializable items
        metrics_copy = results["metrics"].copy()
        if "q_phi_calibration_curve" in metrics_copy:
            # Already serializable
            pass
        json.dump(metrics_copy, f, indent=2, default=str)
    
    # Save human-readable report
    report_path = os.path.join(output_dir, f"{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(format_metrics_report(results["metrics"]))
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOKRATES model")
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
    args = parser.parse_args()
    
    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    
    # Load model
    print(f"Loading model from {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load test data
    print(f"Loading test data from {args.data}")
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
    results = evaluate_model(model, tokenizer, problems, solver, config)
    
    # Print results
    print("\n" + "=" * 60)
    print(format_metrics_report(results["metrics"]))
    
    # Save results
    save_results(results, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()

