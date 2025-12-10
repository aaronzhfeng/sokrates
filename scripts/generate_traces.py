#!/usr/bin/env python3
"""
Stage 1: Generate and verify optionized traces.

Outputs:
- traces_{timestamp}.jsonl: All generated traces with solver verification
- traces_summary.json: Statistics about generation

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_traces.py \
        --model outputs/sft/latest/final \
        --data data/processed/prontoqa_train.jsonl \
        --output outputs/traces/run1 \
        --num-problems 1500 \
        --samples-per-problem 2

    # Multi-GPU (data parallel)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/generate_traces.py \
        --model outputs/sft/latest/final \
        --data data/processed/prontoqa_train.jsonl \
        --output outputs/traces/run1 \
        --num-problems 1500 \
        --samples-per-problem 2 \
        --num-gpus 4
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.structures import LogicalState, FOLFormula, OptionizedTrace
from src.inference.generate_trace import TraceGenerator, GenerationConfig
from src.solvers.prontoqa_solver import PrOntoQASolver
from src.solvers.folio_solver import FOLIOSolver


@dataclass
class TraceGenerationConfig:
    """Configuration for trace generation."""
    max_steps: int = 6
    max_thought_tokens: int = 60
    max_action_tokens: int = 25
    temperature: float = 0.0  # Greedy
    do_sample: bool = False
    num_problems: int = 0  # 0 = all
    samples_per_problem: int = 2
    seed: int = 42


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    model_path = Path(model_path)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        # Try loading from base model
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
    
    # Load model
    if (model_path / "adapter_config.json").exists():
        with open(model_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        
        print(f"Loading adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def load_problems(data_path: str, num_problems: int = 0, seed: int = 42) -> list[LogicalState]:
    """Load problems from JSONL file."""
    problems = []
    
    with open(data_path) as f:
        for line in f:
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
    
    # Subsample if needed
    if num_problems > 0 and num_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, num_problems)
        print(f"Sampled {num_problems} problems from {len(problems)} total")
    
    return problems


def trace_to_dict(trace: OptionizedTrace, problem: LogicalState) -> dict:
    """Convert trace to JSON-serializable dict.
    
    Includes full problem context (premises, conclusion) for DPO prompt reconstruction.
    """
    return {
        "problem_id": trace.problem_id,
        # Include problem context for DPO prompt reconstruction
        "premises": problem.nl_premises,
        "conclusion": problem.target_conclusion,
        "label": problem.label,
        "final_answer": trace.final_answer,
        "correct": trace.final_answer == problem.label,
        "num_steps": len(trace.steps),
        "steps": [
            {
                "step_idx": step.step_idx,
                "thought": step.thought,
                "action": step.to_action_string(),  # Use method, not attribute
                "option_type": step.option_type.name if step.option_type else None,
                "option_args": step.option_args,
                "result_formula": str(step.result_formula) if step.result_formula else None,
                "solver_valid": step.solver_valid,
                "solver_error": getattr(step, 'solver_error', None),  # May not exist
            }
            for step in trace.steps
        ],
        "all_steps_valid": all(s.solver_valid for s in trace.steps if s.solver_valid is not None),
        "valid_step_count": sum(1 for s in trace.steps if s.solver_valid),
        "total_step_count": len([s for s in trace.steps if s.solver_valid is not None]),
    }


def generate_traces_for_gpu(
    gpu_id: int,
    problems: list[LogicalState],
    model_path: str,
    config: TraceGenerationConfig,
    solver_type: str,
) -> list[dict]:
    """Generate traces for a subset of problems on a specific GPU."""
    device = f"cuda:{gpu_id}"
    
    # Load model on this GPU
    model, tokenizer = load_model(model_path, device=device)
    
    # Create generator
    gen_config = GenerationConfig(
        max_steps=config.max_steps,
        max_thought_tokens=config.max_thought_tokens,
        max_action_tokens=config.max_action_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
    )
    generator = TraceGenerator(model, tokenizer, gen_config)
    
    # Get solver
    solver = PrOntoQASolver() if solver_type == "prontoqa" else FOLIOSolver()
    
    # Generate traces
    all_traces = []
    
    for problem in tqdm(problems, desc=f"GPU {gpu_id}", position=gpu_id):
        traces = generator.generate_trace(problem, num_samples=config.samples_per_problem)
        
        for trace in traces:
            # Verify with solver
            solver.verify_trace(trace, problem.label)
            
            # Convert to dict
            trace_dict = trace_to_dict(trace, problem)
            trace_dict["gpu_id"] = gpu_id
            all_traces.append(trace_dict)
    
    return all_traces


def main():
    parser = argparse.ArgumentParser(description="Generate optionized traces")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--data", type=str, required=True, help="Path to problems JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-problems", type=int, default=0, help="Number of problems (0=all)")
    parser.add_argument("--samples-per-problem", type=int, default=2, help="Traces per problem")
    parser.add_argument("--max-steps", type=int, default=6, help="Max steps per trace")
    parser.add_argument("--max-thought-tokens", type=int, default=60)
    parser.add_argument("--max-action-tokens", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0, help="0=greedy")
    parser.add_argument("--solver-type", type=str, default="prontoqa", choices=["prontoqa", "folio"])
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for parallel generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    print(f"Loading problems from {args.data}")
    problems = load_problems(args.data, args.num_problems, args.seed)
    print(f"Loaded {len(problems)} problems")
    
    # Config
    config = TraceGenerationConfig(
        max_steps=args.max_steps,
        max_thought_tokens=args.max_thought_tokens,
        max_action_tokens=args.max_action_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        num_problems=args.num_problems,
        samples_per_problem=args.samples_per_problem,
        seed=args.seed,
    )
    
    # Save config
    config_path = output_dir / "generation_config.json"
    with open(config_path, "w") as f:
        json.dump({
            **asdict(config),
            "model": args.model,
            "data": args.data,
            "solver_type": args.solver_type,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Trace Generation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Problems: {len(problems)}")
    print(f"Samples per problem: {config.samples_per_problem}")
    print(f"Expected traces: {len(problems) * config.samples_per_problem}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Generate traces
    start_time = datetime.now()
    
    if args.num_gpus == 1:
        # Single GPU
        all_traces = generate_traces_for_gpu(
            gpu_id=0,
            problems=problems,
            model_path=args.model,
            config=config,
            solver_type=args.solver_type,
        )
    else:
        # Multi-GPU with multiprocessing
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        # Split problems across GPUs
        problems_per_gpu = len(problems) // args.num_gpus
        problem_splits = []
        for i in range(args.num_gpus):
            start_idx = i * problems_per_gpu
            end_idx = start_idx + problems_per_gpu if i < args.num_gpus - 1 else len(problems)
            problem_splits.append(problems[start_idx:end_idx])
        
        # Run in parallel
        with mp.Pool(args.num_gpus) as pool:
            results = pool.starmap(
                generate_traces_for_gpu,
                [(i, problem_splits[i], args.model, config, args.solver_type) 
                 for i in range(args.num_gpus)]
            )
        
        all_traces = []
        for r in results:
            all_traces.extend(r)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Save traces
    traces_path = output_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for trace in all_traces:
            f.write(json.dumps(trace) + "\n")
    print(f"\nSaved {len(all_traces)} traces to {traces_path}")
    
    # Compute and save summary
    valid_traces = sum(1 for t in all_traces if t["all_steps_valid"])
    correct_traces = sum(1 for t in all_traces if t["correct"])
    total_steps = sum(t["total_step_count"] for t in all_traces)
    valid_steps = sum(t["valid_step_count"] for t in all_traces)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "num_problems": len(problems),
        "samples_per_problem": config.samples_per_problem,
        "total_traces": len(all_traces),
        "valid_traces": valid_traces,
        "valid_trace_rate": valid_traces / len(all_traces) if all_traces else 0,
        "correct_traces": correct_traces,
        "accuracy": correct_traces / len(all_traces) if all_traces else 0,
        "total_steps": total_steps,
        "valid_steps": valid_steps,
        "step_validity_rate": valid_steps / total_steps if total_steps else 0,
        "traces_per_second": len(all_traces) / elapsed if elapsed else 0,
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Generation Complete")
    print(f"{'='*60}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Valid traces: {valid_traces} ({summary['valid_trace_rate']:.1%})")
    print(f"Correct answers: {correct_traces} ({summary['accuracy']:.1%})")
    print(f"Step validity: {valid_steps}/{total_steps} ({summary['step_validity_rate']:.1%})")
    print(f"Time: {elapsed:.1f}s ({summary['traces_per_second']:.2f} traces/sec)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

