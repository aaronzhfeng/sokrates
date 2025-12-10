#!/usr/bin/env python3
"""
Optimized trace generation using vLLM inference engine.

3-5x faster than HuggingFace generate() with tensor parallelism support.

NOTE: On B200 (Blackwell) GPUs, vLLM's LoRA Triton kernels don't work.
      Use merged models instead:
      1. python scripts/merge_lora_adapter.py --adapter outputs/sft/latest/final --output outputs/sft/latest/merged
      2. python scripts/generate_traces_vllm.py --model outputs/sft/latest/merged ...

Usage:
    # Single GPU with merged model
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_traces_vllm.py \
        --model outputs/sft/latest/merged \
        --data data/processed/prontoqa_train.jsonl \
        --output outputs/traces/run1 \
        --num-problems 100

    # Multi-GPU tensor parallel (splits model, not replicates)
    CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/generate_traces_vllm.py \
        --model outputs/sft/latest/merged \
        --data data/processed/prontoqa_train.jsonl \
        --output outputs/traces/run1 \
        --num-problems 1500 \
        --tensor-parallel-size 4
"""

import argparse
import json
import os
import re
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.structures import LogicalState, FOLFormula
from src.solvers.prontoqa_solver import PrOntoQASolver


@dataclass
class VLLMGenerationConfig:
    """Configuration for vLLM trace generation."""
    max_steps: int = 10
    max_tokens_per_step: int = 200
    temperature: float = 0.7  # Need diversity for DPO!
    top_p: float = 0.95
    num_problems: int = 0  # 0 = all
    samples_per_problem: int = 2
    seed: int = 42


def build_prompt(state: LogicalState) -> str:
    """
    Build the prompt for generation.
    
    Structure:
    1. System instructions (helps guide the model)
    2. Problem in EXACT training format (triggers learned behavior)
    """
    # System instructions to guide the model
    instructions = """You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN.

For each reasoning step, output:
Thought: <explain which premises you're using and why>
Action: <Option type="RULE" args="[premise_indices]" />

Available rules: MODUS_PONENS, MODUS_TOLLENS, CONCLUDE
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


def load_problems(data_path: str, num_problems: int = 0, start_idx: int = 0, seed: int = 42) -> List[LogicalState]:
    """Load problems from JSONL file.
    
    Args:
        data_path: Path to JSONL file
        num_problems: Number of problems to load (0=all)
        start_idx: Starting index for slicing (for data parallel)
        seed: Random seed for sampling
    """
    all_problems = []
    
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            
            formulas = []
            for j, nl in enumerate(item.get("premises", [])):
                formulas.append(FOLFormula(
                    id=j,
                    nl_text=nl,
                    fol_string="",
                    source="premise",
                ))
            
            state = LogicalState(
                problem_id=item["problem_id"],
                nl_premises=item.get("premises", []),
                fol_formulas=formulas,
                target_conclusion=item.get("conclusion", item.get("query", "")),
                label=item.get("label", "UNKNOWN").upper(),
            )
            all_problems.append(state)
    
    # Apply start_idx slicing first (for data parallel)
    if start_idx > 0:
        all_problems = all_problems[start_idx:]
        print(f"Starting from index {start_idx}")
    
    # Then limit number of problems
    if num_problems > 0 and num_problems < len(all_problems):
        problems = all_problems[:num_problems]  # Sequential, not random for data parallel
        print(f"Taking {num_problems} problems (indices {start_idx} to {start_idx + num_problems})")
    else:
        problems = all_problems
    
    return problems


def parse_trace_output(output_text: str, problem: LogicalState) -> dict:
    """Parse vLLM output into trace dict format."""
    steps = []
    lines = output_text.strip().split('\n')
    
    current_thought = ""
    step_idx = 0
    final_answer = "UNKNOWN"
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Thought:"):
            current_thought = line[8:].strip()
        
        elif line.startswith("Action:"):
            action_str = line[7:].strip()
            
            # Parse action
            match = re.match(r'<Option type="(\w+)" args="\[([\d,\s]*)\]" />', action_str)
            if match:
                option_type = match.group(1)
                args_str = match.group(2)
                args = [int(x.strip()) for x in args_str.split(",") if x.strip()]
                
                steps.append({
                    "step_idx": step_idx,
                    "thought": current_thought,
                    "action": action_str,
                    "option_type": option_type,
                    "option_args": args,
                    "solver_valid": None,  # Will be set by solver
                    "solver_error": None,
                })
                
                # Check for CONCLUDE
                if option_type == "CONCLUDE" and args:
                    final_answer = ["TRUE", "FALSE", "UNKNOWN"][min(args[0], 2)]
                
                step_idx += 1
                current_thought = ""
        
        elif line.startswith("Final Answer:"):
            final_answer = line[13:].strip().upper()
            if final_answer not in ["TRUE", "FALSE", "UNKNOWN"]:
                final_answer = "UNKNOWN"
    
    return {
        "problem_id": problem.problem_id,
        "premises": problem.nl_premises,
        "conclusion": problem.target_conclusion,
        "label": problem.label,
        "final_answer": final_answer,
        "correct": final_answer == problem.label,
        "num_steps": len(steps),
        "steps": steps,
        "all_steps_valid": False,  # Will be updated by solver
        "valid_step_count": 0,
        "total_step_count": len(steps),
    }


def verify_traces_with_solver(
    traces: list[dict],
    problems: list[LogicalState],
) -> list[dict]:
    """Verify traces with PrOntoQA solver."""
    problem_lookup = {p.problem_id: p for p in problems}
    
    for trace in tqdm(traces, desc="Verifying"):
        problem = problem_lookup.get(trace["problem_id"])
        if not problem:
            continue
        
        solver = PrOntoQASolver()
        context = ". ".join(problem.nl_premises)
        solver.parse_context(context)
        
        valid_count = 0
        for step in trace["steps"]:
            if step["option_type"] == "CONCLUDE":
                # CONCLUDE is valid if the answer matches ground truth
                step["solver_valid"] = trace["correct"]
            else:
                # For intermediate steps, check if derivation is plausible
                # The solver checks if the entity/category in thought is derivable
                thought_lower = step["thought"].lower()
                
                # Try to extract "X is Y" from thought
                match = re.search(r'(\w+) is (?:a |an )?(\w+)', thought_lower)
                if match:
                    entity, category = match.groups()
                    step["solver_valid"] = solver.check_query(entity, category)
                else:
                    # Can't parse, assume valid (permissive)
                    step["solver_valid"] = True
            
            if step["solver_valid"]:
                valid_count += 1
        
        trace["valid_step_count"] = valid_count
        trace["total_step_count"] = len(trace["steps"])
        trace["all_steps_valid"] = all(s["solver_valid"] for s in trace["steps"]) if trace["steps"] else False
    
    return traces


def generate_traces_vllm(
    model_path: str,
    problems: List[LogicalState],
    config: VLLMGenerationConfig,
    tensor_parallel_size: int = 1,
) -> List[dict]:
    """
    Generate traces using vLLM for optimized inference.
    
    Args:
        model_path: Path to model (base or LoRA adapter)
        problems: List of problems to solve
        config: Generation configuration
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        List of trace dicts
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    
    model_path = Path(model_path)
    
    # Check if this is a LoRA adapter
    is_lora = (model_path / "adapter_config.json").exists()
    
    if is_lora:
        with open(model_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        print(f"Loading base model: {base_model_name}")
        print(f"With LoRA adapter: {model_path}")
        
        llm = LLM(
            model=base_model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=2048,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        
        lora_request = LoRARequest(
            lora_name="sft_adapter",
            lora_int_id=1,
            lora_local_path=str(model_path),
        )
    else:
        print(f"Loading model: {model_path}")
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=2048,
            trust_remote_code=True,
        )
        lora_request = None
    
    # Sampling params - MUST use temperature > 0 for DPO diversity!
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_per_step * config.max_steps,
        stop=["=== ", "\n\n\n", "Final Answer:"],
    )
    
    # Build all prompts
    print(f"Building prompts for {len(problems)} problems × {config.samples_per_problem} samples...")
    all_prompts = []
    prompt_to_problem = []
    
    for problem in problems:
        prompt = build_prompt(problem)
        for _ in range(config.samples_per_problem):
            all_prompts.append(prompt)
            prompt_to_problem.append(problem)
    
    print(f"Total prompts: {len(all_prompts)}")
    
    # Generate all at once (vLLM handles batching internally)
    print("Generating traces with vLLM (this is the fast part!)...")
    if lora_request:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(all_prompts, sampling_params)
    
    # Parse outputs
    print("Parsing outputs...")
    all_traces = []
    for i, output in enumerate(tqdm(outputs, desc="Parsing")):
        problem = prompt_to_problem[i]
        generated_text = output.outputs[0].text
        trace_dict = parse_trace_output(generated_text, problem)
        trace_dict["raw_output"] = generated_text  # Keep for debugging
        all_traces.append(trace_dict)
    
    return all_traces


def main():
    parser = argparse.ArgumentParser(description="Generate traces with vLLM (3-5x faster)")
    parser.add_argument("--model", type=str, required=True, help="Path to SFT model or LoRA adapter")
    parser.add_argument("--data", type=str, required=True, help="Path to problems JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-problems", type=int, default=0, help="Number of problems (0=all)")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting problem index (for data parallel)")
    parser.add_argument("--samples-per-problem", type=int, default=2, help="Samples per problem")
    parser.add_argument("--max-steps", type=int, default=10, help="Max reasoning steps")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (>0 for DPO!)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs for tensor parallelism")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-verify", action="store_true", help="Skip solver verification")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    print(f"Loading problems from {args.data}")
    problems = load_problems(args.data, args.num_problems, args.start_idx, args.seed)
    print(f"Loaded {len(problems)} problems")
    
    config = VLLMGenerationConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        num_problems=args.num_problems,
        samples_per_problem=args.samples_per_problem,
        seed=args.seed,
    )
    
    # Generate
    print("\n" + "="*60)
    print("vLLM Trace Generation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Problems: {len(problems)}")
    print(f"Samples/problem: {config.samples_per_problem}")
    print(f"Expected traces: {len(problems) * config.samples_per_problem}")
    print(f"Temperature: {config.temperature}")
    print(f"Tensor parallel: {args.tensor_parallel_size} GPUs")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    all_traces = generate_traces_vllm(
        model_path=args.model,
        problems=problems,
        config=config,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    gen_time = (datetime.now() - start_time).total_seconds()
    print(f"\nGeneration completed in {gen_time:.1f}s ({len(all_traces)/gen_time:.2f} traces/sec)")
    
    # Verify with solver
    if not args.skip_verify:
        print("\nVerifying with solver...")
        verify_start = datetime.now()
        all_traces = verify_traces_with_solver(all_traces, problems)
        verify_time = (datetime.now() - verify_start).total_seconds()
        print(f"Verification completed in {verify_time:.1f}s")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save traces
    traces_path = output_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for trace in all_traces:
            # Remove raw_output to save space (optional)
            trace_copy = {k: v for k, v in trace.items() if k != "raw_output"}
            f.write(json.dumps(trace_copy) + "\n")
    
    # Calculate stats
    valid_traces = sum(1 for t in all_traces if t.get("all_steps_valid", False))
    correct_traces = sum(1 for t in all_traces if t["correct"])
    total_steps = sum(t["total_step_count"] for t in all_traces)
    valid_steps = sum(t["valid_step_count"] for t in all_traces)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "num_problems": len(problems),
        "samples_per_problem": config.samples_per_problem,
        "total_traces": len(all_traces),
        "valid_traces": valid_traces,
        "correct_traces": correct_traces,
        "accuracy": correct_traces / len(all_traces) if all_traces else 0,
        "step_validity": valid_steps / total_steps if total_steps else 0,
        "generation_time_seconds": gen_time,
        "total_time_seconds": total_time,
        "traces_per_second": len(all_traces) / gen_time if gen_time else 0,
        "config": asdict(config),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("vLLM Generation Complete")
    print("="*60)
    print(f"Total traces: {len(all_traces)}")
    print(f"Correct answers: {correct_traces} ({correct_traces/len(all_traces)*100:.1f}%)")
    print(f"Valid traces: {valid_traces} ({valid_traces/len(all_traces)*100:.1f}%)")
    print(f"Step validity: {valid_steps}/{total_steps} ({valid_steps/total_steps*100:.1f}%)" if total_steps else "N/A")
    print(f"Generation time: {gen_time:.1f}s ({len(all_traces)/gen_time:.2f} traces/sec)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output: {traces_path}")
    print("="*60)


if __name__ == "__main__":
    main()

