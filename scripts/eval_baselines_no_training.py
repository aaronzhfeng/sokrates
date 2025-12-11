#!/usr/bin/env python3
"""
Evaluate baselines that DON'T require training:
1. Base CoT (few-shot chain-of-thought prompting)
2. Self-Consistency (sample k=8, majority vote)

These use the base Qwen3-8B model with prompting only.

Usage:
    python scripts/eval_baselines_no_training.py --baseline base_cot --data prontoqa
    python scripts/eval_baselines_no_training.py --baseline self_consistency --data prontoqa
    python scripts/eval_baselines_no_training.py --baseline all --data all
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams


# Few-shot examples for PrOntoQA
PRONTOQA_FEW_SHOT = """You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Example 1:
Premises:
  [0] Rex is a cat.
  [1] Every cat is a mammal.
Conclusion: Rex is a mammal.

Let me think step by step:
- From premise [0], Rex is a cat.
- From premise [1], every cat is a mammal.
- Since Rex is a cat, and every cat is a mammal, Rex must be a mammal.
- This matches the conclusion.

Answer: TRUE

Example 2:
Premises:
  [0] Sam is a bird.
  [1] Every bird can fly.
  [2] Every mammal is warm-blooded.
Conclusion: Sam is warm-blooded.

Let me think step by step:
- From premise [0], Sam is a bird.
- From premise [1], every bird can fly, so Sam can fly.
- Premise [2] tells us about mammals, but Sam is a bird, not a mammal.
- We cannot determine if Sam is warm-blooded from the given premises.

Answer: UNKNOWN

Example 3:
Premises:
  [0] Max is a dog.
  [1] Every dog is loyal.
  [2] No loyal animal is a cat.
Conclusion: Max is a cat.

Let me think step by step:
- From premise [0], Max is a dog.
- From premise [1], every dog is loyal, so Max is loyal.
- From premise [2], no loyal animal is a cat.
- Since Max is loyal, Max cannot be a cat.
- This contradicts the conclusion.

Answer: FALSE

---

Now solve this problem:

"""

# Few-shot examples for FOLIO
FOLIO_FEW_SHOT = """You are a logical reasoning assistant. Given premises and a conclusion, determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Example 1:
Premises:
  [0] All students study hard.
  [1] John is a student.
Conclusion: John studies hard.

Let me think step by step:
- From premise [1], John is a student.
- From premise [0], all students study hard.
- Since John is a student, John must study hard.
- This matches the conclusion.

Answer: TRUE

Example 2:
Premises:
  [0] Some birds can swim.
  [1] Penguins are birds.
Conclusion: Penguins can swim.

Let me think step by step:
- From premise [1], penguins are birds.
- From premise [0], some birds can swim, but not necessarily all.
- We cannot determine if penguins specifically can swim from "some birds."

Answer: UNKNOWN

---

Now solve this problem:

"""


def load_data(data_path: str) -> list:
    """Load problems from JSONL file."""
    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def format_problem(problem: dict, few_shot: str) -> str:
    """Format a problem with few-shot examples."""
    # Build premise section
    premises = problem.get("premises", problem.get("nl_premises", []))
    if isinstance(premises, str):
        premises = [p.strip() for p in premises.split("\n") if p.strip()]
    
    premise_text = "Premises:\n"
    for i, p in enumerate(premises):
        premise_text += f"  [{i}] {p}\n"
    
    conclusion = problem.get("conclusion", problem.get("target_conclusion", ""))
    
    prompt = few_shot + premise_text + f"Conclusion: {conclusion}\n\nLet me think step by step:\n"
    return prompt


def extract_answer(text: str) -> str:
    """Extract TRUE/FALSE/UNKNOWN from model output."""
    text_upper = text.upper()
    
    # Look for explicit "Answer: X" pattern
    if "ANSWER:" in text_upper:
        after_answer = text_upper.split("ANSWER:")[-1].strip()
        for ans in ["TRUE", "FALSE", "UNKNOWN"]:
            if ans in after_answer[:20]:
                return ans
    
    # Look for answer at end
    last_100 = text_upper[-100:]
    for ans in ["TRUE", "FALSE", "UNKNOWN"]:
        if ans in last_100:
            return ans
    
    return "UNKNOWN"


def evaluate_base_cot(
    llm: LLM,
    problems: list,
    few_shot: str,
    output_dir: Path
) -> dict:
    """Evaluate with basic chain-of-thought (single sample)."""
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy
        max_tokens=512,
        stop=["---", "\n\nPremises:"]
    )
    
    # Prepare prompts
    prompts = [format_problem(p, few_shot) for p in problems]
    
    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate
    results = []
    correct = 0
    
    for prob, output in zip(problems, outputs):
        response = output.outputs[0].text
        predicted = extract_answer(response)
        
        label = prob.get("label", prob.get("answer", "")).upper()
        if label in ["0", "1", "2"]:
            label = ["TRUE", "FALSE", "UNKNOWN"][int(label)]
        
        is_correct = predicted == label
        if is_correct:
            correct += 1
        
        results.append({
            "problem_id": prob.get("id", ""),
            "predicted": predicted,
            "label": label,
            "correct": is_correct,
            "response": response[:500]
        })
    
    accuracy = correct / len(problems) if problems else 0
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    summary = {
        "method": "base_cot",
        "total": len(problems),
        "correct": correct,
        "accuracy": accuracy
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def evaluate_self_consistency(
    llm: LLM,
    problems: list,
    few_shot: str,
    output_dir: Path,
    k: int = 8
) -> dict:
    """Evaluate with self-consistency (sample k times, majority vote)."""
    
    sampling_params = SamplingParams(
        temperature=0.7,  # Need diversity for self-consistency
        max_tokens=512,
        n=k,  # Sample k times
        stop=["---", "\n\nPremises:"]
    )
    
    # Prepare prompts
    prompts = [format_problem(p, few_shot) for p in problems]
    
    print(f"Generating {len(prompts)} × {k} = {len(prompts) * k} responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate with majority voting
    results = []
    correct = 0
    
    for prob, output in zip(problems, outputs):
        # Extract all k answers
        answers = []
        for o in output.outputs:
            ans = extract_answer(o.text)
            answers.append(ans)
        
        # Majority vote
        vote_counts = Counter(answers)
        predicted = vote_counts.most_common(1)[0][0]
        
        label = prob.get("label", prob.get("answer", "")).upper()
        if label in ["0", "1", "2"]:
            label = ["TRUE", "FALSE", "UNKNOWN"][int(label)]
        
        is_correct = predicted == label
        if is_correct:
            correct += 1
        
        results.append({
            "problem_id": prob.get("id", ""),
            "predicted": predicted,
            "label": label,
            "correct": is_correct,
            "all_answers": answers,
            "vote_counts": dict(vote_counts)
        })
    
    accuracy = correct / len(problems) if problems else 0
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    summary = {
        "method": "self_consistency",
        "k": k,
        "total": len(problems),
        "correct": correct,
        "accuracy": accuracy
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate no-training baselines")
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["base_cot", "self_consistency", "all"],
                        help="Which baseline to run")
    parser.add_argument("--data", type=str, required=True,
                        choices=["prontoqa", "folio", "all"],
                        help="Which dataset to evaluate on")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model to use")
    parser.add_argument("--output-dir", type=str, default="outputs/eval/baselines",
                        help="Output directory")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of samples for self-consistency")
    parser.add_argument("--gpu", type=int, default=2,
                        help="GPU to use")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index for problems (for data parallel)")
    parser.add_argument("--num-problems", type=int, default=0,
                        help="Number of problems to process (0=all)")
    args = parser.parse_args()
    
    # Only set GPU if not already set by shell
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Initialize model
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8  # Lower to allow multiple parallel instances
    )
    
    # Determine datasets
    datasets = []
    if args.data in ["prontoqa", "all"]:
        datasets.append(("prontoqa", "data/processed/prontoqa_test.jsonl", PRONTOQA_FEW_SHOT))
    if args.data in ["folio", "all"]:
        datasets.append(("folio", "data/processed/folio_validation.jsonl", FOLIO_FEW_SHOT))
    
    # Determine baselines
    baselines = []
    if args.baseline in ["base_cot", "all"]:
        baselines.append("base_cot")
    if args.baseline in ["self_consistency", "all"]:
        baselines.append("self_consistency")
    
    # Run evaluations
    all_results = {}
    
    for dataset_name, data_path, few_shot in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        problems = load_data(data_path)
        
        # Apply slicing for data parallelism
        if args.num_problems > 0:
            end_idx = min(args.start_idx + args.num_problems, len(problems))
            problems = problems[args.start_idx:end_idx]
            print(f"Processing problems {args.start_idx} to {end_idx} ({len(problems)} problems)")
        else:
            print(f"Loaded {len(problems)} problems")
        
        for baseline in baselines:
            print(f"\n[{baseline}] Running on {dataset_name}...")
            output_dir = Path(args.output_dir) / f"{dataset_name}_{baseline}"
            
            if baseline == "base_cot":
                summary = evaluate_base_cot(llm, problems, few_shot, output_dir)
            else:  # self_consistency
                summary = evaluate_self_consistency(llm, problems, few_shot, output_dir, k=args.k)
            
            print(f"  Accuracy: {summary['accuracy']:.1%}")
            all_results[f"{dataset_name}_{baseline}"] = summary
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name}: {result['accuracy']:.1%}")
    
    # Save combined summary
    combined_path = Path(args.output_dir) / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to: {combined_path}")


if __name__ == "__main__":
    main()

