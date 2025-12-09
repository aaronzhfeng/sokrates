"""
Direct Preference Optimization (DPO) for SOKRATES.

Implements the DPO training loop using solver-induced preferences
over optionized traces.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import Dataset

from src.data.structures import OptionizedTrace, PreferencePair, LogicalState


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # "sigmoid" or "hinge"
    
    # Training
    output_dir: str = "outputs/dpo"
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024
    
    # LoRA (assumes model already has LoRA from SFT)
    use_peft: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    
    # Hardware
    bf16: bool = True


class PreferencePairDataset(Dataset):
    """Dataset of preference pairs for DPO training."""
    
    def __init__(
        self,
        pairs: list[PreferencePair],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
    ):
        """
        Initialize the dataset.
        
        Args:
            pairs: List of PreferencePair objects
            tokenizer: The tokenizer
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        dpo_format = pair.to_dpo_format()
        
        return {
            "prompt": dpo_format["prompt"],
            "chosen": dpo_format["chosen"],
            "rejected": dpo_format["rejected"],
        }


def build_preference_pairs(
    problems: list[LogicalState],
    traces_per_problem: dict[str, list[OptionizedTrace]],
    require_valid_winner: bool = True,
) -> list[PreferencePair]:
    """
    Build preference pairs from sampled traces.
    
    For each problem:
    - Winner: A trace with all valid steps AND correct answer
    - Loser: A trace with invalid steps OR wrong answer
    
    Args:
        problems: List of problem states
        traces_per_problem: Dict mapping problem_id to list of traces
        require_valid_winner: Only create pairs if there's a fully valid trace
        
    Returns:
        List of PreferencePair objects
    """
    pairs = []
    
    for problem in problems:
        traces = traces_per_problem.get(problem.problem_id, [])
        if not traces:
            continue
        
        # Separate valid and invalid traces
        valid_traces = [t for t in traces if t.trace_valid]
        invalid_traces = [t for t in traces if not t.trace_valid]
        
        if require_valid_winner and not valid_traces:
            # Skip if no valid trace exists
            continue
        
        # Create pairs
        if valid_traces and invalid_traces:
            # Standard case: valid vs invalid
            winner = valid_traces[0]  # Could also sample randomly
            for loser in invalid_traces:
                pairs.append(PreferencePair(
                    problem_id=problem.problem_id,
                    prompt=problem.to_prompt(),
                    winner=winner,
                    loser=loser,
                ))
        elif not require_valid_winner and len(traces) >= 2:
            # Fallback: compare by step validity rate
            sorted_traces = sorted(
                traces,
                key=lambda t: t.step_validity_rate,
                reverse=True,
            )
            # Use highest validity as winner, lowest as loser
            pairs.append(PreferencePair(
                problem_id=problem.problem_id,
                prompt=problem.to_prompt(),
                winner=sorted_traces[0],
                loser=sorted_traces[-1],
            ))
    
    return pairs


def train_dpo(
    model,
    tokenizer,
    train_pairs: list[PreferencePair],
    config: Optional[DPOConfig] = None,
    ref_model=None,
):
    """
    Run DPO training.
    
    Args:
        model: The model to train (should be SFT model)
        tokenizer: The tokenizer
        train_pairs: List of preference pairs
        config: DPO configuration
        ref_model: Reference model (if None, uses a copy of model)
    """
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    
    if config is None:
        config = DPOConfig()
    
    # Prepare dataset
    dataset = PreferencePairDataset(
        train_pairs, tokenizer,
        config.max_length, config.max_prompt_length,
    )
    
    # Convert to HF dataset format
    from datasets import Dataset as HFDataset
    
    data_dicts = [dataset[i] for i in range(len(dataset))]
    hf_dataset = HFDataset.from_list(data_dicts)
    
    # Set up DPO config
    dpo_config = TRLDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        bf16=config.bf16,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        report_to="wandb",
        remove_unused_columns=False,
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model(os.path.join(config.output_dir, "final"))
    
    return trainer


def run_dpo_iteration(
    model,
    tokenizer,
    problems: list[LogicalState],
    solver,
    generator,
    config: Optional[DPOConfig] = None,
    num_samples: int = 8,
    iteration: int = 0,
) -> tuple:
    """
    Run a single DPO iteration: sample → verify → build prefs → train.
    
    Args:
        model: Current model
        tokenizer: Tokenizer
        problems: List of problems to train on
        solver: FOL solver for verification
        generator: TraceGenerator for sampling
        config: DPO configuration
        num_samples: Traces to sample per problem
        iteration: Current iteration number
        
    Returns:
        Tuple of (trained_model, stats)
    """
    from src.inference.generate_trace import sample_traces_for_dpo
    
    if config is None:
        config = DPOConfig()
    config.output_dir = f"{config.output_dir}/iter_{iteration}"
    
    print(f"DPO Iteration {iteration}")
    print("=" * 50)
    
    # 1. Sample traces
    print(f"Sampling {num_samples} traces per problem...")
    sampled = sample_traces_for_dpo(generator, problems, solver, num_samples)
    
    # Convert to dict format
    traces_per_problem = {
        problem.problem_id: traces
        for problem, traces in sampled
    }
    
    # 2. Build preference pairs
    print("Building preference pairs...")
    pairs = build_preference_pairs(problems, traces_per_problem)
    print(f"Created {len(pairs)} preference pairs")
    
    if not pairs:
        print("No preference pairs created, skipping DPO")
        return model, {"num_pairs": 0}
    
    # 3. Run DPO training
    print("Running DPO training...")
    trainer = train_dpo(model, tokenizer, pairs, config)
    
    # Collect stats
    stats = {
        "num_pairs": len(pairs),
        "num_problems": len(problems),
        "traces_sampled": sum(len(traces) for traces in traces_per_problem.values()),
        "valid_traces": sum(
            sum(1 for t in traces if t.trace_valid)
            for traces in traces_per_problem.values()
        ),
    }
    
    return model, stats

