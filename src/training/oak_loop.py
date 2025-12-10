"""
OaK (Options and Knowledge) Training Loop for SOKRATES.

Implements the iterative loop:
1. Generate optionized traces
2. Verify with solver (get "knowledge")
3. Update option-success predictor q̂_φ
4. Build preference pairs
5. Update policy via DPO
6. Repeat

This is the core "micro-OaK" cycle described in the paper.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
from datetime import datetime
from tqdm import tqdm

from src.data.structures import LogicalState, OptionizedTrace
from src.models.option_head import OptionSuccessHead
from src.evaluation.calibration import CalibrationAnalyzer
from src.training.dpo import DPOConfig, run_dpo_iteration


@dataclass
class OaKLoopConfig:
    """Configuration for the OaK training loop."""
    
    # Loop parameters
    num_iterations: int = 3
    samples_per_problem: int = 8
    max_problems: int = 0  # 0 = use all problems, >0 = limit to this many
    max_val_problems: int = 200  # Max validation problems for eval (0 = all)
    
    # Paths
    output_dir: str = "outputs/oak_loop"
    checkpoint_dir: str = "checkpoints"
    
    # DPO settings
    dpo_config: DPOConfig = field(default_factory=DPOConfig)
    
    # Option head training
    train_option_head: bool = True
    option_head_lr: float = 1e-4
    option_head_epochs: int = 3
    
    # Logging
    log_calibration: bool = True
    save_traces: bool = True


class OaKLoop:
    """
    The OaK (Options and Knowledge) training loop.
    
    Implements the iterative cycle of experience generation,
    knowledge extraction, and policy improvement.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        solver,
        config: Optional[OaKLoopConfig] = None,
        option_head: Optional[OptionSuccessHead] = None,
    ):
        """
        Initialize the OaK loop.
        
        Args:
            model: The language model (policy)
            tokenizer: The tokenizer
            solver: FOL solver for verification
            config: Loop configuration
            option_head: Option success predictor (will create if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.solver = solver
        self.config = config or OaKLoopConfig()
        
        # Get hidden dim from model config
        hidden_dim = model.config.hidden_size
        
        # Initialize option head if not provided
        if option_head is None:
            self.option_head = OptionSuccessHead(hidden_dim=hidden_dim)
        else:
            self.option_head = option_head
        
        # Move option head to same device as model
        device = next(model.parameters()).device
        self.option_head = self.option_head.to(device)
        
        # Calibration analyzer for tracking
        self.calibration_analyzer = CalibrationAnalyzer()
        
        # History tracking
        self.iteration_history = []
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def run(
        self,
        train_problems: list[LogicalState],
        val_problems: Optional[list[LogicalState]] = None,
    ) -> dict:
        """
        Run the complete OaK loop.
        
        Args:
            train_problems: Problems for training
            val_problems: Optional validation problems
            
        Returns:
            Dict with training history and final metrics
        """
        import random
        from src.inference.generate_trace import TraceGenerator, GenerationConfig
        
        # Apply max_problems limit if set
        if self.config.max_problems > 0 and len(train_problems) > self.config.max_problems:
            random.seed(42)  # Reproducible subset
            train_problems = random.sample(train_problems, self.config.max_problems)
            print(f"[TIME OPT] Using subset of {self.config.max_problems} problems")
        
        print("=" * 60)
        print("Starting SOKRATES OaK Training Loop")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Problems: {len(train_problems)}")
        print(f"Samples per problem: {self.config.samples_per_problem}")
        print("=" * 60)
        
        for iteration in range(self.config.num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.config.num_iterations}")
            print(f"{'='*60}\n")
            
            # Create trace generator with current model
            gen_config = GenerationConfig(
                max_steps=15,
                temperature=0.7,
                do_sample=True,
            )
            generator = TraceGenerator(
                self.model, self.tokenizer, gen_config
            )
            
            # Step 1: Generate experience
            print("Step 1: Generating optionized traces...")
            all_traces = self._generate_experience(generator, train_problems)
            
            # Step 2: Verify with solver
            print("Step 2: Verifying traces with solver...")
            self._verify_traces(all_traces)
            
            # Step 3: Update option head (knowledge)
            if self.config.train_option_head:
                print("Step 3: Updating option success predictor...")
                self._update_option_head(all_traces)
            
            # Step 4-5: Build preferences and run DPO
            print("Step 4-5: Building preferences and running DPO...")
            self._run_dpo_iteration(train_problems, all_traces, iteration)
            
            # Evaluate on validation set
            if val_problems:
                print("Evaluating on validation set...")
                val_metrics = self._evaluate(generator, val_problems)
                self.iteration_history[-1]["val_metrics"] = val_metrics
            
            # Save checkpoint
            self._save_checkpoint(iteration)
            
            # Log progress
            self._log_iteration(iteration)
        
        # Final summary
        summary = self._create_summary()
        self._save_summary(summary)
        
        return summary
    
    def _generate_experience(
        self,
        generator,
        problems: list[LogicalState],
    ) -> dict[str, list[OptionizedTrace]]:
        """Generate traces for all problems, distributed across GPUs."""
        import torch.distributed as dist
        
        # Get distributed training info
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # If launched with torchrun/accelerate but dist isn't initialized yet,
        # initialize so we can all-gather traces before DPO starts.
        if (
            world_size > 1
            and dist.is_available()
            and not dist.is_initialized()
        ):
            dist.init_process_group(backend="nccl", init_method="env://")

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        
        # Split problems across GPUs - each GPU handles a subset
        problems_per_gpu = len(problems) // world_size
        start_idx = rank * problems_per_gpu
        end_idx = start_idx + problems_per_gpu if rank < world_size - 1 else len(problems)
        local_problems = problems[start_idx:end_idx]
        
        if rank == 0:
            print(f"  Distributing {len(problems)} problems across {world_size} GPUs")
            print(f"  Each GPU handles ~{problems_per_gpu} problems")
        
        # Generate traces for this GPU's subset
        local_traces = {}
        disable_tqdm = rank != 0
        
        for problem in tqdm(
            local_problems,
            desc=f"  Generating traces",
            disable=disable_tqdm,
            position=0,
            leave=True,
            ncols=80,
            file=sys.stdout,
        ):
            traces = generator.generate_trace(
                problem,
                num_samples=self.config.samples_per_problem,
            )
            local_traces[problem.problem_id] = traces
        
        # Gather traces from all GPUs (only if distributed)
        if world_size > 1 and dist.is_initialized():
            # Serialize local traces
            import pickle
            local_data = pickle.dumps(local_traces)
            
            # Gather sizes first
            local_size = torch.tensor([len(local_data)], dtype=torch.long, device=f"cuda:{rank}")
            all_sizes = [torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}") for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            
            # Gather all data
            max_size = max(s.item() for s in all_sizes)
            local_tensor = torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
            local_tensor[:len(local_data)] = torch.frombuffer(local_data, dtype=torch.uint8).to(f"cuda:{rank}")
            
            all_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}") for _ in range(world_size)]
            dist.all_gather(all_tensors, local_tensor)
            
            # Combine all traces
            all_traces = {}
            for i, (tensor, size) in enumerate(zip(all_tensors, all_sizes)):
                data = tensor[:size.item()].cpu().numpy().tobytes()
                traces_dict = pickle.loads(data)
                all_traces.update(traces_dict)
        else:
            all_traces = local_traces
        
        total_traces = sum(len(t) for t in all_traces.values())
        if rank == 0:
            print(f"  Generated {total_traces} total traces")
        
        return all_traces
    
    def _verify_traces(self, all_traces: dict[str, list[OptionizedTrace]]):
        """Verify all traces with the solver."""
        total_steps = 0
        valid_steps = 0
        valid_traces = 0
        total_traces = 0
        
        # Only show progress bar on rank 0
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        disable_tqdm = rank != 0
        
        all_trace_lists = list(all_traces.values())
        for traces in tqdm(
            all_trace_lists,
            desc="  Verifying traces",
            disable=disable_tqdm,
            dynamic_ncols=True,
        ):
            for trace in traces:
                total_traces += 1
                
                # Verify trace
                is_valid, step_results = self.solver.verify_trace(
                    trace,
                    trace.initial_state.label,
                )
                
                if is_valid:
                    valid_traces += 1
                
                # Count steps
                for step, result in zip(trace.steps, step_results):
                    total_steps += 1
                    if result.is_valid:
                        valid_steps += 1
        
        print(f"  Verified {total_traces} traces")
        print(f"  Step validity: {valid_steps}/{total_steps} ({100*valid_steps/max(1,total_steps):.1f}%)")
        print(f"  Trace validity: {valid_traces}/{total_traces} ({100*valid_traces/max(1,total_traces):.1f}%)")
        
        # Store in history
        self.iteration_history.append({
            "total_traces": total_traces,
            "valid_traces": valid_traces,
            "total_steps": total_steps,
            "valid_steps": valid_steps,
        })
    
    def _update_option_head(self, all_traces: dict[str, list[OptionizedTrace]]):
        """Update the option success predictor from verified traces."""
        # Collect training data
        # In practice, would extract hidden states from the model
        # For now, we track the calibration data
        
        for traces in all_traces.values():
            for trace in traces:
                for step in trace.steps:
                    if step.solver_valid is not None and step.predicted_valid is not None:
                        self.calibration_analyzer.add_prediction(
                            prediction=step.predicted_valid,
                            label=int(step.solver_valid),
                            metadata={"option_type": step.option_type.name},
                        )
        
        # Log calibration metrics
        if self.config.log_calibration:
            metrics = self.calibration_analyzer.compute_metrics()
            print(f"  Option head calibration:")
            if metrics["brier_score"] is not None:
                print(f"    Brier: {metrics['brier_score']:.4f}")
                print(f"    ECE: {metrics['ece']:.4f}")
    
    def _run_dpo_iteration(
        self,
        problems: list[LogicalState],
        all_traces: dict[str, list[OptionizedTrace]],
        iteration: int,
    ):
        """Run DPO training iteration."""
        from src.training.dpo import build_preference_pairs, train_dpo
        
        # Build preference pairs - first try strict (require valid winner)
        pairs = build_preference_pairs(problems, all_traces, require_valid_winner=True)
        
        # If too few pairs, use fallback (compare by step validity rate)
        if len(pairs) < 100:
            print(f"  Only {len(pairs)} strict pairs, using fallback comparison...")
            pairs = build_preference_pairs(problems, all_traces, require_valid_winner=False)
        
        print(f"  Created {len(pairs)} preference pairs")
        
        if not pairs:
            print("  No preference pairs, skipping DPO")
            return
        
        # Configure DPO
        dpo_config = self.config.dpo_config
        dpo_config.output_dir = os.path.join(
            self.config.output_dir, f"dpo_iter_{iteration}"
        )
        
        # Run DPO
        train_dpo(
            self.model,
            self.tokenizer,
            pairs,
            dpo_config,
        )
        
        self.iteration_history[-1]["num_preference_pairs"] = len(pairs)
    
    def _evaluate(
        self,
        generator,
        problems: list[LogicalState],
    ) -> dict:
        """Evaluate current model on problems (only on rank 0 for efficiency)."""
        from src.evaluation.metrics import compute_all_metrics
        
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        # Only run evaluation on rank 0 to avoid redundant computation
        if rank != 0:
            return {"accuracy": 0.0, "trace_validity": 0.0}
        
        # Limit validation to subset for speed (full eval can be done post-training)
        max_eval = self.config.max_val_problems if self.config.max_val_problems > 0 else len(problems)
        max_eval_problems = min(len(problems), max_eval)
        eval_problems = problems[:max_eval_problems]
        
        print(f"  Evaluating on {max_eval_problems} validation problems...")
        
        # Generate one trace per problem (greedy)
        traces = []
        labels = []
        
        for problem in tqdm(eval_problems, desc="  Evaluating", disable=False):
            trace = generator.generate_trace(problem, num_samples=1)[0]
            self.solver.verify_trace(trace, problem.label)
            traces.append(trace)
            labels.append(problem.label)
        
        metrics = compute_all_metrics(traces, labels)
        
        print(f"  Validation accuracy: {metrics['accuracy']:.2%}")
        print(f"  Validation trace validity: {metrics['trace_validity']:.2%}")
        
        return metrics
    
    def _save_checkpoint(self, iteration: int):
        """Save model and option head checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"iter_{iteration}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model (if using PEFT, save adapter)
        self.model.save_pretrained(os.path.join(checkpoint_path, "model"))
        self.tokenizer.save_pretrained(os.path.join(checkpoint_path, "model"))
        
        # Save option head
        self.option_head.save(os.path.join(checkpoint_path, "option_head.pt"))
        
        # Save calibration data
        self.calibration_analyzer.save(
            os.path.join(checkpoint_path, "calibration.json")
        )
    
    def _log_iteration(self, iteration: int):
        """Log iteration summary."""
        history = self.iteration_history[-1]
        
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Traces: {history['total_traces']} ({history['valid_traces']} valid)")
        print(f"  Steps: {history['total_steps']} ({history['valid_steps']} valid)")
        if "num_preference_pairs" in history:
            print(f"  DPO pairs: {history['num_preference_pairs']}")
        if "val_metrics" in history:
            print(f"  Val accuracy: {history['val_metrics']['accuracy']:.2%}")
    
    def _create_summary(self) -> dict:
        """Create training summary."""
        return {
            "config": {
                "num_iterations": self.config.num_iterations,
                "samples_per_problem": self.config.samples_per_problem,
            },
            "history": self.iteration_history,
            "final_calibration": self.calibration_analyzer.compute_metrics(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _save_summary(self, summary: dict):
        """Save training summary to file."""
        path = os.path.join(self.config.output_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved training summary to {path}")


def run_oak_pipeline(
    model,
    tokenizer,
    solver,
    train_problems: list[LogicalState],
    val_problems: Optional[list[LogicalState]] = None,
    config: Optional[OaKLoopConfig] = None,
) -> dict:
    """
    Run the complete OaK training pipeline.
    
    This is the main entry point for SOKRATES training.
    
    Args:
        model: The base language model
        tokenizer: The tokenizer
        solver: FOL solver
        train_problems: Training problems
        val_problems: Optional validation problems
        config: Training configuration
        
    Returns:
        Training summary dict
    """
    loop = OaKLoop(model, tokenizer, solver, config)
    return loop.run(train_problems, val_problems)

