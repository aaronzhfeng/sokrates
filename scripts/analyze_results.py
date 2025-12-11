#!/usr/bin/env python3
"""
Post-training analysis script for SOKRATES.

Generates:
1. Training curves and summary statistics
2. Paper-ready figures
3. LaTeX tables
4. JSON results for downstream processing
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import plotting utilities
from scripts.plot_training import (
    load_sft_history,
    load_oak_metrics,
    plot_sft_combined,
    plot_oak_iterations,
    plot_reliability_diagram,
    plot_model_comparison,
    plot_per_option_validity,
    COLORS,
)


def compute_sft_statistics(history: list[dict]) -> dict:
    """Compute summary statistics from SFT training."""
    train_losses = [e['loss'] for e in history if 'loss' in e]
    eval_losses = [e['eval_loss'] for e in history if 'eval_loss' in e]
    grad_norms = [e['grad_norm'] for e in history if 'grad_norm' in e]
    
    # Find convergence point (first time loss drops below 0.05)
    convergence_step = None
    for e in history:
        if 'loss' in e and e['loss'] < 0.05:
            convergence_step = e['step']
            break
    
    return {
        'total_steps': max(e['step'] for e in history if 'step' in e),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_eval_loss': eval_losses[-1] if eval_losses else None,
        'min_train_loss': min(train_losses) if train_losses else None,
        'initial_train_loss': train_losses[0] if train_losses else None,
        'loss_reduction': (train_losses[0] - train_losses[-1]) / train_losses[0] if train_losses else None,
        'convergence_step': convergence_step,
        'final_grad_norm': grad_norms[-1] if grad_norms else None,
        'mean_grad_norm': np.mean(grad_norms) if grad_norms else None,
    }


def compute_oak_statistics(metrics: list[dict]) -> dict:
    """Compute summary statistics from OaK-DPO training."""
    if not metrics:
        return {}
    
    iterations = [m.get('iteration', i) for i, m in enumerate(metrics)]
    accuracies = [m.get('accuracy', m.get('val_accuracy', 0)) for m in metrics]
    step_validities = [m.get('step_validity', m.get('overall_step_validity', 0)) for m in metrics]
    trace_validities = [m.get('trace_validity', 0) for m in metrics]
    
    return {
        'num_iterations': len(metrics),
        'initial_accuracy': accuracies[0] if accuracies else None,
        'final_accuracy': accuracies[-1] if accuracies else None,
        'accuracy_improvement': (accuracies[-1] - accuracies[0]) if len(accuracies) >= 2 else None,
        'initial_step_validity': step_validities[0] if step_validities else None,
        'final_step_validity': step_validities[-1] if step_validities else None,
        'step_validity_improvement': (step_validities[-1] - step_validities[0]) if len(step_validities) >= 2 else None,
        'initial_trace_validity': trace_validities[0] if trace_validities else None,
        'final_trace_validity': trace_validities[-1] if trace_validities else None,
        'trace_validity_improvement': (trace_validities[-1] - trace_validities[0]) if len(trace_validities) >= 2 else None,
        'best_accuracy': max(accuracies) if accuracies else None,
        'best_iteration': iterations[np.argmax(accuracies)] if accuracies else None,
    }


def generate_latex_table(
    sft_stats: dict,
    oak_stats: dict,
    output_path: str,
):
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{SOKRATES Training Results on PrOntoQA}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{SFT Baseline} & \textbf{OaK-DPO Final} \\",
        r"\midrule",
    ]
    
    # SFT metrics
    if sft_stats.get('final_train_loss'):
        lines.append(f"Final Training Loss & {sft_stats['final_train_loss']:.4f} & -- \\\\")
    if sft_stats.get('convergence_step'):
        lines.append(f"Convergence Step & {sft_stats['convergence_step']} & -- \\\\")
    
    lines.append(r"\midrule")
    
    # OaK metrics
    if oak_stats.get('initial_accuracy') and oak_stats.get('final_accuracy'):
        lines.append(f"Answer Accuracy & {oak_stats['initial_accuracy']:.1%} & {oak_stats['final_accuracy']:.1%} \\\\")
    if oak_stats.get('initial_step_validity') and oak_stats.get('final_step_validity'):
        lines.append(f"Step Validity & {oak_stats['initial_step_validity']:.1%} & {oak_stats['final_step_validity']:.1%} \\\\")
    if oak_stats.get('initial_trace_validity') and oak_stats.get('final_trace_validity'):
        lines.append(f"Trace Validity & {oak_stats['initial_trace_validity']:.1%} & {oak_stats['final_trace_validity']:.1%} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to {output_path}")


def generate_results_json(
    sft_stats: dict,
    oak_stats: dict,
    output_path: str,
):
    """Generate JSON results file for programmatic access."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'sft': sft_stats,
        'oak_dpo': oak_stats,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"JSON results saved to {output_path}")


def print_summary(sft_stats: dict, oak_stats: dict):
    """Print human-readable summary to console."""
    print("\n" + "=" * 60)
    print("SOKRATES TRAINING SUMMARY")
    print("=" * 60)
    
    print("\n📊 SFT Training:")
    if sft_stats:
        print(f"   Total Steps: {sft_stats.get('total_steps', 'N/A')}")
        print(f"   Initial Loss: {sft_stats.get('initial_train_loss', 'N/A'):.4f}")
        print(f"   Final Loss: {sft_stats.get('final_train_loss', 'N/A'):.4f}")
        if sft_stats.get('loss_reduction'):
            print(f"   Loss Reduction: {sft_stats['loss_reduction']:.1%}")
        if sft_stats.get('convergence_step'):
            print(f"   Converged at Step: {sft_stats['convergence_step']}")
    else:
        print("   No SFT data available")
    
    print("\n🔄 OaK-DPO Training:")
    if oak_stats:
        print(f"   Iterations: {oak_stats.get('num_iterations', 'N/A')}")
        print(f"   Accuracy: {oak_stats.get('initial_accuracy', 0):.1%} → {oak_stats.get('final_accuracy', 0):.1%}")
        if oak_stats.get('accuracy_improvement'):
            print(f"            (Δ = {oak_stats['accuracy_improvement']:+.1%})")
        print(f"   Step Validity: {oak_stats.get('initial_step_validity', 0):.1%} → {oak_stats.get('final_step_validity', 0):.1%}")
        if oak_stats.get('step_validity_improvement'):
            print(f"            (Δ = {oak_stats['step_validity_improvement']:+.1%})")
        print(f"   Trace Validity: {oak_stats.get('initial_trace_validity', 0):.1%} → {oak_stats.get('final_trace_validity', 0):.1%}")
        if oak_stats.get('trace_validity_improvement'):
            print(f"            (Δ = {oak_stats['trace_validity_improvement']:+.1%})")
    else:
        print("   No OaK-DPO data available")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze SOKRATES training results")
    parser.add_argument(
        "--sft-dir",
        type=str,
        help="Path to SFT experiment directory",
    )
    parser.add_argument(
        "--oak-dir",
        type=str,
        help="Path to OaK-DPO experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for figures",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show
    fmt = args.format
    
    sft_stats = {}
    oak_stats = {}
    
    # Process SFT results
    if args.sft_dir:
        print(f"\n📁 Loading SFT results from {args.sft_dir}")
        try:
            history = load_sft_history(args.sft_dir)
            sft_stats = compute_sft_statistics(history)
            
            # Generate plots
            plot_sft_combined(
                history, 
                output_path=str(output_dir / f"sft_training.{fmt}"),
                show=show
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Process OaK-DPO results
    if args.oak_dir:
        print(f"\n📁 Loading OaK-DPO results from {args.oak_dir}")
        try:
            metrics = load_oak_metrics(args.oak_dir)
            oak_stats = compute_oak_statistics(metrics)
            
            if metrics:
                plot_oak_iterations(
                    metrics,
                    output_path=str(output_dir / f"oak_iterations.{fmt}"),
                    show=show
                )
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Generate comparison if both available
    if sft_stats and oak_stats:
        # Create mock metrics for comparison plot
        sft_metrics = {
            'accuracy': oak_stats.get('initial_accuracy', 0),
            'step_validity': oak_stats.get('initial_step_validity', 0),
            'trace_validity': oak_stats.get('initial_trace_validity', 0),
        }
        dpo_metrics = {
            'accuracy': oak_stats.get('final_accuracy', 0),
            'step_validity': oak_stats.get('final_step_validity', 0),
            'trace_validity': oak_stats.get('final_trace_validity', 0),
        }
        
        plot_model_comparison(
            sft_metrics,
            dpo_metrics,
            output_path=str(output_dir / f"model_comparison.{fmt}"),
            show=show
        )
    
    # Generate outputs
    if sft_stats or oak_stats:
        generate_latex_table(
            sft_stats, oak_stats,
            str(output_dir / "results_table.tex")
        )
        generate_results_json(
            sft_stats, oak_stats,
            str(output_dir / "results.json")
        )
    
    # Print summary
    print_summary(sft_stats, oak_stats)
    
    print(f"\n📂 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

