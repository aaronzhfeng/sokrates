#!/usr/bin/env python3
"""
Generate all figures for the SOKRATES paper.

This script creates publication-ready figures from training outputs:
- Figure 2: SFT training loss curve
- Figure 3: OaK-DPO iteration progress (accuracy, validity)
- Figure 4: Model comparison (SFT baseline vs final DPO)
- Figure 5: Calibration diagram (if available)

Usage:
    python scripts/generate_paper_figures.py --output-dir paper/figures
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.plot_training import (
    load_sft_history,
    load_oak_metrics,
    COLORS,
)

# Override styles for paper
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,  # Set True if LaTeX is available
})


def figure_sft_loss(history: list[dict], output_path: str):
    """
    Figure 2: SFT Training Loss Curve
    
    Shows rapid convergence of the SFT model on optionized proof traces.
    """
    train_steps = [e['step'] for e in history if 'loss' in e]
    train_losses = [e['loss'] for e in history if 'loss' in e]
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    
    ax.plot(train_steps, train_losses, color=COLORS['primary'], linewidth=1.5)
    
    # Highlight convergence region
    converge_idx = next((i for i, l in enumerate(train_losses) if l < 0.05), len(train_losses))
    if converge_idx < len(train_steps):
        ax.axvline(x=train_steps[converge_idx], color=COLORS['light'], 
                   linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f'Converged\n(step {train_steps[converge_idx]})', 
                    xy=(train_steps[converge_idx], 0.3),
                    xytext=(train_steps[converge_idx] + 100, 1.0),
                    fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], alpha=0.5))
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 5)
    
    # Add final loss annotation
    final_loss = train_losses[-1]
    ax.annotate(f'Final: {final_loss:.4f}', 
                xy=(train_steps[-1], final_loss),
                xytext=(train_steps[-1] * 0.7, final_loss * 3),
                fontsize=8,
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_oak_progress(metrics: list[dict], output_path: str):
    """
    Figure 3: OaK-DPO Iteration Progress
    
    Shows improvement in accuracy and validity across OaK iterations.
    """
    if not metrics:
        print("  Skipping OaK figure (no metrics)")
        return
    
    iterations = list(range(len(metrics)))
    accuracies = [m.get('accuracy', m.get('val_accuracy', 0)) for m in metrics]
    step_validities = [m.get('step_validity', m.get('overall_step_validity', 0)) for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Accuracy
    bars1 = ax1.bar(iterations, accuracies, color=COLORS['primary'], alpha=0.8, 
                    edgecolor=COLORS['neutral'], linewidth=0.5)
    ax1.set_xlabel('OaK Iteration')
    ax1.set_ylabel('Answer Accuracy')
    ax1.set_xticks(iterations)
    ax1.set_xticklabels([f'{i}' for i in iterations])
    ax1.set_ylim(0, 1)
    
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    # Step Validity
    bars2 = ax2.bar(iterations, step_validities, color=COLORS['valid'], alpha=0.8,
                    edgecolor=COLORS['neutral'], linewidth=0.5)
    ax2.set_xlabel('OaK Iteration')
    ax2.set_ylabel('Step Validity Rate')
    ax2.set_xticks(iterations)
    ax2.set_xticklabels([f'{i}' for i in iterations])
    ax2.set_ylim(0, 1)
    
    for bar, val in zip(bars2, step_validities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    # Add improvement annotation
    if len(accuracies) >= 2:
        acc_delta = accuracies[-1] - accuracies[0]
        if acc_delta > 0:
            ax1.annotate(f'+{acc_delta:.1%}', 
                        xy=(len(iterations)-1, accuracies[-1]),
                        xytext=(len(iterations)-0.5, accuracies[-1] + 0.15),
                        fontsize=9, color=COLORS['valid'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_model_comparison(sft_metrics: dict, dpo_metrics: dict, output_path: str):
    """
    Figure 4: SFT Baseline vs OaK-DPO Final Comparison
    
    Bar chart comparing key metrics between models.
    """
    metrics = ['Accuracy', 'Step Validity', 'Trace Validity']
    sft_values = [
        sft_metrics.get('accuracy', 0),
        sft_metrics.get('step_validity', 0),
        sft_metrics.get('trace_validity', 0),
    ]
    dpo_values = [
        dpo_metrics.get('accuracy', 0),
        dpo_metrics.get('step_validity', 0),
        dpo_metrics.get('trace_validity', 0),
    ]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sft_values, width, label='SFT Baseline',
                   color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, dpo_values, width, label='OaK-DPO',
                   color=COLORS['secondary'], alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Improvement indicators
    for i, (s, d) in enumerate(zip(sft_values, dpo_values)):
        if d > s and d > 0:
            improvement = (d - s) / (s + 1e-10) * 100
            ax.annotate(f'↑{improvement:.0f}%',
                       xy=(x[i] + width/2, d + 0.05),
                       ha='center', fontsize=7, color=COLORS['valid'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_combined(history: list[dict], oak_metrics: list[dict], output_path: str):
    """
    Figure 5: Combined Training Overview
    
    2-panel figure showing SFT convergence and OaK improvement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Panel A: SFT Loss
    train_steps = [e['step'] for e in history if 'loss' in e]
    train_losses = [e['loss'] for e in history if 'loss' in e]
    
    ax1.plot(train_steps, train_losses, color=COLORS['primary'], linewidth=1.5)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_yscale('log')
    ax1.set_title('(a) SFT Training', fontsize=10)
    
    # Panel B: OaK Iterations
    if oak_metrics:
        iterations = list(range(len(oak_metrics)))
        accuracies = [m.get('accuracy', m.get('val_accuracy', 0)) for m in oak_metrics]
        
        ax2.bar(iterations, accuracies, color=COLORS['secondary'], alpha=0.8)
        ax2.set_xlabel('OaK Iteration')
        ax2.set_ylabel('Answer Accuracy')
        ax2.set_xticks(iterations)
        ax2.set_ylim(0, 1)
        ax2.set_title('(b) OaK-DPO Progress', fontsize=10)
        
        for i, val in enumerate(accuracies):
            ax2.text(i, val + 0.02, f'{val:.1%}', ha='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'OaK metrics\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) OaK-DPO Progress', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--sft-dir",
        type=str,
        default="outputs/sft/latest",
        help="Path to SFT experiment directory",
    )
    parser.add_argument(
        "--oak-dir",
        type=str,
        default="outputs/oak_dpo/latest",
        help="Path to OaK-DPO experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    
    print("\n" + "=" * 50)
    print("SOKRATES Paper Figure Generation")
    print("=" * 50)
    
    # Load data
    history = None
    oak_metrics = None
    
    try:
        print(f"\n📁 Loading SFT data from {args.sft_dir}")
        history = load_sft_history(args.sft_dir)
        print(f"   Loaded {len(history)} entries")
    except Exception as e:
        print(f"   ⚠️  Could not load SFT data: {e}")
    
    try:
        print(f"\n📁 Loading OaK-DPO data from {args.oak_dir}")
        oak_metrics = load_oak_metrics(args.oak_dir)
        print(f"   Loaded {len(oak_metrics)} entries")
    except Exception as e:
        print(f"   ⚠️  Could not load OaK data: {e}")
    
    print(f"\n📊 Generating figures...")
    
    # Generate figures
    if history:
        figure_sft_loss(history, str(output_dir / f"fig2_sft_loss.{fmt}"))
    
    if oak_metrics:
        figure_oak_progress(oak_metrics, str(output_dir / f"fig3_oak_progress.{fmt}"))
    
    if oak_metrics and len(oak_metrics) >= 2:
        sft_metrics = {
            'accuracy': oak_metrics[0].get('accuracy', oak_metrics[0].get('val_accuracy', 0)),
            'step_validity': oak_metrics[0].get('step_validity', oak_metrics[0].get('overall_step_validity', 0)),
            'trace_validity': oak_metrics[0].get('trace_validity', 0),
        }
        dpo_metrics = {
            'accuracy': oak_metrics[-1].get('accuracy', oak_metrics[-1].get('val_accuracy', 0)),
            'step_validity': oak_metrics[-1].get('step_validity', oak_metrics[-1].get('overall_step_validity', 0)),
            'trace_validity': oak_metrics[-1].get('trace_validity', 0),
        }
        figure_model_comparison(sft_metrics, dpo_metrics, str(output_dir / f"fig4_comparison.{fmt}"))
    
    if history:
        figure_combined(history, oak_metrics or [], str(output_dir / f"fig5_combined.{fmt}"))
    
    print(f"\n✅ Figures saved to {output_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

