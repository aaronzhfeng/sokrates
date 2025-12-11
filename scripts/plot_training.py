#!/usr/bin/env python3
"""
Plotting utilities for SOKRATES training results.

Creates publication-quality figures for:
- SFT training curves (loss, learning rate, gradient norm)
- OaK-DPO iteration metrics (accuracy, validity)
- Calibration plots (reliability diagrams)
- Model comparison (SFT baseline vs DPO)
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# SOKRATES color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'light': '#E8E8E8',        # Light gray
    'valid': '#2ECC71',        # Green
    'invalid': '#E74C3C',      # Red
}


def load_sft_history(exp_dir: str) -> list[dict]:
    """Load SFT training history from experiment directory."""
    exp_path = Path(exp_dir)
    
    # Try training_history.json first
    history_file = exp_path / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    
    # Try trainer_state.json from checkpoint
    for checkpoint_dir in sorted(exp_path.glob("checkpoint-*"), reverse=True):
        state_file = checkpoint_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                return data.get("log_history", [])
    
    raise FileNotFoundError(f"No training history found in {exp_dir}")


def load_oak_metrics(exp_dir: str) -> list[dict]:
    """Load OaK-DPO metrics from experiment directory."""
    exp_path = Path(exp_dir)
    metrics_file = exp_path / "metrics.jsonl"
    
    metrics = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return metrics


def plot_sft_loss(history: list[dict], output_path: Optional[str] = None, show: bool = True):
    """
    Plot SFT training loss curve.
    
    Args:
        history: Training history from HuggingFace Trainer
        output_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    # Extract training and eval losses
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot training loss
    ax.plot(train_steps, train_losses, color=COLORS['primary'], 
            linewidth=1.5, label='Training Loss', alpha=0.9)
    
    # Plot eval loss if available
    if eval_losses:
        ax.scatter(eval_steps, eval_losses, color=COLORS['secondary'], 
                   s=80, marker='*', label='Validation Loss', zorder=5)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('SOKRATES SFT Training Loss')
    ax.legend(loc='upper right')
    
    # Add convergence annotation
    final_loss = train_losses[-1] if train_losses else 0
    ax.axhline(y=final_loss, color=COLORS['light'], linestyle='--', alpha=0.7)
    ax.annotate(f'Final: {final_loss:.4f}', 
                xy=(train_steps[-1], final_loss),
                xytext=(train_steps[-1] * 0.7, final_loss + 0.3),
                fontsize=9, color=COLORS['neutral'],
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], alpha=0.5))
    
    # Log scale for y-axis if range is large
    if train_losses and max(train_losses) / min(train_losses) > 10:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_sft_learning_rate(history: list[dict], output_path: Optional[str] = None, show: bool = True):
    """Plot learning rate schedule during SFT."""
    steps = []
    lrs = []
    
    for entry in history:
        if 'learning_rate' in entry and 'step' in entry:
            steps.append(entry['step'])
            lrs.append(entry['learning_rate'])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(steps, lrs, color=COLORS['accent'], linewidth=2)
    ax.fill_between(steps, lrs, alpha=0.2, color=COLORS['accent'])
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('SOKRATES SFT Learning Rate Schedule')
    
    # Format y-axis for scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_sft_gradient_norm(history: list[dict], output_path: Optional[str] = None, show: bool = True):
    """Plot gradient norm during SFT training."""
    steps = []
    grad_norms = []
    
    for entry in history:
        if 'grad_norm' in entry and 'step' in entry:
            steps.append(entry['step'])
            grad_norms.append(entry['grad_norm'])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(steps, grad_norms, color=COLORS['success'], linewidth=1, alpha=0.7)
    
    # Add smoothed line
    if len(grad_norms) > 10:
        window = min(20, len(grad_norms) // 5)
        smoothed = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color=COLORS['success'], 
                linewidth=2, label='Smoothed (MA-20)')
        ax.legend()
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('SOKRATES SFT Gradient Norm')
    
    # Log scale if needed
    if grad_norms and max(grad_norms) / (min(grad_norms) + 1e-10) > 20:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_sft_combined(history: list[dict], output_path: Optional[str] = None, show: bool = True):
    """Create a combined 2x2 plot of all SFT metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Extract data
    train_steps = [e['step'] for e in history if 'loss' in e]
    train_losses = [e['loss'] for e in history if 'loss' in e]
    eval_steps = [e['step'] for e in history if 'eval_loss' in e]
    eval_losses = [e['eval_loss'] for e in history if 'eval_loss' in e]
    lr_steps = [e['step'] for e in history if 'learning_rate' in e]
    lrs = [e['learning_rate'] for e in history if 'learning_rate' in e]
    grad_steps = [e['step'] for e in history if 'grad_norm' in e]
    grad_norms = [e['grad_norm'] for e in history if 'grad_norm' in e]
    epochs = [e['epoch'] for e in history if 'epoch' in e and 'loss' in e]
    
    # (0,0) Loss curve
    ax = axes[0, 0]
    ax.plot(train_steps, train_losses, color=COLORS['primary'], linewidth=1.5, label='Train')
    if eval_losses:
        ax.scatter(eval_steps, eval_losses, color=COLORS['secondary'], s=60, marker='*', label='Val')
    if max(train_losses) / (min(train_losses) + 1e-10) > 10:
        ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # (0,1) Loss vs Epoch
    ax = axes[0, 1]
    if epochs:
        ax.plot(epochs, train_losses[:len(epochs)], color=COLORS['primary'], linewidth=1.5)
        if max(train_losses) / (min(train_losses) + 1e-10) > 10:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        
        # Add vertical lines for epoch boundaries
        for ep in range(1, int(max(epochs)) + 1):
            ax.axvline(x=ep, color=COLORS['light'], linestyle=':', alpha=0.5)
    
    # (1,0) Learning Rate
    ax = axes[1, 0]
    ax.plot(lr_steps, lrs, color=COLORS['accent'], linewidth=2)
    ax.fill_between(lr_steps, lrs, alpha=0.2, color=COLORS['accent'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    # (1,1) Gradient Norm
    ax = axes[1, 1]
    ax.plot(grad_steps, grad_norms, color=COLORS['success'], linewidth=1, alpha=0.6)
    if len(grad_norms) > 10:
        window = min(20, len(grad_norms) // 5)
        smoothed = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
        ax.plot(grad_steps[window-1:], smoothed, color=COLORS['success'], linewidth=2, label='Smoothed')
        ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient L2 Norm')
    if grad_norms and max(grad_norms) / (min(grad_norms) + 1e-10) > 20:
        ax.set_yscale('log')
    
    fig.suptitle('SOKRATES SFT Training Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_oak_iterations(metrics: list[dict], output_path: Optional[str] = None, show: bool = True):
    """
    Plot OaK-DPO metrics across iterations.
    
    Args:
        metrics: List of metric dicts from OaK loop iterations
        output_path: Path to save figure
        show: Whether to display
    """
    if not metrics:
        print("No OaK metrics to plot")
        return None
    
    iterations = []
    accuracies = []
    step_validities = []
    trace_validities = []
    
    for m in metrics:
        if 'iteration' in m:
            iterations.append(m['iteration'])
            accuracies.append(m.get('accuracy', m.get('val_accuracy', 0)))
            step_validities.append(m.get('step_validity', m.get('overall_step_validity', 0)))
            trace_validities.append(m.get('trace_validity', 0))
    
    if not iterations:
        print("No iteration data found in metrics")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    x = np.arange(len(iterations))
    
    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, accuracies, color=COLORS['primary'], alpha=0.8, edgecolor=COLORS['neutral'])
    ax.set_xlabel('OaK Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Answer Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Iter {i}' for i in iterations])
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Step Validity
    ax = axes[1]
    bars = ax.bar(x, step_validities, color=COLORS['valid'], alpha=0.8, edgecolor=COLORS['neutral'])
    ax.set_xlabel('OaK Iteration')
    ax.set_ylabel('Step Validity Rate')
    ax.set_title('Step-Level Validity')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Iter {i}' for i in iterations])
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, step_validities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Trace Validity
    ax = axes[2]
    bars = ax.bar(x, trace_validities, color=COLORS['secondary'], alpha=0.8, edgecolor=COLORS['neutral'])
    ax.set_xlabel('OaK Iteration')
    ax.set_ylabel('Trace Validity Rate')
    ax.set_title('Full Trace Validity')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Iter {i}' for i in iterations])
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, trace_validities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('SOKRATES OaK-DPO Training Progress', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_reliability_diagram(
    predictions: list[float],
    labels: list[int],
    n_bins: int = 10,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = "Calibration (Reliability Diagram)"
):
    """
    Plot reliability diagram for calibration analysis.
    
    Args:
        predictions: Predicted probabilities [0, 1]
        labels: Binary outcomes (0 or 1)
        n_bins: Number of bins
        output_path: Path to save figure
        show: Whether to display
        title: Plot title
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        else:
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        
        if mask.sum() > 0:
            bin_accuracies.append(labels[mask].mean())
            bin_confidences.append(predictions[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            bin_counts.append(0)
    
    # Compute ECE
    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if acc is not None and conf is not None:
            ece += (count / len(predictions)) * abs(acc - conf)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    
    # Bar chart of calibration
    valid_centers = [c for c, a in zip(bin_centers, bin_accuracies) if a is not None]
    valid_accs = [a for a in bin_accuracies if a is not None]
    valid_confs = [c for c in bin_confidences if c is not None]
    
    bar_width = 0.8 / n_bins
    bars = ax.bar(valid_centers, valid_accs, width=bar_width, 
                  color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['neutral'],
                  label='Observed Accuracy')
    
    # Gap visualization
    for center, acc, conf in zip(valid_centers, valid_accs, valid_confs):
        if acc is not None and conf is not None:
            gap_color = COLORS['invalid'] if acc < conf else COLORS['valid']
            ax.plot([conf, conf], [acc, conf], color=gap_color, linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Accuracy)')
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    # Add count annotations
    for center, count in zip(bin_centers, bin_counts):
        if count > 0:
            ax.text(center, -0.05, f'n={count}', ha='center', fontsize=8, color=COLORS['neutral'])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_model_comparison(
    sft_metrics: dict,
    dpo_metrics: dict,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison between SFT baseline and final DPO model.
    
    Args:
        sft_metrics: Metrics dict from SFT evaluation
        dpo_metrics: Metrics dict from DPO evaluation
        output_path: Path to save figure
        show: Whether to display
    """
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Step Validity', 'step_validity'),
        ('Trace Validity', 'trace_validity'),
    ]
    
    # Extract values
    labels = []
    sft_values = []
    dpo_values = []
    
    for label, key in metrics_to_compare:
        labels.append(label)
        
        if key == 'step_validity':
            sft_val = sft_metrics.get(key, {}).get('overall', 0) if isinstance(sft_metrics.get(key), dict) else sft_metrics.get(key, 0)
            dpo_val = dpo_metrics.get(key, {}).get('overall', 0) if isinstance(dpo_metrics.get(key), dict) else dpo_metrics.get(key, 0)
        else:
            sft_val = sft_metrics.get(key, 0)
            dpo_val = dpo_metrics.get(key, 0)
        
        sft_values.append(sft_val)
        dpo_values.append(dpo_val)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars_sft = ax.bar(x - width/2, sft_values, width, 
                       label='SFT Baseline', color=COLORS['primary'], alpha=0.8)
    bars_dpo = ax.bar(x + width/2, dpo_values, width,
                       label='OaK-DPO Final', color=COLORS['secondary'], alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('SOKRATES: SFT Baseline vs OaK-DPO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars_sft, bars_dpo]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # Add improvement arrows
    for i, (sft_v, dpo_v) in enumerate(zip(sft_values, dpo_values)):
        if dpo_v > sft_v:
            improvement = (dpo_v - sft_v) / (sft_v + 1e-10) * 100
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(x[i], max(sft_v, dpo_v) + 0.08),
                       ha='center', fontsize=9, color=COLORS['valid'], fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_per_option_validity(
    per_option_rates: dict[str, float],
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot validity rates per option type.
    
    Args:
        per_option_rates: Dict mapping option type name to validity rate
        output_path: Path to save figure
        show: Whether to display
    """
    if not per_option_rates:
        print("No per-option data to plot")
        return None
    
    # Sort by validity rate
    sorted_items = sorted(per_option_rates.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_items]
    rates = [rate for _, rate in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
    
    y = np.arange(len(names))
    
    # Color based on validity
    colors = [COLORS['valid'] if r >= 0.7 else 
              COLORS['accent'] if r >= 0.4 else 
              COLORS['invalid'] for r in rates]
    
    bars = ax.barh(y, rates, color=colors, alpha=0.8, edgecolor=COLORS['neutral'])
    
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Solver Validity Rate')
    ax.set_title('SOKRATES: Step Validity by Option Type')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.02, bar.get_y() + bar.get_height()/2,
                f'{rate:.1%}', va='center', fontsize=9)
    
    # Add threshold lines
    ax.axvline(x=0.7, color=COLORS['valid'], linestyle='--', alpha=0.3, label='Good (70%)')
    ax.axvline(x=0.4, color=COLORS['accent'], linestyle='--', alpha=0.3, label='Moderate (40%)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot SOKRATES training results")
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
        default="outputs/figures",
        help="Directory to save figures",
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
    
    # Plot SFT results
    if args.sft_dir:
        print(f"\n=== Plotting SFT results from {args.sft_dir} ===")
        try:
            history = load_sft_history(args.sft_dir)
            print(f"Loaded {len(history)} training entries")
            
            plot_sft_loss(history, output_dir / f"sft_loss.{fmt}", show=show)
            plot_sft_learning_rate(history, output_dir / f"sft_lr.{fmt}", show=show)
            plot_sft_gradient_norm(history, output_dir / f"sft_grad_norm.{fmt}", show=show)
            plot_sft_combined(history, output_dir / f"sft_combined.{fmt}", show=show)
            
        except Exception as e:
            print(f"Error plotting SFT results: {e}")
    
    # Plot OaK-DPO results
    if args.oak_dir:
        print(f"\n=== Plotting OaK-DPO results from {args.oak_dir} ===")
        try:
            metrics = load_oak_metrics(args.oak_dir)
            print(f"Loaded {len(metrics)} metric entries")
            
            if metrics:
                plot_oak_iterations(metrics, output_dir / f"oak_iterations.{fmt}", show=show)
            else:
                print("No OaK metrics found")
                
        except Exception as e:
            print(f"Error plotting OaK results: {e}")
    
    print(f"\n=== Figures saved to {output_dir} ===")


if __name__ == "__main__":
    main()

