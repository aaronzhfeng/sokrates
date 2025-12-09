"""
Evaluation metrics for SOKRATES.

Includes metrics for:
- Task-level accuracy
- Step and trace validity
- Calibration of q̂_φ predictions
"""

import numpy as np
from typing import Optional
from collections import defaultdict


def compute_accuracy(
    predictions: list[str],
    labels: list[str],
) -> float:
    """
    Compute final answer accuracy.
    
    Args:
        predictions: List of predicted labels ("TRUE", "FALSE", "UNKNOWN")
        labels: List of ground truth labels
        
    Returns:
        Accuracy as a fraction
    """
    if not predictions:
        return 0.0
    
    correct = sum(
        1 for pred, label in zip(predictions, labels)
        if pred.upper().strip() == label.upper().strip()
    )
    return correct / len(predictions)


def compute_step_validity(traces: list) -> dict[str, float]:
    """
    Compute step validity metrics across traces.
    
    Args:
        traces: List of OptionizedTrace objects with solver_valid labels
        
    Returns:
        Dict with:
        - "overall": fraction of all steps that are valid
        - "per_trace_mean": mean validity rate per trace
        - "per_option_type": validity rate per option type
    """
    total_steps = 0
    valid_steps = 0
    per_trace_rates = []
    per_option_counts = defaultdict(lambda: {"valid": 0, "total": 0})
    
    for trace in traces:
        trace_valid = 0
        trace_total = 0
        
        for step in trace.steps:
            if step.solver_valid is not None:
                total_steps += 1
                trace_total += 1
                
                option_name = step.option_type.name
                per_option_counts[option_name]["total"] += 1
                
                if step.solver_valid:
                    valid_steps += 1
                    trace_valid += 1
                    per_option_counts[option_name]["valid"] += 1
        
        if trace_total > 0:
            per_trace_rates.append(trace_valid / trace_total)
    
    # Compute per-option rates
    per_option_rates = {}
    for opt_name, counts in per_option_counts.items():
        if counts["total"] > 0:
            per_option_rates[opt_name] = counts["valid"] / counts["total"]
    
    return {
        "overall": valid_steps / total_steps if total_steps > 0 else 0.0,
        "per_trace_mean": np.mean(per_trace_rates) if per_trace_rates else 0.0,
        "per_option_type": per_option_rates,
        "total_steps": total_steps,
        "valid_steps": valid_steps,
    }


def compute_trace_validity(
    traces: list,
    labels: Optional[list[str]] = None,
) -> float:
    """
    Compute full-trace validity rate.
    
    A trace is valid if:
    1. All steps are solver-valid
    2. The final answer matches ground truth (if labels provided)
    
    Args:
        traces: List of OptionizedTrace objects
        labels: Optional list of ground truth labels
        
    Returns:
        Fraction of traces that are fully valid
    """
    if not traces:
        return 0.0
    
    valid_traces = 0
    
    for i, trace in enumerate(traces):
        # Check all steps are valid
        all_steps_valid = all(
            step.solver_valid for step in trace.steps
            if step.solver_valid is not None
        )
        
        # Check final answer
        answer_correct = True
        if labels and i < len(labels):
            answer_correct = trace.final_answer.upper() == labels[i].upper()
        
        if all_steps_valid and answer_correct:
            valid_traces += 1
    
    return valid_traces / len(traces)


def compute_brier_score(
    predictions: list[float],
    labels: list[int],
) -> float:
    """
    Compute Brier score for calibration.
    
    Brier = (1/n) * Σ(predicted_prob - actual_outcome)²
    
    Lower is better. Range: [0, 1]
    
    Args:
        predictions: List of predicted probabilities [0, 1]
        labels: List of binary outcomes (0 or 1)
        
    Returns:
        Brier score
    """
    if not predictions:
        return 1.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    return np.mean((predictions - labels) ** 2)


def compute_ece(
    predictions: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Bins predictions by confidence and measures the gap between
    predicted probability and actual accuracy in each bin.
    
    Lower is better. Range: [0, 1]
    
    Args:
        predictions: List of predicted probabilities [0, 1]
        labels: List of binary outcomes (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    if not predictions:
        return 1.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find predictions in this bin
        bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        
        if i == n_bins - 1:  # Last bin includes upper boundary
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        
        if bin_mask.sum() == 0:
            continue
        
        # Average confidence in bin
        avg_confidence = predictions[bin_mask].mean()
        
        # Actual accuracy in bin
        avg_accuracy = labels[bin_mask].mean()
        
        # Weight by bin size
        bin_weight = bin_mask.sum() / len(predictions)
        
        ece += bin_weight * abs(avg_confidence - avg_accuracy)
    
    return ece


def compute_calibration_curve(
    predictions: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data for plotting.
    
    Args:
        predictions: List of predicted probabilities
        labels: List of binary outcomes
        n_bins: Number of bins
        
    Returns:
        Dict with "mean_predicted", "fraction_positive", "bin_counts"
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        
        if i == n_bins - 1:
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        
        if bin_mask.sum() == 0:
            mean_predicted.append(None)
            fraction_positive.append(None)
            bin_counts.append(0)
        else:
            mean_predicted.append(predictions[bin_mask].mean())
            fraction_positive.append(labels[bin_mask].mean())
            bin_counts.append(bin_mask.sum())
    
    return {
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def compute_all_metrics(
    traces: list,
    labels: list[str],
    q_phi_predictions: Optional[list[tuple[float, int]]] = None,
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        traces: List of OptionizedTrace objects
        labels: Ground truth labels
        q_phi_predictions: Optional list of (prediction, label) tuples for q̂_φ
        
    Returns:
        Dict with all metrics
    """
    # Extract final answers
    predictions = [trace.final_answer for trace in traces]
    
    results = {
        "accuracy": compute_accuracy(predictions, labels),
        "step_validity": compute_step_validity(traces),
        "trace_validity": compute_trace_validity(traces, labels),
        "num_traces": len(traces),
    }
    
    # Add calibration metrics if q̂_φ predictions provided
    if q_phi_predictions:
        preds, labs = zip(*q_phi_predictions)
        results["q_phi_brier"] = compute_brier_score(list(preds), list(labs))
        results["q_phi_ece"] = compute_ece(list(preds), list(labs))
        results["q_phi_calibration_curve"] = compute_calibration_curve(
            list(preds), list(labs)
        )
    
    return results


def format_metrics_report(metrics: dict) -> str:
    """
    Format metrics dict as a human-readable report.
    
    Args:
        metrics: Dict from compute_all_metrics
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "SOKRATES Evaluation Report",
        "=" * 50,
        "",
        "Task-Level Metrics:",
        f"  Accuracy: {metrics['accuracy']:.2%}",
        f"  Number of traces: {metrics['num_traces']}",
        "",
        "Step-Level Metrics:",
        f"  Overall step validity: {metrics['step_validity']['overall']:.2%}",
        f"  Per-trace mean validity: {metrics['step_validity']['per_trace_mean']:.2%}",
        f"  Total steps: {metrics['step_validity']['total_steps']}",
        f"  Valid steps: {metrics['step_validity']['valid_steps']}",
        "",
        "Trace-Level Metrics:",
        f"  Full trace validity: {metrics['trace_validity']:.2%}",
    ]
    
    # Per-option breakdown
    if metrics['step_validity'].get('per_option_type'):
        lines.append("")
        lines.append("Per-Option Validity:")
        for opt_name, rate in sorted(metrics['step_validity']['per_option_type'].items()):
            lines.append(f"  {opt_name}: {rate:.2%}")
    
    # Calibration metrics
    if 'q_phi_brier' in metrics:
        lines.extend([
            "",
            "Calibration Metrics (q̂_φ):",
            f"  Brier Score: {metrics['q_phi_brier']:.4f}",
            f"  ECE: {metrics['q_phi_ece']:.4f}",
        ])
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)

