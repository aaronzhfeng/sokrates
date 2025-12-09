"""
Calibration analysis utilities for the option success predictor q̂_φ.

Provides tools for analyzing and visualizing how well the model's
predicted probabilities match actual outcomes.
"""

import numpy as np
from typing import Optional
import json


class CalibrationAnalyzer:
    """
    Analyzes calibration of probability predictions.
    
    Tracks predictions over time and computes various calibration metrics.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize the calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins
        self.predictions = []
        self.labels = []
        self.metadata = []  # Optional metadata per prediction
    
    def add_prediction(
        self,
        prediction: float,
        label: int,
        metadata: Optional[dict] = None,
    ):
        """
        Add a single prediction-label pair.
        
        Args:
            prediction: Predicted probability [0, 1]
            label: Actual outcome (0 or 1)
            metadata: Optional dict with additional info (option type, etc.)
        """
        self.predictions.append(prediction)
        self.labels.append(label)
        self.metadata.append(metadata or {})
    
    def add_batch(
        self,
        predictions: list[float],
        labels: list[int],
        metadata: Optional[list[dict]] = None,
    ):
        """Add multiple predictions at once."""
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(predictions))
    
    def compute_metrics(self) -> dict:
        """
        Compute all calibration metrics.
        
        Returns:
            Dict with Brier score, ECE, and calibration curve data
        """
        if not self.predictions:
            return {
                "brier_score": None,
                "ece": None,
                "calibration_curve": None,
                "num_samples": 0,
            }
        
        preds = np.array(self.predictions)
        labs = np.array(self.labels)
        
        # Brier score
        brier = np.mean((preds - labs) ** 2)
        
        # ECE
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        curve_data = {
            "bin_centers": [],
            "mean_confidence": [],
            "actual_accuracy": [],
            "bin_counts": [],
        }
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            bin_center = (bin_lower + bin_upper) / 2
            
            if i == self.n_bins - 1:
                mask = (preds >= bin_lower) & (preds <= bin_upper)
            else:
                mask = (preds >= bin_lower) & (preds < bin_upper)
            
            bin_count = mask.sum()
            curve_data["bin_centers"].append(bin_center)
            curve_data["bin_counts"].append(int(bin_count))
            
            if bin_count > 0:
                bin_confidence = preds[mask].mean()
                bin_accuracy = labs[mask].mean()
                curve_data["mean_confidence"].append(float(bin_confidence))
                curve_data["actual_accuracy"].append(float(bin_accuracy))
                
                bin_weight = bin_count / len(preds)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
            else:
                curve_data["mean_confidence"].append(None)
                curve_data["actual_accuracy"].append(None)
        
        return {
            "brier_score": float(brier),
            "ece": float(ece),
            "calibration_curve": curve_data,
            "num_samples": len(self.predictions),
            "mean_prediction": float(preds.mean()),
            "mean_label": float(labs.mean()),
        }
    
    def compute_per_option_metrics(self) -> dict:
        """
        Compute calibration metrics broken down by option type.
        
        Requires metadata with "option_type" field.
        
        Returns:
            Dict mapping option type to calibration metrics
        """
        # Group by option type
        by_option = {}
        for pred, label, meta in zip(self.predictions, self.labels, self.metadata):
            opt_type = meta.get("option_type", "unknown")
            if opt_type not in by_option:
                by_option[opt_type] = {"predictions": [], "labels": []}
            by_option[opt_type]["predictions"].append(pred)
            by_option[opt_type]["labels"].append(label)
        
        # Compute metrics per option
        results = {}
        for opt_type, data in by_option.items():
            preds = np.array(data["predictions"])
            labs = np.array(data["labels"])
            
            results[opt_type] = {
                "brier_score": float(np.mean((preds - labs) ** 2)),
                "accuracy": float((preds.round() == labs).mean()),
                "num_samples": len(preds),
                "mean_prediction": float(preds.mean()),
                "actual_rate": float(labs.mean()),
            }
        
        return results
    
    def get_reliability_diagram_data(self) -> dict:
        """
        Get data formatted for plotting a reliability diagram.
        
        Returns:
            Dict with data ready for matplotlib/plotly
        """
        metrics = self.compute_metrics()
        curve = metrics["calibration_curve"]
        
        return {
            "x": [c for c, a in zip(curve["mean_confidence"], curve["actual_accuracy"]) if c is not None],
            "y": [a for c, a in zip(curve["mean_confidence"], curve["actual_accuracy"]) if a is not None],
            "perfect_line": [0, 1],
            "ece": metrics["ece"],
            "brier": metrics["brier_score"],
        }
    
    def reset(self):
        """Clear all stored predictions."""
        self.predictions = []
        self.labels = []
        self.metadata = []
    
    def save(self, path: str):
        """Save calibration data to JSON file."""
        data = {
            "predictions": self.predictions,
            "labels": self.labels,
            "metadata": self.metadata,
            "n_bins": self.n_bins,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "CalibrationAnalyzer":
        """Load calibration data from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        analyzer = cls(n_bins=data.get("n_bins", 10))
        analyzer.predictions = data["predictions"]
        analyzer.labels = data["labels"]
        analyzer.metadata = data.get("metadata", [{}] * len(data["predictions"]))
        
        return analyzer


def compare_calibration(
    before: CalibrationAnalyzer,
    after: CalibrationAnalyzer,
) -> dict:
    """
    Compare calibration between two analyzers (e.g., before/after training).
    
    Args:
        before: CalibrationAnalyzer with predictions before intervention
        after: CalibrationAnalyzer with predictions after intervention
        
    Returns:
        Dict with comparison metrics
    """
    before_metrics = before.compute_metrics()
    after_metrics = after.compute_metrics()
    
    return {
        "before": before_metrics,
        "after": after_metrics,
        "brier_improvement": (
            before_metrics["brier_score"] - after_metrics["brier_score"]
            if before_metrics["brier_score"] and after_metrics["brier_score"]
            else None
        ),
        "ece_improvement": (
            before_metrics["ece"] - after_metrics["ece"]
            if before_metrics["ece"] and after_metrics["ece"]
            else None
        ),
    }


def format_calibration_report(metrics: dict) -> str:
    """
    Format calibration metrics as a human-readable report.
    
    Args:
        metrics: Dict from CalibrationAnalyzer.compute_metrics()
        
    Returns:
        Formatted string
    """
    if metrics["brier_score"] is None:
        return "No calibration data available."
    
    lines = [
        "Calibration Report",
        "-" * 30,
        f"Samples: {metrics['num_samples']}",
        f"Brier Score: {metrics['brier_score']:.4f}",
        f"ECE: {metrics['ece']:.4f}",
        f"Mean Prediction: {metrics['mean_prediction']:.3f}",
        f"Actual Positive Rate: {metrics['mean_label']:.3f}",
        "",
        "Calibration Curve:",
    ]
    
    curve = metrics["calibration_curve"]
    for i, (center, conf, acc, count) in enumerate(zip(
        curve["bin_centers"],
        curve["mean_confidence"],
        curve["actual_accuracy"],
        curve["bin_counts"],
    )):
        if conf is not None and acc is not None:
            gap = abs(conf - acc)
            lines.append(
                f"  Bin {center:.1f}: conf={conf:.3f}, acc={acc:.3f}, "
                f"gap={gap:.3f}, n={count}"
            )
        else:
            lines.append(f"  Bin {center:.1f}: (empty)")
    
    return "\n".join(lines)

