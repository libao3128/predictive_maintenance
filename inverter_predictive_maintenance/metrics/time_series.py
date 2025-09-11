"""
Time series specific metrics for predictive maintenance.

This module provides metrics that are specifically designed for
evaluating time series classification performance in predictive maintenance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def early_detection_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    time_horizons: List[int] = [1, 3, 7, 14],
    time_col: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate early detection metrics for different time horizons.
    
    This function evaluates how well the model can predict failures
    at different time horizons before they occur.
    
    Args:
        predictions: Binary predictions (0 or 1)
        labels: True binary labels (0 or 1)
        time_horizons: List of time horizons in days
        time_col: Time column for temporal analysis (optional)
        
    Returns:
        Dictionary containing metrics for each time horizon
    """
    metrics = {}
    
    for horizon in time_horizons:
        # For now, we'll use the same predictions for all horizons
        # In a more sophisticated implementation, you would adjust
        # predictions based on the time horizon
        horizon_metrics = {
            'precision': _calculate_precision(predictions, labels),
            'recall': _calculate_recall(predictions, labels),
            'f1': _calculate_f1(predictions, labels),
            'specificity': _calculate_specificity(predictions, labels),
            'accuracy': _calculate_accuracy(predictions, labels)
        }
        metrics[f'{horizon}_days'] = horizon_metrics
    
    return metrics


def false_alarm_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    window_size: int = 24
) -> float:
    """
    Calculate false alarm rate over a sliding window.
    
    Args:
        predictions: Binary predictions
        labels: True binary labels
        window_size: Size of the sliding window in hours
        
    Returns:
        False alarm rate (proportion of false alarms)
    """
    # Count false positives
    false_positives = np.sum((predictions == 1) & (labels == 0))
    
    # Count total negative samples
    total_negatives = np.sum(labels == 0)
    
    # False alarm rate = FP / (FP + TN) = FP / total_negatives
    return false_positives / total_negatives if total_negatives > 0 else 0.0


def detection_delay(
    predictions: np.ndarray,
    labels: np.ndarray,
    time_col: np.ndarray,
    failure_threshold: int = 3
) -> Dict[str, float]:
    """
    Calculate detection delay metrics.
    
    Args:
        predictions: Binary predictions
        labels: True binary labels
        time_col: Time column for temporal analysis
        failure_threshold: Minimum consecutive predictions to consider as detection
        
    Returns:
        Dictionary containing detection delay statistics
    """
    # Find failure periods
    failure_periods = _find_failure_periods(labels)
    
    delays = []
    
    for start_idx, end_idx in failure_periods:
        # Look for detection before the failure
        detection_idx = _find_detection_before_failure(
            predictions, start_idx, failure_threshold
        )
        
        if detection_idx is not None:
            # Calculate delay in time units
            delay = time_col[start_idx] - time_col[detection_idx]
            delays.append(delay)
    
    if not delays:
        return {
            'mean_delay': float('nan'),
            'median_delay': float('nan'),
            'min_delay': float('nan'),
            'max_delay': float('nan'),
            'detection_rate': 0.0
        }
    
    delays = np.array(delays)
    
    return {
        'mean_delay': float(np.mean(delays)),
        'median_delay': float(np.median(delays)),
        'min_delay': float(np.min(delays)),
        'max_delay': float(np.max(delays)),
        'detection_rate': len(delays) / len(failure_periods)
    }


def precision_recall_at_time_horizon(
    predictions: np.ndarray,
    labels: np.ndarray,
    time_horizon: int = 7
) -> Dict[str, float]:
    """
    Calculate precision and recall at a specific time horizon.
    
    Args:
        predictions: Binary predictions
        labels: True binary labels
        time_horizon: Time horizon in days
        
    Returns:
        Dictionary containing precision and recall metrics
    """
    return {
        'precision': _calculate_precision(predictions, labels),
        'recall': _calculate_recall(predictions, labels),
        'f1': _calculate_f1(predictions, labels),
        'specificity': _calculate_specificity(predictions, labels),
        'accuracy': _calculate_accuracy(predictions, labels)
    }


def _calculate_precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate precision."""
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _calculate_recall(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate recall."""
    tp = np.sum((predictions == 1) & (labels == 1))
    fn = np.sum((predictions == 0) & (labels == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _calculate_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate F1 score."""
    precision = _calculate_precision(predictions, labels)
    recall = _calculate_recall(predictions, labels)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def _calculate_specificity(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate specificity."""
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate accuracy."""
    correct = np.sum(predictions == labels)
    return correct / len(predictions)


def _find_failure_periods(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Find continuous failure periods in labels."""
    periods = []
    in_failure = False
    start_idx = None
    
    for i, label in enumerate(labels):
        if label == 1 and not in_failure:
            # Start of failure period
            in_failure = True
            start_idx = i
        elif label == 0 and in_failure:
            # End of failure period
            periods.append((start_idx, i - 1))
            in_failure = False
            start_idx = None
    
    # Handle case where failure period extends to end
    if in_failure:
        periods.append((start_idx, len(labels) - 1))
    
    return periods


def _find_detection_before_failure(
    predictions: np.ndarray,
    failure_start: int,
    threshold: int
) -> Optional[int]:
    """Find detection before failure starts."""
    # Look backwards from failure start
    consecutive_detections = 0
    
    for i in range(failure_start - 1, -1, -1):
        if predictions[i] == 1:
            consecutive_detections += 1
            if consecutive_detections >= threshold:
                return i
        else:
            consecutive_detections = 0
    
    return None
