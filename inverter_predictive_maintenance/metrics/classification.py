"""
Classification metrics for predictive maintenance.

This module provides metrics specifically designed for evaluating
classification performance in predictive maintenance tasks.
"""

import numpy as np
from typing import Tuple, Dict, Union, List


def cal_topK_metrics(
    logits: np.ndarray, 
    labels: np.ndarray, 
    top_k: Union[int, Tuple[int, ...]] = (1, 5)
) -> Dict[str, float]:
    """
    Calculate Top-K accuracy metrics for ranking-based evaluation.
    
    This function is particularly useful for predictive maintenance where
    we want to identify the top K most likely failure cases.
    
    Args:
        logits: Model output scores/logits
        labels: True binary labels (0 or 1)
        top_k: K values to evaluate (single int or tuple of ints)
        
    Returns:
        Dictionary containing precision@k and recall@k for each k
        
    Raises:
        ValueError: If inputs have incompatible shapes or invalid values
    """
    # Ensure inputs are numpy arrays
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    
    # Validate inputs
    if logits.shape != labels.shape:
        raise ValueError(f"logits shape {logits.shape} != labels shape {labels.shape}")
    
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only values 0 and 1")
    
    # Convert single k to tuple
    if isinstance(top_k, int):
        top_k = (top_k,)
    
    # Sort by logits (descending order)
    order = np.argsort(-logits)
    y_true_sorted = labels[order]
    
    # Count total positive samples
    total_pos = int((labels == 1).sum())
    
    # Calculate metrics for each k
    topk_metrics = {}
    for k in top_k:
        k_eff = min(k, len(y_true_sorted))
        
        if k_eff == 0:
            p_at_k = float('nan')
            r_at_k = float('nan')
        else:
            # Count true positives in top k
            tp_at_k = int(y_true_sorted[:k_eff].sum())
            
            # Precision@k = TP@k / k
            p_at_k = tp_at_k / k_eff
            
            # Recall@k = TP@k / total_positives
            r_at_k = (tp_at_k / total_pos) if total_pos > 0 else float('nan')
        
        topk_metrics[f'prec@{k}'] = p_at_k
        topk_metrics[f'rec@{k}'] = r_at_k
    
    return topk_metrics


def precision_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calculate precision at top K.
    
    Args:
        logits: Model output scores
        labels: True binary labels
        k: Number of top predictions to consider
        
    Returns:
        Precision at k
    """
    metrics = cal_topK_metrics(logits, labels, top_k=(k,))
    return metrics[f'prec@{k}']


def recall_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calculate recall at top K.
    
    Args:
        logits: Model output scores
        labels: True binary labels
        k: Number of top predictions to consider
        
    Returns:
        Recall at k
    """
    metrics = cal_topK_metrics(logits, labels, top_k=(k,))
    return metrics[f'rec@{k}']


def f1_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calculate F1 score at top K.
    
    Args:
        logits: Model output scores
        labels: True binary labels
        k: Number of top predictions to consider
        
    Returns:
        F1 score at k
    """
    p_k = precision_at_k(logits, labels, k)
    r_k = recall_at_k(logits, labels, k)
    
    if p_k == 0 and r_k == 0:
        return 0.0
    
    return 2 * (p_k * r_k) / (p_k + r_k)


def average_precision_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calculate average precision at top K.
    
    This is equivalent to the area under the precision-recall curve
    when considering only the top K predictions.
    
    Args:
        logits: Model output scores
        labels: True binary labels
        k: Number of top predictions to consider
        
    Returns:
        Average precision at k
    """
    # Sort by logits
    order = np.argsort(-logits)
    y_true_sorted = labels[order]
    
    # Take top k
    k_eff = min(k, len(y_true_sorted))
    if k_eff == 0:
        return float('nan')
    
    y_topk = y_true_sorted[:k_eff]
    
    # Calculate precision at each position
    precisions = []
    for i in range(1, k_eff + 1):
        if y_topk[i-1] == 1:  # If current position is positive
            precision = y_topk[:i].sum() / i
            precisions.append(precision)
    
    # Average precision is the mean of precisions at positive positions
    return np.mean(precisions) if precisions else 0.0


def ndcg_at_k(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at top K.
    
    Args:
        logits: Model output scores
        labels: True binary labels
        k: Number of top predictions to consider
        
    Returns:
        NDCG at k
    """
    # Sort by logits
    order = np.argsort(-logits)
    y_true_sorted = labels[order]
    
    # Take top k
    k_eff = min(k, len(y_true_sorted))
    if k_eff == 0:
        return float('nan')
    
    y_topk = y_true_sorted[:k_eff]
    
    # Calculate DCG
    dcg = 0.0
    for i in range(k_eff):
        dcg += y_topk[i] / np.log2(i + 2)  # log2(i+2) because i starts from 0
    
    # Calculate IDCG (ideal DCG)
    # Sort labels in descending order to get ideal ranking
    ideal_labels = np.sort(labels)[::-1]
    idcg = 0.0
    for i in range(min(k_eff, len(ideal_labels))):
        idcg += ideal_labels[i] / np.log2(i + 2)
    
    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0
