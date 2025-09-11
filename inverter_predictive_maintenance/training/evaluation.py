"""
Evaluation utilities for predictive maintenance models.

This module provides functions for evaluating model performance,
including metrics computation and report generation.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
from typing import Tuple, Optional


def get_logits_and_labels(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract logits and labels from a model and dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        
    Returns:
        Tuple of (logits, labels) as numpy arrays
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)  # shape: (batch, 1)
            all_logits.append(logits.cpu().numpy().ravel())
            all_labels.append(y.cpu().numpy().ravel())
    
    return np.concatenate(all_logits), np.concatenate(all_labels)


def generate_report(
    trues: np.ndarray, 
    predictions: np.ndarray, 
    outputs: np.ndarray,
    target_names: Optional[list] = None,
    show_plot: bool = True
) -> dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        trues: True binary labels
        predictions: Binary predictions
        outputs: Continuous prediction scores/probabilities
        target_names: Names for target classes
        show_plot: Whether to display ROC curve plot
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if target_names is None:
        target_names = ['Normal', 'Failure']
    
    # Classification report
    print("Classification Report:")
    print(classification_report(trues, predictions, target_names=target_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(trues, predictions)
    print(cm)
    
    # ROC AUC
    roc_auc = roc_auc_score(trues, outputs)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # ROC curve plot
    if show_plot:
        fpr, tpr, _ = roc_curve(trues, outputs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Return metrics dictionary
    metrics = {
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(trues, predictions, target_names=target_names, output_dict=True)
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    device: str = 'cuda',
    criterion: Optional[torch.nn.Module] = None,
    target_names: Optional[list] = None,
    show_plot: bool = True
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        threshold: Classification threshold for binary predictions
        device: Device to run inference on
        criterion: Loss function for computing test loss
        target_names: Names for target classes
        show_plot: Whether to display plots
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Get predictions and true labels
    trues, scores, avg_loss = _get_predictions_and_loss(
        model, test_loader, device, criterion
    )
    
    # Convert scores to binary predictions
    predictions = (scores > threshold).astype(int)
    
    # Generate comprehensive report
    metrics = generate_report(trues, predictions, scores, target_names, show_plot)
    
    # Add additional metrics
    if avg_loss is not None:
        metrics['test_loss'] = avg_loss
    
    metrics['threshold'] = threshold
    metrics['positive_rate'] = float(predictions.mean())
    metrics['true_positive_rate'] = float(trues.mean())
    
    return metrics


def _get_predictions_and_loss(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    criterion: Optional[torch.nn.Module]
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Get predictions and compute loss."""
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    trues, scores = [], []

    with torch.inference_mode():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device).float()
            logits = model(X_test).squeeze()
            
            if criterion is not None:
                total_loss += criterion(logits, y_test).item()
            
            trues.append(y_test.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())

    avg_loss = total_loss / len(test_loader) if criterion is not None else None
    return np.concatenate(trues), np.concatenate(scores), avg_loss
