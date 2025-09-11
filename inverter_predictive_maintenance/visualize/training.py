"""
Training visualization utilities for predictive maintenance.

This module provides functions for visualizing training progress,
model outputs, and evaluation metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import precision_recall_curve
from typing import Union, List, Optional


def visualize_log(log: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Visualize training log with loss and metrics curves.
    
    Args:
        log: DataFrame containing training logs
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training and validation loss
    axes[0, 0].plot(log['epoch'], log['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in log.columns:
        axes[0, 0].plot(log['epoch'], log['val_loss'], label='Validation Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUCPR
    if 'aucpr' in log.columns:
        axes[0, 1].plot(log['epoch'], log['aucpr'], label='AUCPR', marker='o', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUCPR')
        axes[0, 1].set_title('Area Under Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Precision@K metrics
    if 'prec@50' in log.columns:
        axes[1, 0].plot(log['epoch'], log['prec@50'], label='P@50', marker='o')
        axes[1, 0].plot(log['epoch'], log['prec@100'], label='P@100', marker='s')
        axes[1, 0].plot(log['epoch'], log['prec@200'], label='P@200', marker='^')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision@K')
        axes[1, 0].set_title('Precision at Top K')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall@K metrics
    if 'rec@50' in log.columns:
        axes[1, 1].plot(log['epoch'], log['rec@50'], label='R@50', marker='o')
        axes[1, 1].plot(log['epoch'], log['rec@100'], label='R@100', marker='s')
        axes[1, 1].plot(log['epoch'], log['rec@200'], label='R@200', marker='^')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall@K')
        axes[1, 1].set_title('Recall at Top K')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_outputs_distribution(
    probs_calibrated: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    title: str = "Validation Outputs Distribution"
) -> None:
    """
    Plot distribution of model outputs by class.
    
    Args:
        probs_calibrated: Calibrated probability outputs
        labels: True binary labels
        title: Title for the plot
    """
    df = pd.DataFrame({
        "scores": probs_calibrated,
        "labels": labels
    })
    
    fig = px.histogram(
        df, 
        x="scores", 
        color="labels", 
        barmode="stack", 
        histnorm="probability", 
        title=title
    )
    fig.update_xaxes(range=[0, 1])
    fig.show()


def plot_precision_recall(
    trues: Union[np.ndarray, List[int]],
    prob: Union[np.ndarray, List[float]],
    title: str = "Precision-Recall Curve"
) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        trues: True binary labels
        prob: Predicted probabilities
        title: Title for the plot
    """
    precision, recall, thresholds = precision_recall_curve(trues, prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add baseline (random classifier)
    baseline = np.mean(trues)
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.7, label=f'Random (P={baseline:.3f})')
    plt.legend()
    
    plt.show()


def plot_roc_curve(
    trues: Union[np.ndarray, List[int]],
    prob: Union[np.ndarray, List[float]],
    title: str = "ROC Curve"
) -> None:
    """
    Plot ROC curve.
    
    Args:
        trues: True binary labels
        prob: Predicted probabilities
        title: Title for the plot
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(trues, prob)
    auc = roc_auc_score(trues, prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(
    trues: Union[np.ndarray, List[int]],
    predictions: Union[np.ndarray, List[int]],
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix with annotations.
    
    Args:
        trues: True binary labels
        predictions: Predicted binary labels
        title: Title for the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(trues, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: List[float],
    title: str = "Feature Importance",
    top_k: int = 20
) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        title: Title for the plot
        top_k: Number of top features to show
    """
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    top_features = [feature_names[i] for i in sorted_indices[:top_k]]
    top_scores = [importance_scores[i] for i in sorted_indices[:top_k]]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_scores)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
