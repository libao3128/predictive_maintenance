"""
Training utilities for predictive maintenance models.

This module provides the core training and testing loops for neural network models.
"""

import torch
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import average_precision_score
from ..metrics.classification import cal_topK_metrics


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = 'cuda',
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[torch.nn.Module] = None,
    num_epochs: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 100,
    save_interval: int = 1,
    save_path: Optional[str] = None,
    resume: bool = True
) -> pd.DataFrame:
    """
    General PyTorch training loop for CNN+LSTM binary classification model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        device: Device to run training on ('cuda' or 'cpu')
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of training epochs
        scheduler: Learning rate scheduler (optional)
        log_interval: Interval for logging training progress
        save_interval: Interval for saving model checkpoints
        save_path: Path to save model and logs
        resume: Whether to resume training from existing checkpoint
        
    Returns:
        DataFrame containing training logs
        
    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If training fails
    """
    # Validate inputs
    if optimizer is None:
        raise ValueError("optimizer is required")
    if criterion is None:
        raise ValueError("criterion is required")
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Setup save path
    if save_path is None:
        save_path = time.strftime('model/local/%m%d_%H%M/', time.localtime())
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize or resume training log
    log_path = os.path.join(save_path, 'training_log.csv')
    if resume and os.path.exists(log_path):
        log = pd.read_csv(log_path)
        cur_epoch = int(log['epoch'].max() + 1)
        max_aucpr = log['aucpr'].max()
        print(f"Resuming training from epoch {cur_epoch}")
    else:
        log = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'aucpr', 'accuracy', 'time'])
        cur_epoch = 1
        max_aucpr = float('-inf')

    # Training loop
    for epoch in range(cur_epoch, cur_epoch + num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, log_interval, epoch
        )
        
        print(f"🔁 Epoch {epoch} finished. Avg Train Loss: {train_loss:.4f}")
        end_time = time.time()

        # Validation phase
        if val_loader is not None:
            val_metrics = _validate_epoch(model, val_loader, criterion, device, epoch)
            
            # Save best model based on AUCPR
            if val_metrics['aucpr'] > max_aucpr:
                max_aucpr = val_metrics['aucpr']
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f"Best model saved at epoch {epoch} with AUC-PR {val_metrics['aucpr']:.4f}")
            
            # Update log
            log = _update_training_log(log, epoch, train_loss, val_metrics, end_time - start_time)
        else:
            # No validation, just log training loss
            log = _update_training_log(log, epoch, train_loss, None, end_time - start_time)

        # Save checkpoint
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
            log.to_csv(log_path, index=False)
            
    # Final save
    log.to_csv(log_path, index=False)
    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
    print("🏁 Training completed.")
    
    return log


def _train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    log_interval: int,
    epoch: int
) -> float:
    """Train for one epoch."""
    total_loss = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(X)
        output = output.squeeze(-1)
        
        loss = criterion(output, y)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        if batch_idx % log_interval == 0:
            print(f"[Epoch {epoch}] Step {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def _validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch and compute metrics."""
    y_true_val, y_score_val, avg_val_loss = test_loop(model, val_loader, device, criterion)
    
    # Compute metrics
    pos_rate = float((y_true_val == 1).mean())
    aucpr = average_precision_score(y_true_val, y_score_val) if pos_rate > 0 else float('nan')
    ap_uplift = aucpr / pos_rate if pos_rate > 0 else float('nan')
    
    # Top-K metrics
    topk_list = [50, 100, 200]
    top_k_metrics = cal_topK_metrics(y_score_val, y_true_val, top_k=topk_list)
    
    # Print summary
    k_str = " | ".join([f"P@{k}:{top_k_metrics[f'prec@{k}']:.3f} R@{k}:{top_k_metrics[f'rec@{k}']:.3f}" for k in topk_list])
    print(f"✅ avg_loss: {avg_val_loss:.4f} | AUC-PR: {aucpr:.4f} | baseline: {pos_rate:.4f} | uplift: {ap_uplift:.2f}x | {k_str}")
    
    # Prepare metrics dictionary
    metrics = {
        'val_loss': avg_val_loss,
        'aucpr': aucpr,
        'baseline_pos_rate': pos_rate,
        'ap_uplift': ap_uplift
    }
    metrics.update(top_k_metrics)
    
    return metrics


def _update_training_log(
    log: pd.DataFrame,
    epoch: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]],
    epoch_time: float
) -> pd.DataFrame:
    """Update training log with new epoch data."""
    # Define required columns
    required_cols = ['epoch', 'train_loss', 'val_loss', 'aucpr', 'accuracy', 'baseline_pos_rate', 'ap_uplift', 'time']
    topk_list = [50, 100, 200]
    for k in topk_list:
        required_cols += [f'prec@{k}', f'rec@{k}']
    
    # Add missing columns
    for col in required_cols:
        if col not in log.columns:
            log[col] = np.nan
    
    # Create new row
    row = {
        'epoch': epoch,
        'train_loss': train_loss,
        'time': epoch_time
    }
    
    if val_metrics is not None:
        row.update(val_metrics)
    else:
        # Fill with NaN for validation metrics
        for col in ['val_loss', 'aucpr', 'baseline_pos_rate', 'ap_uplift']:
            row[col] = np.nan
        for k in topk_list:
            row[f'prec@{k}'] = np.nan
            row[f'rec@{k}'] = np.nan
    
    # Add row to log
    log.loc[len(log)] = row
    return log


def test_loop(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    criterion: Optional[torch.nn.Module] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Run inference on test data and compute metrics.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        device: Device to run inference on
        criterion: Loss function for computing test loss
        
    Returns:
        Tuple of (true_labels, predicted_scores, average_loss)
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    trues, scores = [], []

    with torch.inference_mode():
        for X_test, y_test in tqdm(test_loader, desc="Testing"):
            X_test, y_test = X_test.to(device), y_test.to(device).float()
            logits = model(X_test).squeeze()
            
            if criterion is not None:
                total_loss += criterion(logits, y_test).item()
            
            trues.append(y_test.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())

    avg_test_loss = total_loss / len(test_loader) if criterion is not None else None
    return np.concatenate(trues), np.concatenate(scores), avg_test_loss
