"""
Loss functions for predictive maintenance models.

This module contains specialized loss functions designed for
imbalanced classification tasks in predictive maintenance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Focal Loss is designed to down-weight easy examples and focus on hard examples,
    which is particularly useful for imbalanced datasets where the majority class
    dominates the loss function.
    
    The focal loss is defined as:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the predicted probability for the true class
    - α_t is the weighting factor for class t
    - γ is the focusing parameter
    
    Args:
        alpha (float): Weighting factor for positive class (default: 0.75)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('mean', 'sum', or 'none')
        
    References:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', or 'none')
            
        Raises:
            ValueError: If alpha is not between 0 and 1, or gamma is negative,
                       or reduction is not valid
        """
        super().__init__()
        
        # Validate inputs
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs of shape (N,) or (N, 1)
            targets: Binary targets of shape (N,) or (N, 1) with values in {0, 1}
            
        Returns:
            Computed focal loss
            
        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        # Ensure inputs are 1D
        if logits.dim() > 1:
            logits = logits.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Validate shapes
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {logits.shape} != targets shape {targets.shape}")
        
        # Validate target values
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("targets must contain only values 0 and 1")
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute probabilities with numerical stability
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        
        # Compute p_t (probability of true class)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Compute weighting factor
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def __repr__(self) -> str:
        """String representation of the loss function."""
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    
    This loss function applies different weights to positive and negative classes
    to address class imbalance in binary classification tasks.
    
    Args:
        pos_weight (float): Weight for positive class
        reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class
            reduction: Reduction method
            
        Raises:
            ValueError: If pos_weight is negative or reduction is invalid
        """
        super().__init__()
        
        if pos_weight < 0:
            raise ValueError("pos_weight must be non-negative")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Raw model outputs
            targets: Binary targets
            
        Returns:
            Computed weighted BCE loss
        """
        # Ensure inputs are 1D
        if logits.dim() > 1:
            logits = logits.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Compute BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction=self.reduction
        )
        
        return loss

    def __repr__(self) -> str:
        """String representation of the loss function."""
        return f"WeightedBCELoss(pos_weight={self.pos_weight}, reduction='{self.reduction}')"
