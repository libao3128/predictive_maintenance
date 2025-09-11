"""
Neural network architectures for predictive maintenance.

This module contains the CNN-LSTM model architecture designed for
time series classification tasks in predictive maintenance.
"""

import torch
import torch.nn as nn
from typing import Optional


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for time series classification.
    
    This model combines convolutional neural networks (CNN) for extracting
    short-term patterns within time windows and LSTM for learning temporal
    dependencies across the sequence.
    
    Architecture:
        1. CNN module: Extracts local patterns within time windows
        2. LSTM module: Learns temporal dependencies
        3. Classifier: Maps LSTM output to binary classification
    
    Args:
        num_features (int): Number of input features per time step
        cnn_out_channels (int): Number of output channels from CNN layer
        lstm_hidden_size (int): Hidden size of LSTM layers
        lstm_layers (int): Number of LSTM layers
        dropout (float): Dropout rate for regularization
        
    Attributes:
        cnn (nn.Sequential): CNN module for feature extraction
        lstm (nn.LSTM): LSTM module for temporal modeling
        classifier (nn.Sequential): Classification head
    """
    
    def __init__(
        self,
        num_features: int,
        cnn_out_channels: int = 32,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3
    ) -> None:
        """
        Initialize CNN-LSTM model.
        
        Args:
            num_features: Number of input features per time step
            cnn_out_channels: Number of output channels from CNN layer
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(CNNLSTMModel, self).__init__()
        
        # Validate inputs
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if cnn_out_channels <= 0:
            raise ValueError("cnn_out_channels must be positive")
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        # CNN module: extract short-term patterns within time windows
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout)
        )

        # LSTM module: learn temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # Note: Sigmoid is not applied here to allow for logits output
            # Apply torch.sigmoid() during inference if probabilities are needed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Output tensor of shape (batch_size, 1) containing logits
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        batch_size, seq_len, num_features = x.shape
        
        # Reshape for CNN: (batch_size, num_features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN forward pass: (batch_size, cnn_out_channels, seq_len)
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch_size, seq_len, cnn_out_channels)
        x = x.permute(0, 2, 1)
        
        # LSTM forward pass: (batch_size, seq_len, lstm_hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take output from last time step: (batch_size, lstm_hidden_size)
        last_time_step = lstm_out[:, -1, :]
        
        # Classification: (batch_size, 1)
        out = self.classifier(last_time_step)
        
        return out

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate embeddings from the LSTM layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            LSTM embeddings of shape (batch_size, seq_len, lstm_hidden_size)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        # Reshape for CNN
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # Get LSTM output
        lstm_out, _ = self.lstm(x)
        
        return lstm_out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            threshold: Classification threshold
            
        Returns:
            Binary predictions tensor of shape (batch_size, 1)
        """
        probabilities = self.predict_proba(x)
        predictions = (probabilities > threshold).float()
        return predictions
