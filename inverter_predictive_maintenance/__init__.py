"""
Inverter Predictive Maintenance Package

A comprehensive Python package for predictive maintenance of solar plant inverters,
providing tools for data preprocessing, model training, evaluation, and visualization.

This package is developed as part of the UCLA MEng Capstone Program in collaboration
with MN8 Energy (a Goldman Sachs-backed renewable energy company).

Main modules:
- dataset: Dataset classes for time series data
- model: Neural network models and loss functions  
- preprocess: Data preprocessing utilities
- training: Training utilities and loops
- metrics: Evaluation metrics
- visualize: Visualization tools
- postprocess: Post-processing utilities
"""

__version__ = "1.0.0"
__author__ = "UCLA MEng Capstone Team"
__email__ = "leo900527@gmail.com"
__description__ = "Predictive maintenance system for solar plant inverters using deep learning"

# Import main classes for easy access
from .dataset import (
    InverterTimeSeriesDataset,
    InverterTimeSeriesDataset_metadata,
    PositiveInverterTimeSeriesDataset,
    NegativeInverterTimeSeriesDataset,
    PositiveInverterTimeSeriesDataset_metadata,
    NegativeInverterTimeSeriesDataset_metadata,
    combine_dataset,
    combine_dataset_metadata
)

from .model import CNNLSTMModel, FocalLoss

from .metrics import cal_topK_metrics

from .training import train_loop, test_loop, get_logits_and_labels, generate_report, evaluate_model

__all__ = [
    # Dataset classes
    "InverterTimeSeriesDataset",
    "InverterTimeSeriesDataset_metadata", 
    "PositiveInverterTimeSeriesDataset",
    "NegativeInverterTimeSeriesDataset",
    "PositiveInverterTimeSeriesDataset_metadata",
    "NegativeInverterTimeSeriesDataset_metadata",
    "combine_dataset",
    "combine_dataset_metadata",
    
    # Model classes
    "CNNLSTMModel",
    "FocalLoss",
    
    # Metrics
    "cal_topK_metrics",
    
    # Training utilities
    "train_loop",
    "test_loop", 
    "get_logits_and_labels",
    "generate_report",
    "evaluate_model"
]
