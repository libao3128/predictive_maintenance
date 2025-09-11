"""
Utility functions for dataset operations.

This module provides utility functions for combining and manipulating datasets.
"""

import numpy as np
import pandas as pd
import torch
from typing import List
from .base import InverterTimeSeriesDataset, InverterTimeSeriesDataset_metadata


def combine_dataset(datasets: List[InverterTimeSeriesDataset]) -> InverterTimeSeriesDataset:
    """
    Combine multiple InverterTimeSeriesDataset instances into one.
    
    This function efficiently combines multiple datasets while optimizing memory usage
    by avoiding the creation of large intermediate tensors.
    
    Args:
        datasets: List of InverterTimeSeriesDataset instances to combine
        
    Returns:
        Combined dataset instance
        
    Raises:
        ValueError: If any item in datasets is not an InverterTimeSeriesDataset instance
    """
    # Validate input
    for dataset in datasets:
        if not isinstance(dataset, InverterTimeSeriesDataset):
            raise ValueError("All items in datasets must be instances of InverterTimeSeriesDataset.")

    # Use list to avoid creating large intermediate tensors
    combined_X = []
    combined_y = []
    
    for ds in datasets:
        # Convert to numpy for efficient concatenation if needed
        if isinstance(ds.X, torch.Tensor):
            combined_X.append(ds.X.cpu().numpy())
        else:
            combined_X.append(np.array(ds.X))
            
        if isinstance(ds.y, torch.Tensor):
            combined_y.append(ds.y.cpu().numpy())
        else:
            combined_y.append(np.array(ds.y))

    # Concatenate arrays
    combined_X = np.concatenate(combined_X, axis=0)
    combined_y = np.concatenate(combined_y, axis=0)

    # Create new dataset instance
    new_dataset = InverterTimeSeriesDataset.from_X_y(combined_X, combined_y)
    return new_dataset


def combine_dataset_metadata(datasets: List[InverterTimeSeriesDataset_metadata]) -> InverterTimeSeriesDataset_metadata:
    """
    Combine multiple InverterTimeSeriesDataset_metadata instances into one.
    
    This function combines datasets along with their metadata, preserving
    the metadata information for each sample.
    
    Args:
        datasets: List of InverterTimeSeriesDataset_metadata instances to combine
        
    Returns:
        Combined dataset instance with metadata
        
    Raises:
        ValueError: If any item in datasets is not an InverterTimeSeriesDataset_metadata instance
    """
    # Validate input
    for ds in datasets:
        if not isinstance(ds, InverterTimeSeriesDataset_metadata):
            raise ValueError("All inputs must be InverterTimeSeriesDataset_metadata (or subclasses).")

    # Handle empty dataset list
    if not datasets:
        return InverterTimeSeriesDataset_metadata.from_X_y(
            np.empty((0,)), np.empty((0,)), pd.DataFrame()
        )

    # Concatenate tensors and metadata
    X = np.concatenate([ds.X.cpu().numpy() for ds in datasets], axis=0)
    y = np.concatenate([ds.y.cpu().numpy() for ds in datasets], axis=0)
    meta = pd.concat([ds.meta_data for ds in datasets], ignore_index=True)

    return InverterTimeSeriesDataset_metadata.from_X_y(X, y, meta_df=meta)
