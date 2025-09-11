"""
Data splitting utilities for predictive maintenance.

This module provides functions for splitting time series data
into training and testing sets based on temporal order.
"""

import pandas as pd
from typing import Tuple


def train_test_split_on_time(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = 'event_local_time'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and testing sets based on time.
    
    This function ensures that the test set contains the most recent data,
    which is important for time series tasks to avoid data leakage.
    
    Args:
        df: Input DataFrame with time series data
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        time_col: Name of the time column for sorting
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        ValueError: If test_size is not between 0 and 1, or time_col is missing
    """
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")
    
    # Sort by time to ensure temporal order
    df = df.sort_values(time_col)
    n = len(df)
    test_n = int(n * test_size)
    
    # Split: train gets earlier data, test gets later data
    train_df = df[:-test_n] if test_n > 0 else df
    test_df = df[-test_n:] if test_n > 0 else df.iloc[0:0]  # Empty DataFrame with same structure
    
    # Print summary
    if len(train_df) > 0 and len(test_df) > 0:
        print(f"Train set size: {len(train_df)}")
        print(f"Train set time range: {train_df[time_col].min()} to {train_df[time_col].max()}")
        print(f"Test set size: {len(test_df)}")
        print(f"Test set time range: {test_df[time_col].min()} to {test_df[time_col].max()}")
    elif len(train_df) > 0:
        print(f"Train set size: {len(train_df)}")
        print(f"Train set time range: {train_df[time_col].min()} to {train_df[time_col].max()}")
        print("Test set is empty")
    else:
        print("Train set is empty")
        print(f"Test set size: {len(test_df)}")
        print(f"Test set time range: {test_df[time_col].min()} to {test_df[time_col].max()}")
    
    return train_df, test_df
