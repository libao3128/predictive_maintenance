"""
Data loading utilities for predictive maintenance.

This module provides functions for loading parquet files and failure session data.
"""

import os
from glob import glob
import pandas as pd
from typing import Optional


def load_parquet_data(parquet_dir: str) -> pd.DataFrame:
    """
    Load all parquet files from a directory and combine them into a single DataFrame.
    
    Args:
        parquet_dir: Directory path containing parquet files
        
    Returns:
        Combined DataFrame with all parquet data
        
    Raises:
        FileNotFoundError: If parquet_dir doesn't exist
        ValueError: If no parquet files found in directory
    """
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory {parquet_dir} does not exist")
    
    # Find all parquet files
    paths = glob(os.path.join(parquet_dir, '*.parquet'))
    if not paths:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    # Load and combine DataFrames
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No parquet files could be loaded successfully")
    
    # Combine all DataFrames
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert event_local_time to datetime
    df['event_local_time'] = pd.to_datetime(df['event_local_time'])
    
    print(f"Loaded {len(paths)} parquet files → {df.shape[0]} rows")
    return df


def load_failure_sessions(
    csv_path: str,
    min_days: int = 3,
    parse_dates: Optional[list] = None,
    dtype: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load failure sessions from CSV file with preprocessing.
    
    Args:
        csv_path: Path to the CSV file
        min_days: Minimum duration in days to keep a session
        parse_dates: List of columns to parse as dates (default: ['start_time', 'end_time'])
        dtype: Dictionary of column dtypes (default: {'device_name': str})
        
    Returns:
        Preprocessed failure sessions DataFrame
        
    Raises:
        FileNotFoundError: If csv_path doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} does not exist")
    
    # Set defaults
    if parse_dates is None:
        parse_dates = ['start_time', 'end_time']
    if dtype is None:
        dtype = {'device_name': str}
    
    # Load CSV
    df = pd.read_csv(
        csv_path,
        parse_dates=parse_dates,
        dtype=dtype
    )
    
    # Remove unnamed columns
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
    
    # Validate required columns
    required_cols = ['start_time', 'end_time', 'device_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Process duration column
    if 'duration' in df.columns:
        df['duration'] = pd.to_timedelta(df['duration'])
    else:
        # Calculate duration if not present
        df['duration'] = pd.to_timedelta(df['end_time'] - df['start_time'])
    
    # Ensure datetime columns are properly formatted
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Handle maintenance column
    if 'maintenance' in df.columns:
        df['maintenance'] = df['maintenance'].fillna(False).astype(bool)
    else:
        df['maintenance'] = False
    
    # Filter by minimum duration
    df = df[df['duration'] > pd.Timedelta(days=min_days)]
    
    print(f"Kept {len(df)} sessions longer than {min_days} days")
    return df
