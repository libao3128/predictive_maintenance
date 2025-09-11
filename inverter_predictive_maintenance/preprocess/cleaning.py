"""
Data cleaning utilities for predictive maintenance.

This module provides functions for cleaning time series data,
including missing value imputation, downsampling, and period exclusion.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union


def exclude_periods_from_data(
    df: pd.DataFrame,
    exclude_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    """
    Exclude specific time periods from the dataset.
    
    Args:
        df: Input DataFrame with time series data
        exclude_periods: List of (start, end) tuples for periods to exclude
        
    Returns:
        DataFrame with excluded periods removed
        
    Raises:
        ValueError: If event_local_time column is missing
    """
    if 'event_local_time' not in df.columns:
        raise ValueError("DataFrame must contain 'event_local_time' column")
    
    inverter_data = df.copy()
    
    # Ensure event_local_time is datetime
    inverter_data['event_local_time'] = pd.to_datetime(inverter_data['event_local_time'])
    
    # Exclude each period
    for start, end in exclude_periods:
        mask = (
            (inverter_data['event_local_time'].dt.floor('D') >= start) &
            (inverter_data['event_local_time'].dt.floor('D') <= end)
        )
        inverter_data = inverter_data[~mask]
    
    # Reset index
    inverter_data = inverter_data.reset_index(drop=True)
    
    print(f"Excluded {len(exclude_periods)} periods, remaining data size: {inverter_data.shape[0]}")
    return inverter_data


def missing_value_imputation(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "event_local_time",
    device_col: str = "device_name",
    short_gap_limit: int = 6,
    long_fill_value: float = 0.0,
    add_missing_mask: bool = True,
) -> pd.DataFrame:
    """
    Perform missing value imputation for multi-device time series data.
    
    This function:
    1. Generates per-step missing masks (optional)
    2. Performs time-based interpolation for short gaps
    3. Fills remaining long gaps with specified value
    
    Args:
        df: Original DataFrame with time series data
        feature_cols: Numerical columns to impute
        time_col: Time column name
        device_col: Device column name
        short_gap_limit: Use interpolation for consecutive missing records within this limit
        long_fill_value: Fill long gaps with this value
        add_missing_mask: Whether to generate missing mask columns
        
    Returns:
        DataFrame with completed imputation and optional mask columns
        
    Raises:
        KeyError: If required columns are missing
        ValueError: If time column has invalid values
    """
    imputed_df = df.copy()

    # Ensure time column is datetime
    imputed_df[time_col] = pd.to_datetime(imputed_df[time_col], errors="coerce")
    
    # Check for invalid time values
    if imputed_df[time_col].isna().any():
        raise ValueError(f"{time_col} has invalid time values, please clean first")

    # Check existence of required columns
    missing_cols = [c for c in [time_col, device_col] + feature_cols if c not in imputed_df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}")

    # Process each device separately
    for device, device_data in imputed_df.groupby(device_col, sort=False):
        # Copy to avoid SettingWithCopy warning
        block = device_data.loc[:, [time_col, device_col] + feature_cols].copy()
        # Remember original index for restoration
        block["_orig_idx"] = block.index

        # Generate per-step missing mask (based on original missing values)
        if add_missing_mask:
            for col in feature_cols:
                imputed_df.loc[block["_orig_idx"], f"{col}_missing"] = block[col].isna().astype(int)

        # Sort by time and use time as index for time-based interpolation
        block = block.sort_values(time_col)
        block = block.set_index(time_col)

        # Short gaps: time interpolation (bidirectional, avoid front/end segments with all NaN)
        if short_gap_limit > 0:
            block[feature_cols] = block[feature_cols].interpolate(
                method="time", limit=short_gap_limit, limit_direction="forward"
            )

        # Long gaps: fill remaining NaN with specified value
        block[feature_cols] = block[feature_cols].fillna(long_fill_value)

        # Restore index and order
        block = block.reset_index()
        block = block.set_index("_orig_idx").sort_index()

        # Write back to imputed_df (only overwrite target feature columns)
        imputed_df.loc[block.index, feature_cols] = block[feature_cols].values

    return imputed_df


def downsample_inverter_raw(
    df: pd.DataFrame,
    freq: str = "30T",
    time_col: str = "event_local_time",
    device_col: str = "device_name",
    energy_as: str = "delta",
    drop_empty_bins: bool = True
) -> pd.DataFrame:
    """
    Downsample original 5-min data based on column semantics.
    
    Aggregation rules:
    - Continuous variables → mean
    - Boolean/connection/heartbeat/status/WORD → max
    - Cumulative variables (ENERGY_*, VARH_*) → delta (default) / last / mean
    - Setpoint/HW_VERSION → last
    
    Args:
        df: Input DataFrame with 5-minute data
        freq: Resampling frequency (pandas offset alias)
        time_col: Time column name
        device_col: Device column name
        energy_as: Aggregation method for energy columns ('delta', 'last', 'mean')
        drop_empty_bins: Whether to drop rows where all continuous variables are NaN
        
    Returns:
        Downsampled DataFrame
        
    Raises:
        ValueError: If time column has invalid values or energy_as is invalid
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} has invalid time, please clean first.")

    # Column classification (by naming rules)
    cols: List[str] = [c for c in df.columns if c not in (time_col, device_col)]

    # Possible "cumulative" columns (energy, varh)
    cumulative_cols = [c for c in cols if any([
        c.startswith("metric.ENERGY_") and c.endswith(".MEASURED"),
        c.startswith("metric.VARH_")   and c.endswith(".MEASURED"),
        c == "metric.ENERGY_DELIVERED.MEASURED",
        c == "metric.ENERGY_RECEIVED.MEASURED",
        c == "metric.VARH_DELIVERED.MEASURED"
    ])]

    # Status/error code/WORD/boolean flag types (including COMM_LINK, HEARTBEAT)
    state_like_cols = [c for c in cols if (
        c.startswith("metric.STATUS_") or
        c.endswith("WORD.MEASURED") or
        c in ["metric.COMM_LINK.MEASURED", "metric.HEARTBEAT.MEASURED"]
    )]

    # Setpoint / version
    last_pref_cols = [c for c in cols if (
        c.endswith("_SETPOINT.MEASURED") or
        c == "metric.HW_VERSION.MEASURED"
    )]

    # Others treated as continuous variables (voltage/current/power/frequency/temperature...)
    assigned = set(cumulative_cols) | set(state_like_cols) | set(last_pref_cols)
    continuous_mean_cols = [c for c in cols if c not in assigned]

    # Aggregation function definitions
    def agg_cumulative(s: pd.Series) -> float:
        """Interval increment: last - first, handle reset/rollover as >=0"""
        if s.dropna().empty:
            return float("nan")
        first = s.iloc[0]
        last  = s.iloc[-1]
        return max(float(last) - float(first), 0.0)

    # Aggregation rules dictionary
    agg: dict = {}

    # Continuous variables → mean
    for c in continuous_mean_cols:
        agg[c] = "mean"

    # Status/WORD/boolean → max
    for c in state_like_cols:
        agg[c] = "max"

    # Setpoint/HW_VERSION → last
    for c in last_pref_cols:
        agg[c] = "last"
        
    # Missing feature columns
    for c in df.columns:
        if c.endswith('_missing'):
            agg[c] = 'mean'

    # Cumulative variables → based on parameter
    if energy_as == "delta":
        for c in cumulative_cols:
            agg[c] = agg_cumulative
    elif energy_as == "last":
        for c in cumulative_cols:
            agg[c] = "last"
    elif energy_as == "mean":
        for c in cumulative_cols:
            agg[c] = "mean"
    else:
        raise ValueError("energy_as must be one of {'delta','last','mean'}")
    
    print(f"Downsampling {len(df)} rows using following method: ")
    print(f"{pd.DataFrame(agg.items(), columns=['Column', 'Aggregation'])}")
    
    rs = pd.DataFrame()
    # Downsample by device
    for device, group in df.groupby(device_col, sort=False):
        # Sort by time
        group = group.sort_values(time_col)

        # Group by time and device, then perform resample aggregation
        group = group.set_index(time_col)
        resampled = group.groupby(device_col).resample(freq).agg(agg).reset_index()

        # Write back to original DataFrame
        if device_col not in resampled.columns:
            resampled[device_col] = device
        rs = pd.concat([rs, resampled], ignore_index=True)

    rs.reset_index(drop=True, inplace=True)

    # Optional: drop rows where all "continuous variables" are NaN
    if drop_empty_bins and continuous_mean_cols:
        mask_all_nan = rs[continuous_mean_cols].isna().all(axis=1)
        rs = rs.loc[~mask_all_nan].copy()

    # Column order (try to stay close to original)
    ordered = [time_col, device_col] + continuous_mean_cols + state_like_cols + last_pref_cols + cumulative_cols
    ordered = [c for c in ordered if c in rs.columns]
    rs = rs.loc[:, ordered].sort_values([device_col, time_col])

    return rs
