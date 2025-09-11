"""
Data labeling utilities for predictive maintenance.

This module provides functions for labeling pre-failure periods and
preparing datasets with appropriate failure indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def label_pre_failure_and_drop(
    inv_grp: pd.DataFrame,
    sess_grp: pd.DataFrame,
    pre_days: int = 5
) -> pd.DataFrame:
    """
    Label pre-failure periods and drop data within failure sessions.
    
    This function:
    1. Labels pre-failure periods (label=1) from non-maintenance sessions
    2. Drops all data within any session window (maintenance or failure)
    3. Labels remaining data as normal operation (label=0)
    
    Args:
        inv_grp: Inverter data grouped by device
        sess_grp: Failure sessions grouped by device
        pre_days: Number of days before failure to label as pre-failure
        
    Returns:
        DataFrame with labeled data and session periods removed
    """
    # Sort inverter data by time
    inv_grp = inv_grp.sort_values('event_local_time')
    et = inv_grp['event_local_time'].values.astype('datetime64[ns]')

    # Handle case with no sessions
    if sess_grp.empty:
        out = inv_grp.copy()
        out['label'] = 0
        return out

    # Sort sessions by start time
    sess_grp = sess_grp.sort_values('start_time')
    starts_all = sess_grp['start_time'].values.astype('datetime64[ns]')
    ends_all = sess_grp['end_time'].values.astype('datetime64[ns]')
    
    # Get maintenance flags
    maint = sess_grp['maintenance'].astype(bool).values if 'maintenance' in sess_grp.columns \
             else np.zeros(len(sess_grp), dtype=bool)

    # Labeling: only NON-maintenance sessions create pre-failure windows
    nm_mask = ~maint
    starts_nm = starts_all[nm_mask]

    n = len(et)
    labels = np.zeros(n, dtype=np.int8)

    if starts_nm.size > 0:
        # Find next non-maintenance session for each time point
        idx_next_nm = np.searchsorted(starts_nm, et, side='right')
        valid_next_nm = idx_next_nm < starts_nm.size
        idxn = idx_next_nm[valid_next_nm]
        
        # Create pre-failure windows
        window_starts = starts_nm - np.timedelta64(pre_days, 'D')

        mask_pre = np.zeros(n, dtype=bool)
        mask_pre[valid_next_nm] = (
            (et[valid_next_nm] >= window_starts[idxn]) &
            (et[valid_next_nm] < starts_nm[idxn])
        )
        labels[mask_pre] = 1

    # Dropping: remove rows that lie in ANY session (maintenance or not)
    idx_next_all = np.searchsorted(starts_all, et, side='right')
    idx_prev_all = idx_next_all - 1
    valid_prev_all = idx_prev_all >= 0

    mask_in_session = np.zeros(n, dtype=bool)
    # Inside if previous session has started and hasn't ended yet
    mask_in_session[valid_prev_all] = (et[valid_prev_all] <= ends_all[idx_prev_all[valid_prev_all]])

    # Keep only data outside sessions
    keep = ~mask_in_session
    inv_grp = inv_grp.iloc[keep].copy()
    inv_grp['label'] = labels[keep]
    
    return inv_grp


def prepare_dataset(
    inverter_df: pd.DataFrame,
    failure_sessions: pd.DataFrame,
    pre_days: int = 5
) -> pd.DataFrame:
    """
    Prepare labeled dataset by applying pre-failure labeling to all devices.
    
    Args:
        inverter_df: DataFrame containing inverter time series data
        failure_sessions: DataFrame containing failure session information
        pre_days: Number of days before failure to label as pre-failure
        
    Returns:
        Labeled DataFrame with pre-failure indicators
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_inv_cols = ['device_name', 'event_local_time']
    required_sess_cols = ['device_name', 'start_time', 'end_time']
    
    missing_inv = [col for col in required_inv_cols if col not in inverter_df.columns]
    missing_sess = [col for col in required_sess_cols if col not in failure_sessions.columns]
    
    if missing_inv:
        raise ValueError(f"Missing columns in inverter_df: {missing_inv}")
    if missing_sess:
        raise ValueError(f"Missing columns in failure_sessions: {missing_sess}")
    
    # Apply labeling to each device
    frames = []
    for dev, grp in inverter_df.groupby('device_name', sort=False):
        grp = grp.sort_values('event_local_time')
        sess = failure_sessions.loc[failure_sessions['device_name'] == dev]\
                              .sort_values('start_time')
        frames.append(label_pre_failure_and_drop(grp, sess, pre_days))
    
    # Combine all device data
    labeled_df = pd.concat(frames, ignore_index=True)
    
    # Print summary statistics
    total_pre_failure = labeled_df['label'].sum()
    total_rows = labeled_df.shape[0]
    pre_failure_rate = total_pre_failure / total_rows if total_rows > 0 else 0
    
    print(f"Dataset preparation complete:")
    print(f"  Total pre-failure rows: {total_pre_failure}")
    print(f"  Total rows: {total_rows}")
    print(f"  Pre-failure rate: {pre_failure_rate:.4f}")
    
    # Handle legacy column name
    if 'failure_label' in labeled_df.columns:
        labeled_df = labeled_df.rename(columns={'failure_label': 'label'})
    
    return labeled_df
