"""
Prediction smoothing utilities for predictive maintenance.

This module provides functions for smoothing model predictions
to reduce noise and improve interpretability.
"""

import numpy as np
from typing import Union, List, Dict, Optional


def smooth_by_consecutive(
    bin_preds: Union[np.ndarray, List[int]],
    group_ids: Optional[Union[np.ndarray, List]] = None,
    min_consecutive: int = 3,
    return_events: bool = True,
) -> Union[np.ndarray, tuple]:
    """
    Smooth binary predictions by keeping only consecutive runs above minimum length.
    
    This function filters out short runs of positive predictions that are likely
    noise, keeping only runs that meet the minimum consecutive length requirement.
    
    Args:
        bin_preds: Binary predictions per window (0/1) after thresholding
        group_ids: Group key per window (e.g., device id or name). If None, treat all as one group
        min_consecutive: Minimum length (in windows) for a run of 1's to be kept
        return_events: If True, also return lists of kept runs and filtered noise runs
        
    Returns:
        If return_events=True: Tuple of (smoothed_predictions, kept_events, noise_runs)
        If return_events=False: Smoothed predictions array
        
    Raises:
        ValueError: If inputs have invalid values or shapes
    """
    # Convert inputs to numpy arrays
    b = np.asarray(bin_preds).astype(np.uint8)
    n = len(b)
    
    # Validate inputs
    if not np.all((b == 0) | (b == 1)):
        raise ValueError("bin_preds must contain only values 0 and 1")
    
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")
    
    # Handle group_ids
    if group_ids is None:
        group_ids = np.zeros(n, dtype=np.int32)
    else:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != n:
            raise ValueError(f"group_ids length {len(group_ids)} != bin_preds length {n}")

    # Initialize output
    smoothed = np.zeros_like(b, dtype=np.uint8)
    kept_events, noise_runs = [], []

    # Process each group independently
    for gid in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        if idx.size == 0:
            continue
        
        seg = b[idx]
        smoothed_seg = _smooth_segment(seg, min_consecutive, return_events)
        
        if return_events:
            smoothed_seg, kept_seg_events, noise_seg_runs = smoothed_seg
            
            # Adjust indices to global coordinates
            for event in kept_seg_events:
                event['group'] = gid
                event['start_idx'] = int(idx[event['start_idx']])
                event['end_idx'] = int(idx[event['end_idx']])
                kept_events.append(event)
            
            for event in noise_seg_runs:
                event['group'] = gid
                event['start_idx'] = int(idx[event['start_idx']])
                event['end_idx'] = int(idx[event['end_idx']])
                noise_runs.append(event)
        else:
            smoothed_seg = smoothed_seg
        
        smoothed[idx] = smoothed_seg

    if return_events:
        return smoothed, kept_events, noise_runs
    return smoothed


def _smooth_segment(
    seg: np.ndarray,
    min_consecutive: int,
    return_events: bool
) -> Union[np.ndarray, tuple]:
    """
    Smooth a single segment of binary predictions.
    
    Args:
        seg: Binary segment to smooth
        min_consecutive: Minimum consecutive length
        return_events: Whether to return event information
        
    Returns:
        Smoothed segment and optionally event information
    """
    smoothed = np.zeros_like(seg, dtype=np.uint8)
    kept_events, noise_runs = [], []
    
    i = 0
    while i < len(seg):
        if seg[i] == 1:
            # Find end of current run
            j = i
            while j < len(seg) and seg[j] == 1:
                j += 1
            
            run_len = j - i
            
            if run_len >= min_consecutive:
                # Keep the run
                smoothed[i:j] = 1
                if return_events:
                    kept_events.append({
                        "start_idx": i,
                        "end_idx": j - 1,
                        "length": run_len,
                    })
            else:
                # Mark as noise
                if return_events:
                    noise_runs.append({
                        "start_idx": i,
                        "end_idx": j - 1,
                        "length": run_len,
                    })
            
            i = j
        else:
            i += 1
    
    if return_events:
        return smoothed, kept_events, noise_runs
    return smoothed


def smooth_by_majority_vote(
    bin_preds: Union[np.ndarray, List[int]],
    window_size: int = 5,
    threshold: float = 0.6
) -> np.ndarray:
    """
    Smooth binary predictions using majority voting within a sliding window.
    
    Args:
        bin_preds: Binary predictions to smooth
        window_size: Size of the sliding window
        threshold: Minimum proportion of 1's in window to output 1
        
    Returns:
        Smoothed binary predictions
    """
    bin_preds = np.asarray(bin_preds)
    n = len(bin_preds)
    smoothed = np.zeros_like(bin_preds)
    
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window_preds = bin_preds[start:end]
        
        # Majority vote
        if np.mean(window_preds) >= threshold:
            smoothed[i] = 1
    
    return smoothed


def smooth_by_median_filter(
    bin_preds: Union[np.ndarray, List[int]],
    kernel_size: int = 5
) -> np.ndarray:
    """
    Smooth binary predictions using median filtering.
    
    Args:
        bin_preds: Binary predictions to smooth
        kernel_size: Size of the median filter kernel
        
    Returns:
        Smoothed binary predictions
    """
    from scipy import ndimage
    
    bin_preds = np.asarray(bin_preds)
    smoothed = ndimage.median_filter(bin_preds, size=kernel_size)
    
    return smoothed.astype(np.uint8)


def remove_isolated_predictions(
    bin_preds: Union[np.ndarray, List[int]],
    min_neighbors: int = 2
) -> np.ndarray:
    """
    Remove isolated positive predictions that don't have enough neighbors.
    
    Args:
        bin_preds: Binary predictions to filter
        min_neighbors: Minimum number of positive neighbors required
        
    Returns:
        Filtered binary predictions
    """
    bin_preds = np.asarray(bin_preds)
    filtered = bin_preds.copy()
    
    for i in range(len(bin_preds)):
        if bin_preds[i] == 1:
            # Count positive neighbors
            neighbors = 0
            for j in range(max(0, i-1), min(len(bin_preds), i+2)):
                if j != i and bin_preds[j] == 1:
                    neighbors += 1
            
            if neighbors < min_neighbors:
                filtered[i] = 0
    
    return filtered
