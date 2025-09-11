"""
Post-processing module for predictive maintenance.

This module provides utilities for post-processing model predictions,
including smoothing, filtering, and event detection.
"""

from .smoothing import smooth_by_consecutive

__all__ = [
    "smooth_by_consecutive"
]
