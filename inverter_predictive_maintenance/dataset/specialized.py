"""
Specialized dataset classes for positive and negative samples.

This module contains dataset classes that filter samples based on their labels,
useful for creating balanced datasets or focusing on specific failure types.
"""

from typing import Union
from .base import InverterTimeSeriesDataset, InverterTimeSeriesDataset_metadata


class PositiveInverterTimeSeriesDataset(InverterTimeSeriesDataset):
    """
    Dataset class that only keeps positive samples (label == 1).
    
    This class is useful for creating datasets focused on failure cases
    or for positive-only training scenarios.
    """
    
    def __init__(self) -> None:
        """Initialize positive-only dataset."""
        super().__init__()
        
    def _keep_example(self, y: Union[int, float]) -> bool:
        """
        Keep only examples with positive labels (y == 1).
        
        Args:
            y: Label value
            
        Returns:
            True if y == 1, False otherwise
        """
        return y == 1


class NegativeInverterTimeSeriesDataset(InverterTimeSeriesDataset):
    """
    Dataset class that only keeps negative samples (label == 0).
    
    This class is useful for creating datasets focused on normal operation
    or for negative-only training scenarios.
    """
    
    def __init__(self) -> None:
        """Initialize negative-only dataset."""
        super().__init__()
    
    def _keep_example(self, y: Union[int, float]) -> bool:
        """
        Keep only examples with negative labels (y == 0).
        
        Args:
            y: Label value
            
        Returns:
            True if y == 0, False otherwise
        """
        return y == 0


class PositiveInverterTimeSeriesDataset_metadata(InverterTimeSeriesDataset_metadata):
    """
    Metadata-enabled dataset class that only keeps positive samples (label == 1).
    
    This class combines the metadata tracking functionality with positive-only filtering.
    """
    
    def _keep_example(self, y: Union[int, float]) -> bool:
        """
        Keep only examples with positive labels (y == 1).
        
        Args:
            y: Label value
            
        Returns:
            True if y == 1, False otherwise
        """
        return y == 1


class NegativeInverterTimeSeriesDataset_metadata(InverterTimeSeriesDataset_metadata):
    """
    Metadata-enabled dataset class that only keeps negative samples (label == 0).
    
    This class combines the metadata tracking functionality with negative-only filtering.
    """
    
    def _keep_example(self, y: Union[int, float]) -> bool:
        """
        Keep only examples with negative labels (y == 0).
        
        Args:
            y: Label value
            
        Returns:
            True if y == 0, False otherwise
        """
        return y == 0
