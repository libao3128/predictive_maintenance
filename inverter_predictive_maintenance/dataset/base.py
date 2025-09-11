"""
Base dataset classes for inverter time series data.

This module contains the core dataset classes that handle windowing
and basic data loading functionality.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from typing import List, Optional, Union, Tuple


def _to_datetime_utc(s: Union[pd.Series, np.ndarray], unit: Optional[str] = None) -> pd.Series:
    """Convert series to UTC datetime."""
    return pd.to_datetime(s, utc=True, errors="coerce", unit=unit)


class InverterTimeSeriesDataset(Dataset):
    """
    Base dataset class for inverter time series data with windowing.
    
    This class handles the creation of sliding windows from time series data,
    with support for different sampling strategies and device grouping.
    
    Attributes:
        X (torch.Tensor): Input features tensor of shape (N, window_size, num_features)
        y (torch.Tensor): Target labels tensor of shape (N,)
        feature_cols (List[str]): List of feature column names
        label_col (str): Name of the label column
        window_size (int): Size of the sliding window
        stride (int): Stride for sliding window creation
    """
    
    def __init__(self) -> None:
        """Initialize empty dataset."""
        self.X = torch.tensor([], dtype=torch.float32)
        self.y = torch.tensor([], dtype=torch.float32)
        self.feature_cols: Optional[List[str]] = None
        self.label_col: Optional[str] = None
        self.window_size: Optional[int] = None
        self.stride: Optional[int] = None

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        window_size: int = 30,
        stride: int = 1
    ) -> 'InverterTimeSeriesDataset':
        """
        Create dataset from DataFrame.
        
        Args:
            dataframe: Input DataFrame with time series data
            feature_cols: List of feature column names
            label_col: Name of the label column
            window_size: Size of the sliding window
            stride: Stride for sliding window creation
            
        Returns:
            Dataset instance
        """
        instance = cls()
        instance.load_from_dataframe(dataframe, feature_cols, label_col, window_size, stride)
        return instance
    
    @classmethod
    def from_X_y(cls, X: Union[np.ndarray, torch.Tensor, List], y: Union[np.ndarray, torch.Tensor, List]) -> 'InverterTimeSeriesDataset':
        """
        Create dataset from X and y arrays.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Dataset instance
        """
        instance = cls()
        instance.load_from_X_y(X, y)
        return instance
    
    def load_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        window_size: int = 30,
        stride: int = 1
    ) -> None:
        """
        Load dataset from DataFrame with windowing.
        
        Args:
            dataframe: Input DataFrame with time series data
            feature_cols: List of feature column names
            label_col: Name of the label column
            window_size: Size of the sliding window
            stride: Stride for sliding window creation
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride
        self.X = []
        self.y = []
               
        # Validate required columns
        required_cols = feature_cols + [label_col, 'event_local_time', 'device_name']
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")

        # Preprocessing checks
        dataframe = dataframe.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        if dataframe.isnull().values.any():
            raise ValueError("DataFrame contains NaN values. Please clean the data before creating the dataset.")

        # Process by device separately
        for device, group in tqdm(dataframe.groupby('device_name'), desc="Processing devices"):
            group = group.reset_index(drop=True)
            times = pd.to_datetime(group['event_local_time'])
            values = group[feature_cols].values  # shape: (T, F)
            labels = group[label_col].values     # shape: (T,)

            # Try to find the main time interval for this device
            time_deltas = times.diff().dt.total_seconds().dropna().round()
            if len(time_deltas) == 0:
                continue
            expected_delta = time_deltas.mode()[0]  # Most common time difference

            # Find all time-continuous segments
            good_indices = (time_deltas == expected_delta).astype(int).to_numpy()
            # First record is considered as continuous start
            runs = np.where(good_indices == 0)[0]
            start = 0
            for end in runs:
                self._add_windows_from_block(values[start:end+1], labels[start:end+1])
                start = end + 1
                
            self._add_windows_from_block(values[start:], labels[start:])  # Last segment

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def load_from_X_y(self, X: Union[np.ndarray, torch.Tensor, List], y: Union[np.ndarray, torch.Tensor, List]) -> None:
        """
        Load dataset from X and y arrays.
        
        Args:
            X: Input features
            y: Target labels
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
            
        # Convert X to tensor
        if isinstance(X, list):
            X = np.array(X)
            self.X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            self.X = X
            
        # Convert y to tensor
        if isinstance(y, list):
            y = np.array(y)
            self.y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, torch.Tensor):
            self.y = y

    def _add_windows_from_block(self, X_block: np.ndarray, y_block: np.ndarray) -> None:
        """
        Add windows from a continuous block of data.
        
        Args:
            X_block: Feature data block
            y_block: Label data block
        """
        if len(X_block) < self.window_size:
            return

        windows_X = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        windows_y = y_block[self.window_size - 1::self.stride]

        # Filter out records with label == -1
        for x, y in zip(windows_X, windows_y):
            if y == -1:
                continue
            if self._keep_example(y):
                self.X.append(x)
                self.y.append(y)

    def _keep_example(self, y: Union[int, float]) -> bool:
        """
        Determine whether to keep an example based on its label.
        
        Args:
            y: Label value
            
        Returns:
            True if example should be kept, False otherwise
        """
        return y != -1

    def under_sample(self, sampling_strategy: str = 'auto') -> None:
        """
        Apply undersampling to balance the dataset.
        
        Args:
            sampling_strategy: Strategy for undersampling
        """
        from imblearn.under_sampling import RandomUnderSampler
        
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
        X_ind, self.y = rus.fit_resample(np.array(range(len(self.X))).reshape(-1, 1), self.y)
        X_ind = X_ind.flatten()
        self.X = np.array(self.X)[X_ind]
        
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        return self.X[idx], self.y[idx]


class InverterTimeSeriesDataset_metadata(Dataset):
    """
    Dataset class with metadata tracking for inverter time series data.
    
    This class extends the base dataset functionality to include metadata
    about each window, such as device name, time range, and label information.
    
    Attributes:
        X (torch.Tensor): Input features tensor
        y (torch.Tensor): Target labels tensor
        meta_data (pd.DataFrame): Metadata for each window
    """
    
    def __init__(self) -> None:
        """Initialize empty dataset with metadata."""
        self.X = torch.tensor([], dtype=torch.float32)
        self.y = torch.tensor([], dtype=torch.float32)
        self.meta_data = pd.DataFrame()
        self.feature_cols: Optional[List[str]] = None
        self.label_col: Optional[str] = None
        self.window_size: Optional[int] = None
        self.stride: Optional[int] = None

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        window_size: int = 30,
        stride: int = 1
    ) -> 'InverterTimeSeriesDataset_metadata':
        """Create dataset from DataFrame with metadata tracking."""
        inst = cls()
        inst.load_from_dataframe(dataframe, feature_cols, label_col, window_size, stride)
        return inst

    @classmethod
    def from_X_y(
        cls,
        X: Union[np.ndarray, torch.Tensor, List],
        y: Union[np.ndarray, torch.Tensor, List],
        meta_df: Optional[pd.DataFrame] = None
    ) -> 'InverterTimeSeriesDataset_metadata':
        """Create dataset from X, y, and metadata."""
        inst = cls()
        inst.load_from_X_y(X, y, meta_df)
        return inst

    def load_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        window_size: int = 30,
        stride: int = 1,
        drop_label_neg1: bool = True
    ) -> None:
        """
        Load dataset from DataFrame with metadata tracking.
        
        Args:
            dataframe: Input DataFrame
            feature_cols: List of feature column names
            label_col: Name of the label column
            window_size: Size of the sliding window
            stride: Stride for sliding window creation
            drop_label_neg1: Whether to drop samples with label -1
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride

        # Check required columns
        need_cols = set(feature_cols + [label_col, 'event_local_time', 'device_name'])
        missing = [c for c in need_cols if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")

        df = dataframe.copy()
        df['event_local_time'] = _to_datetime_utc(df['event_local_time'], unit='ms')
        df = df.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        
        if df.isnull().values.any():
            df = df.dropna(subset=['event_local_time'] + feature_cols + [label_col])

        X_list, y_list, meta_records = [], [], []

        # Process by device
        for dev, g in tqdm(df.groupby('device_name'), desc="Processing devices"):
            g = g.reset_index(drop=True)
            times = g['event_local_time'].to_numpy()
            values = g[feature_cols].to_numpy()  # (T, F)
            labels = g[label_col].to_numpy()    # (T,)

            if len(times) < self.window_size:
                continue

            # Determine main sampling interval (using mode)
            deltas = pd.Series(times).diff().dt.total_seconds().dropna().round()
            if len(deltas) == 0:
                continue
            expected = deltas.mode().iloc[0]

            # Find continuous segments (gap != expected is considered a break)
            gap = (pd.Series(times).diff().dt.total_seconds().fillna(expected).round() != expected).to_numpy()
            start_idx = 0
            for i in range(1, len(times)):
                if gap[i]:
                    self._append_windows_of_block(
                        dev, times[start_idx:i], values[start_idx:i], labels[start_idx:i],
                        X_list, y_list, meta_records, drop_label_neg1
                    )
                    start_idx = i
            # Last segment
            self._append_windows_of_block(
                dev, times[start_idx:], values[start_idx:], labels[start_idx:],
                X_list, y_list, meta_records, drop_label_neg1
            )

        self.X = torch.tensor(np.stack(X_list), dtype=torch.float32) if X_list else torch.empty((0,), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32) if y_list else torch.empty((0,), dtype=torch.float32)
        self.meta_data = pd.DataFrame(meta_records) if meta_records else pd.DataFrame(
            columns=["device", "start", "end", "length", "label"]
        )

    def _append_windows_of_block(
        self,
        dev: str,
        times: np.ndarray,
        values: np.ndarray,
        labels: np.ndarray,
        X_list: List,
        y_list: List,
        meta_records: List,
        drop_label_neg1: bool = True
    ) -> None:
        """Append windows from a continuous block with metadata."""
        T = len(times)
        W = self.window_size
        if T < W:
            return

        # Create windows
        values_windows = sliding_window_view(values, (W, values.shape[1]))[::self.stride, 0, :]
        labels_at_end = labels[W - 1::self.stride]

        # Corresponding time ranges
        start_times = times[0::self.stride][0:len(values_windows)]
        end_times = times[W - 1::self.stride]

        for x, y, t0, t1 in zip(values_windows, labels_at_end, start_times, end_times):
            if drop_label_neg1 and (y == -1):
                continue
            if self._keep_example(y):
                X_list.append(x)
                y_list.append(float(y))
                meta_records.append({
                    "device": dev,
                    "start": pd.Timestamp(t0),
                    "end": pd.Timestamp(t1),
                    "length": W,
                    "label": int(y) if (y in (0, 1)) else -1
                })

    def _keep_example(self, y: Union[int, float]) -> bool:
        """Determine whether to keep an example based on its label."""
        return y != -1

    def load_from_X_y(
        self,
        X: Union[np.ndarray, torch.Tensor, List],
        y: Union[np.ndarray, torch.Tensor, List],
        meta_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Load dataset from X, y, and metadata."""
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            self.X = torch.tensor(np.asarray(X), dtype=torch.float32)

        if isinstance(y, torch.Tensor):
            self.y = y.float()
        else:
            self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

        self.meta_data = meta_df.copy() if isinstance(meta_df, pd.DataFrame) else pd.DataFrame()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.X[idx], self.y[idx]
