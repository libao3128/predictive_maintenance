import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

class InverterTimeSeriesDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([], dtype=torch.float32)
        self.y = torch.tensor([], dtype=torch.float32)

    @classmethod
    def from_dataframe(cls, dataframe, feature_cols, label_col='label', window_size=30, stride=1):
        """
        從 DataFrame 建立資料集。
        """
        instance = cls()
        instance.load_from_dataframe(dataframe, feature_cols, label_col, window_size, stride)
        return instance
    
    @classmethod
    def from_X_y(cls, X, y):
        """
        從 X 和 y 建立資料集。
        """
        instance = cls()
        instance.load_from_X_y(X, y)
        return instance
    
    def load_from_dataframe(self, dataframe, feature_cols, label_col='label', window_size=30, stride=1):
        """
        重新載入資料集，適用於需要更改參數的情況。
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride
        self.X = []
        self.y = []
               
        # 預處理檢查
        dataframe = dataframe.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        if dataframe.isnull().values.any():
            raise ValueError("DataFrame contains NaN values. Please clean the data before creating the dataset.")

        for col in feature_cols + [label_col, 'event_local_time', 'device_name']:
            if col not in dataframe.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # 依 device 分開處理
        for device, group in tqdm(dataframe.groupby('device_name'), desc="Processing devices"):
            group = group.reset_index(drop=True)
            times = pd.to_datetime(group['event_local_time'])
            values = group[feature_cols].values  # shape: (T, F)
            labels = group[label_col].values     # shape: (T,)

            # 嘗試找出該 device 的主要時間間隔
            time_deltas = times.diff().dt.total_seconds().dropna().round()
            if len(time_deltas) == 0:
                continue
            expected_delta = time_deltas.mode()[0]  # 最常見的時間差

            # 找出所有時間連續的區段
            good_indices = (time_deltas == expected_delta).astype(int).to_numpy()
            # 第一筆視為連續開始
            runs = np.where(good_indices == 0)[0]
            start = 0
            for end in runs:
                self._add_windows_from_block(values[start:end+1], labels[start:end+1])
                start = end + 1
                
            self._add_windows_from_block(values[start:], labels[start:])  # 最後一段

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def load_from_X_y(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if isinstance(X, list):
            X = np.array(X)
            self.X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            self.X = X
            
        if isinstance(y, list):
            y = np.array(y)
            self.y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, torch.Tensor):
            self.y = y

    def _add_windows_from_block(self, X_block, y_block):
        if len(X_block) < self.window_size:
            return

        windows_X = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        windows_y = y_block[self.window_size - 1::self.stride]

        # 過濾掉 label == -1 的
        for x, y in zip(windows_X, windows_y):
            if y == -1:
                continue
            self.X.append(x)
            self.y.append(y)
            
    def under_sample(self, sampling_strategy='auto'):
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
        X_ind, self.y = rus.fit_resample(np.array(range(len(self.X))).reshape(-1, 1), self.y)
        X_ind = X_ind.flatten()
        self.X = np.array(self.X)[X_ind]
        
        self.X =  torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositiveInverterTimeSeriesDataset(InverterTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        
    def _add_windows_from_block(self, X_block, y_block):
        if len(X_block) < self.window_size:
            return

        windows_X = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        windows_y = y_block[self.window_size - 1::self.stride]

        # 過濾掉 label == -1 的
        for x, y in zip(windows_X, windows_y):
            if y == -1 or y == 0:  # 只保留 label == 1 的樣本
                continue
            self.X.append(x)
            self.y.append(y)
        
class NegativeInverterTimeSeriesDataset(InverterTimeSeriesDataset):
    def __init__(self):
        super().__init__()
    
    def _add_windows_from_block(self, X_block, y_block):
        if len(X_block) < self.window_size:
            return

        windows_X = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        windows_y = y_block[self.window_size - 1::self.stride]

        # 過濾掉 label == -1 的
        for x, y in zip(windows_X, windows_y):
            if y == -1 or y == 1:  # 只保留 label ==  0 的樣本
                continue
            self.X.append(x)
            self.y.append(y)
        
def combine_dataset(datasets):
    """
    Combine multiple InverterTimeSeriesDataset instances into one.
    """
    combined_X = []
    combined_y = []
    
    for dataset in datasets:
        if not isinstance(dataset, InverterTimeSeriesDataset):
            raise ValueError("All items in datasets must be instances of InverterTimeSeriesDataset.")
        combined_X.append(dataset.X)
        combined_y.append(dataset.y)
    
    combined_X = torch.cat(combined_X, dim=0)
    combined_y = torch.cat(combined_y, dim=0)

    new_dataset = InverterTimeSeriesDataset.from_X_y(combined_X, combined_y)
    return new_dataset