import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

class InverterTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, feature_cols, label_col='label', window_size=30, stride=1):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride
        self.samples = []

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
                self._add_windows_from_block(values[start:end+1], labels[start:end+1], self.samples)
                start = end + 1
                
            self._add_windows_from_block(values[start:], labels[start:], self.samples)  # 最後一段

    def _add_windows_from_block(self, X_block, y_block, output_samples):
        if len(X_block) < self.window_size:
            return

        windows_X = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        windows_y = y_block[self.window_size - 1::self.stride]

        # 過濾掉 label == -1 的
        for x, y in zip(windows_X, windows_y):
            if y == -1:
                continue
            output_samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
