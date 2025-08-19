import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

class InverterTimeSeriesDataset_test(Dataset):
    def __init__(self):
        # 永遠回傳 meta，所以不需要 return_meta 參數
        self.X_list = []
        self.y_list = []
        self.dev_list = []   # device_name
        self.ts_list  = []   # timestamp(ms)

        self.X = torch.empty(0)
        self.y = torch.empty(0)
        self.dev_arr = None
        self.t_ms_arr = None

        self.feature_cols = None
        self.label_col = None
        self.window_size = None
        self.stride = None

    @classmethod
    def from_dataframe(cls, dataframe, feature_cols, label_col='label',
                       window_size=30, stride=1):
        inst = cls()
        inst.load_from_dataframe(dataframe, feature_cols, label_col, window_size, stride)
        inst._finalize()
        return inst

    def load_from_dataframe(self, dataframe, feature_cols, label_col='label',
                            window_size=30, stride=1):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride

        df = dataframe.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        must = feature_cols + [label_col, 'event_local_time', 'device_name']
        for c in must:
            if c not in df.columns:
                raise KeyError(f"Column '{c}' not in dataframe")

        for dev, g in df.groupby('device_name'):
            g = g.reset_index(drop=True)
            times = pd.to_datetime(g['event_local_time'], utc=True, errors='coerce')
            vals  = g[feature_cols].values
            labs  = g[label_col].values

            deltas = times.diff().dt.total_seconds().dropna().round()
            if len(deltas) == 0:
                continue
            expected = deltas.mode()[0]
            good = (deltas == expected).astype(int).to_numpy()
            runs = np.where(good == 0)[0]

            start = 0
            for end in runs:
                self._add_windows_from_block(dev, times[start:end+1], vals[start:end+1], labs[start:end+1])
                start = end + 1
            self._add_windows_from_block(dev, times[start:], vals[start:], labs[start:])

    def _add_windows_from_block(self, dev, times, X_block, y_block):
        if len(X_block) < self.window_size:
            return

        winX = sliding_window_view(X_block, (self.window_size, X_block.shape[1]))[::self.stride, 0, :]
        winY = y_block[self.window_size - 1::self.stride]
        tail_times = times[self.window_size - 1::self.stride]
        tail_ms = (tail_times.view('int64') // 10**6).astype(np.int64)

        for x, y, t_ms in zip(winX, winY, tail_ms):
            if y == -1:
                continue
            self.X_list.append(x)
            self.y_list.append(y)
            self.dev_list.append(dev)
            self.ts_list.append(t_ms)

    def _finalize(self):
        if len(self.X_list) == 0:
            self.X = torch.empty(0, dtype=torch.float32)
            self.y = torch.empty(0, dtype=torch.float32)
            self.dev_arr = np.array([], dtype=object)
            self.t_ms_arr = np.array([], dtype=np.int64)
            return

        self.X = torch.tensor(np.stack(self.X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float32)
        self.dev_arr = np.array(self.dev_list, dtype=object)
        self.t_ms_arr = np.array(self.ts_list, dtype=np.int64)

        self.X_list, self.y_list, self.dev_list, self.ts_list = [], [], [], []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 永遠回傳 meta
        return self.X[idx], self.y[idx], self.dev_arr[idx], self.t_ms_arr[idx]
