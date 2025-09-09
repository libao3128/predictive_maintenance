import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def _to_datetime_utc(s, unit=None):
    return pd.to_datetime(s, utc=True, errors="coerce", unit=unit)



class InverterTimeSeriesDataset_metadata(Dataset):
    """
    基底：建立 window 化資料，並在建立時同步蒐集 meta_data。
    - __getitem__ 仍只回傳 (X, y) 以便訓練不受影響
    - self.meta_data: DataFrame，記錄每個 window 的 device / start / end / length / label
    """
    def __init__(self):
        self.X = torch.tensor([], dtype=torch.float32)
        self.y = torch.tensor([], dtype=torch.float32)
        self.meta_data = pd.DataFrame()

        self.feature_cols = None
        self.label_col = None
        self.window_size = None
        self.stride = None

    @classmethod
    def from_dataframe(cls, dataframe, feature_cols, label_col='label',
                       window_size=30, stride=1):
        inst = cls()
        inst.load_from_dataframe(dataframe, feature_cols, label_col, window_size, stride)
        return inst

    @classmethod
    def from_X_y(cls, X, y, meta_df=None):
        inst = cls()
        inst.load_from_X_y(X, y, meta_df)
        return inst

    # ---------------- core build ----------------
    def load_from_dataframe(self, dataframe, feature_cols, label_col='label',
                            window_size=30, stride=1, drop_label_neg1=True):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride

        # 檢查欄位
        need_cols = set(feature_cols + [label_col, 'event_local_time', 'device_name'])
        missing = [c for c in need_cols if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")

        df = dataframe.copy()
        df['event_local_time'] = _to_datetime_utc(df['event_local_time'], unit='ms')
        df = df.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        if df.isnull().values.any():
            # 可依你需求決定要不要 raise；這裡保守一點：
            df = df.dropna(subset=['event_local_time'] + feature_cols + [label_col])

        X_list, y_list, meta_records = [], [], []

        # 依 device 切
        for dev, g in tqdm(df.groupby('device_name'), desc="Processing devices"):
            g = g.reset_index(drop=True)
            times = g['event_local_time'].to_numpy()
            values = g[feature_cols].to_numpy()  # (T, F)
            labels = g[label_col].to_numpy()    # (T,)

            if len(times) < self.window_size:
                continue

            # 判斷主要採樣間隔（用眾數）
            deltas = pd.Series(times).diff().dt.total_seconds().dropna().round()
            if len(deltas) == 0:
                continue
            expected = deltas.mode().iloc[0]

            # 找出連續區段（gap != expected 視為斷開）
            gap = (pd.Series(times).diff().dt.total_seconds().fillna(expected).round() != expected).to_numpy()
            # gap==True 表示新的區段開始；我們要把每段逐段做 window
            start_idx = 0
            for i in range(1, len(times)):
                if gap[i]:
                    self._append_windows_of_block(
                        dev, times[start_idx:i], values[start_idx:i], labels[start_idx:i],
                        X_list, y_list, meta_records, drop_label_neg1
                    )
                    start_idx = i
            # 最後一段
            self._append_windows_of_block(
                dev, times[start_idx:], values[start_idx:], labels[start_idx:],
                X_list, y_list, meta_records, drop_label_neg1
            )

        self.X = torch.tensor(np.stack(X_list), dtype=torch.float32) if X_list else torch.empty((0,), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32) if y_list else torch.empty((0,), dtype=torch.float32)
        self.meta_data = pd.DataFrame(meta_records) if meta_records else pd.DataFrame(
            columns=["device", "start", "end", "length", "label"]
        )

    def _append_windows_of_block(self, dev, times, values, labels,
                                 X_list, y_list, meta_records, drop_label_neg1=True):
        T = len(times)
        W = self.window_size
        if T < W:
            return

        # window 化
        # values_windows: (N, W, F)
        values_windows = sliding_window_view(values, (W, values.shape[1]))[::self.stride, 0, :]
        # 每個 window 的標籤：取 window 尾端的 label
        labels_at_end = labels[W - 1::self.stride]

        # 對應的時間範圍（每個 window 的 start / end）
        start_times = times[0::self.stride][0:len(values_windows)]
        end_times = times[W - 1::self.stride]

        for x, y, t0, t1 in zip(values_windows, labels_at_end, start_times, end_times):
            if drop_label_neg1 and (y == -1):
                continue
            # 交給子類別決定是否要保留（正/負）
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

    def _keep_example(self, y):
        """基底類別：保留所有 (y != -1)。子類別覆寫以達到 positive/negative 過濾。"""
        return (y != -1)

    # ---------------- other loaders ----------------
    def load_from_X_y(self, X, y, meta_df=None):
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            self.X = torch.tensor(np.asarray(X), dtype=torch.float32)

        if isinstance(y, torch.Tensor):
            self.y = y.float()
        else:
            self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

        self.meta_data = meta_df.copy() if isinstance(meta_df, pd.DataFrame) else pd.DataFrame()

    # ---------------- Dataset API ----------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 訓練/驗證沿用 (X, y) 的介面；meta 用 self.meta_data 另外讀
        return self.X[idx], self.y[idx]


class PositiveInverterTimeSeriesDataset_metadata(InverterTimeSeriesDataset_metadata):
    """只保留 y==1 的 window，並同步維護 meta_data"""
    def _keep_example(self, y):
        return (y == 1)


class NegativeInverterTimeSeriesDataset_metadata(InverterTimeSeriesDataset_metadata):
    """只保留 y==0 的 window，並同步維護 meta_data"""
    def _keep_example(self, y):
        return (y == 0)


def combine_dataset_metadata(datasets):
    """
    將多個 InverterTimeSeriesDataset 合併，包含 meta_data。
    """
    for ds in datasets:
        if not isinstance(ds, InverterTimeSeriesDataset_metadata):
            raise ValueError("All inputs must be InverterTimeSeriesDataset_metadata (or subclasses).")

    X = np.concatenate([ds.X.cpu().numpy() for ds in datasets], axis=0) if datasets else np.empty((0,))
    y = np.concatenate([ds.y.cpu().numpy() for ds in datasets], axis=0) if datasets else np.empty((0,))
    meta = pd.concat([ds.meta_data for ds in datasets], ignore_index=True) if datasets else pd.DataFrame()

    return InverterTimeSeriesDataset_metadata.from_X_y(X, y, meta_df=meta)
