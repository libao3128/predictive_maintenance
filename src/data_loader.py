import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class InverterTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, feature_cols, label_col='label', window_size=30, stride=1):
        """
        dataframe: 預先標記過的 DataFrame，需包含時間順序、特徵欄與 label 欄
        feature_cols: 要當作 input features 的欄位清單
        label_col: 對應的 target 欄位（預設為 'label'）
        window_size: 每筆樣本的時間序列長度
        stride: 每幾步擷取一次 window（預設為1）
        """
        self.data = dataframe.sort_values(['device_name', 'event_local_time']).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        
        # check if feature_cols and label_col exist in dataframe
        for col in feature_cols + [label_col]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if dataframe.isnull().values.any():
            raise ValueError("DataFrame contains NaN values. Please clean the data before creating the dataset.")

        # 依 device 分開處理 sliding window
        for device, group in tqdm(self.data.groupby('device_name'), desc="Processing devices"):
            values = group[feature_cols].values
            labels = group[label_col].values
            times = group['event_local_time']

            for i in range(0, len(group) - window_size, stride):
                # 檢查 window 內的時間是否連續
                window_times = times[i:i+window_size]
                # 假設 event_local_time 是 pandas.Timestamp 或 datetime
                time_diffs = window_times.diff().dropna().dt.total_seconds().values
                # 檢查是否所有時間間隔都相同
                if not (time_diffs==time_diffs[0]).all():
                    continue

                window_X = values[i:i+window_size]
                window_y = labels[i+window_size-1]  # 預測最後一點的 label
                if window_y == -1:
                    # is in failure session, skip
                    continue
                assert window_X.shape == (window_size, len(feature_cols)), f"Window shape mismatch: {window_X.shape} != ({window_size}, {len(feature_cols)})"
                self.samples.append((window_X, window_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
