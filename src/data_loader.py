import torch
from torch.utils.data import Dataset

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

        # 依 device 分開處理 sliding window
        for device, group in self.data.groupby('device_name'):
            values = group[feature_cols].values
            labels = group[label_col].values

            for i in range(0, len(group) - window_size, stride):
                window_X = values[i:i+window_size]
                window_y = labels[i+window_size-1]  # 預測最後一點的 label
                self.samples.append((window_X, window_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
