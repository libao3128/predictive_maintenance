import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, num_features, cnn_out_channels=32, lstm_hidden_size=64, lstm_layers=1, dropout=0.3):
        super(CNNLSTMModel, self).__init__()

        # CNN 模塊：擷取時間窗口內的短期模式
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout)
        )

        # LSTM 模塊：學習時間依賴
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            #nn.Sigmoid()  # Output: 機率值 (0~1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)           # → (batch_size, num_features, seq_len)
        x = self.cnn(x)                  # → (batch_size, cnn_out_channels, seq_len)
        x = x.permute(0, 2, 1)           # → (batch_size, seq_len, cnn_out_channels)
        lstm_out, _ = self.lstm(x)       # → (batch_size, seq_len, lstm_hidden_size)
        last_time_step = lstm_out[:, -1, :]  # 取最後時間步的輸出
        out = self.classifier(last_time_step)  # → (batch_size, 1)
        return out
