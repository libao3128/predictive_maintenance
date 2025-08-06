import torch
from torch.utils.data import DataLoader

def train_loop(model,
               train_loader: DataLoader,
               val_loader: DataLoader = None,
               device='cuda',
               optimizer=None,
               criterion=None,
               num_epochs=10,
               time_step=30,
               scheduler=None,
               log_interval=100):
    """
    通用 PyTorch 訓練迴圈，適用 CNN+LSTM 二分類模型
    """

    model = model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device).float()

            optimizer.zero_grad()
            output = model(X)
            output = output.squeeze()  # [B] if needed

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch}/{num_epochs}] Step {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"🔁 Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}")

        # 驗證階段
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device).float()
                    output = model(X_val).squeeze()
                    val_loss += criterion(output, y_val).item()

                    pred = torch.sigmoid(output) > 0.5
                    correct += (pred == y_val).sum().item()
                    total += y_val.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"✅ Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%}")

        # scheduler step if used
        if scheduler is not None:
            scheduler.step()

    print("🏁 Training completed.")

if __name__ == "__main__":
    from .model import CNNLSTMModel as Model
    from .data_loader import InverterTimeSeriesDataset


    # 假設你有一個處理過的 dataframe：
    # - 已有時間順序
    # - 已補值/標準化
    # - 有 label 欄位
    # - 有 feature 欄位（你已選好）

    feature_cols = ['AC_POWER']  # 你要用的欄位，例如 ['AC_POWER', 'DC_VOLTAGE', ...]
    num_features = len(feature_cols)
    label_col = 'label'

    dataset = InverterTimeSeriesDataset(
        dataframe=processed_df,
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=30,
        stride=1
    )

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    model = Model(
        num_features=num_features,
        cnn_out_channels=32,
        lstm_hidden_size=64,
        lstm_layers=1,
        dropout=0.3
    )
    
    train_loop(
        model=model,
        train_loader=train_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        criterion=torch.nn.BCELoss(),
        num_epochs=10,
        log_interval=10
    )