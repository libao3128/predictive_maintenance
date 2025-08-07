import torch
from torch.utils.data import DataLoader

def train_loop(model,
               train_loader: DataLoader,
               val_loader: DataLoader = None,
               device='cuda',
               optimizer=None,
               criterion=None,
               num_epochs=10,
               scheduler=None,
               log_interval=100):
    """
    通用 PyTorch 訓練迴圈，適用 CNN+LSTM 二分類模型
    """

    model = model.to(device)
    print(f"Model moved to {device}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            #print(X.shape, y.shape)  # Debugging shape
            X = X.to(device, non_blocking=True)  # 建議加 non_blocking=True
            y = y.to(device, non_blocking=True)
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
                    X_val = X_val.to(device, non_blocking=True)  # 建議加 non_blocking=True
                    y_val = y_val.to(device, non_blocking=True)
                    output = model(X_val).squeeze()
                    val_loss += criterion(output, y_val).item()

                    pred = output > 0.5
                    correct += (pred == y_val).sum().item()
                    total += y_val.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"✅ Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%}")

        # scheduler step if used
        if scheduler is not None:
            scheduler.step()

    print("🏁 Training completed.")
    
def test_loop(model,
            test_loader: DataLoader,
            device='cuda',
            criterion=None):
    """
    測試迴圈，用於評估模型在測試集上的表現
    """
    import numpy as np
    model = model.to(device)
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    trues = []
    predictions = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device).float()
            output = model(X_test).squeeze()

            loss = criterion(output, y_test)
            total_loss += loss.item()

            pred = output>0.5  # 二分類預測
            trues.append(y_test.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            correct += (pred == y_test).sum().item()
            total += y_test.size(0)

    avg_test_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f"🔍 Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%}")
    return np.concatenate(trues), np.concatenate(predictions)