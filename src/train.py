import torch
from torch.utils.data import DataLoader
import time
import os
import pandas as pd

def train_loop(model,
               train_loader: DataLoader,
               val_loader: DataLoader = None,
               device='cuda',
               optimizer=None,
               criterion=None,
               num_epochs=10,
               scheduler=None,
               log_interval=100,
               save_interval=1,
               save_path=None):
    """
    通用 PyTorch 訓練迴圈，適用 CNN+LSTM 二分類模型
    """

    model = model.to(device)
    print(f"Model moved to {device}")
    if save_path is None:
        save_path = time.strftime('model/%m%d_%H%M/', time.localtime())
    os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(save_path+'/training_log.csv'):
        log = pd.read_csv(save_path+'/training_log.csv')
        cur_epoch = int(log['epoch'].max() + 1)
        min_val_loss = log['val_loss'].min()
        print(f"Resuming training from epoch {cur_epoch}")
    else:
        log = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 'time'])
        cur_epoch = 1
        min_val_loss = float('inf')

    for epoch in range(cur_epoch, cur_epoch + num_epochs):
        start_time = time.time()
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
        end_time = time.time()

        if (epoch) % save_interval == 0:
            torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
            log.to_csv(f'{save_path}/training_log.csv', index=False)
            #print(f"Model saved at epoch {epoch}")

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

                    pred = torch.sigmoid(output) > 0.5
                    correct += (pred == y_val).sum().item()
                    total += y_val.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"✅ Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%}")
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{save_path}/best_model.pth')
                print(f"Best model saved at epoch {epoch} with loss {avg_val_loss:.4f}")

        # scheduler step if used
        if scheduler is not None:
            scheduler.step()

        log.loc[len(log)] = [epoch, avg_train_loss, avg_val_loss if val_loader else None, accuracy if val_loader else None, end_time - start_time]

    log.to_csv(f'{save_path}/training_log.csv', index=False)
    torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
    print("🏁 Training completed.")
    
    return log
    
def test_loop(model,
            test_loader: DataLoader,
            best_threshold=0.5,
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
    outputs = []
    predictions = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device).float()
            output = model(X_test).squeeze()
            outputs.append(torch.sigmoid(output).cpu().numpy())

            loss = criterion(output, y_test)
            total_loss += loss.item()

            pred = torch.sigmoid(output) > best_threshold  # 二分類預測
            trues.append(y_test.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            correct += (pred == y_test).sum().item()
            total += y_test.size(0)

    avg_test_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f"🔍 Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2%}")
    return np.concatenate(trues), np.concatenate(predictions), np.concatenate(outputs)