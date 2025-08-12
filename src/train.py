import torch
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

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
        log = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'aucpr', 'accuracy', 'time'])
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
            y_true_val, y_score_val = [], []

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device, non_blocking=True)  # 建議加 non_blocking=True
                    y_val = y_val.to(device, non_blocking=True).float()
                    output = model(X_val).squeeze()
                    val_loss += criterion(output, y_val).item()

                    # 收集真實標籤和預測分數用於 AUC-PR 計算
                    y_true_val.append(y_val.detach().cpu().numpy())
                    y_score_val.append(torch.sigmoid(output).detach().cpu().numpy())

                    pred = torch.sigmoid(output) > 0.5
                    correct += (pred == y_val).sum().item()
                    total += y_val.size(0)

            # 計算驗證指標
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            # 計算 AUC-PR
            y_true_val = np.concatenate(y_true_val)
            y_score_val = np.concatenate(y_score_val)
            aucpr = average_precision_score(y_true_val, y_score_val)
            
            print(f"✅ Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%} | AUC-PR: {aucpr:.4f}")
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{save_path}/best_model.pth')
                print(f"Best model saved at epoch {epoch} with loss {avg_val_loss:.4f}")

        # scheduler step if used
        if scheduler is not None:
            scheduler.step()

        log.loc[len(log)] = [epoch, avg_train_loss, avg_val_loss if val_loader else None, aucpr if val_loader else None, accuracy if val_loader else None, end_time - start_time]

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
        for X_test, y_test in tqdm(test_loader, desc="Testing"):
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

def generate_report(trues, predictions, outputs):
    """
    評估模型性能，計算 AUC-PR 和其他指標
    """
    print(classification_report(trues, predictions , target_names=['Normal', 'Failure']))
    print(confusion_matrix(trues, predictions ))
    roc_auc = roc_auc_score(trues, outputs)
    print(f"ROC AUC: {roc_auc:.4f}")
    curve = roc_curve(trues, outputs)

    plt.figure(figsize=(8, 6))
    plt.plot(curve[0], curve[1], label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

def evaluate_model(model,
            test_loader: DataLoader,
            best_threshold=0.5,
            device='cuda',
            criterion=None):
    trues, predictions, outputs = test_loop(model, test_loader, best_threshold, device, criterion)
    generate_report(trues, predictions, outputs)