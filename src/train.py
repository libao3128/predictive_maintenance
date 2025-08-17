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
from src.metrics import cal_topK_metrics

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
        save_path = time.strftime('model/local/%m%d_%H%M/', time.localtime())
    os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(save_path+'/training_log.csv'):
        log = pd.read_csv(save_path+'/training_log.csv')
        cur_epoch = int(log['epoch'].max() + 1)
        max_aucpr = log['aucpr'].max()
        print(f"Resuming training from epoch {cur_epoch}")
    else:
        log = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'aucpr', 'accuracy', 'time'])
        cur_epoch = 1
        max_aucpr = float('-inf')

    for epoch in range(cur_epoch, cur_epoch + num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            #print(X.shape, y.shape)  # Debugging shape
            X = X.to(device, non_blocking=True)  # 建議加 non_blocking=True
            y = y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(X)
            output = output.squeeze(-1)  # [B] if needed

            loss = criterion(output, y)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  
            # scheduler step if used
            if scheduler is not None:
                scheduler.step()

            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch}/{cur_epoch + num_epochs - 1}] Step {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"🔁 Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}")
        end_time = time.time()


        # ===== 驗證階段 =====
        if val_loader is not None:
            y_true_val, y_score_val, avg_val_loss = test_loop(model, val_loader, device, criterion)

            # ---- 主指標：AUCPR 與 baseline ----
            pos_rate = float((y_true_val == 1).mean())  # baseline（random classifier）
            if pos_rate == 0.0:
                aucpr = float('nan')
                ap_uplift = float('nan')
            else:
                aucpr = average_precision_score(y_true_val, y_score_val)
                ap_uplift = aucpr / pos_rate

            # ---- Top-K 指標 ----
            # 你可以自訂多個 K；例如抓最高分的 50/100/200 筆
            topk_list = [50, 100, 200]
            top_k_metrics = cal_topK_metrics(y_score_val, y_true_val, top_k=topk_list)
            
            # ---- 印出摘要 ----
            k_str = " | ".join([f"P@{k}:{top_k_metrics[f'prec@{k}']:.3f} R@{k}:{top_k_metrics[f'rec@{k}']:.3f}" for k in topk_list])
            print(
                f"✅ avg_loss: {avg_val_loss:.4f} | AUC-PR: {aucpr:.4f} | baseline: {pos_rate:.4f} | uplift: {ap_uplift:.2f}x | {k_str}"
            )
            # ---- 儲存 best（仍以 val_loss 為準；你也可改成以 aucpr 為準）----
            if aucpr > max_aucpr:
                max_aucpr = aucpr
                torch.save(model.state_dict(), f'{save_path}/best_model.pth')
                print(f"Best model saved at epoch {epoch} with AUC-PR {aucpr:.4f}")
            # ---- 將新指標寫入 log ----
            # 動態補上新欄位（若第一次寫入）
            required_cols = ['epoch', 'train_loss', 'val_loss', 'aucpr', 'accuracy', 'baseline_pos_rate', 'ap_uplift', 'time']
            for k in topk_list:
                required_cols += [f'prec@{k}', f'rec@{k}']
            for col in required_cols:
                if col not in log.columns:
                    log[col] = np.nan  # 先補欄位
            row = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'aucpr': aucpr,
                'baseline_pos_rate': pos_rate,
                'ap_uplift': ap_uplift,
                'time': end_time - start_time
            }
            for k in topk_list:
                row[f'prec@{k}'] = top_k_metrics[f'prec@{k}']
                row[f'rec@{k}']  = top_k_metrics[f'rec@{k}']
            log.loc[len(log)] = row

        if (epoch) % save_interval == 0:
            torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
            log.to_csv(f'{save_path}/training_log.csv', index=False)
            #print(f"Model saved at epoch {epoch}")
            
    log.to_csv(f'{save_path}/training_log.csv', index=False)
    torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
    print("🏁 Training completed.")
    
    return log

def test_loop(model,
            test_loader: DataLoader,
            device='cuda',
            criterion=None):
    """
    Runs inference on the test_loader and computes average loss and prediction scores.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test/validation set.
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
        criterion (callable, optional): Loss function. Defaults to None.

    Returns:
        tuple: (np.ndarray of true labels, np.ndarray of predicted scores, float average loss)
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    trues, scores = [], []
    avg_test_loss = None

    with torch.inference_mode():
        for X_test, y_test in tqdm(test_loader, desc="Testing"):
            X_test, y_test = X_test.to(device), y_test.to(device).float()
            logits = model(X_test).squeeze()
            if criterion is not None:
                total_loss += criterion(logits, y_test).item()
            trues.append(y_test.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())

    if criterion is not None:
        avg_test_loss = total_loss / len(test_loader)

    return np.concatenate(trues), np.concatenate(scores), avg_test_loss

def get_logits_and_labels(model, dataloader, device='cuda'):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)  # shape: (batch, 1)
            all_logits.append(logits.cpu().numpy().ravel())
            all_labels.append(y.cpu().numpy().ravel())
    return np.concatenate(all_logits), np.concatenate(all_labels)

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
