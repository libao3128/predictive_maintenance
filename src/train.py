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
                print(f"[Epoch {epoch}/{cur_epoch + num_epochs}] Step {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"🔁 Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}")
        end_time = time.time()

        if (epoch) % save_interval == 0:
            torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
            log.to_csv(f'{save_path}/training_log.csv', index=False)
            #print(f"Model saved at epoch {epoch}")

        # ===== 驗證階段 =====
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            y_true_val_list, y_score_val_list = [], []

            # 你可以自訂多個 K；例如抓最高分的 50/100/200 筆
            topk_list = [50, 100, 200]

            with torch.inference_mode():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device, non_blocking=True)
                    y_val = y_val.float().view(-1).to(device, non_blocking=True)

                    # 若資料含 -1（in-session），先過濾掉
                    valid_mask = (y_val == 0) | (y_val == 1)
                    if valid_mask.sum() == 0:
                        continue
                    
                    X_val = X_val[valid_mask]
                    y_val = y_val[valid_mask]

                    logits = model(X_val).view(-1)
                    val_loss += criterion(logits, y_val).item()

                    probs = torch.sigmoid(logits)  # [B]

                    # 收集到 CPU 做整體指標
                    y_true_val_list.append(y_val.detach().cpu().numpy().astype(np.int32))
                    y_score_val_list.append(probs.detach().cpu().numpy())

                    # Accuracy（僅供參考）
                    pred = (probs > 0.5).int()
                    correct += (pred == y_val.int()).sum().item()
                    total += y_val.size(0)

            if total > 0:
                avg_val_loss = val_loss / len(val_loader)
                accuracy = correct / total

                y_true_val = np.concatenate(y_true_val_list).reshape(-1)
                y_score_val = np.concatenate(y_score_val_list).reshape(-1)

                # ---- 主指標：AUCPR 與 baseline ----
                pos_rate = float((y_true_val == 1).mean())  # baseline（random classifier）
                if pos_rate == 0.0:
                    aucpr = float('nan')
                    ap_uplift = float('nan')
                else:
                    aucpr = average_precision_score(y_true_val, y_score_val)
                    ap_uplift = aucpr / pos_rate

                # ---- Top-K 指標 ----
                # 先依分數排序（由高到低）
                order = np.argsort(-y_score_val)
                y_true_sorted = y_true_val[order]

                total_pos = int((y_true_val == 1).sum())
                topk_metrics = {}
                for k in topk_list:
                    k_eff = min(k, len(y_true_sorted))
                    if k_eff == 0:
                        p_at_k = float('nan')
                        r_at_k = float('nan')
                    else:
                        tp_at_k = int(y_true_sorted[:k_eff].sum())
                        p_at_k = tp_at_k / k_eff
                        r_at_k = (tp_at_k / total_pos) if total_pos > 0 else float('nan')
                    topk_metrics[f'prec@{k}'] = p_at_k
                    topk_metrics[f'rec@{k}']  = r_at_k

                # ---- 印出摘要 ----
                k_str = " | ".join([f"P@{k}:{topk_metrics[f'prec@{k}']:.3f} R@{k}:{topk_metrics[f'rec@{k}']:.3f}" for k in topk_list])
                print(
                    f"✅ Validation Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2%} | "
                    f"AUC-PR: {aucpr:.4f} | baseline: {pos_rate:.4f} | uplift: {ap_uplift:.2f}x | {k_str}"
                )

                # ---- 儲存 best（仍以 val_loss 為準；你也可改成以 aucpr 為準）----
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    torch.save(model.state_dict(), f'{save_path}/best_model.pth')
                    print(f"Best model saved at epoch {epoch} with loss {avg_val_loss:.4f}")

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
                    'accuracy': accuracy,
                    'baseline_pos_rate': pos_rate,
                    'ap_uplift': ap_uplift,
                    'time': end_time - start_time
                }
                for k in topk_list:
                    row[f'prec@{k}'] = topk_metrics[f'prec@{k}']
                    row[f'rec@{k}']  = topk_metrics[f'rec@{k}']
                log.loc[len(log)] = row

            else:
                print("⚠️ No valid samples in validation after filtering labels.")
                # 也把空值寫進 log 以保持 epoch 對齊
                if 'baseline_pos_rate' not in log.columns:
                    log['baseline_pos_rate'] = np.nan
                    log['ap_uplift'] = np.nan
                log.loc[len(log)] = [epoch, avg_train_loss, np.nan, np.nan, np.nan, end_time - start_time]

        # scheduler step if used
        if scheduler is not None:
            scheduler.step()


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