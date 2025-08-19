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


@torch.inference_mode()
def test_loop_testonly(
    model,
    test_loader,
    *,
    threshold: float = 0.5,
    device: str = "cuda",
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
    trues, preds, scores = [], [], []
    records = []
    avg_test_loss = None

    for batch in tqdm(test_loader, desc="Testing"):
        # 這裡假設 test dataset 永遠回 (X, y, device_name, timestamp)
        X_test, y_test, dev_ids, ts_arr = batch

        X_test = X_test.to(device, non_blocking=True)
        y_test = y_test.to(device, non_blocking=True).float()

        logits = model(X_test).squeeze()
        prob = torch.sigmoid(logits)

        if criterion is not None:
            total_loss += criterion(logits, y_test).item()

        # 收集 numpy 版輸出
        y_np = y_test.detach().cpu().numpy().reshape(-1)
        p_np = prob.detach().cpu().numpy().reshape(-1)
        pred_np = (p_np >= threshold).astype(np.int32)

        trues.append(y_np)
        scores.append(p_np)
        preds.append(pred_np)

        # 組合 meta
        dev_ids = np.asarray(dev_ids)
        ts_np   = np.asarray(ts_arr)

        # 時間轉換（若是整數，預設當毫秒；若是字串/日期型，直接 to_datetime）
        if np.issubdtype(ts_np.dtype, np.integer):
            ts_dt = pd.to_datetime(ts_np, unit="ms", utc=True, errors="coerce")
        else:
            ts_dt = pd.to_datetime(ts_np, utc=True, errors="coerce")

        for d, t, yt, yp in zip(dev_ids, ts_dt, y_np, p_np):
            records.append({
                "device_name": str(d),
                "score_time": t,
                "y_true": float(yt),
                "y_score": float(yp),
            })

    # 聚合
    y_true  = np.concatenate(trues)  if trues  else np.array([])
    y_pred  = np.concatenate(preds)  if preds  else np.array([])
    y_score = np.concatenate(scores) if scores else np.array([])

    avg_test_loss = None
    if criterion is not None and len(test_loader) > 0:
        avg_test_loss = total_loss / len(test_loader)

    df_results = pd.DataFrame(
        records,
        columns=["device_name", "score_time", "y_true", "y_score"]
    )

    return y_true, y_pred, y_score, avg_test_loss, df_results
    
    
def generate_report_test_only(y_true, y_pred, y_score):
    """
    印出分類報告、混淆矩陣、ROC AUC，並畫 ROC 曲線
    """
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Failure']))
    print(confusion_matrix(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_score)
    print(f"ROC AUC: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    

#def _to_utc_timestamp(ts_batch):
#    """
#    ts_batch 可以是 torch.Tensor / np.ndarray / list：
#    - 若像 1736294400000 這種 13 位數（>=1e12），視為毫秒 --> unit='ms'
#    - 若像 1736294400    這種 10 位數（>=1e9 且 <1e12），視為秒   --> unit='s'
#    轉成 tz-aware (UTC) 的 pandas.Timestamp 陣列。
#    """
#    if isinstance(ts_batch, torch.Tensor):
#        ts_arr = ts_batch.detach().cpu().numpy()
#    else:
#        ts_arr = np.asarray(ts_batch)
#
#    ts_arr = ts_arr.astype("int64")
#
#    # 判斷單位
#    max_abs = np.nanmax(np.abs(ts_arr)) if ts_arr.size else 0
#    if max_abs >= 1e12:
#        unit = "ms"
#    elif max_abs >= 1e9:
#        unit = "s"
#    else:
#        # 其他單位就照毫秒處理（依你的資料情況而定）
#        unit = "ms"
#
#    return pd.to_datetime(ts_arr, unit=unit, utc=True, errors="coerce")

def get_logits_and_labels_for_test(model, dataloader, device: str = "cuda"):
    model = model.to(device)
    model.eval()

    all_logits, all_labels = [], []
    dev_names, ts_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            # 預期 batch 結構為 (X, y, device_name, timestamp)
            X, y, dev, ts = batch

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            logits = model(X).squeeze()
            logits_np = logits.detach().cpu().numpy().reshape(-1)
            labels_np = y.detach().cpu().numpy().reshape(-1)

            all_logits.append(logits_np)
            all_labels.append(labels_np)


            # 收集 meta
            dev_arr = np.asarray(dev).reshape(-1)
            ts_arr = np.asarray(ts).reshape(-1)

            # timestamp 轉 datetime（若是毫秒可用 unit='ms'）
            ts_arr = pd.to_datetime(ts_arr, unit='ms', utc=True, errors="coerce")

            B = logits_np.shape[0]
            if dev_arr.shape[0] != B:
                dev_arr = np.resize(dev_arr, B)
            if ts_arr.shape[0] != B:
                ts_arr = np.resize(ts_arr, B)

            dev_names.extend(dev_arr.astype(str).tolist())
            ts_list.extend(ts_arr.tolist())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])

    df_meta = pd.DataFrame({
        "device_name": dev_names,
        "timestamp": ts_list,
    })

    return logits, labels, df_meta
    
def evaluate_model_test_only(model,
            test_loader: DataLoader,
            best_threshold=0.5,
            device='cuda',
            criterion=None):
    
    y_true, y_pred, y_score, avg_loss = test_loop_testonly(
        model, test_loader, threshold=best_threshold, device=device, criterion=criterion
    )
    if avg_loss is not None:
        print(f"Test Loss: {avg_loss:.4f}")
    generate_report_test_only(y_true, y_pred, y_score)
