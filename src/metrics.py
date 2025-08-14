import numpy as np

def cal_topK_metrics(logits, labels, top_k=(1, 5)):
    """
    計算 Top-K 準確率
    """
    order = np.argsort(-logits)
    y_true_sorted = labels[order]
    total_pos = int((labels == 1).sum())
    topk_metrics = {}
    for k in top_k:
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

    return topk_metrics