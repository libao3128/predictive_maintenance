import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def smooth_by_consecutive(
    bin_preds: np.ndarray,
    group_ids: np.ndarray | None = None,
    min_consecutive: int = 3,
    return_events: bool = True,
):
    """
    Keep only runs of 1's with length >= min_consecutive. Shorter runs are set to 0.
    If group_ids is provided, smoothing is applied within each group (e.g., per device).

    Parameters
    ----------
    bin_preds : array-like of shape (N,)
        Binary predictions per window (0/1) after thresholding.
    group_ids : array-like of shape (N,), optional
        Group key per window (e.g., device id or name). If None, treat all as one group.
    min_consecutive : int
        Minimum length (in windows) for a run of 1's to be kept.
    return_events : bool
        If True, also return lists of kept runs and filtered noise runs.

    Returns
    -------
    smoothed : np.ndarray
        Binary predictions after smoothing.
    kept_events : list[dict]
        Kept runs (only if return_events=True).
    noise_runs : list[dict]
        Runs that were removed as noise (only if return_events=True).
    """
    b = np.asarray(bin_preds).astype(np.uint8)
    n = len(b)
    if group_ids is None:
        group_ids = np.zeros(n, dtype=np.int32)
    else:
        group_ids = np.asarray(group_ids)

    smoothed = np.zeros_like(b, dtype=np.uint8)
    kept_events, noise_runs = [], []

    # process each group independently
    for gid in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        if idx.size == 0:
            continue
        seg = b[idx]

        i = 0
        while i < len(seg):
            if seg[i] == 1:
                j = i
                while j < len(seg) and seg[j] == 1:
                    j += 1
                run_len = j - i
                if run_len >= min_consecutive:
                    smoothed[idx[i:j]] = 1
                    kept_events.append({
                        "group": gid,
                        "start_idx": int(idx[i]),
                        "end_idx": int(idx[j-1]),
                        "length": int(run_len),
                    })
                else:
                    noise_runs.append({
                        "group": gid,
                        "start_idx": int(idx[i]),
                        "end_idx": int(idx[j-1]),
                        "length": int(run_len),
                    })
                i = j
            else:
                i += 1

    if return_events:
        return smoothed, kept_events, noise_runs
    return smoothed

