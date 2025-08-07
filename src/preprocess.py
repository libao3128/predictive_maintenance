import os
from glob import glob

import numpy as np
import pandas as pd



def load_parquet_data(parquet_dir: str) -> pd.DataFrame:
    paths = glob(os.path.join(parquet_dir, '*.parquet'))
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(paths)} parquet files → {df.shape[0]} rows")
    return df


def load_failure_sessions(csv_path: str, min_days: int = 3) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=['start_time', 'end_time'],
        dtype={'device_name': str}
    )
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
    df['duration'] = pd.to_timedelta(df['duration'])
    df = df[df['duration'] > pd.Timedelta(days=min_days)]
    print(f"Kept {len(df)} sessions longer than {min_days} days")
    return df


def label_pre_failure_and_drop(inv_grp, sess_grp, pre_days=5):
    """
    inv_grp: DataFrame for a single device, sorted by event_local_time
    sess_grp: DataFrame of failure sessions for that device, sorted by start_time
    returns inv_grp filtered (real failures dropped) with new column 'failure_label'
    """
    # 1) arrays of times
    et = inv_grp['event_local_time'].values.astype('datetime64[ns]')
    starts = sess_grp['start_time'].values.astype('datetime64[ns]')
    ends   = sess_grp['end_time'].values.astype('datetime64[ns]')

    n = len(et)
    labels = np.zeros(n, dtype=np.int8)

    if len(starts)>0:
        # 2) for each event, find index of next session start > event
        idx_next = np.searchsorted(starts, et, side='right')

        # 3) pre-failure: event in [start-5d, start)
        window_starts = starts - np.timedelta64(pre_days, 'D')
        valid_next = idx_next < len(starts)
        idxn = idx_next[valid_next]
        mask_pre = np.zeros(n, bool)
        mask_pre[valid_next] = (
            (et[valid_next] >= window_starts[idxn]) &
            (et[valid_next] <  starts[idxn])
        )
        labels[mask_pre] = 1

        # 4) drop real failures: find most recent session start ≤ event
        idx_prev = idx_next - 1
        valid_prev = idx_prev >= 0
        idxp = idx_prev[valid_prev]
        mask_fail = np.zeros(n, bool)
        mask_fail[valid_prev] = (et[valid_prev] <= ends[idxp])

        # 5) apply drop
        keep = ~mask_fail
        inv_grp = inv_grp.iloc[keep].copy()
        labels = labels[keep]
    else:
        # no sessions: everyone is label 0, keep all
        inv_grp = inv_grp.copy()

    inv_grp['label'] = labels
    return inv_grp


def prepare_dataset(inverter_df: pd.DataFrame,
                    failure_sessions: pd.DataFrame,
                    pre_days: int = 5) -> pd.DataFrame:
    # run label_pre_failure_and_drop over each device
    frames = []
    for dev, grp in inverter_df.groupby('device_name', sort=False):
        grp = grp.sort_values('event_local_time')
        sess = failure_sessions.loc[failure_sessions['device_name'] == dev]\
                              .sort_values('start_time')
        frames.append(label_pre_failure_and_drop(grp, sess, pre_days))
    labeled_df = pd.concat(frames, ignore_index=True)
    print("Total pre-failure rows:", labeled_df['label'].sum())
    print("Total rows:", labeled_df.shape[0])
    labeled_df = labeled_df.rename(columns={'failure_label':'label'})
    return labeled_df



