import os
from glob import glob
from typing import List
import numpy as np
import pandas as pd



def load_parquet_data(parquet_dir: str) -> pd.DataFrame:
    paths = glob(os.path.join(parquet_dir, '*.parquet'))
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df['event_local_time'] = pd.to_datetime(df['event_local_time'])
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
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
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


def exclude_periods_from_data(df, exclude_periods):
    inverter_data = df.copy()
    for start, end in exclude_periods:
        inverter_data = inverter_data[~((inverter_data['event_local_time'].dt.floor('D') >= start) & (inverter_data['event_local_time'].dt.floor('D') <= end))]
    return inverter_data

def train_test_split_on_time(df: pd.DataFrame, test_size: float = 0.2, time_col: str = 'event_local_time') -> tuple:
    """
    Split the DataFrame into training and testing sets based on time.
    """
    df = df.sort_values(time_col)
    n = len(df)
    test_n = int(n * test_size)
    train_df = df[:-test_n]
    test_df = df[-test_n:]
    print(f"Train set size: {len(train_df)} Train set time range: {train_df['event_local_time'].min()} to {train_df['event_local_time'].max()}")
    print(f"Test set size: {len(test_df)} Test set time range: {test_df['event_local_time'].min()} to {test_df['event_local_time'].max()}")
    return train_df, test_df

def missing_value_imputation(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "event_local_time",
    device_col: str = "device_name",
    short_gap_limit: int = 6,   # 5 分鐘資料 -> 6 筆 ≈ 30 分鐘內用插值
    long_fill_value: float = 0.0,
    add_missing_mask: bool = True,
) -> pd.DataFrame:
    """
    針對多裝置時間序列做缺失補值：
      1) 先產生 per-step 缺失 mask（可選）
      2) 每個裝置內，以時間排序後對 feature 做「時間型插值」(limit=short_gap_limit)
      3) 尚未補到的長缺失以指定值（預設 0）補齊

    參數：
      - df: 原始 DataFrame，需包含 time_col 與 device_col
      - feature_cols: 要補值的數值欄位
      - time_col: 時間欄位名稱（需可轉為 datetime）
      - device_col: 裝置欄位名稱
      - short_gap_limit: 連續缺失筆數在此上限以內使用插值
      - long_fill_value: 插值後仍為 NaN 的長缺失以此值補
      - add_missing_mask: 是否為每個 feature 產生 *_missing 的 0/1 mask 欄位

    回傳：
      - 完成補值與（可選）新增 mask 的 DataFrame
    """
    imputed_df = df.copy()

    # 確保時間欄為 datetime
    imputed_df[time_col] = pd.to_datetime(imputed_df[time_col], errors="coerce")

    # 需要的欄位存在性檢查
    missing_cols = [c for c in [time_col, device_col] + feature_cols if c not in imputed_df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}")

    for device, device_data in imputed_df.groupby(device_col, sort=False):
        # 複製避免 SettingWithCopy
        block = device_data.loc[:, [time_col, device_col] + feature_cols].copy()
        # 記住原始索引以便放回
        block["_orig_idx"] = block.index

        # 產生 per-step 缺失 mask（基於原始缺失）
        if add_missing_mask:
            for col in feature_cols:
                imputed_df.loc[block["_orig_idx"], f"{col}_missing"] = block[col].isna().astype(int)

        # 依時間排序並以時間為索引做 time-based interpolate
        block = block.sort_values(time_col)
        block = block.set_index(time_col)

        # 僅對目標特徵做處理
        # 短缺失：時間插值（雙向皆可，避免前段或尾段全 NaN 無法補）
        block[feature_cols] = block[feature_cols].interpolate(
            method="time", limit=short_gap_limit
        ).interpolate(method="time", limit_direction="both")

        # 長缺失：仍為 NaN 的以指定值補齊
        block[feature_cols] = block[feature_cols].fillna(long_fill_value)

        # 還原索引與順序
        block = block.reset_index()
        block = block.set_index("_orig_idx").sort_index()

        # 寫回 imputed_df（僅覆蓋目標特徵欄）
        imputed_df.loc[block.index, feature_cols] = block[feature_cols].values

    return imputed_df
