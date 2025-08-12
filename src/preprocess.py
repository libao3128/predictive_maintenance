import os
from glob import glob
import numpy as np
import pandas as pd
from typing import Dict, List



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
    is_maintenance = sess_grp['maintenance'].values

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
    inverter_data = inverter_data.reset_index(drop=True)
    print(f"Excluded {len(exclude_periods)} periods, remaining data size: {inverter_data.shape[0]}")
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
        if short_gap_limit > 0:
            block[feature_cols] = block[feature_cols].interpolate(
                method="time", limit=short_gap_limit, limit_direction="forward"
            )

        # 長缺失：仍為 NaN 的以指定值補齊
        block[feature_cols] = block[feature_cols].fillna(long_fill_value)

        # 還原索引與順序
        block = block.reset_index()
        block = block.set_index("_orig_idx").sort_index()

        # 寫回 imputed_df（僅覆蓋目標特徵欄）
        imputed_df.loc[block.index, feature_cols] = block[feature_cols].values

    return imputed_df

def downsample_inverter_raw(
    df: pd.DataFrame,
    freq: str = "30T",
    time_col: str = "event_local_time",
    device_col: str = "device_name",
    energy_as: str = "delta",   # "delta" | "last" | "mean"
    drop_empty_bins: bool = True
) -> pd.DataFrame:
    """
    依欄位語意對原始 5-min 資料做下採樣（不重造衍生特徵）。
    規則：
      - 連續量 → mean
      - 布林/連線/心跳/狀態/WORD → max
      - 累積量(ENERGY_*, VARH_*) → delta(預設) / last / mean
      - Setpoint/HW_VERSION → last
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} 有無效時間，請先清理。")

    # ==== 欄位分類（依名稱規則）====
    cols: List[str] = [c for c in df.columns if c not in (time_col, device_col)]

    # 可能的「累積量」欄位（energy, varh）
    cumulative_cols = [c for c in cols if any([
        c.startswith("metric.ENERGY_") and c.endswith(".MEASURED"),
        c.startswith("metric.VARH_")   and c.endswith(".MEASURED"),
        c == "metric.ENERGY_DELIVERED.MEASURED",
        c == "metric.ENERGY_RECEIVED.MEASURED",
        c == "metric.VARH_DELIVERED.MEASURED"
    ])]

    # 狀態/錯誤碼/WORD/布林旗標類（含 COMM_LINK、HEARTBEAT）
    state_like_cols = [c for c in cols if (
        c.startswith("metric.STATUS_") or
        c.endswith("WORD.MEASURED") or
        c in ["metric.COMM_LINK.MEASURED", "metric.HEARTBEAT.MEASURED"]
    )]

    # Setpoint / 版本
    last_pref_cols = [c for c in cols if (
        c.endswith("_SETPOINT.MEASURED") or
        c == "metric.HW_VERSION.MEASURED"
    )]

    # 其餘視為連續量（電壓/電流/功率/頻率/溫度...）
    assigned = set(cumulative_cols) | set(state_like_cols) | set(last_pref_cols)
    continuous_mean_cols = [c for c in cols if c not in assigned]

    # ==== 聚合函式定義 ====
    def agg_cumulative(s: pd.Series) -> float:
        """區間增量：last - first，處理重置/回捲為 >=0"""
        if s.dropna().empty:
            return float("nan")
        first = s.iloc[0]
        last  = s.iloc[-1]
        return max(float(last) - float(first), 0.0)

    # 聚合規則字典
    agg: Dict[str, object] = {}

    # 連續量 → mean
    for c in continuous_mean_cols:
        agg[c] = "mean"

    # 狀態/WORD/布林 → max
    for c in state_like_cols:
        agg[c] = "max"

    # Setpoint/HW_VERSION → last
    for c in last_pref_cols:
        agg[c] = "last"

    # 累積量 → 依參數
    if energy_as == "delta":
        for c in cumulative_cols:
            agg[c] = agg_cumulative
    elif energy_as == "last":
        for c in cumulative_cols:
            agg[c] = "last"
    elif energy_as == "mean":
        for c in cumulative_cols:
            agg[c] = "mean"
    else:
        raise ValueError("energy_as must be one of {'delta','last','mean'}")
    
    print(f"Downsampling {len(df)} rows using following method: ")
    print(f"{pd.DataFrame(agg.items(), columns=['Column', 'Aggregation'])}")
    
    
    rs = pd.DataFrame()
    # ==== 分裝置下採樣 ====
    for device, group in df.groupby(device_col, sort=False):
        # 依時間排序
        group = group.sort_values(time_col)

        # 依時間與裝置分組，並做 resample 聚合
        group = group.set_index(time_col)
        resampled = group.groupby(device_col).resample(freq).agg(agg).reset_index()

        # 寫回原始 DataFrame
        if device_col not in resampled.columns:
            resampled[device_col] = device
        rs = pd.concat([rs, resampled], ignore_index=True)

    rs.reset_index(drop=True, inplace=True)

    # optional: 丟掉該時間窗所有「連續量」皆 NaN 的列（通常代表窗內沒資料）
    if drop_empty_bins and continuous_mean_cols:
        mask_all_nan = rs[continuous_mean_cols].isna().all(axis=1)
        rs = rs.loc[~mask_all_nan].copy()

    # 欄位順序（盡量貼近原始）
    ordered = [time_col, device_col] + continuous_mean_cols + state_like_cols + last_pref_cols + cumulative_cols
    ordered = [c for c in ordered if c in rs.columns]
    rs = rs.loc[:, ordered].sort_values([device_col, time_col])

    return rs
