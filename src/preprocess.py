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

    
    if 'maintenance' in df.columns:
        df['maintenance'] = df['maintenance'].fillna(False).astype(bool)
    else:
        df['maintenance'] = False

    df = df[df['duration'] > pd.Timedelta(days=min_days)]
    print(f"Kept {len(df)} sessions longer than {min_days} days")
    return df


def label_pre_failure_and_drop(inv_grp, sess_grp, pre_days=5):
    """
    - Pre-failure labels (1) come only from NON-maintenance sessions.
    - Rows inside ANY session window (maintenance or not) are dropped.
    """
    inv_grp = inv_grp.sort_values('event_local_time')
    et = inv_grp['event_local_time'].values.astype('datetime64[ns]')

    if sess_grp.empty:
        out = inv_grp.copy()
        out['label'] = 0
        return out

    sess_grp = sess_grp.sort_values('start_time')
    starts_all = sess_grp['start_time'].values.astype('datetime64[ns]')
    ends_all   = sess_grp['end_time'].values.astype('datetime64[ns]')
    maint      = sess_grp['maintenance'].astype(bool).values if 'maintenance' in sess_grp.columns \
                 else np.zeros(len(sess_grp), dtype=bool)

    # ---- Labeling: only NON-maintenance sessions create pre-failure windows
    nm_mask   = ~maint
    starts_nm = starts_all[nm_mask]

    n = len(et)
    labels = np.zeros(n, dtype=np.int8)

    if starts_nm.size > 0:
        idx_next_nm   = np.searchsorted(starts_nm, et, side='right')        # next non-maint start idx
        valid_next_nm = idx_next_nm < starts_nm.size
        idxn          = idx_next_nm[valid_next_nm]
        window_starts = starts_nm - np.timedelta64(pre_days, 'D')

        mask_pre = np.zeros(n, dtype=bool)
        mask_pre[valid_next_nm] = (
            (et[valid_next_nm] >= window_starts[idxn]) &
            (et[valid_next_nm] <  starts_nm[idxn])
        )
        labels[mask_pre] = 1

    # ---- Dropping: remove rows that lie in ANY session (maintenance or not)
    idx_next_all   = np.searchsorted(starts_all, et, side='right')
    idx_prev_all   = idx_next_all - 1
    valid_prev_all = idx_prev_all >= 0

    mask_in_session = np.zeros(n, dtype=bool)
    # inside if previous session has started and hasn't ended yet
    mask_in_session[valid_prev_all] = (et[valid_prev_all] <= ends_all[idx_prev_all[valid_prev_all]])

    keep = ~mask_in_session
    inv_grp = inv_grp.iloc[keep].copy()
    inv_grp['label'] = labels[keep]
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
    short_gap_limit: int = 6,   # 5-minute data -> 6 records ≈ 30 minutes for interpolation
    long_fill_value: float = 0.0,
    add_missing_mask: bool = True,
) -> pd.DataFrame:
    """
    Missing value imputation for multi-device time series:
      1) First generate per-step missing mask (optional)
      2) Within each device, sort by time and perform "time-based interpolation" on features (limit=short_gap_limit)
      3) Fill remaining long gaps with specified value (default 0)

    Parameters:
      - df: Original DataFrame, must contain time_col and device_col
      - feature_cols: Numerical columns to impute
      - time_col: Time column name (must be convertible to datetime)
      - device_col: Device column name
      - short_gap_limit: Use interpolation for consecutive missing records within this limit
      - long_fill_value: Fill long gaps that remain NaN after interpolation with this value
      - add_missing_mask: Whether to generate 0/1 mask columns *_missing for each feature

    Returns:
      - DataFrame with completed imputation and (optionally) new mask columns
    """
    imputed_df = df.copy()

    # Ensure time column is datetime
    imputed_df[time_col] = pd.to_datetime(imputed_df[time_col], errors="coerce")

    # Check existence of required columns
    missing_cols = [c for c in [time_col, device_col] + feature_cols if c not in imputed_df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}")

    for device, device_data in imputed_df.groupby(device_col, sort=False):
        # Copy to avoid SettingWithCopy
        block = device_data.loc[:, [time_col, device_col] + feature_cols].copy()
        # Remember original index for restoration
        block["_orig_idx"] = block.index

        # Generate per-step missing mask (based on original missing values)
        if add_missing_mask:
            for col in feature_cols:
                imputed_df.loc[block["_orig_idx"], f"{col}_missing"] = block[col].isna().astype(int)

        # Sort by time and use time as index for time-based interpolation
        block = block.sort_values(time_col)
        block = block.set_index(time_col)

        # Only process target features
        # Short gaps: time interpolation (bidirectional, avoid front/end segments with all NaN)
        if short_gap_limit > 0:
            block[feature_cols] = block[feature_cols].interpolate(
                method="time", limit=short_gap_limit, limit_direction="forward"
            )

        # Long gaps: fill remaining NaN with specified value
        block[feature_cols] = block[feature_cols].fillna(long_fill_value)

        # Restore index and order
        block = block.reset_index()
        block = block.set_index("_orig_idx").sort_index()

        # Write back to imputed_df (only overwrite target feature columns)
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
    Downsample original 5-min data based on column semantics (without recreating derived features).
    Rules:
      - Continuous variables → mean
      - Boolean/connection/heartbeat/status/WORD → max
      - Cumulative variables (ENERGY_*, VARH_*) → delta (default) / last / mean
      - Setpoint/HW_VERSION → last
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} has invalid time, please clean first.")

    # ==== Column classification (by naming rules) ====
    cols: List[str] = [c for c in df.columns if c not in (time_col, device_col)]

    # Possible "cumulative" columns (energy, varh)
    cumulative_cols = [c for c in cols if any([
        c.startswith("metric.ENERGY_") and c.endswith(".MEASURED"),
        c.startswith("metric.VARH_")   and c.endswith(".MEASURED"),
        c == "metric.ENERGY_DELIVERED.MEASURED",
        c == "metric.ENERGY_RECEIVED.MEASURED",
        c == "metric.VARH_DELIVERED.MEASURED"
    ])]

    # Status/error code/WORD/boolean flag types (including COMM_LINK, HEARTBEAT)
    state_like_cols = [c for c in cols if (
        c.startswith("metric.STATUS_") or
        c.endswith("WORD.MEASURED") or
        c in ["metric.COMM_LINK.MEASURED", "metric.HEARTBEAT.MEASURED"]
    )]

    # Setpoint / version
    last_pref_cols = [c for c in cols if (
        c.endswith("_SETPOINT.MEASURED") or
        c == "metric.HW_VERSION.MEASURED"
    )]

    # Others treated as continuous variables (voltage/current/power/frequency/temperature...)
    assigned = set(cumulative_cols) | set(state_like_cols) | set(last_pref_cols)
    continuous_mean_cols = [c for c in cols if c not in assigned]

    # ==== Aggregation function definitions ====
    def agg_cumulative(s: pd.Series) -> float:
        """Interval increment: last - first, handle reset/rollover as >=0"""
        if s.dropna().empty:
            return float("nan")
        first = s.iloc[0]
        last  = s.iloc[-1]
        return max(float(last) - float(first), 0.0)

    # Aggregation rules dictionary
    agg: Dict[str, object] = {}

    # Continuous variables → mean
    for c in continuous_mean_cols:
        agg[c] = "mean"

    # Status/WORD/boolean → max
    for c in state_like_cols:
        agg[c] = "max"

    # Setpoint/HW_VERSION → last
    for c in last_pref_cols:
        agg[c] = "last"
        
    # missing feature
    for c in df.columns:
        if c.endswith('_missing'):
            agg[c] = 'mean'

    # Cumulative variables → based on parameter
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
    # ==== Downsample by device ====
    for device, group in df.groupby(device_col, sort=False):
        # Sort by time
        group = group.sort_values(time_col)

        # Group by time and device, then perform resample aggregation
        group = group.set_index(time_col)
        resampled = group.groupby(device_col).resample(freq).agg(agg).reset_index()

        # Write back to original DataFrame
        if device_col not in resampled.columns:
            resampled[device_col] = device
        rs = pd.concat([rs, resampled], ignore_index=True)

    rs.reset_index(drop=True, inplace=True)

    # optional: drop rows where all "continuous variables" are NaN in that time window (usually means no data in window)
    if drop_empty_bins and continuous_mean_cols:
        mask_all_nan = rs[continuous_mean_cols].isna().all(axis=1)
        rs = rs.loc[~mask_all_nan].copy()

    # Column order (try to stay close to original)
    ordered = [time_col, device_col] + continuous_mean_cols + state_like_cols + last_pref_cols + cumulative_cols
    ordered = [c for c in ordered if c in rs.columns]
    rs = rs.loc[:, ordered].sort_values([device_col, time_col])

    return rs
