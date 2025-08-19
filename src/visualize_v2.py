# --- 覆蓋 src/visualize_v2.py 內的函式 ---

import os
from typing import Optional, List
import pandas as pd
import plotly.graph_objects as go

def _ensure_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _safe_name(x: str) -> str:
    return "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(x)])

def plot_probs_per_device_interactive(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    device_col: str = "device_name",
    prob_col: str = "y_prob",
    threshold: float | None = 0.5,
    resample_rule: str | None = None,
    output_dir: str = "plots_prob_interactive",

    # sessions（同一個 DataFrame, 用旗標分兩類）
    failure_sessions_df: Optional[pd.DataFrame] = None,
    fs_device_col: str = "device_name",
    fs_start_col: str = "start_time",
    fs_end_col: str = "end_time",
    fs_flag_col: str = "maintenance",   # True=maintenance, False=failure
    fs_min_duration: str = "10min",

    # 顏色（可調）
    failure_fill: str = "rgba(120,46,139,0.22)",   # 紫色
    failure_edge: str = "rgba(120,46,139,0.6)",
    maint_fill:   str = "rgba(255,105,180,0.22)",  # 粉紅
    maint_edge:   str = "rgba(255,105,180,0.6)",

    # 顯示範圍
    limit_to_data_range: bool = True,
    y_fixed_0_to_1: bool = True,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    # 檢查基礎欄位
    for c in (time_col, device_col, prob_col):
        if c not in df.columns:
            raise KeyError(f"df 缺少欄位：{c}")

    data = df.copy()
    data[time_col] = _ensure_datetime_utc(data[time_col])
    data = data.dropna(subset=[time_col, device_col, prob_col]).sort_values([device_col, time_col])

    if resample_rule:
        parts = []
        for dev, sub in data.groupby(device_col):
            s = sub.set_index(time_col).sort_index()
            r = s[prob_col].resample(resample_rule).mean().to_frame(prob_col)
            r[device_col] = dev
            r = r.reset_index().rename(columns={time_col: time_col})
            parts.append(r)
        data = (pd.concat(parts, ignore_index=True)
                  .dropna(subset=[prob_col])
                  .sort_values([device_col, time_col]))

    # sessions 前處理
    fs = None
    if failure_sessions_df is not None and len(failure_sessions_df) > 0:
        fs = failure_sessions_df.rename(
            columns={fs_device_col: "_dev", fs_start_col: "_start", fs_end_col: "_end"}
        ).copy()
        fs["_start"] = _ensure_datetime_utc(fs["_start"])
        fs["_end"]   = _ensure_datetime_utc(fs["_end"])
        fs = fs.dropna(subset=["_dev", "_start", "_end"])
        fs = fs[fs["_end"] > fs["_start"]]

        # 最短顯示長度
        min_td = pd.to_timedelta(fs_min_duration)
        fs = fs[(fs["_end"] - fs["_start"]) >= min_td]

        # 保留 maintenance 旗標（若沒有就全部視為 failure）
        if fs_flag_col in failure_sessions_df.columns:
            fs["_maint"] = failure_sessions_df[fs_flag_col].values
        else:
            fs["_maint"] = False

    out_paths: List[str] = []

    for dev, sub in data.groupby(device_col):
        sub = sub.sort_values(time_col)
        if sub.empty:
            continue

        x_min = sub[time_col].min()
        x_max = sub[time_col].max()

        fig = go.Figure()
        # 機率線
        fig.add_trace(go.Scatter(x=sub[time_col], y=sub[prob_col], mode="lines", name="Probability"))

        # 門檻線
        if threshold is not None:
            fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"threshold={threshold}")

        # 疊區段
        if fs is not None:
            fs_dev = fs.loc[fs["_dev"] == dev].copy()
            if not fs_dev.empty:
                if limit_to_data_range:
                    fs_dev = fs_dev[(fs_dev["_end"] >= x_min) & (fs_dev["_start"] <= x_max)].copy()

                # 分類：maintenance / failure
                for is_maint, group in fs_dev.groupby("_maint"):
                    fill = maint_fill if is_maint else failure_fill
                    edge = maint_edge if is_maint else failure_edge
                    label = "Maintenance" if is_maint else "Failure Session"
                    for _, r in group.iterrows():
                        s = max(r["_start"], x_min) if limit_to_data_range else r["_start"]
                        e = min(r["_end"],   x_max) if limit_to_data_range else r["_end"]
                        if e <= s:
                            continue
                        fig.add_vrect(
                            x0=s, x1=e,
                            fillcolor=fill,
                            line=dict(color=edge, width=1),
                            annotation_text=label,
                            annotation_position="top left"
                        )

        # 版面
        fig.update_layout(
            title=f"{dev} — probability over time",
            xaxis_title="Time",
            yaxis_title="Probability",
            xaxis=dict(rangeslider=dict(visible=True),
                       rangeselector=dict(buttons=[
                           dict(count=7, step="day",   stepmode="backward", label="1w"),
                           dict(count=1, step="month", stepmode="backward", label="1m"),
                           dict(count=3, step="month", stepmode="backward", label="3m"),
                           dict(count=6, step="month", stepmode="backward", label="6m"),
                           dict(count=1, step="year",  stepmode="backward", label="1y"),
                           dict(step="all")
                       ])),
        )
        if y_fixed_0_to_1:
            fig.update_yaxes(range=[0, 1])

        # 輸出
        out_path = os.path.join(output_dir, f"device_{_safe_name(dev)}.html")
        fig.write_html(out_path)
        out_paths.append(out_path)

    return out_paths
