import plotly.express as px
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def _plot_device_series(g: pd.DataFrame,
                        feature_cols,
                        device: str,
                        fs_by_dev: dict,
                        outdir: str,
                        title: str,
                        ts_col: str = "ts"):
    """單一裝置繪圖與輸出 HTML。"""
    fig = px.line(g, x=ts_col, y=feature_cols, title=f'{device} {title}')
    start_time, end_time = g[ts_col].min(), g[ts_col].max()

    for _, row in fs_by_dev.get(device, pd.DataFrame()).iterrows():
        if (row['end_time'] < start_time) or (row['start_time'] > end_time):
            continue
        color = "gray" if row.get('maintenance', False) else "red"
        session_id = row.get('session_id', '')
        annotation_text = f"Session: {session_id}" if session_id else "Failure Session"
        if row.get('maintenance', False):
            annotation_text += " (Maintenance)"
        fig.add_vrect(
            x0=row['start_time'],
            x1=row['end_time'],
            fillcolor=color,
            opacity=0.5,
            annotation_text=annotation_text,
            annotation_position="top left"
        )

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Mean Value',
        legend_title='Features',
        title_x=0.5
    )
    fig.write_html(f'{outdir}/{device}.html', full_html=False, include_plotlyjs='cdn')


def visualize_mean_values(inverter_data: pd.DataFrame,
                          failure_sessions: pd.DataFrame,
                          feature_cols,
                          folder_path: str = 'visualization',
                          title: str = 'Mean Values of Features',
                          time_col: str = 'event_local_time',
                          device_col: str = 'device_name',
                          freq: str | None = 'H',
                          workers: int = 8) -> str:
    """
    通用可視化：
      - freq=None => 直接以原始時間點繪圖 (相當於原本的 visualize_raw_mean_values)
      - freq='H'  => 依小時聚合平均 (相當於原本的 visualize_hourly_mean_values)
      - 也可傳入其他 pandas offset alias，如 '30T', 'D' 等

    回傳輸出資料夾路徑。
    """
    # 預處理與可選聚合
    cols = [time_col, device_col] + list(feature_cols)
    df = inverter_data[cols].copy()
    df.rename(columns={time_col: 'ts'}, inplace=True)

    if freq is not None:
        df['ts'] = pd.to_datetime(df['ts']).dt.floor(freq)
        # 依裝置 + ts 聚合平均（只聚合數值欄）
        df = (df.groupby([device_col, 'ts'], as_index=False)[feature_cols]
                .mean(numeric_only=True))

    # 依裝置切分失效/維修區段
    fs_by_dev = {d: g for d, g in failure_sessions.groupby(device_col)}

    # 輸出路徑
    freq_tag = 'raw' if freq is None else freq
    outdir = f'{folder_path}/{title} ({freq_tag})'
    os.makedirs(outdir, exist_ok=True)

    # 多執行緒輸出
    devices = df[device_col].unique().tolist()

    def _worker(device: str):
        g = df[df[device_col] == device].sort_values('ts')
        if g.empty:
            return
        _plot_device_series(g, feature_cols, device, fs_by_dev, outdir, title, ts_col='ts')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(_worker, devices), total=len(devices), desc="Writing HTML"))

    print(f"Visualization saved at {outdir}/*.html")
    return outdir

    
def visualize_failure_timeline(
    failure_sessions: pd.DataFrame,
    *,
    device_subset=None,                 # e.g., ['INV 01','INV 02']
    order_by="total_downtime",          # 'total_downtime' | 'first_start' | 'name'
    height_per_device=30,
    min_visible_hours=12,               # 短事件顯示的最小寬度（只影響視覺，不影響原始值）
    title="Failure Sessions Timeline"
):
    """
    需要欄位：
      start_time, end_time (datetime-like), device_name (str),
      maintenance (bool), session_id (str/int 可選)
    """

    df = failure_sessions.copy()

    # --- 時間欄位正規化 ---
    for c in ["start_time", "end_time"]:
        if not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time", "device_name"])

    # --- 衍生欄位 ---
    df["duration_hours"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600.0
    df["maintenance_label"] = np.where(df["maintenance"].astype(bool),
                                       "Planned (maintenance)",
                                       "Unplanned failure")

    # 子集
    if device_subset is not None:
        df = df[df["device_name"].isin(device_subset)]

    # --- 排序 ---
    if order_by == "total_downtime":
        order = (df.groupby("device_name")["duration_hours"]
                   .sum()
                   .sort_values(ascending=False)
                   .index.tolist())
    elif order_by == "first_start":
        order = (df.groupby("device_name")["start_time"]
                   .min()
                   .sort_values()
                   .index.tolist())
    else:  # name
        order = sorted(df["device_name"].unique())

    # --- 視覺最小寬度（避免短事件看起來消失）---
    # 視覺上把太短的區段填到 min_visible_hours，但 hover 仍顯示真實 duration
    min_delta = pd.to_timedelta(min_visible_hours, unit="h")
    df["x_start_vis"] = df["start_time"]
    df["x_end_vis"]   = df["end_time"]
    too_short = (df["end_time"] - df["start_time"]) < min_delta
    df.loc[too_short, "x_end_vis"] = df.loc[too_short, "start_time"] + min_delta
    df["visual_padded"] = too_short

    # --- 畫圖 ---
    height = max(420, int(height_per_device * len(order) + 140))

    color_map = {
        "Planned (maintenance)": "#6b7280",  # 深灰 (比原本更有對比)
        "Unplanned failure":     "#2563eb",  # 飽和藍
    }

    labels = {
        "device_name": "Device",
        "maintenance_label": "Type",
        "x_start_vis": "Start",
        "x_end_vis": "End",
        "duration_hours": "Duration (hrs)",
        "session_id": "Session ID",
    }

    fig = px.timeline(
        df,
        x_start="x_start_vis",
        x_end="x_end_vis",
        y="device_name",
        color="maintenance_label",
        color_discrete_map=color_map,
        category_orders={"device_name": order},
        hover_data={
            "session_id": True if "session_id" in df.columns else False,
            "start_time": "|%Y-%m-%d %H:%M",
            "end_time":   "|%Y-%m-%d %H:%M",
            "duration_hours": ':.2f',
            "visual_padded": True,        # 告知是否做了視覺補寬
            "device_name": False,
            "maintenance_label": False
        },
        labels=labels,
        title=title,
    )

    # y 軸與版面
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        height=height,
        legend_title="Event Type",
        bargap=0.25,
        margin=dict(l=70, r=30, t=60, b=40),
    )

    # x 軸：grid + rangeselector + rangeslider
    fig.update_xaxes(
        showgrid=True,
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1w",  step="day",   stepmode="backward"),
                dict(count=1,  label="1m",  step="month", stepmode="backward"),
                dict(count=3,  label="3m",  step="month", stepmode="backward"),
                dict(count=6,  label="6m",  step="month", stepmode="backward"),
                dict(count=1,  label="1y",  step="year",  stepmode="backward"),
                dict(step="all")
            ]
        )
    )

    # 長條外框與透明度
    fig.update_traces(
        marker_line_color="rgba(30,30,60,0.55)",
        marker_line_width=1.5,
        opacity=0.98,
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Start: %{customdata[1]|%Y-%m-%d %H:%M}<br>" +
            "End: %{customdata[2]|%Y-%m-%d %H:%M}<br>" +
            "Duration (hrs): %{customdata[3]:.2f}<br>"
        )
    )

    fig.show()