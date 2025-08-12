import plotly.express as px
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def visualize_hourly_mean_values(inverter_data, failure_sessions, feature_cols, folder_path='visualization', title='Hourly Mean Values of Features'):
    """
    Visualizes the hourly mean values of specified features for each inverter device,
    highlighting failure sessions.

    Parameters:
    - inverter_data: DataFrame containing inverter data with 'event_local_time', 'device_name', and feature columns.
    - failure_sessions: DataFrame containing failure sessions with 'start_time', 'end_time', and 'device_name'.
    - feature_cols: List of feature column names to visualize.
    """


    for device in tqdm(inverter_data['device_name'].unique(), desc="Visualizing devices"):
        single_inverter_data = inverter_data[inverter_data['device_name'] == device][['event_local_time', 'device_name'] + feature_cols]
        single_inverter_data['hour'] = single_inverter_data['event_local_time'].dt.floor('H')
        hourly_inverter_data = single_inverter_data.groupby('hour').mean(numeric_only=True).reset_index()
        fig = px.line(hourly_inverter_data, x='hour', y=feature_cols, title=f'Hourly Mean Values of Features for {device}')
        
        start_time, end_time = single_inverter_data['event_local_time'].min(), single_inverter_data['event_local_time'].max()
        for id, row in failure_sessions[failure_sessions['device_name'] == device].iterrows():
            if (row['end_time'] < start_time) or (row['start_time'] > end_time):
                #print(f"Warning: Failure session {id} for device {device} is out of bounds of the data range.")
                continue
            fig.add_vrect(
                x0=row['start_time'], x1=row['end_time'],
                fillcolor="red", opacity=0.5,
                annotation_text="Failure Session", annotation_position="top left"
            )
        fig.update_layout(
            xaxis_title='Hour',
            yaxis_title='Mean Value',
            legend_title='Features',
            title_x=0.5,  # Center the title
            title = device + title
        )
        os.makedirs(f'{folder_path}/{title}', exist_ok=True)
        fig.write_html(f'{folder_path}/{title}/{device}.html')
    print(f"Visualization saved at {folder_path}/{title}/*.html")
    
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