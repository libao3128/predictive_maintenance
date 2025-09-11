"""
Time series visualization utilities for predictive maintenance.

This module provides functions for visualizing time series data,
failure sessions, and device-specific patterns.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union


def visualize_mean_values(
    inverter_data: pd.DataFrame,
    failure_sessions: pd.DataFrame,
    feature_cols: List[str],
    folder_path: str = 'visualization',
    title: str = 'Mean Values of Features',
    time_col: str = 'event_local_time',
    device_col: str = 'device_name',
    freq: Optional[str] = 'H',
    workers: int = 8
) -> str:
    """
    Create visualizations of mean feature values with failure session overlays.
    
    This function generates interactive HTML plots for each device showing
    feature values over time with failure sessions highlighted.
    
    Args:
        inverter_data: DataFrame containing time series data
        failure_sessions: DataFrame containing failure session information
        feature_cols: List of feature columns to visualize
        folder_path: Base folder for saving visualizations
        title: Title for the visualization
        time_col: Name of the time column
        device_col: Name of the device column
        freq: Aggregation frequency (None for raw data, 'H' for hourly, etc.)
        workers: Number of worker threads for parallel processing
        
    Returns:
        Path to the output directory containing HTML files
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = [time_col, device_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in inverter_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in inverter_data: {missing_cols}")
    
    # Preprocessing and optional aggregation
    cols = [time_col, device_col] + list(feature_cols)
    df = inverter_data[cols].copy()
    df.rename(columns={time_col: 'ts'}, inplace=True)

    if freq is not None:
        df['ts'] = pd.to_datetime(df['ts']).dt.floor(freq)
        # Aggregate by device + ts (only aggregate numerical columns)
        df = (df.groupby([device_col, 'ts'], as_index=False)[feature_cols]
                .mean(numeric_only=True))

    # Split failure/maintenance segments by device
    fs_by_dev = {d: g for d, g in failure_sessions.groupby(device_col)}

    # Output path
    freq_tag = 'raw' if freq is None else freq
    outdir = f'{folder_path}/{title} ({freq_tag})'
    os.makedirs(outdir, exist_ok=True)

    # Multi-threaded output
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


def _plot_device_series(
    g: pd.DataFrame,
    feature_cols: List[str],
    device: str,
    fs_by_dev: dict,
    outdir: str,
    title: str,
    ts_col: str = "ts"
) -> None:
    """Create a single device plot with failure session overlays."""
    fig = px.line(g, x=ts_col, y=feature_cols, title=f'{device} {title}')
    start_time, end_time = g[ts_col].min(), g[ts_col].max()

    # Add failure session overlays
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


def visualize_failure_timeline(
    failure_sessions: pd.DataFrame,
    device_subset: Optional[List[str]] = None,
    order_by: str = "total_downtime",
    height_per_device: int = 30,
    min_visible_hours: int = 12,
    title: str = "Failure Sessions Timeline"
) -> None:
    """
    Create a timeline visualization of failure sessions.
    
    This function generates an interactive timeline showing failure sessions
    for all devices, with different colors for planned vs unplanned failures.
    
    Args:
        failure_sessions: DataFrame containing failure session data
        device_subset: List of device names to include (None for all)
        order_by: Method for ordering devices ('total_downtime', 'first_start', 'name')
        height_per_device: Height in pixels per device row
        min_visible_hours: Minimum visible width for short events
        title: Title for the visualization
        
    Raises:
        ValueError: If required columns are missing
    """
    df = failure_sessions.copy()

    # Validate required columns
    required_cols = ["start_time", "end_time", "device_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Time column normalization
    for c in ["start_time", "end_time"]:
        if not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time", "device_name"])

    # Derived columns
    df["duration_hours"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600.0
    df["maintenance_label"] = np.where(df["maintenance"].astype(bool),
                                       "Planned (maintenance)",
                                       "Unplanned failure")

    # Subset devices if specified
    if device_subset is not None:
        df = df[df["device_name"].isin(device_subset)]

    # Sorting
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

    # Visual minimum width (avoid short events appearing to disappear)
    min_delta = pd.to_timedelta(min_visible_hours, unit="h")
    df["x_start_vis"] = df["start_time"]
    df["x_end_vis"] = df["end_time"]
    too_short = (df["end_time"] - df["start_time"]) < min_delta
    df.loc[too_short, "x_end_vis"] = df.loc[too_short, "start_time"] + min_delta
    df["visual_padded"] = too_short

    # Plotting
    height = max(420, int(height_per_device * len(order) + 140))

    color_map = {
        "Planned (maintenance)": "#6b7280",  # Dark gray
        "Unplanned failure": "#2563eb",      # Saturated blue
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
            "end_time": "|%Y-%m-%d %H:%M",
            "duration_hours": ':.2f',
            "visual_padded": True,
            "device_name": False,
            "maintenance_label": False
        },
        labels=labels,
        title=title,
    )

    # Layout updates
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        height=height,
        legend_title="Event Type",
        bargap=0.25,
        margin=dict(l=70, r=30, t=60, b=40),
    )

    # x-axis: grid + rangeselector + rangeslider
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

    # Bar outline and transparency
    fig.update_traces(
        marker_line_color="rgba(30,30,60,0.55)",
        marker_line_width=1.5,
        opacity=0.98,
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Session ID: %{customdata[0]}<br>" +
            "Start: %{customdata[1]|%Y-%m-%d %H:%M}<br>" +
            "End: %{customdata[2]|%Y-%m-%d %H:%M}<br>" +
            "Duration (hrs): %{customdata[3]:.2f}<br>"
        )
    )

    fig.show()
