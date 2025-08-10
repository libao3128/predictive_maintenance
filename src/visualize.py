import plotly.express as px
import os

def visualize_hourly_mean_values(inverter_data, failure_sessions, feature_cols, file_path='visualization/hourly_mean_values'):
    """
    Visualizes the hourly mean values of specified features for each inverter device,
    highlighting failure sessions.

    Parameters:
    - inverter_data: DataFrame containing inverter data with 'event_local_time', 'device_name', and feature columns.
    - failure_sessions: DataFrame containing failure sessions with 'start_time', 'end_time', and 'device_name'.
    - feature_cols: List of feature column names to visualize.
    """


    for device in inverter_data['device_name'].unique():
        single_inverter_data = inverter_data[inverter_data['device_name'] == device][['event_local_time', 'device_name'] + feature_cols]
        single_inverter_data['hour'] = single_inverter_data['event_local_time'].dt.floor('H')
        hourly_inverter_data = single_inverter_data.groupby('hour').mean(numeric_only=True).reset_index()
        fig = px.line(hourly_inverter_data, x='hour', y=feature_cols, title=f'Hourly Mean Values of Features for {device}')
        for id, row in failure_sessions[failure_sessions['device_name'] == device].iterrows():
            fig.add_vrect(
                x0=row['start_time'], x1=row['end_time'],
                fillcolor="red", opacity=0.5,
                annotation_text="Failure Session", annotation_position="top left"
            )
        os.makedirs(file_path, exist_ok=True)
        fig.write_html(f'{file_path}/{device}.html')
        print(f"Visualization saved for device: {device} at {file_path}/{device}.html")