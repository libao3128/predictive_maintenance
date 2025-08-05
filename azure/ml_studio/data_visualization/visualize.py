from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.express as px
import pandas as pd

voltage_metrics = [
    "metric.AC_VOLTAGE_AB.MEASURED",
    "metric.AC_VOLTAGE_BC.MEASURED",
    "metric.AC_VOLTAGE_CA.MEASURED",
    "metric.AC_VOLTAGE_HI_SETPOINT.MEASURED",
    "metric.AC_VOLTAGE_LO_SETPOINT.MEASURED",
    "metric.DC_VOLTAGE.MEASURED",
    "metric.DC_VOLTAGE_BUS.MEASURED",
    "metric.DC_VOLTAGE_N.MEASURED",
    "metric.DC_VOLTAGE_P.MEASURED",
    "metric.DC_BATT_VOLTAGE_BUS.MEASURED"
]
current_metrics = [
    "metric.AC_CURRENT_A.MEASURED",
    "metric.AC_CURRENT_B.MEASURED",
    "metric.AC_CURRENT_C.MEASURED",
    "metric.AC_CURRENT_MAX.MEASURED",
    "metric.DC_CURRENT.MEASURED",
    "metric.DC_CURRENT_AVG.MEASURED",
    "metric.DC_CURRENT_MAX.MEASURED"
]
power_metrics = [
    "metric.AC_POWER.MEASURED",
    "metric.AC_POWER_LIMIT_SETPOINT.MEASURED",
    "metric.DC_POWER.MEASURED",
    "metric.VAR.MEASURED",
    "metric.SVA.MEASURED",
    "metric.SVA_LIMIT_SETPOINT.MEASURED",
    "metric.VAR_LIMIT_SETPOINT.MEASURED"
]
energy_metrics = [
    "metric.ENERGY_DELIVERED.MEASURED",
    "metric.ENERGY_DELIVERED_DAILY.MEASURED",
    "metric.ENERGY_DELIVERED_MONTHLY.MEASURED",
    "metric.ENERGY_RECEIVED.MEASURED",
    "metric.VARH_DELIVERED.MEASURED",
    "metric.VARH_DELIVERED_DAILY.MEASURED",
    "metric.VARH_DELIVERED_MONTHLY.MEASURED"
]
electrical_params = [
    "metric.FREQUENCY.MEASURED",
    "metric.POWER_FACTOR.MEASURED"
]
status_metrics = [
    "metric.STATUS_AC_MOD_ADMISSION_TEMP.MEASURED",
    "metric.STATUS_CURRENT_NORMATIVE.MEASURED",
    "metric.STATUS_FAULT_MODULE.MEASURED",
    "metric.STATUS_FAULT_WORD.MEASURED",
    "metric.STATUS_IGBT_MAX_TEMP.MEASURED",
    "metric.STATUS_INSUL_MON_AC.MEASURED",
    "metric.STATUS_INSUL_MON_DC.MEASURED",
    "metric.STATUS_INTERNAL_HUMIDITY.MEASURED",
    "metric.STATUS_INTERNAL_INPUT_WORD.MEASURED",
    "metric.STATUS_INTERNAL_OUTPUT_WORD.MEASURED",
    "metric.STATUS_INTERNAL_TEMP.MEASURED",
    "metric.STATUS_LV_PRESSURE.MEASURED",
    "metric.STATUS_MOD_MAX_TEMP.MEASURED",
    "metric.STATUS_MV_PRESSURE.MEASURED",
    "metric.STATUS_POWER_SOURCE_1.MEASURED",
    "metric.STATUS_POWER_SOURCE_2.MEASURED",
    "metric.STATUS_WARNING_MODULE.MEASURED",
    "metric.STATUS_WARNING_WORD.MEASURED",
    "metric.STATUS_WORD.MEASURED"
]
insulation_metrics = [
    "metric.INSUL_MON_AC_RESISTOR.MEASURED",
    "metric.INSUL_MON_DC_RESISTOR.MEASURED"
]
misc_metrics = [
    "metric.HEARTBEAT.MEASURED",
    "metric.HW_VERSION.MEASURED",
    "metric.COMM_LINK.MEASURED"
]


def visualize_metrics(inverter_data, failure_sessions, device_name, metric_columns, session_to_visualize=0):
    filtered_inverter_data = inverter_data[inverter_data['device_name'] == device_name].copy()
    filtered_failure_sessions = failure_sessions[failure_sessions['device_name'] == device_name].copy()

    if len(filtered_failure_sessions) == 0:
        print(f"No failure sessions found for device {device_name}.")
        return
    
    print(f"Device {device_name} has {len(filtered_failure_sessions)} failure sessions.")
    if session_to_visualize >= len(filtered_failure_sessions):
        print(f"Session {session_to_visualize} is out of range for device {device_name}.")
        return
    print(f"Visualizing session {session_to_visualize} for device {device_name}...")
    session = filtered_failure_sessions.iloc[session_to_visualize]
    
    filtered_inverter_data = filtered_inverter_data[
        (filtered_inverter_data['event_local_time'] >= session['start_time']-pd.Timedelta('60 days')) &
        (filtered_inverter_data['event_local_time'] <= session['end_time']+pd.Timedelta('1 days'))
    ]
    
    filtered_inverter_data.sort_values(by='event_local_time', inplace=True)
    
    fig = make_subplots(rows=len(metric_columns), cols=1, shared_xaxes=True, vertical_spacing=0.01,
                        subplot_titles=metric_columns)
    
    for i, metric in enumerate(metric_columns):
        fig.add_trace(
            px.line(filtered_inverter_data, x='event_local_time', y=metric, title=metric, height=200).data[0],
            row=i + 1, col=1
        )
        fig.add_vrect(
            x0=session['start_time'], x1=session['end_time'],
            fillcolor="red", opacity=0.2, line_width=0,
            row=i + 1, col=1
        )
    fig.update_layout(height=200 * len(metric_columns), title_text=f"Failure Session {session_to_visualize} for Device {device_name}", hovermode='x unified')
    #fig.show()
    return fig

def visualize_failure_sessions(ac_power_data, failure_sessions, device_name, year_to_plot=None):   
    inverter_ac_power = ac_power_data[ac_power_data['device_name'] == device_name]
    failure_data = failure_sessions[failure_sessions['device_name'] == device_name]
    
    if year_to_plot is None:
        year_to_plot = failure_data['start_time'].dt.year.unique()
        year_to_plot = range(min(year_to_plot), max(year_to_plot) + 1)
    elif isinstance(year_to_plot, int):
        year_to_plot = [year_to_plot]
    inverter_ac_power = inverter_ac_power[inverter_ac_power['event_local_time'].dt.year.isin(year_to_plot)]
    failure_data = failure_data[(failure_data['start_time'].dt.year.isin(year_to_plot)) | (failure_data['end_time'].dt.year.isin(year_to_plot))]
        

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces for inverter AC power and total power
    fig.add_trace(go.Scatter(x=inverter_ac_power['event_local_time'], 
                             y=inverter_ac_power['total_power'], 
                             mode='lines', 
                             name='Total Power', 
                             line=dict(width=2),
                             opacity=0.2),
                  secondary_y=True)
    # Add traces for inverter AC power
    fig.add_trace(go.Scatter(x=inverter_ac_power ['event_local_time'], 
                             y=inverter_ac_power ['inverter_ac_power'], 
                             mode='lines', 
                             name=device_name, 
                             line=dict(width=2)),
                  secondary_y=False)
    
    # Add vertical rectangles for failure sessions
    for _, session in failure_data.iterrows():
        annotation_text = 'Duration:<br>' + str(session['duration'])[:-3]  # Format duration to exclude microseconds
        fig.add_vrect(
            x0=session['start_time'],
            x1=session['end_time'],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text=annotation_text,
            annotation_position="top left",
            name = 'Failure Session'
        )
        
    title = f'AC Power for {device_name}'
    if year_to_plot is not None:
        title += f' in {year_to_plot}'

    fig.update_layout(title=title,
                      xaxis_title='Time',
                      yaxis_title='AC Power (W)')

    fig.update_traces(opacity=0.7, secondary_y=True)
    return fig