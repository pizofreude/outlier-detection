"""
KPI Cards Component
==================

Reusable KPI card components for the dashboard.
"""

import streamlit as st
import plotly.graph_objects as go

def create_kpi_card(title, value, delta=None, delta_color="normal", format_func=None):
    """Create a KPI card with optional delta indicator."""
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = str(value)
    
    if delta is not None:
        st.metric(
            label=title,
            value=formatted_value,
            delta=delta,
            delta_color=delta_color
        )
    else:
        st.metric(
            label=title,
            value=formatted_value
        )

def create_gauge_chart(value, title, min_val=0, max_val=100, thresholds=None):
    """Create a gauge chart for KPIs."""
    if thresholds is None:
        thresholds = [30, 70]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': thresholds[1]},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, thresholds[0]], 'color': "lightgray"},
                {'range': [thresholds[0], thresholds[1]], 'color': "yellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds[1]
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_status_indicator(status, label):
    """Create a status indicator with color coding."""
    status_colors = {
        'healthy': '🟢',
        'warning': '🟡', 
        'critical': '🔴',
        'unknown': '⚪'
    }
    
    color_class = {
        'healthy': 'status-good',
        'warning': 'status-warning',
        'critical': 'status-danger',
        'unknown': ''
    }
    
    icon = status_colors.get(status.lower(), '⚪')
    css_class = color_class.get(status.lower(), '')
    
    st.markdown(f"""
    <div class="{css_class}">
        {icon} {label}: {status.title()}
    </div>
    """, unsafe_allow_html=True)

def create_metric_cards_row(metrics_dict, cols=4):
    """Create a row of metric cards."""
    cols = st.columns(cols)
    
    for i, (key, value) in enumerate(metrics_dict.items()):
        with cols[i % len(cols)]:
            if isinstance(value, dict):
                create_kpi_card(
                    title=value.get('title', key),
                    value=value.get('value', 0),
                    delta=value.get('delta'),
                    delta_color=value.get('delta_color', 'normal'),
                    format_func=value.get('format_func')
                )
            else:
                create_kpi_card(title=key.replace('_', ' ').title(), value=value)

def create_kpi_grid(*metric_dicts, cols=4, title=None):
    """Create a grid of KPI cards with optional title.
    
    Args:
        *metric_dicts: One or more dictionaries containing metrics
        cols: Number of columns per row
        title: Optional title for the grid
    """
    if title:
        st.subheader(title)
    
    # Merge all metric dictionaries
    merged_metrics = {}
    for metrics_dict in metric_dicts:
        if isinstance(metrics_dict, dict):
            merged_metrics.update(metrics_dict)
    
    # Group metrics into rows
    metric_items = list(merged_metrics.items())
    
    # Create rows of columns
    for i in range(0, len(metric_items), cols):
        row_metrics = metric_items[i:i+cols]
        columns = st.columns(len(row_metrics))
        
        for j, (key, value) in enumerate(row_metrics):
            with columns[j]:
                if isinstance(value, dict):
                    create_kpi_card(
                        title=value.get('title', key),
                        value=value.get('value', 0),
                        delta=value.get('delta'),
                        delta_color=value.get('delta_color', 'normal'),
                        format_func=value.get('format_func')
                    )
                else:
                    # Format the key to be more readable
                    formatted_key = key.replace('_', ' ').title()
                    
                    # Apply default formatting based on value type
                    if isinstance(value, float):
                        if 0 <= value <= 1:
                            # Percentage
                            formatted_value = f"{value:.1%}"
                        elif value > 1000:
                            # Large numbers
                            formatted_value = f"{value:,.0f}"
                        else:
                            # Regular floats
                            formatted_value = f"{value:.2f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    else:
                        formatted_value = str(value)
                    
                    st.metric(label=formatted_key, value=formatted_value)