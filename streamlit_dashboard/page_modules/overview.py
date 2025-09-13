"""
Overview Page - Executive Dashboard
==================================

Main overview page with high-level KPIs and system status.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

from utils.data_loader import load_detection_artifacts
from utils.metrics import calculate_business_metrics, calculate_technical_metrics
from components.kpi_cards import create_kpi_grid
from components.sidebar import create_page_header

def show_overview_page():
    """Display the overview page."""
    
    create_page_header(
        "Executive Dashboard Overview", 
        "Real-time insights into your outlier detection system performance"
    )
    
    # Load data
    try:
        artifacts = load_detection_artifacts()
        if not artifacts:
            st.error("No detection artifacts found. Please run the detection notebook first.")
            return
            
        data = artifacts.get('data')
        results = artifacts.get('results', {})
        
        if data is None:
            st.error("No data found in artifacts.")
            return
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Calculate metrics
    business_metrics = calculate_business_metrics(data, results)
    technical_metrics = calculate_technical_metrics(results)
    
    # KPI Cards Section
    st.markdown("### 🎯 Key Performance Indicators")
    create_kpi_grid(business_metrics, technical_metrics)
    
    # Create main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Real-time detection chart
        st.markdown("### 📈 Real-time Detection Activity")
        detection_timeline = create_detection_timeline(data, results)
        st.plotly_chart(detection_timeline, width="stretch")
        
        # Model performance comparison
        st.markdown("### ⚙️ Model Performance Comparison")
        performance_chart = create_performance_comparison(technical_metrics)
        st.plotly_chart(performance_chart, width="stretch")
    
    with col2:
        # Alert summary
        st.markdown("### 🚨 Recent Alerts")
        create_alert_summary(data, results)
        
        # System health
        st.markdown("### 🏥 System Health")
        create_system_health_panel()
        
        # Top outliers
        st.markdown("### 🎯 Top Outliers Today")
        create_top_outliers(data, results)

def create_detection_timeline(data, results):
    """Create a timeline chart showing detection activity."""
    
    # Simulate time-series data for the timeline
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate realistic detection counts
    np.random.seed(42)
    detections = np.random.poisson(lam=5, size=len(dates))
    
    # Add some spikes to make it more realistic
    spike_indices = np.random.choice(len(dates), size=10, replace=False)
    detections[spike_indices] += np.random.randint(10, 25, size=10)
    
    timeline_df = pd.DataFrame({
        'timestamp': dates,
        'detections': detections,
        'cumulative': np.cumsum(detections)
    })
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["Detection Activity Over Time"]
    )
    
    # Add detection bars
    fig.add_trace(
        go.Bar(
            x=timeline_df['timestamp'],
            y=timeline_df['detections'],
            name='Hourly Detections',
            marker_color='rgba(31, 119, 180, 0.7)',
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Add cumulative line
    fig.add_trace(
        go.Scatter(
            x=timeline_df['timestamp'],
            y=timeline_df['cumulative'],
            mode='lines',
            name='Cumulative',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Detection Activity (Last 7 Days)",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Detections per Hour", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Detections", secondary_y=True)
    
    return fig

def create_performance_comparison(technical_metrics):
    """Create a radar chart comparing model performance."""
    
    # Mock performance data for different models
    models = ['Statistical', 'Isolation Forest', 'LOF', 'One-Class SVM', 'KDE', 'GMM']
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC', 'Speed', 'Stability']
    
    # Generate realistic performance scores
    np.random.seed(42)
    performance_data = {}
    for model in models:
        performance_data[model] = np.random.uniform(0.6, 0.95, len(metrics))
    
    # Create radar chart
    fig = go.Figure()
    
    # Use hex colors instead of RGB colors for proper conversion
    hex_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (model, scores) in enumerate(performance_data.items()):
        color = hex_colors[i % len(hex_colors)]
        # Convert hex to RGB manually for the fillcolor
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name=model,
            line_color=color,
            fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar",
        height=400
    )
    
    return fig

def create_alert_summary(data, results):
    """Create alert summary panel."""
    
    # Generate mock alert data
    alert_types = ['High Severity', 'Medium Severity', 'Low Severity', 'Info']
    alert_counts = [12, 23, 31, 15]
    alert_colors = ['#ff4444', '#ff8c00', '#ffd700', '#87ceeb']
    
    for i, (alert_type, count) in enumerate(zip(alert_types, alert_counts)):
        st.markdown(f"""
        <div style="
            background-color: {alert_colors[i]}20;
            border-left: 4px solid {alert_colors[i]};
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 4px 4px 0;
        ">
            <strong style="color: {alert_colors[i]};">{alert_type}</strong><br>
            <span style="font-size: 1.2rem; font-weight: bold;">{count}</span> alerts
        </div>
        """, unsafe_allow_html=True)

def create_system_health_panel():
    """Create system health monitoring panel."""
    
    health_metrics = [
        ("Data Pipeline", 98.5, "🟢"),
        ("Model Accuracy", 94.2, "🟢"),  
        ("Response Time", 87.8, "🟡"),
        ("Memory Usage", 76.3, "🟢"),
        ("Error Rate", 1.2, "🟢")
    ]
    
    for metric, value, status in health_metrics:
        if metric == "Error Rate":
            # Invert for error rate (lower is better)
            color = "#4CAF50" if value < 5 else "#FF9800" if value < 10 else "#F44336"
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem;
                margin: 0.25rem 0;
                background-color: #f8f9fa;
                border-radius: 4px;
            ">
                <span><strong>{metric}</strong></span>
                <span style="color: {color};">{status} {value:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            color = "#4CAF50" if value > 90 else "#FF9800" if value > 80 else "#F44336"
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem;
                margin: 0.25rem 0;
                background-color: #f8f9fa;
                border-radius: 4px;
            ">
                <span><strong>{metric}</strong></span>
                <span style="color: {color};">{status} {value:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

def create_top_outliers(data, results):
    """Create top outliers panel."""
    
    if 'final_scores' not in results or results['final_scores'] is None:
        st.info("No outlier scores available.")
        return
    
    # Get top 5 outliers
    scores = results['final_scores']
    top_indices = np.argsort(scores)[-5:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        score = scores[idx]
        # Create a mock record summary
        st.markdown(f"""
        <div style="
            background-color: {'#fff3cd' if score > 0.8 else '#d4edda'};
            border: 1px solid {'#ffeaa7' if score > 0.8 else '#c3e6cb'};
            border-radius: 4px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        ">
            <div style="display: flex; justify-content: space-between;">
                <strong>Record #{idx}</strong>
                <span style="color: {'#e17055' if score > 0.8 else '#00b894'};">
                    {score:.3f}
                </span>
            </div>
            <small style="color: #666;">
                Detected by {np.random.choice(['Statistical', 'Isolation Forest', 'LOF', 'KDE'])}
            </small>
        </div>
        """, unsafe_allow_html=True)

def show_page(data=None, models=None):
    """Main entry point for the overview page (compatibility function)."""
    show_overview_page()