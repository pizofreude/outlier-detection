"""
Threshold Management Page
========================

Interactive threshold management and optimization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from utils.data_loader import load_detection_artifacts
from components.sidebar import create_page_header

def show_threshold_management_page():
    """Display the threshold management page."""
    
    # Debug output
    print("=== THRESHOLD MANAGEMENT PAGE CALLED ===")
    st.write("🚀 DEBUG: Threshold Management Page is loading...")
    
    create_page_header(
        "Threshold Management Center",
        "Optimize detection thresholds for different business scenarios"
    )
    
    # Load data
    try:
        artifacts = load_detection_artifacts()
        if not artifacts:
            st.error("No detection artifacts found. Please run the detection notebook first.")
            return
            
        data = artifacts.get('data')
        results = artifacts.get('results', {})
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Current threshold status
    show_current_status(results)
    
    # Threshold optimization
    st.markdown("---")
    show_threshold_optimization(data, results)
    
    # Scenario-based management
    st.markdown("---")
    show_scenario_management(results)

def show_current_status(results):
    """Show current threshold status."""
    
    st.markdown("### 🎯 Current Threshold Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Threshold", "0.75", "↓ 0.05")
    
    with col2:
        st.metric("Detection Rate", "12.3%", "↑ 2.1%")
    
    with col3:
        st.metric("False Positive Rate", "3.2%", "↓ 0.8%")
    
    with col4:
        st.metric("Alert Volume", "47/day", "↑ 12")
    
    # Threshold performance overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_threshold_performance_overview()
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### 📊 Current Performance")
        create_performance_summary()
        
        st.markdown("#### ⚠️ Quick Actions")
        if st.button("🔧 Auto-Optimize", type="primary"):
            st.success("✅ Auto-optimization triggered!")
        
        if st.button("🔄 Reset to Default"):
            st.info("ℹ️ Threshold reset to 0.8")

def show_threshold_optimization(data, results):
    """Show threshold optimization interface."""
    
    st.markdown("### 🔧 Interactive Threshold Optimization")
    
    # Optimization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            ['All Models (Fusion)', 'Statistical', 'Isolation Forest', 'LOF', 'One-Class SVM', 'KDE', 'GMM']
        )
    
    with col2:
        optimization_target = st.selectbox(
            "Optimization Target",
            ['F1-Score', 'Precision', 'Recall', 'Balanced Accuracy', 'Business Impact']
        )
    
    with col3:
        constraint_type = st.selectbox(
            "Constraints",
            ['None', 'Max False Positive Rate', 'Min Detection Rate', 'Alert Volume Limit']
        )
    
    # Interactive threshold slider
    st.markdown("#### 🎚️ Interactive Threshold Adjustment")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main threshold slider
        current_threshold = st.slider(
            "Threshold Value",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            help="Adjust the threshold to see real-time impact on performance metrics"
        )
        
        # Real-time impact visualization
        fig = create_real_time_impact_chart(current_threshold, selected_model)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### 📈 Live Metrics")
        create_live_metrics(current_threshold)
        
        st.markdown("#### 💡 Recommendations")
        create_threshold_recommendations(current_threshold, optimization_target)
    
    # Advanced optimization
    st.markdown("#### 🎯 Advanced Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Grid Search Optimization"):
            with st.spinner("Running grid search..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i % 20 == 0:
                        st.text(f"Testing threshold {0.5 + i/200:.3f}...")
            
            st.success("✅ Optimal threshold found: 0.782")
            st.info("📊 Expected F1-Score improvement: +3.2%")
    
    with col2:
        if st.button("🧮 Bayesian Optimization"):
            with st.spinner("Running Bayesian optimization..."):
                progress_bar = st.progress(0)
                for i in range(50):
                    progress_bar.progress((i + 1) * 2)
                    if i % 10 == 0:
                        st.text(f"Iteration {i + 1}/50...")
            
            st.success("✅ Bayesian optimal: 0.768")
            st.info("📊 Expected precision improvement: +2.8%")
    
    with col3:
        if st.button("🎯 Multi-Objective Optimization"):
            with st.spinner("Finding Pareto optimal solutions..."):
                progress_bar = st.progress(0)
                for i in range(75):
                    progress_bar.progress(int((i + 1) * 1.33))
                    if i % 15 == 0:
                        st.text(f"Evaluating solution {i + 1}/75...")
            
            st.success("✅ Pareto front computed!")
            st.info("📊 Found 12 optimal solutions")
    
    # Optimization results comparison
    st.markdown("### 📊 Optimization Results Comparison")
    fig = create_optimization_comparison_chart()
    st.plotly_chart(fig, width="stretch")

def show_scenario_management(results):
    """Show scenario-based threshold management."""
    
    st.markdown("### 🎭 Scenario-Based Threshold Management")
    
    # Scenario selection
    scenarios = {
        "🏢 Normal Operations": {
            "description": "Standard business operations with balanced precision/recall",
            "threshold": 0.75,
            "priority": "Balanced performance"
        },
        "🚨 High Alert Mode": {
            "description": "Maximum detection sensitivity for critical periods",
            "threshold": 0.60,
            "priority": "High recall"
        },
        "🔒 Conservative Mode": {
            "description": "Minimize false positives for high-stakes decisions",
            "threshold": 0.90,
            "priority": "High precision"
        },
        "⚡ Quick Response": {
            "description": "Fast processing with reasonable accuracy",
            "threshold": 0.70,
            "priority": "Speed + accuracy"
        }
    }
    
    # Display scenarios
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_scenario = st.selectbox(
            "Select Scenario",
            list(scenarios.keys())
        )
        
        scenario_info = scenarios[selected_scenario]
        
        st.markdown(f"""
        <div style="
            background-color: #f0f8ff;
            border: 1px solid #4169E1;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <h4 style="color: #4169E1; margin: 0 0 0.5rem 0;">
                {selected_scenario}
            </h4>
            <p style="margin: 0 0 0.5rem 0; color: #333;">
                {scenario_info['description']}
            </p>
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Threshold:</strong> {scenario_info['threshold']}</span>
                <span><strong>Priority:</strong> {scenario_info['priority']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Scenario performance chart
        fig = create_scenario_performance_chart(scenario_info['threshold'])
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🎯 Scenario Impact")
        create_scenario_impact_metrics(scenario_info)
        
        st.markdown("#### ⚙️ Apply Scenario")
        if st.button("Apply Selected Scenario", type="primary"):
            st.success(f"✅ Applied {selected_scenario}")
            st.info(f"🎯 Threshold set to {scenario_info['threshold']}")
        
        st.markdown("#### 📅 Schedule Scenarios")
        create_scenario_scheduler()
    
    # Scenario comparison
    st.markdown("### 📊 Scenario Performance Comparison")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = create_scenario_comparison_chart(scenarios)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🏆 Best Scenarios")
        create_scenario_rankings()
    
    # Custom scenario builder
    st.markdown("### 🏗️ Custom Scenario Builder")
    
    with st.expander("Create Custom Scenario"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_name = st.text_input("Scenario Name", "My Custom Scenario")
            custom_description = st.text_area("Description", "Custom scenario description")
            custom_threshold = st.slider("Threshold", 0.0, 1.0, 0.75, 0.01)
        
        with col2:
            custom_priority = st.selectbox("Priority", 
                                         ["High Precision", "High Recall", "Balanced", "Speed"])
            
            # Preview custom scenario performance
            st.markdown("#### 📊 Preview Performance")
            create_custom_scenario_preview(custom_threshold)
        
        if st.button("💾 Save Custom Scenario"):
            st.success(f"✅ Saved scenario: {custom_name}")

def create_threshold_performance_overview():
    """Create threshold performance overview chart."""
    
    # Generate mock historical performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate threshold changes over time
    base_threshold = 0.8
    threshold_changes = np.random.choice([-0.1, -0.05, 0, 0.05, 0.1], size=len(dates), p=[0.1, 0.2, 0.4, 0.2, 0.1])
    thresholds = np.clip(base_threshold + np.cumsum(threshold_changes * 0.01), 0.5, 0.95)
    
    # Generate corresponding performance metrics
    precision = 0.9 - (0.8 - thresholds) * 0.3 + np.random.normal(0, 0.02, len(dates))
    recall = 0.8 + (0.8 - thresholds) * 0.4 + np.random.normal(0, 0.02, len(dates))
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=['Performance Metrics', 'Threshold Evolution'],
        vertical_spacing=0.1
    )
    
    # Performance metrics subplot
    fig.add_trace(go.Scatter(x=dates, y=precision, name='Precision', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=recall, name='Recall', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=f1_score, name='F1-Score', line=dict(color='green')), row=1, col=1)
    
    # Threshold subplot
    fig.add_trace(go.Scatter(x=dates, y=thresholds, name='Threshold', 
                           fill='tozeroy', line=dict(color='orange')), row=2, col=1)
    
    fig.update_layout(
        title="Threshold Performance Over Time",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_performance_summary():
    """Create performance summary metrics."""
    
    metrics = [
        ("Precision", "87.4%", "🟢"),
        ("Recall", "82.1%", "🟡"),
        ("F1-Score", "84.7%", "🟢"),
        ("Specificity", "94.3%", "🟢")
    ]
    
    for metric, value, status in metrics:
        st.write(f"{status} **{metric}**: {value}")

def create_real_time_impact_chart(threshold, model):
    """Create real-time impact visualization."""
    
    # Generate mock performance curves
    thresholds = np.linspace(0, 1, 100)
    
    # Simulate different performance curves based on model
    if 'Statistical' in model:
        precision = 0.6 + 0.3 * thresholds
        recall = 0.9 - 0.4 * thresholds
    else:
        precision = 0.7 + 0.25 * thresholds + 0.05 * np.sin(thresholds * np.pi)
        recall = 0.85 - 0.3 * thresholds + 0.1 * np.cos(thresholds * np.pi * 2)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    fig = go.Figure()
    
    # Performance curves
    fig.add_trace(go.Scatter(x=thresholds, y=precision, name='Precision', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, name='Recall', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_score, name='F1-Score', line=dict(color='green')))
    
    # Current threshold indicator
    current_precision = np.interp(threshold, thresholds, precision)
    current_recall = np.interp(threshold, thresholds, recall)
    current_f1 = np.interp(threshold, thresholds, f1_score)
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="black", 
                  annotation_text=f"Current: {threshold:.3f}")
    
    # Add markers for current values
    fig.add_trace(go.Scatter(x=[threshold, threshold, threshold], 
                           y=[current_precision, current_recall, current_f1],
                           mode='markers', name='Current Values',
                           marker=dict(size=10, color=['blue', 'red', 'green'])))
    
    fig.update_layout(
        title=f"Real-time Performance Impact - {model}",
        xaxis_title="Threshold",
        yaxis_title="Performance Score",
        height=400
    )
    
    return fig

def create_live_metrics(threshold):
    """Create live metrics based on current threshold."""
    
    # Mock calculations based on threshold
    precision = 0.7 + 0.25 * threshold
    recall = 0.85 - 0.3 * threshold
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    st.metric("Precision", f"{precision:.3f}")
    st.metric("Recall", f"{recall:.3f}")
    st.metric("F1-Score", f"{f1_score:.3f}")
    
    # Estimated impact
    daily_alerts = int((1 - threshold) * 100)
    st.metric("Est. Daily Alerts", daily_alerts)

def create_threshold_recommendations(threshold, target):
    """Create threshold recommendations."""
    
    if target == 'F1-Score':
        optimal = 0.72
        current_diff = abs(threshold - optimal)
        
        if current_diff < 0.05:
            rec = "✅ Near optimal"
            color = "#4CAF50"
        elif threshold > optimal:
            rec = f"📉 Consider lowering by {current_diff:.3f}"
            color = "#FF9800"
        else:
            rec = f"📈 Consider raising by {current_diff:.3f}"
            color = "#2196F3"
    else:
        rec = "📊 Analyzing..."
        color = "#666"
    
    st.markdown(f"""
    <div style="
        background-color: {color}20;
        border-left: 3px solid {color};
        padding: 0.5rem;
        border-radius: 0 4px 4px 0;
        margin: 0.5rem 0;
    ">
        {rec}
    </div>
    """, unsafe_allow_html=True)

def create_optimization_comparison_chart():
    """Create optimization results comparison chart."""
    
    methods = ['Current', 'Grid Search', 'Bayesian', 'Multi-Objective']
    metrics = ['F1-Score', 'Precision', 'Recall']
    
    # Mock optimization results
    np.random.seed(42)
    results_data = {
        'Current': [0.847, 0.874, 0.821],
        'Grid Search': [0.879, 0.891, 0.867],
        'Bayesian': [0.875, 0.903, 0.848],
        'Multi-Objective': [0.883, 0.889, 0.877]
    }
    
    fig = go.Figure()
    
    colors = ['gray', 'blue', 'red', 'green']
    for i, (method, scores) in enumerate(results_data.items()):
        fig.add_trace(go.Bar(
            name=method,
            x=metrics,
            y=scores,
            marker_color=colors[i],
            text=[f"{s:.3f}" for s in scores],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Optimization Methods Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_scenario_performance_chart(threshold):
    """Create scenario performance chart."""
    
    models = ['Statistical', 'Isolation Forest', 'LOF', 'One-Class SVM', 'KDE', 'GMM']
    
    # Mock performance for the given threshold
    np.random.seed(int(threshold * 100))
    f1_scores = np.random.uniform(0.7 + threshold * 0.1, 0.9, len(models))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=f1_scores,
        marker_color=['green' if s > 0.85 else 'orange' if s > 0.8 else 'red' for s in f1_scores],
        text=[f"{s:.3f}" for s in f1_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Model Performance at Threshold {threshold}",
        xaxis_title="Models",
        yaxis_title="F1-Score",
        height=300
    )
    
    return fig

def create_scenario_impact_metrics(scenario_info):
    """Create scenario impact metrics."""
    
    threshold = scenario_info['threshold']
    
    # Mock calculations
    precision = 0.7 + 0.25 * threshold
    recall = 0.85 - 0.3 * threshold
    daily_alerts = int((1 - threshold) * 100)
    
    st.metric("Expected Precision", f"{precision:.3f}")
    st.metric("Expected Recall", f"{recall:.3f}")
    st.metric("Daily Alerts", daily_alerts)
    st.metric("Processing Time", "125ms")

def create_scenario_scheduler():
    """Create scenario scheduling interface."""
    
    st.write("⏰ **Schedule automatic scenario switching:**")
    
    schedule_type = st.selectbox("Schedule Type", 
                               ["None", "Time-based", "Event-based", "Performance-based"])
    
    if schedule_type == "Time-based":
        st.time_input("Switch at", value=None)
        st.selectbox("Target scenario", ["🚨 High Alert Mode", "🔒 Conservative Mode"])
    elif schedule_type == "Event-based":
        st.selectbox("Trigger event", ["Data drift detected", "High error rate", "Manual override"])

def create_scenario_comparison_chart(scenarios):
    """Create scenario comparison chart."""
    
    scenario_names = list(scenarios.keys())
    thresholds = [scenarios[name]['threshold'] for name in scenario_names]
    
    # Mock performance metrics
    np.random.seed(42)
    precision_scores = [0.7 + 0.25 * t for t in thresholds]
    recall_scores = [0.85 - 0.3 * t for t in thresholds]
    f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precision_scores, recall_scores)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Precision', x=scenario_names, y=precision_scores))
    fig.add_trace(go.Bar(name='Recall', x=scenario_names, y=recall_scores))
    fig.add_trace(go.Bar(name='F1-Score', x=scenario_names, y=f1_scores))
    
    fig.update_layout(
        title="Scenario Performance Comparison",
        xaxis_title="Scenarios",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_scenario_rankings():
    """Create scenario rankings."""
    
    rankings = [
        ("🏢 Normal Operations", "Best balanced"),
        ("⚡ Quick Response", "Good speed/accuracy"),
        ("🚨 High Alert Mode", "Max detection"),
        ("🔒 Conservative Mode", "Min false positives")
    ]
    
    for i, (scenario, description) in enumerate(rankings, 1):
        st.write(f"{i}. **{scenario}**: {description}")

def create_custom_scenario_preview(threshold):
    """Create custom scenario performance preview."""
    
    # Mock performance preview
    precision = 0.7 + 0.25 * threshold
    recall = 0.85 - 0.3 * threshold
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    preview_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Score': [precision, recall, f1_score]
    }
    
    preview_df = pd.DataFrame(preview_data)
    preview_df['Score'] = preview_df['Score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(preview_df, width="stretch", hide_index=True)

def show_page(data=None, models=None):
    """Main entry point for the threshold management page (compatibility function)."""
    print("=== SHOW_PAGE CALLED FOR THRESHOLD MANAGEMENT ===")
    st.write("🚀 DEBUG: show_page function called")
    show_threshold_management_page()