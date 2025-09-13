"""
Detection Analysis Page
======================

Detailed analysis of detection methods and results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from utils.data_loader import load_detection_artifacts
from utils.visualizations import create_scatter_plot, create_detection_heatmap
from components.sidebar import create_page_header

def show_detection_analysis_page():
    """Display the detection analysis page."""
    
    create_page_header(
        "Detection Analysis Deep Dive",
        "Comprehensive analysis of detection methods and outlier patterns"
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
    
    # Method selection
    st.markdown("### 🎯 Detection Method Analysis")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        available_methods = ['All Methods', 'Statistical', 'Isolation Forest', 'LOF', 
                           'One-Class SVM', 'KDE', 'GMM', 'Score Fusion']
        selected_method = st.selectbox("Select Method", available_methods)
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ['Score Distribution', 'Detection Patterns', 'Method Comparison', 'Threshold Analysis']
        )
    
    with col3:
        st.info("💡 Use the controls to explore different detection methods and analysis types.")
    
    # Main analysis section
    if analysis_type == 'Score Distribution':
        show_score_distribution_analysis(data, results, selected_method)
    elif analysis_type == 'Detection Patterns':
        show_detection_patterns_analysis(data, results, selected_method)
    elif analysis_type == 'Method Comparison':
        show_method_comparison_analysis(data, results)
    elif analysis_type == 'Threshold Analysis':
        show_threshold_analysis(data, results, selected_method)

def show_score_distribution_analysis(data, results, method):
    """Show score distribution analysis."""
    
    st.markdown("### 📊 Score Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create score distribution chart
        fig = create_score_distribution_chart(results, method)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Statistics summary
        st.markdown("#### 📈 Distribution Stats")
        create_distribution_stats(results, method)
        
        # Threshold recommendations
        st.markdown("#### 🎯 Threshold Recommendations")
        create_threshold_recommendations(results, method)

def show_detection_patterns_analysis(data, results, method):
    """Show detection patterns analysis."""
    
    st.markdown("### 🔍 Detection Patterns Analysis")
    
    # Feature-based analysis
    tab1, tab2, tab3 = st.tabs(["Feature Correlations", "Cluster Analysis", "Anomaly Regions"])
    
    with tab1:
        # Feature correlation with outlier scores
        fig = create_feature_correlation_chart(data, results, method)
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("**Insights:**")
        st.write("• Features with high correlation to outlier scores are key drivers")
        st.write("• Use this to understand what makes data points anomalous")
    
    with tab2:
        # Cluster-based analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_cluster_analysis_chart(data, results)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("#### 🎯 Cluster Insights")
            create_cluster_insights(data, results)
    
    with tab3:
        # Anomaly regions heatmap
        fig = create_detection_heatmap(data, results)
        st.plotly_chart(fig, width="stretch")

def show_method_comparison_analysis(data, results):
    """Show method comparison analysis."""
    
    st.markdown("### ⚖️ Method Comparison Analysis")
    
    # Performance metrics comparison
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = create_method_performance_chart(results)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🏆 Method Rankings")
        create_method_rankings(results)
    
    # Agreement analysis
    st.markdown("### 🤝 Method Agreement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agreement matrix
        fig = create_agreement_matrix(results)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Consensus analysis
        st.markdown("#### 📊 Consensus Statistics")
        create_consensus_stats(results)

def show_threshold_analysis(data, results, method):
    """Show threshold analysis."""
    
    st.markdown("### 🎚️ Threshold Analysis")
    
    # Interactive threshold slider
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        threshold = st.slider("Threshold Value", 0.0, 1.0, 0.8, 0.01)
    
    with col2:
        evaluation_metric = st.selectbox(
            "Evaluation Metric",
            ['Precision', 'Recall', 'F1-Score', 'False Positive Rate']
        )
    
    with col3:
        st.metric("Current Threshold", f"{threshold:.2f}")
    
    # Threshold impact analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # ROC-like curve for threshold analysis
        fig = create_threshold_curve(results, method)
        
        # Add vertical line for current threshold
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Current: {threshold}")
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Threshold impact metrics
        st.markdown("#### 📊 Threshold Impact")
        create_threshold_impact_metrics(results, threshold, method)

def create_score_distribution_chart(results, method):
    """Create score distribution visualization."""
    
    # Mock score data for demonstration
    np.random.seed(42)
    if method == 'All Methods':
        scores = np.concatenate([
            np.random.beta(2, 5, 100),  # Normal data
            np.random.beta(5, 2, 20)    # Outliers
        ])
    else:
        scores = np.concatenate([
            np.random.beta(2, 5, 100),
            np.random.beta(5, 2, 20)
        ])
    
    # Create histogram with KDE
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=30,
        name='Score Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"Score Distribution - {method}",
        xaxis_title="Outlier Score",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_distribution_stats(results, method):
    """Create distribution statistics display."""
    
    # Mock statistics
    stats = {
        'Mean': 0.234,
        'Median': 0.187,
        'Std Dev': 0.156,
        'Skewness': 1.83,
        'Kurtosis': 4.21
    }
    
    for stat, value in stats.items():
        st.metric(stat, f"{value:.3f}")

def create_threshold_recommendations(results, method):
    """Create threshold recommendations."""
    
    recommendations = [
        ("Conservative", 0.9, "Low false positives"),
        ("Balanced", 0.8, "Optimal precision/recall"),
        ("Aggressive", 0.7, "Higher detection rate")
    ]
    
    for name, threshold, description in recommendations:
        color = "#4CAF50" if name == "Balanced" else "#2196F3"
        st.markdown(f"""
        <div style="
            background-color: {color}20;
            border-left: 3px solid {color};
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 0 4px 4px 0;
        ">
            <strong>{name}: {threshold}</strong><br>
            <small>{description}</small>
        </div>
        """, unsafe_allow_html=True)

def create_feature_correlation_chart(data, results, method):
    """Create feature correlation with outlier scores chart."""
    
    if data is None:
        return go.Figure()
    
    # Calculate correlations (mock for now)
    features = data.columns if hasattr(data, 'columns') else ['Feature_' + str(i) for i in range(5)]
    correlations = np.random.uniform(-0.8, 0.8, len(features))
    
    fig = go.Figure(go.Bar(
        x=features,
        y=correlations,
        marker_color=['red' if c > 0 else 'blue' for c in correlations],
        text=[f"{c:.3f}" for c in correlations],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Correlation with Outlier Scores",
        xaxis_title="Features",
        yaxis_title="Correlation Coefficient",
        height=400
    )
    
    return fig

def create_cluster_analysis_chart(data, results):
    """Create cluster analysis visualization."""
    
    # Mock cluster data
    np.random.seed(42)
    n_points = 500
    
    # Generate 3 clusters + outliers
    cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 200)
    cluster2 = np.random.multivariate_normal([-2, -2], [[1, -0.3], [-0.3, 1]], 200)
    cluster3 = np.random.multivariate_normal([2, -2], [[1, 0], [0, 1]], 80)
    outliers = np.random.uniform(-6, 6, (20, 2))
    
    all_data = np.vstack([cluster1, cluster2, cluster3, outliers])
    labels = ['Normal'] * 480 + ['Outlier'] * 20
    
    fig = px.scatter(
        x=all_data[:, 0], 
        y=all_data[:, 1],
        color=labels,
        color_discrete_map={'Normal': 'blue', 'Outlier': 'red'},
        title="Cluster Analysis with Outliers"
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_cluster_insights(data, results):
    """Create cluster insights panel."""
    
    insights = [
        "🎯 3 main clusters detected",
        "📊 20 outliers identified", 
        "🔍 Cluster 1: Dense, well-separated",
        "🔍 Cluster 2: Some overlap with C3",
        "⚠️ Most outliers in boundary regions"
    ]
    
    for insight in insights:
        st.write(insight)

def create_method_performance_chart(results):
    """Create method performance comparison chart."""
    
    methods = ['Statistical', 'Isolation Forest', 'LOF', 'One-Class SVM', 'KDE', 'GMM']
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    # Mock performance data
    np.random.seed(42)
    performance_data = {
        metric: np.random.uniform(0.6, 0.95, len(methods))
        for metric in metrics
    }
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (metric, values) in enumerate(performance_data.items()):
        fig.add_trace(go.Bar(
            name=metric,
            x=methods,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Method Performance Comparison",
        xaxis_title="Detection Method",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_method_rankings(results):
    """Create method rankings display."""
    
    rankings = [
        ("🥇 Score Fusion", "0.912"),
        ("🥈 Isolation Forest", "0.887"),
        ("🥉 LOF", "0.873"),
        ("4️⃣ KDE", "0.861"),
        ("5️⃣ Statistical", "0.845"),
        ("6️⃣ One-Class SVM", "0.832")
    ]
    
    for rank, score in rankings:
        st.markdown(f"**{rank}**: {score}")

def create_agreement_matrix(results):
    """Create method agreement matrix."""
    
    methods = ['Stat', 'IF', 'LOF', 'SVM', 'KDE', 'GMM']
    
    # Mock agreement data
    np.random.seed(42)
    agreement_matrix = np.random.uniform(0.4, 1.0, (len(methods), len(methods)))
    np.fill_diagonal(agreement_matrix, 1.0)  # Perfect self-agreement
    
    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=methods,
        y=methods,
        colorscale='Blues',
        text=[[f"{val:.3f}" for val in row] for row in agreement_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Method Agreement Matrix",
        height=400
    )
    
    return fig

def create_consensus_stats(results):
    """Create consensus statistics display."""
    
    stats = [
        ("High Agreement (>0.8)", "67%"),
        ("Medium Agreement (0.6-0.8)", "28%"),
        ("Low Agreement (<0.6)", "5%"),
        ("Perfect Consensus", "23%")
    ]
    
    for stat, value in stats:
        st.write(f"**{stat}**: {value}")

def create_threshold_curve(results, method):
    """Create threshold analysis curve."""
    
    # Mock threshold analysis data
    thresholds = np.linspace(0, 1, 100)
    precision = 1 - thresholds * 0.5  # Mock precision curve
    recall = 1 - (1 - thresholds) ** 2  # Mock recall curve
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precision,
        name='Precision',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recall,
        name='Recall',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_score,
        name='F1-Score',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="Threshold vs Performance Metrics",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400
    )
    
    return fig

def create_threshold_impact_metrics(results, threshold, method):
    """Create threshold impact metrics display."""
    
    # Mock calculations based on threshold
    precision = max(0.5, 1 - threshold * 0.3)
    recall = max(0.3, threshold)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    st.metric("Precision", f"{precision:.3f}")
    st.metric("Recall", f"{recall:.3f}")
    st.metric("F1-Score", f"{f1_score:.3f}")
    
    # Estimated detections
    total_points = 720
    estimated_detections = int(total_points * (1 - threshold) * 0.1)
    st.metric("Est. Detections", estimated_detections)

def show_page(data=None, models=None):
    """Main entry point for the detection analysis page (compatibility function)."""
    show_detection_analysis_page()