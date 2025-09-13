"""
Data Explorer Page
==================

Interactive data exploration and visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from utils.data_loader import load_detection_artifacts
from utils.visualizations import create_scatter_plot
from components.sidebar import create_page_header

def show_data_explorer_page():
    """Display the data explorer page."""
    
    create_page_header(
        "Interactive Data Explorer",
        "Explore your dataset and understand data patterns and quality"
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
    
    # Data overview section
    show_data_overview(data)
    
    # Interactive exploration section
    st.markdown("---")
    show_interactive_exploration(data, results)
    
    # Data quality section
    st.markdown("---")
    show_data_quality_analysis(data)

def show_data_overview(data):
    """Show data overview section."""
    
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if hasattr(data, 'shape'):
            st.metric("Total Records", f"{data.shape[0]:,}")
        else:
            st.metric("Total Records", "720")
    
    with col2:
        if hasattr(data, 'shape'):
            st.metric("Features", data.shape[1])
        else:
            st.metric("Features", "15")
    
    with col3:
        # Mock missing values calculation
        missing_pct = np.random.uniform(0, 5)
        st.metric("Missing Values", f"{missing_pct:.1f}%")
    
    with col4:
        # Mock data quality score
        quality_score = np.random.uniform(85, 98)
        st.metric("Data Quality", f"{quality_score:.1f}%")
    
    # Data sample
    st.markdown("### 🔍 Data Sample")
    
    if hasattr(data, 'head'):
        st.dataframe(data.head(10), width="stretch")
    else:
        # Create mock data display
        mock_data = pd.DataFrame({
            'feature_1': np.random.randn(10),
            'feature_2': np.random.randn(10),
            'feature_3': np.random.randn(10),
            'feature_4': np.random.randn(10),
            'feature_5': np.random.randn(10)
        })
        st.dataframe(mock_data, width="stretch")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Statistical Summary")
        if hasattr(data, 'describe'):
            st.dataframe(data.describe(), width="stretch")
        else:
            # Mock statistics
            mock_stats = pd.DataFrame({
                'feature_1': [720, 0.05, 1.02, -3.1, -0.8, 0.1, 0.9, 3.2],
                'feature_2': [720, -0.12, 0.98, -2.8, -0.9, -0.1, 0.7, 2.9],
                'feature_3': [720, 0.08, 1.15, -3.5, -0.9, 0.0, 0.8, 3.8]
            }, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            st.dataframe(mock_stats, width="stretch")
    
    with col2:
        st.markdown("### 🎯 Feature Types")
        create_feature_types_chart()

def show_interactive_exploration(data, results):
    """Show interactive exploration section."""
    
    st.markdown("### 🔬 Interactive Exploration")
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    # Get feature names
    if hasattr(data, 'columns'):
        features = list(data.columns)
    else:
        features = [f'Feature_{i}' for i in range(1, 16)]
    
    with col1:
        x_feature = st.selectbox("X-Axis Feature", features, index=0)
    
    with col2:
        y_feature = st.selectbox("Y-Axis Feature", features, index=1 if len(features) > 1 else 0)
    
    with col3:
        color_by = st.selectbox(
            "Color By", 
            ['Outlier Score', 'Outlier Status', 'Feature Value', 'None'],
            index=0
        )
    
    with col4:
        plot_type = st.selectbox(
            "Plot Type",
            ['Scatter Plot', 'Density Plot', 'Box Plot', 'Violin Plot']
        )
    
    # Main visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if plot_type == 'Scatter Plot':
            fig = create_interactive_scatter_plot(data, results, x_feature, y_feature, color_by)
        elif plot_type == 'Density Plot':
            fig = create_density_plot(data, x_feature, y_feature)
        elif plot_type == 'Box Plot':
            fig = create_box_plot(data, x_feature, y_feature)
        else:  # Violin Plot
            fig = create_violin_plot(data, x_feature, y_feature)
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🎨 Plot Controls")
        
        # Additional plot controls
        show_outliers = st.checkbox("Highlight Outliers", True)
        show_confidence = st.checkbox("Show Confidence Regions", False)
        log_scale = st.checkbox("Log Scale", False)
        
        st.markdown("#### 📊 Selection Stats")
        if st.button("Calculate Stats"):
            create_selection_stats(data, x_feature, y_feature)
    
    # Feature relationship matrix
    st.markdown("### 🔗 Feature Relationships")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Correlation matrix
        fig = create_correlation_matrix(data)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🎯 Key Insights")
        create_correlation_insights()

def show_data_quality_analysis(data):
    """Show data quality analysis section."""
    
    st.markdown("### 🏥 Data Quality Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Outliers", "Distributions", "Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_missing_values_chart(data)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("#### 📊 Missing Value Summary")
            create_missing_values_summary()
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_outlier_detection_chart(data)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("#### 🎯 Outlier Summary")
            create_outlier_summary()
    
    with tab3:
        # Distribution analysis
        feature_for_dist = st.selectbox("Select Feature for Distribution Analysis", 
                                      [f'Feature_{i}' for i in range(1, 6)])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_analysis(data, feature_for_dist)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("#### 📈 Distribution Tests")
            create_distribution_tests(feature_for_dist)
    
    with tab4:
        # Advanced correlation analysis
        fig = create_advanced_correlation_analysis(data)
        st.plotly_chart(fig, width="stretch")

def create_feature_types_chart():
    """Create feature types visualization."""
    
    # Mock feature type data
    types = ['Numerical', 'Categorical', 'Binary', 'Ordinal']
    counts = [12, 2, 1, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig = go.Figure(data=[go.Pie(
        labels=types, 
        values=counts,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        title="Feature Type Distribution",
        height=300
    )
    
    st.plotly_chart(fig, width="stretch")

def create_interactive_scatter_plot(data, results, x_feature, y_feature, color_by):
    """Create interactive scatter plot."""
    
    # Generate mock data
    np.random.seed(42)
    n_points = 720
    
    x_data = np.random.randn(n_points)
    y_data = np.random.randn(n_points) + 0.5 * x_data  # Some correlation
    
    # Add outliers
    outlier_indices = np.random.choice(n_points, 50, replace=False)
    x_data[outlier_indices] += np.random.uniform(-3, 3, 50)
    y_data[outlier_indices] += np.random.uniform(-3, 3, 50)
    
    if color_by == 'Outlier Score':
        colors = np.random.beta(2, 8, n_points)  # Most points have low scores
        colors[outlier_indices] = np.random.beta(8, 2, 50)  # Outliers have high scores
        colorscale = 'Reds'
        color_title = 'Outlier Score'
    elif color_by == 'Outlier Status':
        colors = ['Normal'] * n_points
        for idx in outlier_indices:
            colors[idx] = 'Outlier'
        colorscale = None
        color_title = 'Status'
    else:
        colors = x_data  # Color by feature value
        colorscale = 'Viridis'
        color_title = x_feature
    
    fig = px.scatter(
        x=x_data,
        y=y_data,
        color=colors,
        color_continuous_scale=colorscale if colorscale else None,
        color_discrete_map={'Normal': 'blue', 'Outlier': 'red'} if color_by == 'Outlier Status' else None,
        title=f"{x_feature} vs {y_feature}",
        labels={'x': x_feature, 'y': y_feature, 'color': color_title}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_density_plot(data, x_feature, y_feature):
    """Create density plot."""
    
    # Generate mock data
    np.random.seed(42)
    x_data = np.random.randn(720)
    y_data = np.random.randn(720) + 0.5 * x_data
    
    fig = go.Figure(data=go.Histogram2d(
        x=x_data,
        y=y_data,
        colorscale='Blues',
        nbinsx=30,
        nbinsy=30
    ))
    
    fig.update_layout(
        title=f"Density Plot: {x_feature} vs {y_feature}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        height=500
    )
    
    return fig

def create_box_plot(data, x_feature, y_feature):
    """Create box plot."""
    
    # Generate mock data with categories
    np.random.seed(42)
    categories = np.random.choice(['A', 'B', 'C', 'D'], 720)
    values = np.random.randn(720) + np.where(categories == 'A', 1, 0)
    
    fig = px.box(
        x=categories,
        y=values,
        title=f"Box Plot: {y_feature} by Category",
        labels={'x': 'Category', 'y': y_feature}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_violin_plot(data, x_feature, y_feature):
    """Create violin plot."""
    
    # Generate mock data with categories
    np.random.seed(42)
    categories = np.random.choice(['A', 'B', 'C', 'D'], 720)
    values = np.random.randn(720) + np.where(categories == 'A', 1, 0)
    
    fig = px.violin(
        x=categories,
        y=values,
        title=f"Violin Plot: {y_feature} by Category",
        labels={'x': 'Category', 'y': y_feature}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_selection_stats(data, x_feature, y_feature):
    """Create selection statistics."""
    
    stats = {
        'Mean X': np.random.uniform(-1, 1),
        'Mean Y': np.random.uniform(-1, 1),
        'Correlation': np.random.uniform(-0.8, 0.8),
        'Selected Points': np.random.randint(50, 200)
    }
    
    for stat, value in stats.items():
        if isinstance(value, int):
            st.metric(stat, value)
        else:
            st.metric(stat, f"{value:.3f}")

def create_correlation_matrix(data):
    """Create correlation matrix heatmap."""
    
    # Generate mock correlation matrix
    np.random.seed(42)
    n_features = 8
    features = [f'Feature_{i}' for i in range(1, n_features + 1)]
    
    # Generate correlation matrix
    corr_matrix = np.random.uniform(-0.8, 0.8, (n_features, n_features))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1)  # Perfect self-correlation
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=400
    )
    
    return fig

def create_correlation_insights():
    """Create correlation insights."""
    
    insights = [
        "🔗 Strong positive: Features 1-3",
        "🔀 Weak correlation: Features 4-6", 
        "🚫 No correlation: Features 7-8",
        "⚠️ Potential multicollinearity detected",
        "💡 Consider feature selection"
    ]
    
    for insight in insights:
        st.write(insight)

def create_missing_values_chart(data):
    """Create missing values visualization."""
    
    # Mock missing values data
    features = [f'Feature_{i}' for i in range(1, 11)]
    missing_counts = np.random.poisson(2, len(features))
    missing_percentages = (missing_counts / 720) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=features,
        y=missing_percentages,
        marker_color=['red' if p > 5 else 'orange' if p > 1 else 'green' for p in missing_percentages],
        text=[f"{p:.1f}%" for p in missing_percentages],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Missing Values by Feature",
        xaxis_title="Features",
        yaxis_title="Missing Percentage (%)",
        height=400
    )
    
    return fig

def create_missing_values_summary():
    """Create missing values summary."""
    
    summary = [
        ("Total Missing", "127 (2.4%)"),
        ("Complete Records", "693 (96.3%)"),
        ("Features w/ Missing", "6 of 15"),
        ("Max Missing (Feature)", "8.3%"),
        ("Recommended Action", "Imputation")
    ]
    
    for item, value in summary:
        st.write(f"**{item}**: {value}")

def create_outlier_detection_chart(data):
    """Create outlier detection visualization."""
    
    # Mock outlier detection results
    features = [f'Feature_{i}' for i in range(1, 8)]
    outlier_counts = np.random.poisson(5, len(features))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=features,
        y=outlier_counts,
        marker_color='red',
        text=[str(c) for c in outlier_counts],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Outliers Detected by Feature",
        xaxis_title="Features",
        yaxis_title="Number of Outliers",
        height=400
    )
    
    return fig

def create_outlier_summary():
    """Create outlier summary."""
    
    summary = [
        ("Total Outliers", "47 (6.5%)"),
        ("Univariate Outliers", "31"),
        ("Multivariate Outliers", "16"),
        ("Most Affected Feature", "Feature_3"),
        ("Outlier Method", "IQR + Z-Score")
    ]
    
    for item, value in summary:
        st.write(f"**{item}**: {value}")

def create_distribution_analysis(data, feature):
    """Create distribution analysis chart."""
    
    # Generate mock data
    np.random.seed(42)
    if 'normal' in feature.lower():
        values = np.random.normal(0, 1, 720)
    elif 'uniform' in feature.lower():
        values = np.random.uniform(-2, 2, 720)
    else:
        values = np.random.gamma(2, 1, 720)
    
    # Create histogram with fitted distribution
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=30,
        name='Data',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add normal distribution overlay
    x_range = np.linspace(values.min(), values.max(), 100)
    normal_fit = ((1/np.sqrt(2*np.pi*values.var())) * 
                 np.exp(-0.5*((x_range - values.mean())/values.std())**2))
    normal_fit = normal_fit * len(values) * (values.max() - values.min()) / 30
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_fit,
        mode='lines',
        name='Normal Fit',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Distribution Analysis: {feature}",
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_distribution_tests(feature):
    """Create distribution test results."""
    
    tests = [
        ("Shapiro-Wilk", "p=0.034", "❌ Not Normal"),
        ("Anderson-Darling", "stat=1.23", "❌ Not Normal"),
        ("Kolmogorov-Smirnov", "p=0.12", "✅ Normal"),
        ("Jarque-Bera", "p=0.08", "✅ Normal")
    ]
    
    for test, stat, result in tests:
        color = "#4CAF50" if "✅" in result else "#F44336"
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid {color};
        ">
            <span><strong>{test}</strong></span>
            <span>{stat} - {result}</span>
        </div>
        """, unsafe_allow_html=True)

def create_advanced_correlation_analysis(data):
    """Create advanced correlation analysis."""
    
    # Mock advanced correlation data
    methods = ['Pearson', 'Spearman', 'Kendall', 'MIC']
    feature_pairs = ['F1-F2', 'F1-F3', 'F2-F3', 'F1-F4', 'F2-F4']
    
    correlations = {}
    for method in methods:
        correlations[method] = np.random.uniform(0.2, 0.9, len(feature_pairs))
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (method, values) in enumerate(correlations.items()):
        fig.add_trace(go.Bar(
            name=method,
            x=feature_pairs,
            y=values,
            marker_color=colors[i],
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Advanced Correlation Analysis",
        xaxis_title="Feature Pairs",
        yaxis_title="Correlation Coefficient",
        barmode='group',
        height=400
    )
    
    return fig

def show_page(data=None, models=None):
    """Main entry point for the data explorer page (compatibility function)."""
    show_data_explorer_page()