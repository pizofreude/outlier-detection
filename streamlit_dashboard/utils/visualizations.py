"""
Visualization Utilities
======================

Advanced plotting utilities for the outlier detection dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

class PlotGenerator:
    """Generate interactive plots for the dashboard."""
    
    def __init__(self):
        # Color palette for consistency
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'ind': '#2ca02c',
            'ood': '#d62728',
            'neutral': '#7f7f7f'
        }
    
    def create_scatter_plot(self, data, detector_results=None, title="Data Distribution"):
        """Create interactive scatter plot of data points."""
        fig = go.Figure()
        
        # Plot training data
        fig.add_trace(go.Scatter(
            x=data['X_train'][:, 0],
            y=data['X_train'][:, 1],
            mode='markers',
            name='Training (IND)',
            marker=dict(
                color=self.colors['primary'],
                size=6,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Training</b><br>X1: %{x:.3f}<br>X2: %{y:.3f}<extra></extra>'
        ))
        
        # Plot IND test data
        fig.add_trace(go.Scatter(
            x=data['X_ind_test'][:, 0],
            y=data['X_ind_test'][:, 1],
            mode='markers',
            name='IND Test',
            marker=dict(
                color=self.colors['success'],
                size=8,
                symbol='triangle-up',
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>IND Test</b><br>X1: %{x:.3f}<br>X2: %{y:.3f}<extra></extra>'
        ))
        
        # Plot OOD data
        fig.add_trace(go.Scatter(
            x=data['X_ood'][:, 0],
            y=data['X_ood'][:, 1],
            mode='markers',
            name='OOD Test',
            marker=dict(
                color=self.colors['danger'],
                size=10,
                symbol='x',
                opacity=0.9,
                line=dict(width=2)
            ),
            hovertemplate='<b>OOD Test</b><br>X1: %{x:.3f}<br>X2: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_anomaly_heatmap(self, X, scores, title="Anomaly Score Heatmap"):
        """Create heatmap of anomaly scores across feature space."""
        # Create a grid for the heatmap
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # For demonstration, create synthetic scores
        grid_scores = np.exp(-((xx - np.mean(X[:, 0]))**2 + (yy - np.mean(X[:, 1]))**2) / 2)
        
        fig = go.Figure(data=go.Heatmap(
            z=grid_scores,
            x=np.linspace(x_min, x_max, 50),
            y=np.linspace(y_min, y_max, 50),
            colorscale='RdYlBu_r',
            hoverongaps=False,
            colorbar=dict(title="Anomaly Score")
        ))
        
        # Overlay data points
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color='black', size=4, opacity=0.7),
            name='Data Points',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_roc_curve(self, y_true, y_scores, title="ROC Curve"):
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=self.colors['primary'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve"):
        """Create Precision-Recall curve plot."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color=self.colors['success'], width=3),
            fill='tonexty'
        ))
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        fig.add_hline(y=baseline, line_dash="dash", 
                     annotation_text=f"Baseline = {baseline:.3f}")
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_score_distribution(self, scores_dict, title="Score Distributions"):
        """Create distribution plots for different detector scores."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (detector, scores) in enumerate(scores_dict.items()):
            fig.add_trace(go.Histogram(
                x=scores,
                name=detector,
                opacity=0.7,
                nbinsx=30,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            template="plotly_white",
            height=500,
            barmode='overlay'
        )
        
        return fig
    
    def create_correlation_heatmap(self, corr_matrix, title="Correlation Matrix"):
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=dict(text=title, font_size=18, x=0.5),
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_detector_comparison(self, benchmark_df, metric='auroc'):
        """Create detector comparison chart."""
        ood_data = benchmark_df[benchmark_df['dataset'] == 'OOD_Test']
        
        fig = go.Figure(data=[
            go.Bar(
                x=ood_data['detector'],
                y=ood_data[metric],
                marker_color=self.colors['primary'],
                text=ood_data[metric].round(3),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=dict(text=f"Detector Performance Comparison ({metric.upper()})", font_size=18, x=0.5),
            xaxis_title="Detector",
            yaxis_title=metric.upper(),
            template="plotly_white",
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_business_metrics_chart(self, metrics_data):
        """Create business metrics visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI Analysis', 'Cost Breakdown', 'Alert Volume', 'Performance Trend'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # ROI Indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics_data.get('roi', 250),
            title={'text': "ROI (%)"},
            gauge={'axis': {'range': [0, 500]},
                   'bar': {'color': self.colors['success']},
                   'steps': [{'range': [0, 100], 'color': "lightgray"},
                            {'range': [100, 300], 'color': "yellow"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 200}}
        ), row=1, col=1)
        
        # Cost Breakdown Pie Chart
        fig.add_trace(go.Pie(
            labels=['Prevention Cost', 'Investigation Cost', 'False Positive Cost'],
            values=[30, 40, 30],
            hole=0.3
        ), row=1, col=2)
        
        # Alert Volume Bar Chart
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        alerts = np.random.randint(20, 80, 7)
        fig.add_trace(go.Bar(
            x=days, y=alerts,
            marker_color=self.colors['warning']
        ), row=2, col=1)
        
        # Performance Trend
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        performance = 0.85 + 0.1 * np.sin(np.arange(30) * 0.2) + np.random.normal(0, 0.02, 30)
        fig.add_trace(go.Scatter(
            x=dates, y=performance,
            mode='lines+markers',
            line=dict(color=self.colors['primary'])
        ), row=2, col=2)
        
        fig.update_layout(
            title_text="Business Intelligence Dashboard",
            height=600,
            showlegend=False
        )
        
        return fig

# Standalone functions for easy import
def create_scatter_plot(data, detector_results=None, title="Data Distribution"):
    """Create interactive scatter plot of data points (standalone function)."""
    plotter = PlotGenerator()
    return plotter.create_scatter_plot(data, detector_results, title)

def create_detection_heatmap(scores_matrix, model_names=None, title="Detection Scores Heatmap"):
    """Create heatmap showing detection scores across models."""
    plotter = PlotGenerator()
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(scores_matrix.shape[1])]
    
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix.T,
        x=[f'Sample {i+1}' for i in range(scores_matrix.shape[0])],
        y=model_names,
        colorscale='Viridis',
        colorbar=dict(title="Outlier Score")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Data Samples",
        yaxis_title="Detection Models",
        height=400
    )
    
    return fig

def create_correlation_heatmap(data, title="Feature Correlation"):
    """Create correlation heatmap of features."""
    if isinstance(data, dict) and 'X_train' in data:
        # Convert numpy array to DataFrame for easier handling
        df = pd.DataFrame(data['X_train'], columns=[f'Feature_{i}' for i in range(data['X_train'].shape[1])])
    else:
        df = pd.DataFrame(data)
    
    correlation_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        height=500
    )
    
    return fig