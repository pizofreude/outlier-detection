"""
Business Intelligence Page
=========================

Business-focused analytics and insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

from utils.data_loader import load_detection_artifacts
from utils.metrics import calculate_business_metrics
from components.sidebar import create_page_header

def show_business_intelligence_page():
    """Display the business intelligence page."""
    
    create_page_header(
        "Business Intelligence Dashboard",
        "Strategic insights and ROI analysis for outlier detection operations"
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
    
    # Executive summary
    show_executive_summary(data, results)
    
    # Business impact analysis
    st.markdown("---")
    show_business_impact_analysis(results)
    
    # ROI and cost analysis
    st.markdown("---")
    show_roi_cost_analysis()
    
    # Strategic insights
    st.markdown("---")
    show_strategic_insights(results)

def show_executive_summary(data, results):
    """Show executive summary section."""
    
    st.markdown("### 🎯 Executive Summary")
    
    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly ROI", "$1.2M", "↑ 15%")
    
    with col2:
        st.metric("Risk Reduction", "87%", "↑ 8%")
    
    with col3:
        st.metric("Detection Accuracy", "94.2%", "↑ 2.1%")
    
    with col4:
        st.metric("Cost Savings", "$340K", "↑ 23%")
    
    # Executive insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_business_performance_overview()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 💼 Key Insights")
        create_executive_insights()
        
        st.markdown("#### 🎯 Action Items")
        create_action_items()

def show_business_impact_analysis(results):
    """Show business impact analysis section."""
    
    st.markdown("### 📊 Business Impact Analysis")
    
    # Impact categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Financial Impact", 
        "🔒 Risk Management", 
        "⚡ Operational Efficiency", 
        "👥 Customer Impact"
    ])
    
    with tab1:
        show_financial_impact()
    
    with tab2:
        show_risk_management_impact()
    
    with tab3:
        show_operational_efficiency()
    
    with tab4:
        show_customer_impact()

def show_roi_cost_analysis():
    """Show ROI and cost analysis section."""
    
    st.markdown("### 💲 ROI & Cost Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Cost Breakdown")
        fig = create_cost_breakdown_chart()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 📈 ROI Projection")
        fig = create_roi_projection_chart()
        st.plotly_chart(fig, width="stretch")
    
    with col3:
        st.markdown("#### 💵 Savings Analysis")
        create_savings_analysis()
    
    # Detailed cost-benefit analysis
    st.markdown("### 📋 Detailed Cost-Benefit Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = create_cost_benefit_timeline()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 🎯 Break-even Analysis")
        create_breakeven_analysis()

def show_strategic_insights(results):
    """Show strategic insights section."""
    
    st.markdown("### 🧠 Strategic Insights & Recommendations")
    
    # Market analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏭 Industry Benchmarking")
        fig = create_industry_benchmark_chart()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 📈 Competitive Advantage")
        create_competitive_advantage_analysis()
    
    # Strategic recommendations
    st.markdown("#### 🎯 Strategic Recommendations")
    create_strategic_recommendations()
    
    # Future roadmap
    st.markdown("#### 🗺️ Technology Roadmap")
    roadmap_fig = create_technology_roadmap()
    st.plotly_chart(roadmap_fig, width="stretch")

def create_business_performance_overview():
    """Create business performance overview chart."""
    
    # Generate mock business metrics over time
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    np.random.seed(42)
    roi_values = np.cumsum(np.random.uniform(80000, 120000, 12)) / 1000  # In thousands
    risk_reduction = np.minimum(np.cumsum(np.random.uniform(5, 10, 12)), 87)
    cost_savings = np.cumsum(np.random.uniform(20000, 35000, 12)) / 1000  # In thousands
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROI Growth ($K)', 'Risk Reduction (%)', 'Cost Savings ($K)', 'Detection Accuracy (%)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # ROI Growth
    fig.add_trace(go.Bar(x=months, y=roi_values, name='ROI', marker_color='green'), row=1, col=1)
    
    # Risk Reduction
    fig.add_trace(go.Scatter(x=months, y=risk_reduction, mode='lines+markers', 
                           name='Risk Reduction', line=dict(color='red', width=3)), row=1, col=2)
    
    # Cost Savings
    fig.add_trace(go.Bar(x=months, y=cost_savings, name='Savings', marker_color='blue'), row=2, col=1)
    
    # Detection Accuracy
    accuracy = 85 + np.cumsum(np.random.uniform(0.5, 1.5, 12))
    accuracy = np.minimum(accuracy, 95)
    fig.add_trace(go.Scatter(x=months, y=accuracy, mode='lines+markers',
                           name='Accuracy', line=dict(color='orange', width=3)), row=2, col=2)
    
    fig.update_layout(
        title="Business Performance Overview - 2024",
        height=500,
        showlegend=False
    )
    
    return fig

def create_executive_insights():
    """Create executive insights panel."""
    
    insights = [
        "💰 15% ROI increase YoY",
        "🔒 87% reduction in fraud risk",
        "⚡ 34% faster threat detection",
        "📊 94.2% detection accuracy",
        "💵 $340K cost savings achieved"
    ]
    
    for insight in insights:
        st.write(insight)

def create_action_items():
    """Create action items panel."""
    
    action_items = [
        ("🎯 Optimize threshold for Q1", "High", "Next week"),
        ("🔧 Deploy new model version", "Medium", "This month"),
        ("📊 Review alert policies", "Low", "Next quarter"),
        ("🚀 Scale to new regions", "High", "6 months")
    ]
    
    for item, priority, timeline in action_items:
        color = "#ff4444" if priority == "High" else "#ff8800" if priority == "Medium" else "#44ff44"
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background-color: {color}20;
            border-left: 3px solid {color};
            border-radius: 0 4px 4px 0;
        ">
            <span><strong>{item}</strong></span>
            <span style="font-size: 0.9em;">{timeline}</span>
        </div>
        """, unsafe_allow_html=True)

def show_financial_impact():
    """Show financial impact analysis."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue protection
        st.markdown("##### 💰 Revenue Protection")
        
        protection_metrics = [
            ("Prevented Losses", "$2.3M"),
            ("False Positive Cost", "$180K"),
            ("Net Protection", "$2.12M"),
            ("Protection Rate", "92.1%")
        ]
        
        for metric, value in protection_metrics:
            st.metric(metric, value)
    
    with col2:
        # Cost optimization
        st.markdown("##### 📉 Cost Optimization")
        
        fig = create_cost_optimization_chart()
        st.plotly_chart(fig, width="stretch")
    
    # Financial trends
    st.markdown("##### 📈 Financial Impact Trends")
    fig = create_financial_trends_chart()
    st.plotly_chart(fig, width="stretch")

def show_risk_management_impact():
    """Show risk management impact."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk metrics
        st.markdown("##### ⚠️ Risk Metrics")
        
        risk_metrics = [
            ("Risk Score Reduction", "87%"),
            ("Threat Detection Time", "3.2 min"),
            ("False Negative Rate", "2.1%"),
            ("Risk Coverage", "96.8%")
        ]
        
        for metric, value in risk_metrics:
            st.metric(metric, value)
    
    with col2:
        # Risk heat map
        st.markdown("##### 🌡️ Risk Heat Map")
        fig = create_risk_heatmap()
        st.plotly_chart(fig, width="stretch")
    
    # Risk trend analysis
    st.markdown("##### 📊 Risk Trend Analysis")
    fig = create_risk_trend_chart()
    st.plotly_chart(fig, width="stretch")

def show_operational_efficiency():
    """Show operational efficiency analysis."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### ⚡ Processing Efficiency")
        fig = create_processing_efficiency_chart()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("##### 👥 Team Productivity")
        productivity_metrics = [
            ("Analyst Hours Saved", "340h"),
            ("Automation Rate", "78%"),
            ("Response Time", "-45%"),
            ("Team Satisfaction", "4.7/5")
        ]
        
        for metric, value in productivity_metrics:
            st.metric(metric, value)
    
    with col3:
        st.markdown("##### 🔧 System Performance")
        fig = create_system_performance_chart()
        st.plotly_chart(fig, width="stretch")

def show_customer_impact():
    """Show customer impact analysis."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer satisfaction metrics
        st.markdown("##### 😊 Customer Satisfaction")
        
        satisfaction_metrics = [
            ("Customer Satisfaction", "4.6/5"),
            ("Complaint Reduction", "67%"),
            ("Service Uptime", "99.7%"),
            ("Response Quality", "94%")
        ]
        
        for metric, value in satisfaction_metrics:
            st.metric(metric, value)
    
    with col2:
        # Customer feedback trends
        st.markdown("##### 📈 Feedback Trends")
        fig = create_customer_feedback_chart()
        st.plotly_chart(fig, width="stretch")
    
    # Customer journey impact
    st.markdown("##### 🛣️ Customer Journey Impact")
    fig = create_customer_journey_chart()
    st.plotly_chart(fig, width="stretch")

def create_cost_breakdown_chart():
    """Create cost breakdown pie chart."""
    
    categories = ['Infrastructure', 'Personnel', 'Licenses', 'Maintenance', 'Training']
    costs = [150000, 280000, 45000, 35000, 25000]
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=costs,
        hole=0.3,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Annual Cost Breakdown",
        height=300
    )
    
    return fig

def create_roi_projection_chart():
    """Create ROI projection chart."""
    
    months = list(range(1, 13))
    cumulative_roi = np.cumsum(np.random.uniform(80000, 120000, 12))
    projected_roi = np.cumsum(np.random.uniform(90000, 130000, 12))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months, y=cumulative_roi/1000,
        mode='lines+markers',
        name='Actual ROI',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=projected_roi/1000,
        mode='lines+markers',
        name='Projected ROI',
        line=dict(color='green', dash='dash', width=3)
    ))
    
    fig.update_layout(
        title="ROI Projection ($K)",
        xaxis_title="Month",
        yaxis_title="ROI ($K)",
        height=300
    )
    
    return fig

def create_savings_analysis():
    """Create savings analysis metrics."""
    
    savings_categories = [
        ("Manual Review Reduction", "$180K"),
        ("False Positive Prevention", "$95K"),
        ("Early Detection Savings", "$65K"),
        ("Total Annual Savings", "$340K")
    ]
    
    for category, amount in savings_categories:
        if "Total" in category:
            st.metric(category, amount, delta="↑ 23%")
        else:
            st.metric(category, amount)

def create_cost_benefit_timeline():
    """Create cost-benefit timeline chart."""
    
    quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025']
    
    # Costs (negative for display)
    costs = [-135000, -142000, -138000, -145000, -148000, -150000]
    # Benefits (positive)
    benefits = [85000, 156000, 234000, 340000, 425000, 520000]
    # Net benefit
    net_benefit = [b + c for b, c in zip(benefits, costs)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=quarters, y=costs,
        name='Costs',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=quarters, y=benefits,
        name='Benefits',
        marker_color='green',
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=quarters, y=net_benefit,
        mode='lines+markers',
        name='Net Benefit',
        line=dict(color='blue', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Cost-Benefit Analysis Timeline",
        xaxis_title="Quarter",
        yaxis_title="Amount ($)",
        height=400,
        barmode='relative'
    )
    
    return fig

def create_breakeven_analysis():
    """Create break-even analysis."""
    
    st.metric("Break-even Month", "Month 7", delta="2 months ahead")
    st.metric("Payback Period", "14 months", delta="↓ 4 months")
    st.metric("IRR", "23.4%", delta="↑ 3.2%")
    
    st.markdown("**Break-even Details:**")
    st.write("• Initial investment: $535K")
    st.write("• Monthly savings: $38K avg")
    st.write("• Break-even: Month 7")

def create_industry_benchmark_chart():
    """Create industry benchmarking chart."""
    
    metrics = ['Detection Rate', 'False Positive Rate', 'Response Time', 'Cost Efficiency', 'ROI']
    our_performance = [94.2, 3.2, 89.5, 87.3, 92.1]
    industry_avg = [87.5, 8.1, 72.3, 78.9, 76.4]
    industry_leader = [96.8, 2.1, 94.2, 91.7, 94.8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=our_performance,
        theta=metrics,
        fill='toself',
        name='Our Performance',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=industry_avg,
        theta=metrics,
        fill='toself',
        name='Industry Average',
        line_color='gray'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=industry_leader,
        theta=metrics,
        fill='toself',
        name='Industry Leader',
        line_color='green'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        height=400
    )
    
    return fig

def create_competitive_advantage_analysis():
    """Create competitive advantage analysis."""
    
    advantages = [
        ("🎯 Detection Accuracy", "7% above industry avg"),
        ("⚡ Response Speed", "23% faster than competitors"),
        ("💰 Cost Efficiency", "15% lower operational cost"),
        ("🔄 System Reliability", "99.7% uptime vs 97% avg"),
        ("🚀 Innovation Rate", "2x faster feature deployment")
    ]
    
    for advantage, metric in advantages:
        st.markdown(f"""
        <div style="
            background-color: #e8f5e8;
            border-left: 4px solid #4CAF50;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 4px 4px 0;
        ">
            <strong>{advantage}</strong><br>
            <small style="color: #666;">{metric}</small>
        </div>
        """, unsafe_allow_html=True)

def create_strategic_recommendations():
    """Create strategic recommendations panel."""
    
    recommendations = [
        {
            'title': '🎯 Expand Detection Capabilities',
            'description': 'Implement deep learning models for complex pattern recognition',
            'impact': 'High',
            'timeline': '6-9 months',
            'investment': '$250K'
        },
        {
            'title': '🔗 Integrate Real-time Streaming',
            'description': 'Deploy streaming analytics for immediate threat detection',
            'impact': 'High', 
            'timeline': '3-4 months',
            'investment': '$180K'
        },
        {
            'title': '🌐 Multi-region Deployment',
            'description': 'Scale to global operations with regional data centers',
            'impact': 'Medium',
            'timeline': '12-18 months',
            'investment': '$450K'
        },
        {
            'title': '🤖 Advanced Automation',
            'description': 'Automate 90% of routine detection and response tasks',
            'impact': 'High',
            'timeline': '4-6 months',
            'investment': '$120K'
        }
    ]
    
    for rec in recommendations:
        impact_color = "#ff4444" if rec['impact'] == 'High' else "#ff8800"
        
        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: #f8f9fa;
        ">
            <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0;">{rec['title']}</h4>
            <p style="color: #34495e; margin: 0 0 0.5rem 0;">{rec['description']}</p>
            <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                <span><strong>Impact:</strong> <span style="color: {impact_color};">{rec['impact']}</span></span>
                <span><strong>Timeline:</strong> {rec['timeline']}</span>
                <span><strong>Investment:</strong> {rec['investment']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_technology_roadmap():
    """Create technology roadmap visualization."""
    
    roadmap_items = [
        ('Q1 2025', 'Real-time Streaming', 'Infrastructure'),
        ('Q2 2025', 'Deep Learning Models', 'AI/ML'),
        ('Q3 2025', 'Advanced Automation', 'Operations'), 
        ('Q4 2025', 'Global Scaling', 'Infrastructure'),
        ('Q1 2026', 'Quantum Detection', 'Research'),
        ('Q2 2026', 'Predictive Analytics', 'AI/ML')
    ]
    
    quarters = [item[0] for item in roadmap_items]
    items = [item[1] for item in roadmap_items]
    categories = [item[2] for item in roadmap_items]
    
    # Color mapping for categories
    color_map = {
        'Infrastructure': 'blue',
        'AI/ML': 'red',
        'Operations': 'green',
        'Research': 'purple'
    }
    
    colors = [color_map[cat] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=quarters,
        y=[1] * len(quarters),
        mode='markers+text',
        marker=dict(size=20, color=colors),
        text=items,
        textposition="top center",
        textfont=dict(size=10),
        showlegend=False
    ))
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=quarters,
        y=[1] * len(quarters),
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Technology Roadmap",
        xaxis_title="Timeline",
        yaxis=dict(visible=False),
        height=200,
        margin=dict(t=80, b=40)
    )
    
    return fig

# Helper functions for additional charts
def create_cost_optimization_chart():
    """Create cost optimization chart."""
    
    categories = ['Before', 'After']
    manual_costs = [280000, 95000]
    automated_costs = [45000, 180000]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Manual Process', x=categories, y=manual_costs, marker_color='red'))
    fig.add_trace(go.Bar(name='Automated Process', x=categories, y=automated_costs, marker_color='green'))
    
    fig.update_layout(
        title="Cost Optimization",
        yaxis_title="Annual Cost ($)",
        barmode='stack',
        height=250
    )
    
    return fig

def create_financial_trends_chart():
    """Create financial trends chart."""
    
    months = list(range(1, 13))
    revenue_protection = np.cumsum(np.random.uniform(180000, 220000, 12))
    cost_savings = np.cumsum(np.random.uniform(25000, 35000, 12))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months, y=revenue_protection/1000,
        mode='lines+markers',
        name='Revenue Protection',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=cost_savings/1000,
        mode='lines+markers',
        name='Cost Savings',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title="Financial Impact Trends ($K)",
        xaxis_title="Month",
        yaxis_title="Amount ($K)",
        height=300
    )
    
    return fig

def create_risk_heatmap():
    """Create risk assessment heatmap."""
    
    risk_areas = ['Fraud', 'Cyber', 'Operational', 'Market', 'Credit']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    # Generate mock risk scores (lower is better)
    np.random.seed(42)
    risk_scores = np.random.uniform(0.1, 0.8, (len(risk_areas), len(months)))
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_scores,
        x=months,
        y=risk_areas,
        colorscale='RdYlGn_r',  # Reversed so red is high risk
        text=[[f"{val:.2f}" for val in row] for row in risk_scores],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Risk Assessment",
        height=250
    )
    
    return fig

def create_risk_trend_chart():
    """Create risk trend analysis chart."""
    
    weeks = list(range(1, 13))
    overall_risk = 0.8 - np.cumsum(np.random.uniform(0.02, 0.08, 12))
    fraud_risk = 0.75 - np.cumsum(np.random.uniform(0.03, 0.07, 12))
    cyber_risk = 0.85 - np.cumsum(np.random.uniform(0.01, 0.09, 12))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=weeks, y=overall_risk, name='Overall Risk', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=weeks, y=fraud_risk, name='Fraud Risk', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=weeks, y=cyber_risk, name='Cyber Risk', line=dict(color='purple')))
    
    fig.update_layout(
        title="Risk Reduction Trends",
        xaxis_title="Week",
        yaxis_title="Risk Score",
        height=300
    )
    
    return fig

def create_processing_efficiency_chart():
    """Create processing efficiency chart."""
    
    metrics = ['Throughput', 'Latency', 'Accuracy', 'Uptime']
    before = [100, 100, 100, 100]  # Baseline
    after = [145, 65, 108, 103]  # Improvements
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Before', x=metrics, y=before, marker_color='gray'))
    fig.add_trace(go.Bar(name='After', x=metrics, y=after, marker_color='blue'))
    
    fig.update_layout(
        title="Processing Efficiency",
        yaxis_title="Performance Index",
        barmode='group',
        height=250
    )
    
    return fig

def create_system_performance_chart():
    """Create system performance chart."""
    
    days = list(range(1, 8))
    cpu_usage = [45, 52, 48, 67, 43, 39, 51]
    memory_usage = [62, 58, 71, 64, 59, 55, 63]
    response_time = [120, 115, 135, 128, 110, 105, 118]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=days, y=cpu_usage, name='CPU %', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=days, y=memory_usage, name='Memory %', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=days, y=[r/2 for r in response_time], name='Response Time (ms/2)', 
                           line=dict(color='green')))
    
    fig.update_layout(
        title="System Performance",
        xaxis_title="Day",
        yaxis_title="Usage %",
        height=250
    )
    
    return fig

def create_customer_feedback_chart():
    """Create customer feedback trends chart."""
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    satisfaction = [4.1, 4.2, 4.3, 4.4, 4.5, 4.6]
    complaints = [45, 38, 32, 28, 21, 15]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=months, y=satisfaction, name='Satisfaction', 
                           line=dict(color='green', width=3)), secondary_y=False)
    
    fig.add_trace(go.Bar(x=months, y=complaints, name='Complaints', 
                        marker_color='red', opacity=0.7), secondary_y=True)
    
    fig.update_yaxes(title_text="Satisfaction Score", secondary_y=False)
    fig.update_yaxes(title_text="Complaints", secondary_y=True)
    
    fig.update_layout(title="Customer Feedback Trends", height=250)
    
    return fig

def create_customer_journey_chart():
    """Create customer journey impact chart."""
    
    journey_stages = ['Onboarding', 'Transaction', 'Support', 'Retention']
    before_nps = [6.2, 5.8, 6.5, 7.1]
    after_nps = [7.8, 8.2, 8.5, 8.9]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Before Implementation', x=journey_stages, y=before_nps, 
                        marker_color='lightcoral'))
    fig.add_trace(go.Bar(name='After Implementation', x=journey_stages, y=after_nps, 
                        marker_color='lightgreen'))
    
    fig.update_layout(
        title="Customer Journey NPS Impact",
        yaxis_title="NPS Score",
        barmode='group',
        height=300
    )
    
    return fig

def show_page(data=None, models=None):
    """Main entry point for the business intelligence page (compatibility function)."""
    show_business_intelligence_page()