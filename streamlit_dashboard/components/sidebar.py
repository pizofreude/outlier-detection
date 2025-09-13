"""
Sidebar Navigation Component
===========================

Navigation sidebar for the dashboard.
"""

import streamlit as st

def create_sidebar():
    """Create the main navigation sidebar."""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🎯 Control Center</h2>
        <p style="font-size: 0.9rem; color: #ffffff;">
            Production Outlier Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    pages = [
        "🏠 Overview",
        "📊 Detection Analysis", 
        "🔍 Data Explorer",
        "⚙️ Model Performance",
        "🎯 Threshold Management",
        "📈 Business Intelligence",
        "🚨 Alert Center"
    ]
    
    # Use radio buttons for navigation
    selected_page = st.sidebar.radio(
        "Navigate to:",
        pages,
        index=0
    )
    
    # System status section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Mock system status
    status_items = [
        ("Data Pipeline", "🟢 Healthy"),
        ("Model Performance", "🟢 Good"),
        ("Alert System", "🟡 Warning"),
        ("Data Quality", "🟢 Excellent")
    ]
    
    for item, status in status_items:
        st.sidebar.markdown(f"**{item}**: {status}")
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Today's Alerts", "47", "↑ 12%")
    with col2:
        st.metric("System Uptime", "99.8%", "↑ 0.1%")
    
    # Settings and info
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("⚙️ Settings"):
        st.checkbox("Auto-refresh", value=True)
        st.selectbox("Refresh Rate", ["30s", "1min", "5min"], index=1)
        st.slider("Alert Threshold", 0.0, 1.0, 0.8, 0.1)
    
    with st.sidebar.expander("ℹ️ System Info"):
        st.markdown("""
        **Version**: 1.0.0  
        **Last Updated**: Sept 2025  
        **Models**: 6 detectors + fusion  
        **Data Points**: 720 total  
        """)
    
    return selected_page

def create_page_header(title, description=None):
    """Create a consistent page header."""
    st.markdown(f"""
    <div style="padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 2rem;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">{title}</h1>
        {f'<p style="color: #666; font-size: 1.1rem; margin: 0;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)