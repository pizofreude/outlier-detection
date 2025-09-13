"""
Production Outlier Detection Control Center
===========================================

A comprehensive Streamlit dashboard for production-ready outlier detection
with business intelligence features, real-time monitoring, and threshold management.

Author: Outlier Detection System
Date: September 2025
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Outlier Detection Control Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the default Streamlit navigation
st.markdown("""
    <style>
        /* Hide all tab-related elements more aggressively */
        .stTabs [data-baseweb="tab-list"] {
            display: none !important;
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        
        /* Hide the entire tabs container */
        .stTabs {
            display: none !important;
        }
        
        /* Hide multipage navigation */
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide the top navigation bar completely */
        .css-1d391kg, .css-1v0mbdj {
            display: none !important;
        }
        
        /* Hide any navigation elements */
        nav[data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide the hamburger menu and footer */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        
        /* Hide the top header navigation */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Hide the top tabs area */
        .css-10trblm {
            display: none !important;
        }
        
        /* Force hide any tab lists */
        [role="tablist"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Import custom utilities
from utils.data_loader import DataLoader, ModelLoader
from utils.visualizations import PlotGenerator
from utils.metrics import MetricsCalculator
from components.kpi_cards import create_kpi_card
from components.sidebar import create_sidebar

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'current_thresholds' not in st.session_state:
        st.session_state.current_thresholds = {
            'zscore': 0.05, 'iqr': 0.05, 'isolation_forest': 0.05,
            'lof': 0.05, 'kde': 0.05, 'gmm': 0.05, 'fusion': 0.05
        }

@st.cache_data
def load_system_data():
    """Load all system data and models with caching."""
    try:
        loader = DataLoader('artifacts/')
        data = loader.load_all_data()
        
        model_loader = ModelLoader('artifacts/')
        models = model_loader.load_all_models()
        
        return data, models, True
    except Exception as e:
        st.error(f"Error loading system data: {e}")
        return None, None, False

def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Outlier Detection Control Center</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading system data and models..."):
            data, models, success = load_system_data()
            if success:
                st.session_state.data = data
                st.session_state.models = models
                st.session_state.data_loaded = True
                st.session_state.models_loaded = True
                st.success("✅ System data loaded successfully!")
            else:
                st.error("❌ Failed to load system data. Please check your artifacts directory.")
                st.stop()
    
    # Sidebar navigation
    page = create_sidebar()
    
    # Debug output
    st.sidebar.write(f"🔍 Selected page: {page}")
    print(f"DEBUG: User selected page: {page}")
    
    # Route to appropriate page with error handling
    try:
        if page == "🏠 Overview":
            print("DEBUG: Loading Overview page")
            from page_modules import overview
            overview.show_page(st.session_state.data, st.session_state.models)
        elif page == "📊 Detection Analysis":
            print("DEBUG: Loading Detection Analysis page")
            from page_modules import detection_analysis
            detection_analysis.show_page(st.session_state.data, st.session_state.models)
        elif page == "🔍 Data Explorer":
            print("DEBUG: Loading Data Explorer page")
            from page_modules import data_explorer
            data_explorer.show_page(st.session_state.data, st.session_state.models)
        elif page == "⚙️ Model Performance":
            print("DEBUG: Loading Model Performance page")
            from page_modules import model_performance
            model_performance.show_page(st.session_state.data, st.session_state.models)
        elif page == "🎯 Threshold Management":
            print("DEBUG: Loading Threshold Management page")
            from page_modules import threshold_management
            threshold_management.show_page(st.session_state.data, st.session_state.models)
        elif page == "📈 Business Intelligence":
            print("DEBUG: Loading Business Intelligence page")
            from page_modules import business_intelligence
            business_intelligence.show_page(st.session_state.data, st.session_state.models)
        elif page == "🚨 Alert Center":
            print("DEBUG: Loading Alert Center page")
            from page_modules import alert_center
            alert_center.show_page(st.session_state.data, st.session_state.models)
    except Exception as e:
        st.error(f"Error loading page '{page}': {str(e)}")
        st.exception(e)  # This will show the full traceback
        print(f"Page error for {page}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()