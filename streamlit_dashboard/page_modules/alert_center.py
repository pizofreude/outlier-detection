"""
Alert Center Page
================

Real-time alert management and monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

from utils.data_loader import load_detection_artifacts
from components.sidebar import create_page_header

def show_alert_center_page():
    """Display the alert center page."""
    
    create_page_header(
        "Alert Management Center",
        "Real-time alert monitoring, investigation, and response coordination"
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
    
    # Real-time alert dashboard
    show_realtime_dashboard(results)
    
    # Alert management interface
    st.markdown("---")
    show_alert_management(data, results)
    
    # Investigation workspace
    st.markdown("---")
    show_investigation_workspace(data, results)

def show_realtime_dashboard(results):
    """Show real-time alert dashboard."""
    
    st.markdown("### 🚨 Real-time Alert Dashboard")
    
    # Alert status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Alerts", "23", "↑ 5")
    
    with col2:
        st.metric("Critical Alerts", "3", "↑ 1")
    
    with col3:
        st.metric("Avg Response Time", "4.2 min", "↓ 1.1 min")
    
    with col4:
        st.metric("Resolution Rate", "87%", "↑ 3%")
    
    # Real-time alert feed
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📡 Live Alert Feed")
        create_live_alert_feed()
    
    with col2:
        st.markdown("#### 🎯 Alert Priority Distribution")
        fig = create_alert_priority_chart()
        st.plotly_chart(fig, width="stretch")
    
    # Alert trends
    st.markdown("#### 📈 Alert Activity Trends")
    fig = create_alert_trends_chart()
    st.plotly_chart(fig, width="stretch")

def show_alert_management(data, results):
    """Show alert management interface."""
    
    st.markdown("### 🛠️ Alert Management")
    
    # Filter and search controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "Critical", "High", "Medium", "Low"]
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status", 
            ["All", "Open", "In Progress", "Resolved", "False Positive"]
        )
    
    with col3:
        time_filter = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24h", "Last 7d", "Last 30d", "All"]
        )
    
    with col4:
        assignee_filter = st.selectbox(
            "Assignee",
            ["All", "Unassigned", "John Smith", "Jane Doe", "Team Alpha"]
        )
    
    # Alert table with actions
    st.markdown("#### 📋 Alert Management Table")
    create_alert_management_table(severity_filter, status_filter)
    
    # Bulk actions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Bulk Assign", type="primary"):
            st.success("✅ Selected alerts assigned to Team Alpha")
    
    with col2:
        if st.button("✅ Mark Resolved"):
            st.success("✅ Selected alerts marked as resolved")
    
    with col3:
        if st.button("❌ Mark False Positive"):
            st.info("ℹ️ Selected alerts marked as false positive")
    
    with col4:
        if st.button("📤 Export Report"):
            st.download_button(
                label="📥 Download CSV",
                data=create_mock_alert_csv(),
                file_name="alerts_report.csv",
                mime="text/csv"
            )

def show_investigation_workspace(data, results):
    """Show investigation workspace."""
    
    st.markdown("### 🔍 Investigation Workspace")
    
    # Alert selection for investigation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Select Alert")
        selected_alert = st.selectbox(
            "Choose alert to investigate:",
            ["ALERT-2024-001: High anomaly score detected",
             "ALERT-2024-002: Unusual pattern in user behavior", 
             "ALERT-2024-003: Potential fraud transaction",
             "ALERT-2024-004: Data quality anomaly",
             "ALERT-2024-005: System performance outlier"]
        )
        
        if st.button("🔍 Start Investigation", type="primary"):
            st.session_state.investigation_started = True
    
    with col2:
        st.markdown("#### 📊 Alert Details")
        create_alert_details_panel(selected_alert)
    
    # Investigation tools
    if hasattr(st.session_state, 'investigation_started') and st.session_state.investigation_started:
        show_investigation_tools(data, results, selected_alert)

def create_live_alert_feed():
    """Create live alert feed component."""
    
    # Simulate real-time alerts
    alerts = [
        {
            "time": "2 min ago",
            "severity": "🔴 Critical",
            "message": "Anomaly detected in payment processing",
            "id": "ALERT-2024-089"
        },
        {
            "time": "7 min ago", 
            "severity": "🟡 Medium",
            "message": "Unusual user login pattern detected",
            "id": "ALERT-2024-088"
        },
        {
            "time": "12 min ago",
            "severity": "🟠 High", 
            "message": "Data quality threshold exceeded",
            "id": "ALERT-2024-087"
        },
        {
            "time": "18 min ago",
            "severity": "🟡 Medium",
            "message": "Performance metric outlier identified",
            "id": "ALERT-2024-086"
        },
        {
            "time": "25 min ago",
            "severity": "🔴 Critical",
            "message": "Potential security breach detected", 
            "id": "ALERT-2024-085"
        }
    ]
    
    # Create scrollable alert feed
    for alert in alerts:
        severity_color = {
            "🔴 Critical": "#ff4444",
            "🟠 High": "#ff8800", 
            "🟡 Medium": "#ffaa00",
            "🟢 Low": "#44aa44"
        }.get(alert["severity"], "#666666")
        
        st.markdown(f"""
        <div style="
            border-left: 4px solid {severity_color};
            background-color: #f8f9fa;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 0 4px 4px 0;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <strong style="color: {severity_color};">{alert["severity"]}</strong>
                <small style="color: #666;">{alert["time"]}</small>
            </div>
            <div style="color: #333;">{alert["message"]}</div>
            <small style="color: #666;">ID: {alert["id"]}</small>
        </div>
        """, unsafe_allow_html=True)

def create_alert_priority_chart():
    """Create alert priority distribution chart."""
    
    priorities = ['Critical', 'High', 'Medium', 'Low']
    counts = [3, 8, 12, 5]
    colors = ['#ff4444', '#ff8800', '#ffaa00', '#44aa44']
    
    fig = go.Figure(data=[go.Pie(
        labels=priorities,
        values=counts,
        marker=dict(colors=colors),
        hole=0.4
    )])
    
    fig.update_layout(
        title="Alert Priority Distribution",
        height=250
    )
    
    return fig

def create_alert_trends_chart():
    """Create alert activity trends chart."""
    
    # Generate mock hourly data for the last 24 hours
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='H')
    
    np.random.seed(42)
    critical_alerts = np.random.poisson(lam=0.5, size=len(hours))
    high_alerts = np.random.poisson(lam=1.2, size=len(hours))
    medium_alerts = np.random.poisson(lam=2.0, size=len(hours))
    low_alerts = np.random.poisson(lam=0.8, size=len(hours))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=hours, y=critical_alerts, name='Critical', marker_color='#ff4444'))
    fig.add_trace(go.Bar(x=hours, y=high_alerts, name='High', marker_color='#ff8800'))
    fig.add_trace(go.Bar(x=hours, y=medium_alerts, name='Medium', marker_color='#ffaa00'))
    fig.add_trace(go.Bar(x=hours, y=low_alerts, name='Low', marker_color='#44aa44'))
    
    fig.update_layout(
        title="Alert Activity - Last 24 Hours",
        xaxis_title="Time",
        yaxis_title="Number of Alerts",
        barmode='stack',
        height=300
    )
    
    return fig

def create_alert_management_table(severity_filter, status_filter):
    """Create alert management table."""
    
    # Generate mock alert data
    np.random.seed(42)
    n_alerts = 50
    
    alert_ids = [f"ALERT-2024-{str(i).zfill(3)}" for i in range(1, n_alerts + 1)]
    severities = np.random.choice(['Critical', 'High', 'Medium', 'Low'], n_alerts, p=[0.1, 0.2, 0.5, 0.2])
    statuses = np.random.choice(['Open', 'In Progress', 'Resolved', 'False Positive'], n_alerts, p=[0.3, 0.2, 0.4, 0.1])
    assignees = np.random.choice(['Unassigned', 'John Smith', 'Jane Doe', 'Team Alpha'], n_alerts, p=[0.2, 0.3, 0.3, 0.2])
    
    # Create timestamps
    timestamps = []
    base_time = datetime.now()
    for i in range(n_alerts):
        hours_ago = np.random.exponential(12)  # Exponential distribution for realistic timing
        timestamp = base_time - timedelta(hours=hours_ago)
        timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M"))
    
    # Create descriptions
    descriptions = [
        f"Anomaly detected in {'payment', 'user behavior', 'system metrics', 'data quality'}[{i % 4}]"
        for i in range(n_alerts)
    ]
    
    # Create DataFrame
    alerts_df = pd.DataFrame({
        'Alert ID': alert_ids,
        'Severity': severities,
        'Status': statuses,
        'Description': descriptions,
        'Timestamp': timestamps,
        'Assignee': assignees,
        'Select': [False] * n_alerts
    })
    
    # Apply filters
    if severity_filter != "All":
        alerts_df = alerts_df[alerts_df['Severity'] == severity_filter]
    
    if status_filter != "All":
        alerts_df = alerts_df[alerts_df['Status'] == status_filter]
    
    # Display table with selection
    edited_df = st.data_editor(
        alerts_df,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select alerts for bulk actions",
                default=False,
            ),
            "Severity": st.column_config.SelectboxColumn(
                "Severity",
                options=["Critical", "High", "Medium", "Low"],
            ),
            "Status": st.column_config.SelectboxColumn(
                "Status", 
                options=["Open", "In Progress", "Resolved", "False Positive"],
            ),
            "Assignee": st.column_config.SelectboxColumn(
                "Assignee",
                options=["Unassigned", "John Smith", "Jane Doe", "Team Alpha"],
            )
        },
        disabled=["Alert ID", "Description", "Timestamp"],
        hide_index=True,
        width="stretch",
        height=400
    )
    
    # Show selection summary
    selected_count = edited_df['Select'].sum()
    if selected_count > 0:
        st.info(f"📌 {selected_count} alerts selected for bulk actions")

def create_mock_alert_csv():
    """Create mock CSV data for download."""
    
    csv_data = """Alert ID,Severity,Status,Description,Timestamp,Assignee
ALERT-2024-001,Critical,Open,Payment processing anomaly,2024-01-15 14:30,John Smith
ALERT-2024-002,High,In Progress,User behavior pattern unusual,2024-01-15 13:45,Jane Doe
ALERT-2024-003,Medium,Resolved,Data quality threshold exceeded,2024-01-15 12:20,Team Alpha
ALERT-2024-004,Low,False Positive,System metric outlier,2024-01-15 11:10,John Smith
ALERT-2024-005,Critical,Open,Security breach potential,2024-01-15 10:55,Unassigned"""
    
    return csv_data

def create_alert_details_panel(selected_alert):
    """Create detailed alert information panel."""
    
    # Extract alert ID from selection
    alert_id = selected_alert.split(':')[0]
    
    # Mock alert details
    alert_details = {
        "ALERT-2024-001": {
            "severity": "🔴 Critical",
            "status": "🔴 Open",
            "timestamp": "2024-01-15 14:30:25",
            "detector": "Statistical + Isolation Forest",
            "confidence": 0.94,
            "affected_records": 47,
            "description": "High anomaly scores detected in payment processing transactions. Multiple statistical and ML models flagged unusual patterns.",
            "features": ["transaction_amount", "user_location", "payment_method"],
            "assignee": "John Smith",
            "priority": "P1"
        },
        "ALERT-2024-002": {
            "severity": "🟠 High", 
            "status": "🟡 In Progress",
            "timestamp": "2024-01-15 13:45:12",
            "detector": "LOF + KDE",
            "confidence": 0.87,
            "affected_records": 23,
            "description": "Unusual user behavior patterns detected. Local outlier factor analysis identified significant deviations.",
            "features": ["login_frequency", "session_duration", "click_patterns"],
            "assignee": "Jane Doe",
            "priority": "P2"
        }
    }
    
    # Get alert details or default
    details = alert_details.get(alert_id, {
        "severity": "🟡 Medium",
        "status": "🔴 Open", 
        "timestamp": "2024-01-15 12:00:00",
        "detector": "Multiple Models",
        "confidence": 0.75,
        "affected_records": 12,
        "description": "Anomaly detected requiring investigation.",
        "features": ["feature_1", "feature_2", "feature_3"],
        "assignee": "Unassigned",
        "priority": "P3"
    })
    
    st.markdown(f"""
    <div style="
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
    ">
        <h4 style="color: #495057; margin-top: 0;">{alert_id}</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div><strong>Severity:</strong> {details['severity']}</div>
            <div><strong>Status:</strong> {details['status']}</div>
            <div><strong>Timestamp:</strong> {details['timestamp']}</div>
            <div><strong>Detector:</strong> {details['detector']}</div>
            <div><strong>Confidence:</strong> {details['confidence']:.2f}</div>
            <div><strong>Affected Records:</strong> {details['affected_records']}</div>
            <div><strong>Assignee:</strong> {details['assignee']}</div>
            <div><strong>Priority:</strong> {details['priority']}</div>
        </div>
        
        <div style="margin-bottom: 1rem;">
            <strong>Description:</strong><br>
            {details['description']}
        </div>
        
        <div>
            <strong>Key Features:</strong> {', '.join(details['features'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_investigation_tools(data, results, selected_alert):
    """Show investigation tools and workspace."""
    
    st.markdown("### 🔬 Investigation Tools")
    
    # Investigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Analysis",
        "🎯 Similar Cases", 
        "📈 Impact Assessment",
        "📝 Case Notes"
    ])
    
    with tab1:
        show_data_analysis_tab(data, results, selected_alert)
    
    with tab2:
        show_similar_cases_tab(selected_alert)
    
    with tab3:
        show_impact_assessment_tab(selected_alert)
    
    with tab4:
        show_case_notes_tab(selected_alert)

def show_data_analysis_tab(data, results, selected_alert):
    """Show data analysis tab for investigation."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Affected Data Points")
        
        # Create mock scatter plot of affected data points
        fig = create_investigation_scatter_plot(selected_alert)
        st.plotly_chart(fig, width="stretch")
        
        # Feature importance
        st.markdown("#### 🎯 Feature Importance")
        fig = create_feature_importance_chart(selected_alert)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 📈 Statistical Summary")
        create_statistical_summary(selected_alert)
        
        st.markdown("#### 🔍 Detection Details")
        create_detection_details(selected_alert)

def show_similar_cases_tab(selected_alert):
    """Show similar cases analysis."""
    
    st.markdown("#### 🔍 Similar Cases Analysis")
    
    # Similar cases based on ML similarity
    similar_cases = [
        {
            "id": "ALERT-2024-067",
            "similarity": 0.94,
            "outcome": "Confirmed Fraud",
            "resolution_time": "2.3 hours",
            "assignee": "Jane Doe"
        },
        {
            "id": "ALERT-2024-045", 
            "similarity": 0.87,
            "outcome": "False Positive",
            "resolution_time": "45 minutes",
            "assignee": "John Smith"
        },
        {
            "id": "ALERT-2024-032",
            "similarity": 0.82,
            "outcome": "Confirmed Anomaly",
            "resolution_time": "1.8 hours", 
            "assignee": "Team Alpha"
        }
    ]
    
    for case in similar_cases:
        outcome_color = "#ff4444" if "Fraud" in case["outcome"] else "#44aa44" if "False" in case["outcome"] else "#ff8800"
        
        st.markdown(f"""
        <div style="
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: #f8f9fa;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <strong>{case['id']}</strong>
                <span style="background-color: #007bff; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                    {case['similarity']:.0%} similar
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                <div><strong>Outcome:</strong> <span style="color: {outcome_color};">{case['outcome']}</span></div>
                <div><strong>Resolution Time:</strong> {case['resolution_time']}</div>
                <div><strong>Assignee:</strong> {case['assignee']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Pattern analysis
    st.markdown("#### 📊 Pattern Analysis")
    fig = create_pattern_analysis_chart()
    st.plotly_chart(fig, width="stretch")

def show_impact_assessment_tab(selected_alert):
    """Show impact assessment analysis."""
    
    st.markdown("#### 💥 Business Impact Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Potential Loss", "$12,500", delta="High Risk")
        st.metric("Affected Users", "47", delta="↑ 12")
    
    with col2:
        st.metric("System Impact", "Medium", delta="Contained")
        st.metric("SLA Risk", "Low", delta="Within limits")
    
    with col3:
        st.metric("Reputation Risk", "Medium", delta="Monitoring") 
        st.metric("Compliance Risk", "Low", delta="Compliant")
    
    # Impact timeline
    st.markdown("#### ⏱️ Impact Timeline")
    fig = create_impact_timeline_chart()
    st.plotly_chart(fig, width="stretch")
    
    # Risk assessment matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Risk Matrix")
        fig = create_risk_matrix()
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("#### 📋 Mitigation Actions")
        mitigation_actions = [
            "🔒 Temporary account restrictions applied",
            "📧 Customer notifications sent",
            "🔍 Enhanced monitoring activated",
            "👥 Incident response team notified"
        ]
        
        for action in mitigation_actions:
            st.write(action)

def show_case_notes_tab(selected_alert):
    """Show case notes and collaboration tools."""
    
    st.markdown("#### 📝 Investigation Notes")
    
    # Existing notes
    notes = [
        {
            "timestamp": "2024-01-15 15:30",
            "author": "John Smith",
            "note": "Initial analysis shows unusual transaction patterns. Flagged for deeper investigation.",
            "type": "Analysis"
        },
        {
            "timestamp": "2024-01-15 15:45",
            "author": "Jane Doe", 
            "note": "Cross-referenced with fraud database. Similar pattern found in case ALERT-2024-067.",
            "type": "Research"
        },
        {
            "timestamp": "2024-01-15 16:10",
            "author": "System",
            "note": "Automated correlation analysis completed. 94% similarity to confirmed fraud case.",
            "type": "System"
        }
    ]
    
    for note in notes:
        note_color = "#e3f2fd" if note["type"] == "System" else "#f3e5f5" if note["type"] == "Research" else "#e8f5e8"
        
        st.markdown(f"""
        <div style="
            background-color: {note_color};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #2196F3;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; color: #666;">
                <span><strong>{note['author']}</strong> - {note['type']}</span>
                <span>{note['timestamp']}</span>
            </div>
            <div>{note['note']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add new note
    st.markdown("#### ➕ Add Investigation Note")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_note = st.text_area("Note", placeholder="Enter investigation findings, analysis, or next steps...")
    
    with col2:
        note_type = st.selectbox("Type", ["Analysis", "Research", "Action", "Decision"])
        
        if st.button("💾 Add Note", type="primary"):
            if new_note:
                st.success("✅ Note added to investigation case")
                st.rerun()

def create_investigation_scatter_plot(selected_alert):
    """Create scatter plot for investigation data analysis."""
    
    # Generate mock data points for investigation
    np.random.seed(hash(selected_alert) % 100)
    
    n_normal = 200
    n_anomalies = 20
    
    # Normal data points
    normal_x = np.random.normal(0, 1, n_normal)
    normal_y = np.random.normal(0, 1, n_normal) + 0.5 * normal_x
    
    # Anomalous data points
    anomaly_x = np.random.uniform(-3, 3, n_anomalies)
    anomaly_y = np.random.uniform(-3, 3, n_anomalies)
    
    # Combine data
    all_x = np.concatenate([normal_x, anomaly_x])
    all_y = np.concatenate([normal_y, anomaly_y])
    labels = ['Normal'] * n_normal + ['Anomaly'] * n_anomalies
    
    fig = px.scatter(
        x=all_x, y=all_y,
        color=labels,
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title="Data Points Analysis - Feature Space",
        labels={'x': 'Feature 1', 'y': 'Feature 2'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_feature_importance_chart(selected_alert):
    """Create feature importance chart for investigation."""
    
    features = ['transaction_amount', 'user_location', 'payment_method', 'session_duration', 'device_info']
    importance = np.random.uniform(0.1, 1.0, len(features))
    importance = importance / np.sum(importance)  # Normalize
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=['red' if imp > 0.3 else 'orange' if imp > 0.15 else 'blue' for imp in importance],
        text=[f"{imp:.3f}" for imp in importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance in Detection",
        xaxis_title="Importance Score",
        height=250
    )
    
    return fig

def create_statistical_summary(selected_alert):
    """Create statistical summary for investigation."""
    
    stats = [
        ("Outlier Score", "0.94", "🔴 High"),
        ("Z-Score", "3.67", "🔴 High"),
        ("P-Value", "0.002", "🔴 Significant"),
        ("Confidence", "94%", "🟢 High")
    ]
    
    for stat, value, indicator in stats:
        st.metric(stat, value, indicator)

def create_detection_details(selected_alert):
    """Create detection details for investigation."""
    
    details = [
        ("Primary Detector", "Isolation Forest"),
        ("Secondary Detector", "Statistical"),
        ("Detection Time", "0.127s"),
        ("Model Version", "v2.1.3"),
        ("Threshold Used", "0.75"),
        ("False Positive Rate", "3.2%")
    ]
    
    for detail, value in details:
        st.write(f"**{detail}**: {value}")

def create_pattern_analysis_chart():
    """Create pattern analysis chart for similar cases."""
    
    patterns = ['Time Pattern', 'Amount Pattern', 'Location Pattern', 'Device Pattern', 'Behavior Pattern']
    similarity_scores = np.random.uniform(0.6, 0.95, len(patterns))
    
    fig = go.Figure(go.Bar(
        x=patterns,
        y=similarity_scores,
        marker_color=['green' if s > 0.9 else 'orange' if s > 0.8 else 'blue' for s in similarity_scores],
        text=[f"{s:.2f}" for s in similarity_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Pattern Similarity Analysis",
        yaxis_title="Similarity Score",
        height=300
    )
    
    return fig

def create_impact_timeline_chart():
    """Create impact timeline chart."""
    
    # Generate mock timeline data
    times = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                         end=datetime.now() + timedelta(hours=2), freq='30T')
    
    # Mock impact metrics over time
    impact_scores = [0, 0.2, 0.5, 0.8, 0.9, 0.85, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0, 0, 0, 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times[:len(impact_scores)],
        y=impact_scores,
        mode='lines+markers',
        fill='tozeroy',
        name='Impact Score',
        line=dict(color='red', width=3)
    ))
    
    # Add current time marker using a scatter point instead of vline
    current_time = datetime.now()
    
    # Add a marker for current time
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[0.5],  # Middle of the impact range
        mode='markers',
        marker=dict(
            size=15,
            color='black',
            symbol='line-ns-open',
            line=dict(width=3, color='black')
        ),
        name='Current Time',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Impact Timeline",
        xaxis_title="Time",
        yaxis_title="Impact Score",
        height=300
    )
    
    return fig

def create_risk_matrix():
    """Create risk assessment matrix."""
    
    # Risk categories
    categories = ['Financial', 'Operational', 'Reputation', 'Compliance']
    
    # Risk levels (Probability vs Impact)
    probability = [0.8, 0.6, 0.4, 0.3]  # High to Low
    impact = [0.9, 0.7, 0.6, 0.2]      # High to Low
    
    # Risk scores (probability * impact)
    risk_scores = [p * i for p, i in zip(probability, impact)]
    
    fig = px.scatter(
        x=probability,
        y=impact,
        size=risk_scores,
        text=categories,
        color=risk_scores,
        color_continuous_scale='Reds',
        size_max=60
    )
    
    fig.update_traces(textposition="middle center")
    fig.update_layout(
        title="Risk Assessment Matrix",
        xaxis_title="Probability",
        yaxis_title="Impact",
        height=300
    )
    
    return fig

def show_page(data=None, models=None):
    """Main entry point for the alert center page (compatibility function)."""
    show_alert_center_page()