import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict import predict_traffic

# Page configuration
st.set_page_config(
    page_title="DarkShield AI | Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Threat Dark Theme CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0A0E17;
        color: #E0E6ED;
    }
    
    /* Header typography */
    .cyber-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00FF66, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: 1px;
    }
    .cyber-subtitle {
        color: #8892B0;
        font-size: 1.0rem;
        margin-bottom: 25px;
    }
    
    /* Cyber Metric Cards */
    .metric-card {
        background-color: #131C2E;
        border-radius: 10px;
        padding: 18px;
        border-left: 4px solid #00FF66;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 10px;
    }
    .metric-card-danger {
        background-color: #1C131D;
        border-radius: 10px;
        padding: 18px;
        border-left: 4px solid #FF3366;
        box-shadow: 0 4px 15px rgba(255, 51, 102, 0.15);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892B0;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 5px;
    }
    
    /* Threat Alert Banner */
    .threat-banner {
        background: linear-gradient(90deg, rgba(255, 51, 102, 0.2), rgba(19, 28, 46, 0.8));
        border: 1px solid #FF3366;
        border-radius: 8px;
        padding: 16px;
        margin: 20px 0;
        color: #FF6685;
        font-weight: 500;
    }
    .safe-banner {
        background: linear-gradient(90deg, rgba(0, 255, 102, 0.15), rgba(19, 28, 46, 0.8));
        border: 1px solid #00FF66;
        border-radius: 8px;
        padding: 16px;
        margin: 20px 0;
        color: #00FF66;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=70)
    st.markdown("### **Threat Detection Engine**")
    st.markdown("Automated classification of Tor / I2P Dark Web traffic vs Normal HTTP/S flow.")
    st.markdown("---")
    
    # Model Selection
    available_models = {
        "Random Forest Classifier (Scikit-Learn)": "models/trained_model.pkl",
        "Deep Neural Network (MLP)": "models/dl_model.pkl"
    }
    
    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        options=list(available_models.keys())
    )
    selected_model_path = available_models[selected_model_name]
    
    # Check if selected model exists, otherwise fallback to trained_model.pkl
    if not os.path.exists(selected_model_path):
        selected_model_path = "models/trained_model.pkl"
        
    preprocessor_path = "models/preprocessor.pkl"
    
    st.markdown("---")
    st.markdown("### **Input Data Source**")
    
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        help="Upload CSV containing: flow_duration, packet_size, inter_arrival_time, byte_count"
    )
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        load_sample = st.button("📁 Load Built-in Sample", use_container_width=True)
    with col_s2:
        clear_data = st.button("🗑️", help="Clear loaded data", use_container_width=True)
        
    if clear_data:
        st.session_state['current_data'] = None
        st.session_state['data_source_name'] = ""
        st.session_state.pop('analysis_results', None)
        st.rerun()

    st.markdown("---")
    st.markdown("### **System Health**")
    model_ready = os.path.exists(selected_model_path) and os.path.exists(preprocessor_path)
    if model_ready:
        st.success("🟢 ML Pipeline: Active & Ready")
    else:
        st.warning("🟡 Pipeline Missing Artifacts")
        st.caption("Run `python src/train.py` to generate model artifacts.")

# ----------------- Main Interface -----------------
st.markdown('<div class="cyber-title">🛡️ DARK WEB TRAFFIC CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subtitle">Real-time deep packet flow inspection & anomaly threat analysis</div>', unsafe_allow_html=True)

# State Management for Data Ingestion
if 'current_data' not in st.session_state:
    st.session_state['current_data'] = None
if 'data_source_name' not in st.session_state:
    st.session_state['data_source_name'] = ""
if 'last_selected_model' not in st.session_state:
    st.session_state['last_selected_model'] = selected_model_path
elif st.session_state['last_selected_model'] != selected_model_path:
    st.session_state['last_selected_model'] = selected_model_path
    st.session_state.pop('analysis_results', None)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if st.session_state.get('last_uploaded_name') != uploaded_file.name:
            st.session_state['last_uploaded_name'] = uploaded_file.name
            st.session_state['current_data'] = df
            st.session_state['data_source_name'] = f"Uploaded: {uploaded_file.name}"
            st.session_state.pop('analysis_results', None)
        st.sidebar.success(f"Active: {uploaded_file.name} ({len(df)} flows)")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
elif load_sample:
    sample_path = "data/sample_traffic.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.session_state['current_data'] = df
        st.session_state['data_source_name'] = "Built-in Sample Traffic"
        st.session_state.pop('analysis_results', None)
        st.sidebar.info(f"Loaded sample dataset ({len(df)} flows)")
    else:
        st.warning("Sample dataset not found at 'data/sample_traffic.csv'.")

input_df = st.session_state.get('current_data')

# Workflow Execution
if input_df is not None:
    # 1. Dataset Preview
    with st.expander("🔍 Dataset Preview (Raw Flows)", expanded=True):
        st.dataframe(input_df.head(10), use_container_width=True)
        st.caption(f"Source: **{st.session_state.get('data_source_name', 'Active Dataset')}** | Total Rows: **{len(input_df)}** | Columns: `{list(input_df.columns)}`")
    
    # 2. Run Classification
    col_btn, col_empty = st.columns([1, 4])
    with col_btn:
        run_analysis = st.button("🚀 Analyze Traffic", type="primary", use_container_width=True)
        
    if run_analysis or 'analysis_results' in st.session_state:
        if run_analysis:
            if not model_ready:
                st.error("Model artifacts not found! Please train the model first by running `python src/train.py`.")
            else:
                with st.spinner("Analyzing packet features and evaluating threat signatures..."):
                    try:
                        results_df = predict_traffic(
                            input_data=input_df,
                            model_path=selected_model_path,
                            preprocessor_path=preprocessor_path
                        )
                        st.session_state['analysis_results'] = results_df
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        
        if 'analysis_results' in st.session_state:
            res_df = st.session_state['analysis_results']
            
            # Metrics Calculations
            total_flows = len(res_df)
            dark_web_count = int((res_df['Prediction'] == 'Dark Web').sum())
            normal_count = int((res_df['Prediction'] == 'Normal').sum())
            dark_web_pct = (dark_web_count / total_flows) * 100 if total_flows > 0 else 0
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Flows Scanned</div>
                    <div class="metric-value">{total_flows}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Normal Traffic</div>
                    <div class="metric-value" style="color: #00FF66;">{normal_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card-danger">
                    <div class="metric-label">Dark Web Detected</div>
                    <div class="metric-value" style="color: #FF3366;">{dark_web_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Threat Ratio</div>
                    <div class="metric-value" style="color: {'#FF3366' if dark_web_pct > 0 else '#00FF66'};">{dark_web_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Threat Alert Banner
            if dark_web_count > 0:
                st.markdown(f"""
                <div class="threat-banner">
                    🚨 <strong>SECURITY ALERT:</strong> Detected <strong>{dark_web_count}</strong> high-risk Dark Web flow(s) exhibiting anonymization signatures (Tor/I2P proxy patterns). Immediate packet containment recommended.
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔬 Forensic Explanation: Why & How These Flows Are Classified As Dark Web", expanded=True):
                    st.markdown("""
                    The classifier detects Dark Web flows using behavioral flow fingerprints rather than decrypting the payload:
                    1. **Fixed 512-Byte Cell Padding (Packet Size):** Tor mandates uniform 512-byte cell fragments to camouflage traffic. Normal browsing shows high variance (1100–1500 bytes MTU), while flagged flows exhibit characteristic truncated packet distributions.
                    2. **Multi-Hop Relay Latency (Inter-Arrival Time):** Tor routes packets across 3 volunteer nodes (Guard $\\to$ Middle $\\to$ Exit). The resulting decryption overhead spikes packet inter-arrival times to **0.3s – 1.5s** (compared to **0.005s – 0.05s** for normal direct traffic).
                    3. **Persistent Circuit Tunnels (Flow Duration):** Tor circuit handshakes are computationally heavy, leading to long-lived tunnels lasting **60s – 300s**, unlike standard ephemeral web requests.
                    """)
            else:
                st.markdown("""
                <div class="safe-banner">
                    ✅ <strong>ALL CLEAR:</strong> No Dark Web or Tor network traffic detected in this session. All analyzed flows match legitimate normal traffic profiles.
                </div>
                """, unsafe_allow_html=True)
                
            # Interactive Visualizations
            st.markdown("### 📊 Threat Intelligence & Visualizations")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Donut Chart: Traffic Classification Distribution
                fig_dist = px.pie(
                    res_df,
                    names='Prediction',
                    title="Traffic Distribution (Normal vs Dark Web)",
                    hole=0.5,
                    color='Prediction',
                    color_discrete_map={'Normal': '#00FF66', 'Dark Web': '#FF3366'}
                )
                fig_dist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#E0E6ED',
                    title_font_size=16
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with chart_col2:
                # Confidence Histogram
                fig_conf = px.histogram(
                    res_df,
                    x='Confidence_Score',
                    color='Prediction',
                    nbins=20,
                    title="Model Prediction Confidence Spread",
                    color_discrete_map={'Normal': '#00FF66', 'Dark Web': '#FF3366'},
                    labels={'Confidence_Score': 'Model Confidence Score (0.5 - 1.0)'}
                )
                fig_conf.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(19,28,46,0.6)',
                    font_color='#E0E6ED',
                    title_font_size=16,
                    xaxis=dict(gridcolor='#233554'),
                    yaxis=dict(gridcolor='#233554')
                )
                st.plotly_chart(fig_conf, use_container_width=True)
                
            # Scatter Plot: Flow Duration vs Packet Size
            st.markdown("#### 📡 Flow Behavior Map (Duration vs Packet Size)")
            fig_scatter = px.scatter(
                res_df,
                x='flow_duration',
                y='packet_size',
                color='Prediction',
                size='byte_count',
                hover_data=['Confidence'],
                color_discrete_map={'Normal': '#00FF66', 'Dark Web': '#FF3366'},
                labels={'flow_duration': 'Flow Duration (s)', 'packet_size': 'Mean Packet Size (Bytes)'}
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(19,28,46,0.6)',
                font_color='#E0E6ED',
                xaxis=dict(gridcolor='#233554'),
                yaxis=dict(gridcolor='#233554')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Classification Results Table
            st.markdown("### 📋 Detailed Inspection Logs")
            
            # Filter and Toggle options
            ctrl_col1, ctrl_col2 = st.columns([2, 1])
            with ctrl_col1:
                filter_choice = st.radio(
                    "Filter Results By:",
                    options=["All Flows", "Dark Web Only (Threats)", "Normal Only"],
                    horizontal=True
                )
            with ctrl_col2:
                show_reasoning = st.toggle("🔍 Show Threat Reasoning", value=True, help="Toggle AI forensic explanation column for each prediction")
            
            display_table = res_df.copy()
            if filter_choice == "Dark Web Only (Threats)":
                display_table = display_table[display_table['Prediction'] == 'Dark Web']
            elif filter_choice == "Normal Only":
                display_table = display_table[display_table['Prediction'] == 'Normal']
                
            if not show_reasoning and 'Threat_Reason' in display_table.columns:
                display_table = display_table.drop(columns=['Threat_Reason'])
                
            st.dataframe(
                display_table,
                use_container_width=True
            )
            
            # Interactive Per-Flow Forensic Reason Inspector
            if show_reasoning and len(display_table) > 0:
                with st.expander("🔎 Interactive Flow Reason Inspector (Inspect individual session)", expanded=False):
                    flow_indices = display_table.index.tolist()
                    selected_flow_idx = st.selectbox(
                        "Select Flow Record Index to inspect:",
                        options=flow_indices,
                        format_func=lambda idx: f"Flow #{idx} | {display_table.loc[idx, 'Prediction']} ({display_table.loc[idx, 'Confidence']})"
                    )
                    selected_row = display_table.loc[selected_flow_idx]
                    
                    r_col1, r_col2 = st.columns([1, 2])
                    with r_col1:
                        st.markdown(f"**Flow Duration:** `{selected_row['flow_duration']} s`")
                        st.markdown(f"**Packet Size:** `{selected_row['packet_size']} B`")
                        st.markdown(f"**Inter-Arrival Time:** `{selected_row['inter_arrival_time']} s`")
                        st.markdown(f"**Byte Count:** `{selected_row['byte_count']} B`")
                    with r_col2:
                        p_color = "#FF3366" if selected_row['Prediction'] == 'Dark Web' else "#00FF66"
                        st.markdown(f"<h4 style='color: {p_color}; margin-top: 0;'>Classification: {selected_row['Prediction']} (Confidence: {selected_row['Confidence']})</h4>", unsafe_allow_html=True)
                        st.info(f"**Forensic Reasoning:**\n\n{selected_row.get('Threat_Reason', 'Standard profile')}")
            
            # CSV Download
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Forensic Audit Report (CSV)",
                data=csv_data,
                file_name="traffic_classification_report.csv",
                mime="text/csv",
                type="secondary"
            )

else:
    # Landing instructions
    st.info("👈 **Get Started:** Upload a network traffic CSV file in the sidebar, or click **'Load Built-in Sample Traffic'** to test the classifier instantly!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Ingestion")
        st.write("Upload raw network flow captures containing packet duration, inter-arrival intervals, and byte counts.")
    with col2:
        st.markdown("### 2. Classification")
        st.write("AI models inspect behavioral packet flow patterns to detect Tor, I2P, and encrypted darknet tunneling.")
    with col3:
        st.markdown("### 3. Mitigation")
        st.write("Receive instant threat alert notifications, behavioral charts, and exportable forensic audit reports.")
