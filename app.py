import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Scholarship AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e293b;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('scholarship_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None

model, scaler = load_artifacts()

# Sidebar - Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997385.png", width=50)
    st.title("Student Profile")
    # Load Dataset for Selection
    df_existing = None
    try:
        df_existing = pd.read_csv('dataset.csv')
    except:
        st.warning("dataset.csv not found for selection mode.")

    input_mode = st.radio("Input Mode", ["Manual Entry", "Select from Dataset"])

    if input_mode == "Select from Dataset" and df_existing is not None:
        max_idx = len(df_existing) - 1
        row_idx = st.number_input(f"Select Record (0 - {max_idx})", 0, max_idx, 0)
        selected_row = df_existing.iloc[row_idx]
        
        # Display selected ground truth
        st.info(f"Actual Status: {selected_row['scholarship']}")
        
        # Overwrite defaults
        def_gpa = float(selected_row['gpa'])
        def_att = int(selected_row['attendance'])
        def_cred = int(selected_row['credit_hours'])
        def_fin = int(selected_row['financial_score'])
    else:
        def_gpa, def_att, def_cred, def_fin = 3.5, 85, 60, 20

    st.divider()
    
    gpa = st.slider("GPA", 0.0, 4.0, def_gpa, 0.01)
    attendance = st.slider("Attendance %", 0, 100, def_att)
    credit_hours = st.slider("Completed Credit Hours", 0, 130, def_cred, help="Total credit hours completed so far.")
    financial_score = st.slider("Financial Score", 0, 100, def_fin, help="Lower score implies higher financial need.")
    
    st.divider()
    
    analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    st.caption("Powered by Neural Networks")

# --- Main Dashboard ---
st.title("🎓 Scholarship Eligibility Dashboard")
st.markdown("Real-time AI assessment based on academic progress and performance.")

# Tabs for Mode Selection
tab1, tab2 = st.tabs(["Search / Individual Profile", "Batch Processing"])

with tab1:
    if analyze_btn:
        if model and scaler:
            # Prediction Logic
            # Hard Rule: Automatic Disqualification for Low GPA or Credits
            if gpa < 2.5 or credit_hours < 100:
                prob = 0.0
                is_eligible = False
            else:
                # Order must match training: gpa, attendance, financial_score, credit_hours
                input_data = np.array([[gpa, attendance, financial_score, credit_hours]])
                input_scaled = scaler.transform(input_data)
                prob = model.predict(input_scaled)[0][0]
                
                is_eligible = prob > 0.5
            confidence = prob if is_eligible else 1 - prob
            
            # Determine Status Color/Text
            status_color = "#10b981" if is_eligible else "#ef4444"
            status_text = "ELIGIBLE" if is_eligible else "NOT ELIGIBLE"
            
            # Columns for Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Eligibility Probability", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': status_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#fee2e2'},
                        {'range': [50, 100], 'color': '#d1fae5'}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            
            fig.update_layout(height=400, margin=dict(t=50,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown(f"""
            <div style="background-color: {status_color}; padding: 2rem; border-radius: 10px; color: white; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <h2 style="color: white; margin:0;">Result</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0; color: white;">{status_text}</h1>
                <p style="opacity: 0.9;">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Factors Analysis
        st.subheader("Factor Breakdown")
        c1, c2, c3, c4 = st.columns(4)
        
        def metric_html(label, value, is_good):
            color = "#10b981" if is_good else "#f59e0b"
            return f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
            
        c1.markdown(metric_html("GPA", gpa, gpa >= 3.0), unsafe_allow_html=True)
        c2.markdown(metric_html("Attendance", f"{attendance}%", attendance >= 75), unsafe_allow_html=True)
        c3.markdown(metric_html("Credit Hours", credit_hours, credit_hours >= 60), unsafe_allow_html=True)
        c4.markdown(metric_html("Financial Score", financial_score, financial_score <= 40), unsafe_allow_html=True)

    else:
        # Default Landing Info
        st.info("👈 Please enter student details in the sidebar to begin, or select the 'Batch Processing' tab above.")
        
        st.markdown("### System Overview")
        st.write("This scholarship eligibility system uses an **Artificial Neural Network (ANN)** to evaluate candidates based on holistic performance metrics. It is designed to assist administrative decision-making with data-driven insights.")
        
        st.divider()
        
        st.markdown("#### 📊 Input Parameters Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
             with st.expander("Academic Performance", expanded=True):
                st.markdown("""
                **1. GPA (Grade Point Average)**
                - **Range:** 0.0 - 4.0
                - **Impact:** High (40% weight)
                - **Rule:** Minimum 2.5 required for eligibility.
                
                **2. Credit Hours**
                - **Range:** 0 - 130
                - **Impact:** High (30% weight)
                - **Rule:** Minimum 100 hours required (Senior standing).
                """)
        
        with col2:
            with st.expander("Behavioral & Financial", expanded=True):
                st.markdown("""
                **3. Attendance**
                - **Range:** 0% - 100%
                - **Impact:** Moderate (20% weight)
                - **significance:** Indicates consistency and dedication.
                
                **4. Financial Score**
                - **Range:** 0 - 100
                - **Impact:** Need-based (10% weight)
                - **Note:** Lower score = Higher Financial Need.
                """)
                
        st.info("ℹ️ **Note:** The system combines these factors using a non-linear model to capture complex relationships, such as high-need students excelling despite lower attendance.")

with tab2:
    st.header("Batch Eligibility Check")
    st.markdown("Upload a CSV file to check eligibility for multiple students at once.")
    
    uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} records successfully.")
            
            if st.button("Process Batch"):
                with st.spinner("Analyzing records..."):
                    # 1. Hard Rules
                    # Rule: Credit Hours < 100 OR GPA < 2.5 => Not Eligible
                    batch_df['predicted_eligibility'] = 'Pending'
                    batch_df['reason'] = ''
                    
                    mask_fail = (batch_df['credit_hours'] < 100) | (batch_df['gpa'] < 2.5)
                    batch_df.loc[mask_fail, 'predicted_eligibility'] = 'Not Eligible'
                    batch_df.loc[mask_fail, 'reason'] = 'Does not meet minimum requirements (GPA < 2.5 or Credits < 100)'
                    
                    # 2. AI Model
                    mask_ai = ~mask_fail
                    candidates = batch_df.loc[mask_ai]
                    
                    if len(candidates) > 0:
                        # Prepare features: gpa, attendance, financial_score, credit_hours
                        features = candidates[['gpa', 'attendance', 'financial_score', 'credit_hours']]
                        features_scaled = scaler.transform(features)
                        
                        probs = model.predict(features_scaled, verbose=0)
                        preds = (probs > 0.5).astype(int).flatten()
                        
                        labels = np.where(preds == 1, 'Eligible', 'Not Eligible')
                        
                        batch_df.loc[mask_ai, 'predicted_eligibility'] = labels
                        batch_df.loc[mask_ai, 'reason'] = 'Evaluated by AI Model'
                    
                    # 3. Results
                    counts = batch_df['predicted_eligibility'].value_counts()
                    
                    # Display Stats
                    st.divider()
                    col_a, col_b = st.columns(2)
                    col_a.metric("Total Processed", len(batch_df))
                    col_a.metric("Eligible Candidates", counts.get('Eligible', 0))
                    col_b.metric("Not Eligible", counts.get('Not Eligible', 0))
                    
                    # Visual breakdown
                    st.subheader("Eligibility Breakdown")
                    st.bar_chart(counts)
                    
                    # 4. Download
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name='batch_eligibility_results.csv',
                        mime='text/csv',
                    )
                    
                    st.dataframe(batch_df.head(50))
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")


    st.divider()

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 2rem; background-color: white; border-radius: 10px; border: 1px solid #e2e8f0;">
        <h4 style="color: #1e293b; margin-bottom: 0.5rem; font-family: 'Helvetica Neue', sans-serif;">Scholarship AI System</h4>
        <p style="color: #64748b; margin-bottom: 1rem; font-size: 0.9rem;">
            Empowering education through transparent and automated evaluation.
        </p>
        <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 1rem;">
            &copy; 2026 Scholarship AI. All rights reserved.
        </p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; font-size: 0.9rem;">
            <span style="color: #64748b;">Presented by: <strong style="color: #3b82f6;">[Munim Abbas]</strong></span>
            <span style="color: #cbd5e1;">|</span>
            <a href="#" style="color: #64748b; text-decoration: none; transition: color 0.3s;">Project Details</a>
            <span style="color: #cbd5e1;">|</span>
            <a href="#" style="color: #64748b; text-decoration: none; transition: color 0.3s;">GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

