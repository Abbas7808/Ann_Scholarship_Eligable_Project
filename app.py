import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scholarship AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Professional UI)
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc; /* Very light slate */
    }

    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Metrics / Factors Card */
    .factor-card {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .factor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .factor-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .factor-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Improvements */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
    .stButton > button {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Logic & Data Loading
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Sidebar (Settings & Inputs)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997385.png", width=60)
    st.markdown("### 🎓 Student Profile")
    st.caption("Enter details to check eligibility.")
    
    # Defaults
    def_gpa, def_att, def_cred, def_fin = 3.5, 85, 60, 20
    
    # Input Mode Selection
    with st.expander("🛠️ Input Method", expanded=True):
        input_mode = st.radio("Choose Mode", ["Manual Entry", "Select from Dataset"], label_visibility="collapsed")
        
    if input_mode == "Select from Dataset":
        try:
            df_existing = pd.read_csv('dataset.csv')
            max_idx = len(df_existing) - 1
            row_idx = st.number_input(f"Record ID (0 - {max_idx})", 0, max_idx, 0)
            selected_row = df_existing.iloc[row_idx]
            
            st.info(f"Actual Status: **{selected_row['scholarship']}**")
            def_gpa = float(selected_row['gpa'])
            def_att = int(selected_row['attendance'])
            def_cred = int(selected_row['credit_hours'])
            def_fin = int(selected_row['financial_score'])
        except:
            st.warning("dataset.csv not found.")

    st.markdown("---")
    
    # Grouped Inputs
    with st.expander("📚 Academic Performance", expanded=True):
        gpa = st.slider("GPA (0.0 - 4.0)", 0.0, 4.0, def_gpa, 0.01)
        credit_hours = st.slider("Credit Hours", 0, 130, def_cred)
        
    with st.expander("👤 Behavioral & Financial", expanded=True):
        attendance = st.slider("Attendance %", 0, 100, def_att)
        financial_score = st.slider("Financial Score", 0, 100, def_fin, help="0 = High Need, 100 = Low Need")

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Eligibility", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Main Content
# -----------------------------------------------------------------------------
st.title("Scholarship AI Predictor")
st.markdown("### Intelligent Assessment System")
st.markdown("This system utilizes an **Artificial Neural Network** to evaluate scholarship candidates based on academic merit, consistency, and financial need.")

# Tabs
tab1, tab2 = st.tabs(["🔍 Individual Analysis", "📂 Batch Processing"])

# --- TAB 1: Individual Analysis ---
with tab1:
    if analyze_btn:
        if model and scaler:
            # Prediction Logic
            # Hard Rules
            if gpa < 2.5 or credit_hours < 100:
                prob = 0.0
                is_eligible = False
                reason = "Did not meet minimum threshold (GPA < 2.5 or Credits < 100)"
            else:
                input_data = np.array([[gpa, attendance, financial_score, credit_hours]])
                input_scaled = scaler.transform(input_data)
                prob = model.predict(input_scaled)[0][0]
                is_eligible = prob > 0.5
                reason = "AI Model Prediction"
            
            confidence = prob if is_eligible else 1 - prob
            
            # --- UI: Result Section ---
            st.markdown("---")
            
            # Determine Styles
            if is_eligible:
                main_color = "#10b981" # Emerald 500
                bg_color = "#ecfdf5"   # Emerald 50
                icon = "✅"
                title_text = "ELIGIBLE FOR SCHOLARSHIP"
                desc_text = "This candidate meets the criteria based on the AI assessment."
            else:
                main_color = "#ef4444" # Red 500
                bg_color = "#fef2f2"   # Red 50
                icon = "❌"
                title_text = "NOT ELIGIBLE"
                desc_text = "This candidate does not meet the required criteria."
            
            # 1. Main Result Card
            cols = st.columns([1, 2, 1])
            with cols[1]:
                st.markdown(f"""
                <div style="background-color: {bg_color}; border: 2px solid {main_color}; padding: 2rem; border-radius: 16px; text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h2 style="color: {main_color}; margin: 0; font-size: 2rem;">{title_text}</h2>
                    <p style="color: #475569; margin-top: 1rem; font-size: 1.1rem;">{desc_text}</p>
                    <div style="margin-top: 1.5rem;">
                        <span style="background-color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; color: {main_color}; border: 1px solid {main_color};">
                            Confidence: {confidence:.1%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 2. Factor Breakdown
            st.markdown("### 📊 Factor Analysis")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            def factor_html(label, value, color_condition):
                color = "#10b981" if color_condition else "#f59e0b"
                return f"""
                <div class="factor-card">
                    <div class="factor-value" style="color: {color};">{value}</div>
                    <div class="factor-label">{label}</div>
                </div>
                """
            
            f_col1.markdown(factor_html("GPA", gpa, gpa >= 3.0), unsafe_allow_html=True)
            f_col2.markdown(factor_html("Attendance", f"{attendance}%", attendance >= 75), unsafe_allow_html=True)
            f_col3.markdown(factor_html("Credits", credit_hours, credit_hours >= 60), unsafe_allow_html=True)
            f_col4.markdown(factor_html("Finance Score", financial_score, financial_score <= 40), unsafe_allow_html=True)

            # 3. Probability Gauge (Optional visual)
            with st.expander("View AI Probability Details"):
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Eligibility Probability"},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': main_color}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Default State / Landing
        st.info("👈 Please enter student details in the sidebar to begin analysis.")
        
        # Methodology Section
        st.markdown("#### How it works")
        col1, col2 = st.columns(2)
        with col1:
             st.markdown("""
             **1. Academic Screening**
             - **GPA:** Must be roughly above 2.5.
             - **Credit Hours:** Must be a Senior (100+ hours).
             """)
        with col2:
             st.markdown("""
             **2. Holistic Review (AI)**
             - **Attendance:** Shows dedication.
             - **Financial Need:** Prioritized (Lower score = higher need).
             """)

# --- TAB 2: Batch Processing ---
with tab2:
    st.header("📂 Batch File Processing")
    st.markdown("Upload a CSV file containing student records to process them in bulk.")
    
    uploaded_file = st.file_uploader("Drop CSV file here", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} records.")
            
            if st.button("Start Batch Analysis"):
                with st.spinner("Processing records..."):
                    # Logic
                    batch_df['Status'] = 'Pending'
                    batch_df['Note'] = ''
                    
                    # 1. Hard Rules
                    mask_fail = (batch_df['credit_hours'] < 100) | (batch_df['gpa'] < 2.5)
                    batch_df.loc[mask_fail, 'Status'] = 'Not Eligible'
                    batch_df.loc[mask_fail, 'Note'] = 'Below Thresholds'
                    
                    # 2. AI Model
                    mask_ai = ~mask_fail
                    candidates = batch_df.loc[mask_ai]
                    
                    if len(candidates) > 0:
                        feats = candidates[['gpa', 'attendance', 'financial_score', 'credit_hours']]
                        feats_scaled = scaler.transform(feats)
                        probs = model.predict(feats_scaled, verbose=0)
                        preds = (probs > 0.5).astype(int).flatten()
                        
                        labels = np.where(preds == 1, 'Eligible', 'Not Eligible')
                        batch_df.loc[mask_ai, 'Status'] = labels
                        batch_df.loc[mask_ai, 'Note'] = 'AI Evaluated'
                    
                    # Display Simple Stats
                    st.markdown("### Results Summary")
                    col_a, col_b = st.columns(2)
                    elig_count = len(batch_df[batch_df['Status']=='Eligible'])
                    col_a.metric("Eligible", elig_count)
                    col_b.metric("Not Eligible", len(batch_df) - elig_count)
                    
                    st.dataframe(batch_df.head(20))
                    
                    # Download
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Report", csv_data, "scholarship_results.csv", "text/csv")
                    
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    Scholarship AI System &copy; 2026 | Developed by <strong>Munim Abbas</strong>
</div>
""", unsafe_allow_html=True)
