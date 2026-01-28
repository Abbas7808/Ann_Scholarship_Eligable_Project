# ========= Importing libraries ==============
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import json
import os

# -----------------------------------------------------------------------------
# 0. Configuration Management
# -----------------------------------------------------------------------------
CONFIG_FILE = 'scholarship_config.json'

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scholarship AI Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-weight: 700; }
    
    .factor-card {
        background-color: white; padding: 1rem; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); text-align: center; border: 1px solid #e2e8f0;
    }
    .factor-value { font-size: 1.5rem; font-weight: 700; }
    .factor-label { color: #64748b; font-size: 0.875rem; font-weight: 500; text-transform: uppercase; }
    
    .stButton > button {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        color: white; border: none; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.9; }
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
        return None, None

model, scaler = load_artifacts()
config = load_config()

# -----------------------------------------------------------------------------
# 3. Sidebar - Navigation & Input
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997385.png", width=60)
    st.markdown("### 🎓 Scholarship Portal")
    
    mode = st.radio("System Mode", ["Student Application", "Admin Panel"], horizontal=True)
    st.markdown("---")

# =============================================================================
# OPTION A: ADMIN PANEL
# =============================================================================
if mode == "Admin Panel":
    st.title("🛡️ Admin Configuration Panel")
    
    password = st.sidebar.text_input("Admin Password", type="password")
    
    if password == "admin123":
        st.success("Authorized Access")
        
        tab_view, tab_add = st.tabs(["📝 Edit Existing Criteria", "➕ Add New Type"])
        
        # --- Edit Existing ---
        with tab_view:
            scholarship_type = st.selectbox("Select Scholarship Type to Edit", list(config.keys()))
            
            if scholarship_type:
                data = config[scholarship_type]
                
                with st.form("edit_form"):
                    new_desc = st.text_area("Description", data.get('description', ''))
                    
                    c1, c2 = st.columns(2)
                    new_min_gpa = c1.number_input("Min GPA", 0.0, 4.0, data.get('min_gpa', 2.0))
                    new_min_att = c2.number_input("Min Attendance %", 0, 100, data.get('min_attendance', 75))
                    
                    c3, c4 = st.columns(2)
                    new_min_cred = c3.number_input("Min Credit Hours", 0, 130, data.get('min_credit_hours', 50))
                    new_max_fin = c4.number_input("Max Financial Score (Strictness)", 0, 100, data.get('max_financial_score', 50), help="Students must be BELOW this score if checking need.")
                    
                    # Documents as comma-separated string
                    current_docs = ", ".join(data.get('required_documents', []))
                    new_docs_str = st.text_area("Required Documents (comma separated)", current_docs)
                    
                    if st.form_submit_button("💾 Save Changes"):
                        config[scholarship_type] = {
                            "description": new_desc,
                            "min_gpa": new_min_gpa,
                            "min_attendance": new_min_att,
                            "min_credit_hours": new_min_cred,
                            "max_financial_score": new_max_fin,
                            "required_documents": [d.strip() for d in new_docs_str.split(',') if d.strip()]
                        }
                        save_config(config)
                        st.success("Configuration updated successfully!")
                        st.rerun()

        # --- Add New ---
        with tab_add:
            with st.form("add_form"):
                new_key = st.text_input("New Scholarship Name (e.g. 'Women in Tech')")
                add_desc = st.text_area("Description")
                
                cur_docs = "National ID, Student Card"
                add_docs = st.text_area("Required Documents", cur_docs)
                
                submit_add = st.form_submit_button("Create Scholarship Type")
                if submit_add and new_key:
                    if new_key in config:
                        st.error("Type already exists!")
                    else:
                        config[new_key] = {
                            "description": add_desc,
                            "min_gpa": 2.5, "min_attendance": 75, "min_credit_hours": 30, "max_financial_score": 100,
                            "required_documents": [d.strip() for d in add_docs.split(',') if d.strip()]
                        }
                        save_config(config)
                        st.success(f"Created {new_key}!")
                        st.rerun()
                
    else:
        st.warning("Please enter the admin password to access configuration.")
        st.info("Hint: admin123")

# =============================================================================
# OPTION B: STUDENT APPLICATION
# =============================================================================
else:
    # Sidebar Inputs
    with st.sidebar:
        selected_type = st.selectbox("Select Scholarship Type", list(config.keys()))
        
        # Defaults
        def_gpa, def_att, def_cred, def_fin = 3.5, 85, 60, 20
        
        # Dataset Filler
        with st.expander("🛠️ Autofill from Database", expanded=False):
            if st.checkbox("Use Existing Student Record"):
                try:
                    df_existing = pd.read_csv('dataset.csv')
                    row_idx = st.number_input("Student ID", 0, len(df_existing)-1, 0)
                    row = df_existing.iloc[row_idx]
                    def_gpa = float(row['gpa'])
                    def_att = int(row['attendance'])
                    def_cred = int(row['credit_hours'])
                    def_fin = int(row['financial_score'])
                    st.success("Loaded!")
                except:
                    st.error("dataset.csv not found")

        st.markdown("### 📝 Your Profile")
        gpa = st.slider("GPA", 0.0, 4.0, def_gpa, 0.01)
        attendance = st.slider("Attendance %", 0, 100, def_att)
        credit_hours = st.slider("Credit Hours", 0, 150, def_cred)
        financial_score = st.slider("Financial Score", 0, 100, def_fin, help="0 = Poor, 100 = Rich")
        
        analyze_btn = st.button("🚀 Check Eligibility", type="primary")

    # Main Area
    st.title("🎓 Intelligent Scholarship Portal")
    
    if selected_type:
        criteria = config[selected_type]
        
        st.markdown(f"""
        <div style="background: #e0f2fe; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #0284c7;">
            <h3 style="margin:0; color: #0284c7;">{selected_type}</h3>
            <p style="margin-top:0.5rem; color: #334155;">{criteria.get('description', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Document Upload Section (Dynamic) ---
        st.markdown("### 📂 Required Documentation")
        required_docs = criteria.get('required_documents', [])
        
        uploaded_files = {}
        missing_docs = False
        
        if not required_docs:
            st.info("No documents required for this pre-screen.")
        else:
            cols = st.columns(2)
            for i, doc_name in enumerate(required_docs):
                with cols[i % 2]:
                    f = st.file_uploader(f"Upload {doc_name}", key=f"doc_{doc_name}")
                    if f:
                        uploaded_files[doc_name] = f
            
            # Check if all present
            if len(uploaded_files) < len(required_docs):
                missing_docs = True

        # --- Analysis Logic ---
        if analyze_btn:
            st.markdown("---")
            
            # 1. Document Check
            if missing_docs:
                st.warning("⚠️ Please upload all required documents to proceed with the application.")
            
            else:
                # 2. Hard Criteria Check (from Config)
                passed_criteria = True
                fail_reasons = []
                
                if gpa < criteria.get('min_gpa', 0):
                    passed_criteria = False
                    fail_reasons.append(f"GPA {gpa} is below minimum {criteria['min_gpa']}")
                
                if attendance < criteria.get('min_attendance', 0):
                    passed_criteria = False
                    fail_reasons.append(f"Attendance {attendance}% is below minimum {criteria['min_attendance']}%")
                
                # Financial check only if score is high (meaning rich) but max allowed is low (need based)
                # If scholarship requires "max_financial_score" of 40, and user has 80 (rich), they fail.
                req_fin_max = criteria.get('max_financial_score', 100)
                if financial_score > req_fin_max:
                    passed_criteria = False
                    fail_reasons.append(f"Financial Score {financial_score} is too high for this Need-Based category.")

                # 3. AI Check (General Suitability)
                if model and scaler:
                    input_scaled = scaler.transform([[gpa, attendance, financial_score, credit_hours]])
                    ai_prob = model.predict(input_scaled, verbose=0)[0][0]
                else:
                    ai_prob = 0.5 # Default if no model
                
                # Combine Checks
                # If specific criteria fail, immediate fail.
                # If specific criteria pass, use AI to give "Holistic" score or flag "At Risk".
                
                final_eligible = passed_criteria and (ai_prob > 0.4) # Slightly different threshold? Or just rely on criteria?
                # Let's say: MUST pass criteria, AND AI must not strongly reject (e.g. < 0.3).
                # Actually, let's keep it simple: Criteria is the GATE, AI is the RANKER.
                
                if not passed_criteria:
                    st.error("❌ Not Eligible")
                    for reason in fail_reasons:
                        st.markdown(f"- {reason}")
                else:
                    # Passed hard rules
                    if ai_prob > 0.6:
                        st.balloons()
                        st.success("✅ Eligible! You have a high chance of approval.")
                        st.markdown(f"**AI Confidence:** {ai_prob:.1%}")
                    elif ai_prob > 0.4:
                        st.warning("⚠️ Conditionally Eligible. You meet requirements, but competition is high.")
                        st.markdown(f"**AI Confidence:** {ai_prob:.1%}")
                    else:
                        st.error("❌ Meeting minimums, but AI analysis suggests low probability based on historical trends.")
                        st.markdown(f"**AI Confidence:** {ai_prob:.1%}")

                    # Show uploaded proofs
                    with st.expander("View Submitted Proofs"):
                        for name, file_obj in uploaded_files.items():
                            st.markdown(f"**{name}:** `{file_obj.name}`")
                            # If image, show it
                            if file_obj.type.startswith('image'):
                                st.image(file_obj, width=200)
