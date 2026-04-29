import streamlit as st
import os
import sys
import time
import tempfile
import pandas as pd
from PIL import Image
import numpy as np

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding import FaceEmbedder
from scripts.inference import run_inference

# Page configuration
st.set_page_config(
    page_title="FaceID - Milestone 4 Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .match-dec {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        text-align: center;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    .same { color: #4ade80; border: 2px solid #4ade80; background: rgba(74, 222, 128, 0.1); }
    .diff { color: #f87171; border: 2px solid #f87171; background: rgba(248, 113, 113, 0.1); }
    
    .stMarkdown pre {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Singleton Model Loader
@st.cache_resource
def get_embedder():
    return FaceEmbedder(model_name="Facenet")

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def load_markdown(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return "File not found."

def main():
    st.markdown("<h1 class='main-title'>FaceID Final Release</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Milestone 4: Hardware-Aware Inference & Professional Documentation</p>", unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Real-time Inference", "📊 Performance Insights", "🛡️ System Card", "📈 Final Metrics"])
    
    # --- TAB 1: INFERENCE ---
    with tab1:
        # Sample selection
        st.markdown("### 🧬 Quick Select Samples")
        samples = {
            "Custom Upload": (None, None),
            "Same Person (Albrecht Mentz)": (
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0000.jpg",
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0001.jpg"
            ),
            "Same Person (Alejandro Toledo)": (
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0000.jpg",
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0001.jpg"
            ),
            "Different People (Albrecht vs Alejandro)": (
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0000.jpg",
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0000.jpg"
            )
        }
        sample_choice = st.selectbox("Pick a pre-loaded pair or use your own:", list(samples.keys()))
        
        col1, col2 = st.columns([1, 1])
        s_img1, s_img2 = samples[sample_choice]
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Subject A")
            if s_img1:
                st.image(s_img1, use_container_width=True)
                file1 = s_img1
            else:
                file1_up = st.file_uploader("Upload image 1", type=['jpg', 'jpeg', 'png'], key="app_img1")
                if file1_up: 
                    st.image(file1_up, use_container_width=True)
                    file1 = save_uploaded_file(file1_up)
                else: file1 = None
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Subject B")
            if s_img2:
                st.image(s_img2, use_container_width=True)
                file2 = s_img2
            else:
                file2_up = st.file_uploader("Upload image 2", type=['jpg', 'jpeg', 'png'], key="app_img2")
                if file2_up: 
                    st.image(file2_up, use_container_width=True)
                    file2 = save_uploaded_file(file2_up)
                else: file2 = None
            st.markdown("</div>", unsafe_allow_html=True)
            
        threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.35, 0.01)
        
        if st.button("Run Verification", use_container_width=True, type="primary"):
            if file1 and file2:
                with st.spinner("Analyzing..."):
                    embedder = get_embedder()
                    try:
                        res = run_inference(file1, file2, threshold, embedder=embedder)
                        cls = "same" if res['decision'] == "SAME" else "diff"
                        st.markdown(f"<div class='match-dec {cls}'>{res['decision']} (Confidence: {res['confidence']*100:.1f}%)</div>", unsafe_allow_html=True)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Similarity Score", f"{res['similarity_score']:.4f}")
                        m2.metric("Total Latency", f"{res['latency_total_ms']:.1f}ms")
                        m3.metric("Extraction Time", f"{res['latency_emb_ms']:.1f}ms")
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                    finally:
                        # Only cleanup if it was a temp file from upload
                        if isinstance(file1, str) and "tmp" in file1 and os.path.exists(file1): os.remove(file1)
                        if isinstance(file2, str) and "tmp" in file2 and os.path.exists(file2): os.remove(file2)
            else:
                st.warning("Please upload images or select a sample.")

    # --- TAB 2: PERFORMANCE ---
    with tab2:
        st.subheader("Hardware-Aware Profiling Results")
        st.info("Measurements taken on local hardware to characterize CPU latency and throughput.")
        
        # Latency breakdown
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("#### Latency Breakdown")
            latency_data = pd.DataFrame({
                "Stage": ["Embedding", "Similarity"],
                "Mean (ms)": [464.76, 0.15]
            })
            st.bar_chart(latency_data.set_index("Stage"))
        
        with l_col2:
            st.markdown("#### Throughput by Batch Size")
            throughput_data = pd.DataFrame({
                "Batch Size": [1, 4, 8, 16],
                "FPS": [2.10, 2.03, 2.04, 2.20]
            })
            st.line_chart(throughput_data.set_index("Batch Size"))
            
        st.markdown("---")
        st.markdown("#### Profiling Summary")
        summary_txt = load_markdown("reports/profiling_summary.txt")
        # Use st.code for high contrast and readability
        st.code(summary_txt, language="markdown")

    # --- TAB 3: SYSTEM CARD ---
    with tab3:
        st.subheader("System Documentation")
        system_card = load_markdown("reports/System_Card.md")
        st.markdown(system_card)

    # --- TAB 4: FINAL METRICS ---
    with tab4:
        st.subheader("Final Evaluation Summary (Milestone 4)")
        e_col1, e_col2, e_col3 = st.columns(3)
        e_col1.metric("Accuracy", "84.6%")
        e_col2.metric("F1-Score", "0.8254")
        e_col3.metric("Pairs Evaluated", "500")
        
        st.markdown("#### ROC Curve Artifact")
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png", caption="ROC Curve for Final Release Model")
        else:
            st.write("ROC curve image not found.")

if __name__ == "__main__":
    main()
