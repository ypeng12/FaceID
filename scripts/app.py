import streamlit as st
import os
import sys
import time
import tempfile
from PIL import Image
import numpy as np

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding import FaceEmbedder
from scripts.inference import run_inference

# Page configuration
st.set_page_config(
    page_title="FaceID - Deep Verification Engine",
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
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 3rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 1rem;
    }
    
    .match-dec {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .same { color: #4ade80; border: 2px solid #4ade80; background: rgba(74, 222, 128, 0.1); }
    .diff { color: #f87171; border: 2px solid #f87171; background: rgba(248, 113, 113, 0.1); }
    
    .metric-label { color: #94a3b8; font-size: 0.8rem; }
    .metric-value { color: #f8fafc; font-size: 1.5rem; font-weight: 600; }
</style>
""", unsafe_content=True)

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

def main():
    st.markdown("<h1 class='main-title'>FaceID Engine</h1>", unsafe_content=True)
    st.markdown("<p class='sub-title'>DeepFace Verification System — Milestone 3 Optimized</p>", unsafe_content=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ System Settings")
        threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.40, 0.01)
        st.info("Lower threshold is more strict (needs higher similarity for 'SAME').")
        
        st.divider()
        st.subheader("Model Info")
        st.code("Backbone: FaceNet\nMetric: Cosine Similarity")
        
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.rerun()

    # Main layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_content=True)
        st.subheader("👤 Subject A")
        file1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="img1")
        if file1:
            st.image(file1, use_container_width=True)
        st.markdown("</div>", unsafe_content=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_content=True)
        st.subheader("👤 Subject B")
        file2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="img2")
        if file2:
            st.image(file2, use_container_width=True)
        st.markdown("</div>", unsafe_content=True)

    # Verification Trigger
    if st.button("🚀 Run Verification", use_container_width=True, type="primary"):
        if not file1 or not file2:
            st.warning("Please upload both images to continue.")
        else:
            with st.spinner("Analyzing facial features..."):
                # Load Singleton Embedder
                embedder = get_embedder()
                
                # Save temp files
                path1 = save_uploaded_file(file1)
                path2 = save_uploaded_file(file2)
                
                if path1 and path2:
                    # Run inference
                    try:
                        res = run_inference(path1, path2, threshold, embedder=embedder)
                        
                        # Display Results
                        st.divider()
                        
                        # Result Decision Card
                        cls = "same" if res['decision'] == "SAME" else "diff"
                        st.markdown(f"<div class='match-dec {cls}'>{res['decision']}</div>", unsafe_content=True)
                        
                        # Metrics Columns
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.markdown("<p class='metric-label'>Similarity Score</p>", unsafe_content=True)
                            st.markdown(f"<p class='metric-value'>{res['similarity_score']:.4f}</p>", unsafe_content=True)
                        with m_col2:
                            st.markdown("<p class='metric-label'>Confidence</p>", unsafe_content=True)
                            st.markdown(f"<p class='metric-value'>{res['confidence']*100:.1f}%</p>", unsafe_content=True)
                        with m_col3:
                            st.markdown("<p class='metric-label'>Total Latency</p>", unsafe_content=True)
                            st.markdown(f"<p class='metric-value'>{res['latency_total_ms']:.1f}ms</p>", unsafe_content=True)
                        with m_col4:
                            st.markdown("<p class='metric-label'>Feature Extraction</p>", unsafe_content=True)
                            st.markdown(f"<p class='metric-value'>{res['latency_emb_ms']:.1f}ms</p>", unsafe_content=True)
                            
                    finally:
                        # Cleanup temp files
                        if os.path.exists(path1): os.remove(path1)
                        if os.path.exists(path2): os.remove(path2)

if __name__ == "__main__":
    main()
