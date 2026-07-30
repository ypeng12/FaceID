import streamlit as st
import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.embedding import FaceEmbedder
from src.similarity import numpy_vectorized_cosine, numpy_vectorized_euclidean
from src.gallery import FaceGallery

# Page configuration
st.set_page_config(
    page_title="FaceID - Real-Time Recognition & Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .status-badge {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        text-align: center;
        padding: 0.8rem 1.2rem;
        border-radius: 0.75rem;
        font-weight: bold;
        margin-top: 1rem;
        letter-spacing: 1px;
    }
    
    .match-same { 
        color: #4ade80; 
        border: 2px solid #4ade80; 
        background: rgba(74, 222, 128, 0.12); 
        box-shadow: 0 0 15px rgba(74, 222, 128, 0.3);
    }
    .match-diff { 
        color: #f87171; 
        border: 2px solid #f87171; 
        background: rgba(248, 113, 113, 0.12); 
        box-shadow: 0 0 15px rgba(248, 113, 113, 0.3);
    }
    
    .identity-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 0.75rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loader
@st.cache_resource
def get_embedder(model_name="Facenet"):
    return FaceEmbedder(model_name=model_name)

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def draw_face_bbox(image_path_or_pil, facial_area, color=(56, 189, 248), label="Face"):
    """Draw bounding box on image using PIL."""
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.copy()
        
    x, y, w, h = facial_area.get("x", 0), facial_area.get("y", 0), facial_area.get("w", 0), facial_area.get("h", 0)
    if w == 0 or h == 0:
        return img
        
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
    if label:
        draw.rectangle([x, max(0, y - 24), x + w, y], fill=color)
        draw.text((x + 6, max(0, y - 20)), label, fill=(0, 0, 0))
    return img

def load_markdown_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return "File not found."

# Pre-populate 1:N Gallery with sample images if available
@st.cache_resource
def init_sample_gallery():
    gallery = FaceGallery()
    embedder = get_embedder("Facenet")
    
    samples_dir = "data/lfw/test"
    if os.path.exists(samples_dir):
        count = 0
        for person_name in os.listdir(samples_dir):
            person_path = os.path.join(samples_dir, person_name)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    first_img = os.path.join(person_path, images[0])
                    try:
                        emb = embedder.compute_embedding(first_img)
                        gallery.enroll(name=person_name.replace("_", " "), embedding=emb, image_path=first_img)
                        count += 1
                    except Exception as e:
                        pass
            if count >= 10: # Limit initial gallery size for speed
                break
    return gallery

def main():
    st.markdown("<h1 class='main-title'>FaceID Recognition Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Deep Learning Face Verification & 1:N Identification Suite</p>", unsafe_allow_html=True)
    
    # Sidebar setup
    st.sidebar.title("⚙️ Engine Settings")
    model_choice = st.sidebar.selectbox("Backbone Model", ["Facenet", "VGG-Face", "ArcFace", "SFace"])
    threshold = st.sidebar.slider("Verification Threshold (Cosine)", 0.0, 1.0, 0.35, 0.01)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Hugging Face Space Live Demo**\nUses FaceNet Inception Architecture for 128D/512D embeddings.")

    # Initialize embedder & gallery
    embedder = get_embedder(model_choice)
    gallery = init_sample_gallery()
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 1:1 Verification", 
        "👤 1:N Gallery Search", 
        "📸 Face Inspector & Vectors", 
        "📊 System Insights & Cards"
    ])
    
    # ==================== TAB 1: 1:1 VERIFICATION ====================
    with tab1:
        st.markdown("### 🔍 1:1 Face Verification")
        st.caption("Compare two face images to verify if they belong to the same person.")
        
        sample_pairs = {
            "Custom Upload": (None, None),
            "Same Identity: Albrecht Mentz": (
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0000.jpg",
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0001.jpg"
            ),
            "Same Identity: Alejandro Toledo": (
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0000.jpg",
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0001.jpg"
            ),
            "Different Identities: Albrecht vs Alejandro": (
                "data/lfw/test/Albrecht_Mentz/Albrecht_Mentz_0000.jpg",
                "data/lfw/test/Alejandro_Toledo/Alejandro_Toledo_0000.jpg"
            )
        }
        
        selected_preset = st.selectbox("Quick Sample Presets:", list(sample_pairs.keys()))
        s_img1, s_img2 = sample_pairs[selected_preset]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Subject A")
            if s_img1 and os.path.exists(s_img1):
                img1_path = s_img1
                st.image(img1_path, use_container_width=True)
            else:
                up1 = st.file_uploader("Upload Image A", type=['jpg', 'jpeg', 'png'], key="v_img1")
                img1_path = save_uploaded_file(up1) if up1 else None
                if up1: st.image(up1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Subject B")
            if s_img2 and os.path.exists(s_img2):
                img2_path = s_img2
                st.image(img2_path, use_container_width=True)
            else:
                up2 = st.file_uploader("Upload Image B", type=['jpg', 'jpeg', 'png'], key="v_img2")
                img2_path = save_uploaded_file(up2) if up2 else None
                if up2: st.image(up2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.button("⚡ Run Verification", type="primary", use_container_width=True):
            if img1_path and img2_path:
                with st.spinner("Extracting facial embeddings & computing metrics..."):
                    t0 = time.time()
                    details1 = embedder.extract_face_details(img1_path)
                    details2 = embedder.extract_face_details(img2_path)
                    t_extract = (time.time() - t0) * 1000
                    
                    emb1, emb2 = details1["embedding"], details2["embedding"]
                    sim_cos = float(numpy_vectorized_cosine(emb1.reshape(1, -1), emb2.reshape(1, -1))[0])
                    dist_euc = float(numpy_vectorized_euclidean(emb1.reshape(1, -1), emb2.reshape(1, -1))[0])
                    
                    is_same = sim_cos >= threshold
                    confidence = min(100.0, max(50.0, (sim_cos / (threshold * 2.0)) * 100.0)) if is_same else min(50.0, (sim_cos / threshold) * 50.0)
                    
                    # Display Result Badge
                    badge_cls = "match-same" if is_same else "match-diff"
                    decision_str = "MATCH: SAME PERSON" if is_same else "NO MATCH: DIFFERENT PERSONS"
                    st.markdown(f"<div class='status-badge {badge_cls}'>{decision_str} (Confidence: {confidence:.1f}%)</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Cosine Similarity", f"{sim_cos:.4f}")
                    m2.metric("Euclidean Distance", f"{dist_euc:.4f}")
                    m3.metric("Decision Threshold", f"{threshold:.2f}")
                    m4.metric("Extraction Latency", f"{t_extract:.1f} ms")
                    
                    # Show Bounding Box overlays
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        boxed1 = draw_face_bbox(img1_path, details1["facial_area"], label="Subject A")
                        st.image(boxed1, caption="Detected Face A", use_container_width=True)
                    with b_col2:
                        boxed2 = draw_face_bbox(img2_path, details2["facial_area"], label="Subject B")
                        st.image(boxed2, caption="Detected Face B", use_container_width=True)
            else:
                st.warning("Please upload or select images for both Subject A and Subject B.")

    # ==================== TAB 2: 1:N GALLERY SEARCH ====================
    with tab2:
        st.markdown("### 👤 1:N Face Database Identification")
        st.caption("Search an unknown query face against enrolled identities in the Face Gallery.")
        
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("➕ Enroll New Identity")
            new_name = st.text_input("Person Name / ID", placeholder="e.g. Elon Musk")
            new_img_up = st.file_uploader("Upload Enrollment Photo", type=['jpg', 'jpeg', 'png'], key="enroll_file")
            
            if st.button("Register to Gallery", use_container_width=True):
                if new_name and new_img_up:
                    saved_p = save_uploaded_file(new_img_up)
                    emb = embedder.compute_embedding(saved_p)
                    gallery.enroll(name=new_name, embedding=emb, image_path=saved_p)
                    st.success(f"Enrolled '{new_name}' successfully! Gallery size: {gallery.count()}")
                else:
                    st.warning("Please enter name and upload photo.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"**Enrolled Identities in Database:** `{gallery.count()}`")

        with c_right:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Query Identity")
            query_up = st.file_uploader("Upload Unknown Face Photo", type=['jpg', 'jpeg', 'png'], key="query_file")
            top_k = st.slider("Top Results (K)", 1, 5, 3)
            
            if query_up:
                st.image(query_up, width=200, caption="Query Input")
                if st.button("🔍 Search Face Gallery", type="primary", use_container_width=True):
                    q_path = save_uploaded_file(query_up)
                    with st.spinner("Searching gallery vectors..."):
                        q_emb = embedder.compute_embedding(q_path)
                        matches = gallery.search(q_emb, top_k=top_k, threshold=threshold)
                        
                        if matches:
                            st.markdown("#### Top Matching Identities:")
                            for rank, match in enumerate(matches, 1):
                                sim = match["similarity"]
                                is_m = match["is_match"]
                                color_bar = "🟢" if is_m else "🔴"
                                
                                res_col1, res_col2 = st.columns([1, 3])
                                with res_col1:
                                    if match["image_path"] and os.path.exists(match["image_path"]):
                                        st.image(match["image_path"], use_container_width=True)
                                with res_col2:
                                    st.markdown(f"### #{rank} {match['name']} {color_bar}")
                                    st.progress(max(0.0, min(1.0, sim)))
                                    st.write(f"Similarity Score: `{sim:.4f}` | Decision: `{'MATCH' if is_m else 'NO MATCH'}`")
                                st.markdown("---")
                        else:
                            st.info("No identities in gallery. Please enroll faces first.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ==================== TAB 3: FACE INSPECTOR & VECTORS ====================
    with tab3:
        st.markdown("### 📸 Face Inspector & Embedding Visualizer")
        st.caption("Inspect facial alignment, bounding box coordinates, and 128D/512D deep feature vectors.")
        
        insp_up = st.file_uploader("Upload Face Image for Analysis", type=['jpg', 'jpeg', 'png'], key="insp_file")
        
        if insp_up:
            insp_path = save_uploaded_file(insp_up)
            details = embedder.extract_face_details(insp_path)
            emb = details["embedding"]
            area = details["facial_area"]
            
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Facial Bounding Box & Detection")
                boxed_img = draw_face_bbox(insp_path, area, label="Detected Face")
                st.image(boxed_img, use_container_width=True)
                st.write(f"**Bounding Box (x, y, w, h):** `{area}`")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_i2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Embedding Vector Statistics")
                st.metric("Vector Dimension", f"{len(emb)}D")
                st.metric("Vector L2 Norm", f"{np.linalg.norm(emb):.4f}")
                st.metric("Mean Value", f"{np.mean(emb):.4f}")
                st.metric("Standard Deviation", f"{np.std(emb):.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("128D Deep Feature Profile (Embedding Heatmap)")
            df_emb = pd.DataFrame({"Feature Dimension": range(len(emb)), "Activation Value": emb})
            st.line_chart(df_emb.set_index("Feature Dimension"))
            st.markdown("</div>", unsafe_allow_html=True)

    # ==================== TAB 4: SYSTEM INSIGHTS ====================
    with tab4:
        st.markdown("### 📊 System Insights & Documentation")
        
        r_tab1, r_tab2, r_tab3 = st.tabs(["🚀 Latency & Profiling", "🛡️ System Card", "📈 ROC & Metrics"])
        
        with r_tab1:
            st.subheader("Hardware-Aware Latency Breakdown")
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.markdown("#### Latency Breakdown (CPU)")
                latency_data = pd.DataFrame({
                    "Stage": ["Embedding Extraction", "Similarity Calculation"],
                    "Mean Latency (ms)": [464.76, 0.15]
                })
                st.bar_chart(latency_data.set_index("Stage"))
            
            with l_col2:
                st.markdown("#### Throughput Sensitivity (FPS vs Batch Size)")
                throughput_data = pd.DataFrame({
                    "Batch Size": [1, 4, 8, 16],
                    "Throughput (FPS)": [2.10, 2.03, 2.04, 2.20]
                })
                st.line_chart(throughput_data.set_index("Batch Size"))
                
            st.markdown("---")
            st.markdown("#### Detailed Profiling Summary")
            summary_txt = load_markdown_file("reports/profiling_summary.txt")
            st.code(summary_txt, language="markdown")

        with r_tab2:
            st.subheader("System Card Documentation")
            sys_card_md = load_markdown_file("reports/System_Card.md")
            st.markdown(sys_card_md)
            
        with r_tab3:
            st.subheader("Model Evaluation Summary")
            e1, e2, e3 = st.columns(3)
            e1.metric("Verification Accuracy", "84.6%")
            e2.metric("F1-Score", "0.8254")
            e3.metric("Evaluated Pairs", "500")
            
            if os.path.exists("reports/roc_curve.png"):
                st.image("reports/roc_curve.png", caption="ROC Curve for Calibrated Model", use_container_width=True)

if __name__ == "__main__":
    main()
