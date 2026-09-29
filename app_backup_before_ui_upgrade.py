import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

from src.utils import (
    MODEL_PATH, DATASET_DIR, OCCUPANCY_THRESHOLDS, calculate_occupancy_stats, get_default_rois
)
from src.seat_counter import SeatCounterEngine

# Streamlit Page Config
st.set_page_config(
    page_title="JUW SMART SPACE - Seat & PC Occupancy Detection",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern University Aesthetics
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #2563EB;
    }
    .metric-val {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Header
st.markdown('<div class="main-title">JUW SMART SPACE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based Empty Seat & PC Occupancy Detection Using Convolutional Neural Network (CNN)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://www.juw.edu.pk/wp-content/uploads/2019/12/logo.png", width=200)
st.sidebar.header("⚙️ Project Controls")
st.sidebar.markdown("**Real-World Context:** Jinnah University for Women (JUW)")

# Check Model File
model_exists = os.path.exists(MODEL_PATH)
if not model_exists:
    st.error("⚠️ Trained CNN model weights not found! Please run `python src/prepare_dataset.py` and `python src/train.py` first.")
    st.stop()

# Initialize Engine
@st.cache_resource
def load_engine():
    return SeatCounterEngine(MODEL_PATH)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading model engine: {e}")
    st.stop()

# Threshold Settings
st.sidebar.subheader("Occupancy Level Thresholds")
low_thresh = st.sidebar.slider("Low Occupancy Upper Threshold (%)", 10, 50, int(OCCUPANCY_THRESHOLDS['LOW']))
med_thresh = st.sidebar.slider("Medium Occupancy Upper Threshold (%)", 51, 90, int(OCCUPANCY_THRESHOLDS['MEDIUM']))

# Source Selection
st.sidebar.subheader("📷 Image Source")
source_type = st.sidebar.radio("Select Input Mode", ["Sample JUW Lab Overview", "Upload Custom Image"])

input_image = None
image_name = "lab_view.jpg"

if source_type == "Sample JUW Lab Overview":
    sample_dir = os.path.join(DATASET_DIR, 'sample_lab_views')
    sample_files = glob.glob(os.path.join(sample_dir, '*.jpg'))
    if sample_files:
        selected_sample = st.sidebar.selectbox("Select Sample Overview View", sample_files, format_func=lambda x: os.path.basename(x))
        input_image = cv2.imread(selected_sample)
        image_name = os.path.basename(selected_sample)
    else:
        st.warning("No sample overview images found. Generating sample lab view...")
        from src.prepare_dataset import generate_sample_lab_overview
        img_np = generate_sample_lab_overview(20, 13, seed=42)
        input_image = img_np
else:
    uploaded_file = st.sidebar.file_uploader("Upload Computer Lab or Classroom Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        input_image = np.array(pil_img)
        image_name = uploaded_file.name

if input_image is None:
    st.info("👈 Please select or upload a computer lab image from the sidebar to begin detection.")
    st.stop()

# Grid ROI Configuration
st.sidebar.subheader("📐 Workstation Grid Layout")
rows = st.sidebar.slider("Grid Rows", 1, 8, 4)
cols = st.sidebar.slider("Grid Columns", 1, 10, 5)

def build_custom_grid(img_shape, r, c):
    h, w = img_shape[:2]
    rois = []
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.08)
    spacing_x = int((w - 2 * margin_x) / c)
    spacing_y = int((h - 2 * margin_y) / r)
    box_w = int(spacing_x * 0.8)
    box_h = int(spacing_y * 0.75)
    count = 1
    for i in range(r):
        for j in range(c):
            x = margin_x + j * spacing_x + int(spacing_x * 0.1)
            y = margin_y + i * spacing_y + int(spacing_y * 0.1)
            rois.append((f"PC-{count:02d}", (x, y, box_w, box_h)))
            count += 1
    return rois

custom_rois = build_custom_grid(input_image.shape, rows, cols)

# Run Prediction Engine
with st.spinner("Analyzing computer lab workstation occupancy using CNN..."):
    annotated_rgb, stats, predictions = engine.analyze_lab_image(input_image, custom_rois)

# Recalculate stats based on custom slider thresholds
occ_pct = stats['occupancy_pct']
if occ_pct <= low_thresh:
    level = "LOW"
    badge_class = "badge-low"
elif occ_pct <= med_thresh:
    level = "MEDIUM"
    badge_class = "badge-medium"
else:
    level = "HIGH"
    badge_class = "badge-high"

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f'''
    <div class="card">
        <div class="metric-lbl">Total PCs</div>
        <div class="metric-val">{stats["total"]}</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="card" style="border-left-color: #EF4444;">
        <div class="metric-lbl">Occupied PCs</div>
        <div class="metric-val" style="color: #EF4444;">{stats["occupied"]}</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="card" style="border-left-color: #10B981;">
        <div class="metric-lbl">Available PCs</div>
        <div class="metric-val" style="color: #10B981;">{stats["available"]}</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="card">
        <div class="metric-lbl">Occupancy Rate</div>
        <div class="metric-val">{stats["occupancy_pct"]}%</div>
    </div>
    ''', unsafe_allow_html=True)

with col5:
    st.markdown(f'''
    <div class="card">
        <div class="metric-lbl">Occupancy Level</div>
        <div style="margin-top: 8px;"><span class="{badge_class}">{level}</span></div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Image Visualization Section
img_col1, img_col2 = st.columns(2)

with img_col1:
    st.subheader("🖼️ Original Lab Image")
    st.image(input_image, width='stretch', caption=f"Source: {image_name}")

with img_col2:
    st.subheader("🔍 AI Workstation Occupancy Detection")
    st.image(annotated_rgb, width='stretch', caption="CNN Region-of-Interest (ROI) Classification")

st.markdown("---")

# Workstation Status Breakdown Table
st.subheader("📋 Individual Workstation Status & Confidence Scores")
df_preds = pd.DataFrame(predictions)
df_display = df_preds[['id', 'status', 'confidence', 'probability']].copy()
df_display.columns = ['Workstation ID', 'Occupancy Status', 'Confidence (%)', 'CNN Raw Probability']

col_tab1, col_tab2 = st.columns([3, 2])
with col_tab1:
    st.dataframe(
        df_display.style.map(
            lambda val: 'color: red; font-weight: bold;' if val == 'OCCUPIED' else 'color: green; font-weight: bold;' if val == 'EMPTY' else '',
            subset=['Occupancy Status']
        ),
        width='stretch',
        height=300
    )

with col_tab2:
    st.markdown("### 📊 Status Breakdown")
    status_counts = df_display['Occupancy Status'].value_counts()
    st.bar_chart(status_counts, color="#2563EB")

st.markdown("---")

# Information & University Project Documentation Tabs
tab1, tab2, tab3 = st.tabs(["📸 How to Add Real JUW Photos", "📹 Future Real-Time Camera Pipeline", "🧠 ANN Model Architecture"])

with tab1:
    st.markdown("""
    ### 📸 Instructions for Adding On-Site JUW Computer Lab Photos
    When real photos of JUW computer labs are captured on campus:
    1. Place cropped empty seat/PC images into: `dataset/raw/juw_real/empty/`
    2. Place cropped occupied seat/PC images into: `dataset/raw/juw_real/occupied/`
    3. Run dataset re-splitting and fine-tuning:
       ```bash
       python src/prepare_dataset.py
       python src/train.py
       python src/evaluate.py
       ```
    The pipeline automatically integrates JUW real images into training, validation, and test datasets without requiring any code changes.
    """)

with tab2:
    st.markdown("""
    ### 📹 Extension to Live RTSP Camera / Video Feed
    The current architecture is fully modularized for live camera streaming:
    ```
    Live Camera / RTSP Stream 
             ↓
    Extract Video Frame (OpenCV VideoCapture)
             ↓
    Extract Defined Workstation ROIs (seat_counter.py)
             ↓
    Batch Inference via Trained CNN (seat_occupancy_cnn.keras)
             ↓
    Calculate Real-time Occupancy Metrics & Display Live Dashboard
    ```
    """)

with tab3:
    st.markdown("""
    ### 🧠 Convolutional Neural Network (CNN) Details
    * **Input Layer**: 128 x 128 x 3 RGB image patch
    * **Convolutional Layers**: 3 Conv2D blocks (32, 64, 128 filters, 3x3 kernels, ReLU)
    * **Pooling**: MaxPooling2D (2x2) after each Conv block
    * **Regularization**: Data Augmentation + Dropout (0.5)
    * **Output Layer**: Dense (1 unit, Sigmoid activation) for Binary Classification (0 = EMPTY, 1 = OCCUPIED)
    * **Loss Function**: Binary Cross-Entropy
    * **Optimizer**: Adam (learning rate = 0.001)
    """)

# Footer
st.markdown("<br><hr><center><small>JUW SMART SPACE Project | Jinnah University for Women | Department of Computer Science & Software Engineering</small></center>", unsafe_allow_html=True)
