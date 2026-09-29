import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

# Import project utilities and engines safely
try:
    from src.utils import (
        MODEL_PATH, DATASET_DIR, RESULTS_DIR, PLOTS_DIR, OCCUPANCY_THRESHOLDS,
        calculate_occupancy_stats, get_default_rois, preprocess_image_crop
    )
    from src.seat_counter import SeatCounterEngine
    from src.predict import predict_single_image
except ImportError:
    from utils import (
        MODEL_PATH, DATASET_DIR, RESULTS_DIR, PLOTS_DIR, OCCUPANCY_THRESHOLDS,
        calculate_occupancy_stats, get_default_rois, preprocess_image_crop
    )
    from seat_counter import SeatCounterEngine
    from predict import predict_single_image

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="JUW SMART SPACE - Seat & PC Occupancy Detection",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Dashboard Styling (CSS)
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #2563EB 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 6px;
    }
    .context-tag {
        font-size: 0.85rem;
        background: rgba(255, 255, 255, 0.18);
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 10px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)
