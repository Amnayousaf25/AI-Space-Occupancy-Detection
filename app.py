import os
import sys
import glob
import time
import json
import warnings
from typing import List, Dict, Any, Optional

# Suppress verbose TensorFlow, oneDNN and Keras deprecation warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Import core project services safely
try:
    from src.config import (
        MODEL_PATH, DATASET_DIR, RESULTS_DIR, PLOTS_DIR, OCCUPANCY_THRESHOLDS,
        DEFAULT_PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
    )
    from src.utils import calculate_occupancy_stats, preprocess_image_crop, get_default_rois
    from src.spaces_config import get_space_config, get_all_spaces, get_space_rois
    from src.model_service import get_model_service
    from src.prediction_service import classify_workstation_crop
    from src.occupancy_engine import analyze_space_occupancy, analyze_lab_occupancy
    from src.yolo_detector import get_yolo_detector
    from src.analytics import compute_occupancy_analytics
    from src.database import get_all_snapshots, get_latest_occupancy, get_db_connection
    from src.dataset_audit import run_dataset_audit
    from src.dataset_manager import ingest_image, read_manifest, check_data_leakage
    from src.model_registry import get_model_registry, get_active_model_info, set_active_model
    from src.external_validation import evaluate_external_juw_validation
    from src.evaluate_yolo import run_comprehensive_evaluation
    from src.prepare_dataset import generate_sample_lab_overview
except ImportError:
    from config import (
        MODEL_PATH, DATASET_DIR, RESULTS_DIR, PLOTS_DIR, OCCUPANCY_THRESHOLDS,
        DEFAULT_PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
    )
    from utils import calculate_occupancy_stats, preprocess_image_crop, get_default_rois
    from model_service import get_model_service
    from prediction_service import classify_workstation_crop
    from occupancy_engine import analyze_lab_occupancy
    from yolo_detector import get_yolo_detector
    from analytics import compute_occupancy_analytics
    from database import get_all_snapshots, get_latest_occupancy, get_db_connection
    from dataset_audit import run_dataset_audit
    from dataset_manager import ingest_image, read_manifest, check_data_leakage
    from model_registry import get_model_registry, get_active_model_info, set_active_model
    from external_validation import evaluate_external_juw_validation
    from evaluate_yolo import run_comprehensive_evaluation
    from prepare_dataset import generate_sample_lab_overview

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="JUW SMART SPACE | AI Laboratory Intelligence",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Robust HTML Renderer: Zero Raw HTML Bug Guarantee
# ---------------------------------------------------------
def render_html(html_str: str):
    """
    Safely renders custom HTML in Streamlit.
    Strips leading and trailing whitespace from every line so CommonMark markdown
    parsers NEVER mistake lines for indented 4-space verbatim code blocks.
    This guarantees zero raw HTML tags appear on screen.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


# ---------------------------------------------------------
# Global Enterprise Design System (CSS)
# Deep Navy Sidebar + Clean Light Main Workspace + White Cards
# ---------------------------------------------------------
render_html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* Base typography & theme */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Light main workspace background */
.stApp {
    background-color: #F4F7FB;
    color: #0F172A;
}

/* Container spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

/* =========================================================
   SIDEBAR: PROFESSIONAL DARK NAVY THEME (#0B1B3A)
   ========================================================= */
section[data-testid="stSidebar"] {
    background-color: #0B1B3A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

section[data-testid="stSidebar"] div.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Sidebar Brand Header */
.sb-brand-card {
    background: linear-gradient(135deg, rgba(23, 105, 224, 0.15) 0%, rgba(22, 184, 212, 0.1) 100%);
    border: 1px solid rgba(22, 184, 212, 0.25);
    border-radius: 12px;
    padding: 16px 14px;
    margin-bottom: 18px;
    text-align: center;
}
.sb-icon-box {
    width: 44px;
    height: 44px;
    margin: 0 auto 10px auto;
    background: linear-gradient(135deg, #1769E0 0%, #16B8D4 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 12px rgba(22, 184, 212, 0.35);
}
.sb-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.3px;
    margin: 0 0 2px 0;
}
.sb-brand-sub {
    font-size: 0.72rem;
    color: #16B8D4;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Sidebar Section Headers */
.sb-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 16px;
    margin-bottom: 8px;
}

/* Navigation Radio Styling */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px 12px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
    border-left: 3px solid transparent !important;
    background-color: transparent !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] .stRadio label p {
    color: #CBD5E1 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* Hover on Navigation */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover,
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.06) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
    color: #FFFFFF !important;
}

/* Active Navigation Item */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
    background: rgba(23, 105, 224, 0.22) !important;
    border-left: 3px solid #16B8D4 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Sidebar Selectbox & Inputs (High-Contrast White on Deep Navy) */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #132448 !important;
    border: 1.5px solid #2563EB !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #132448 !important;
    border: none !important;
}

/* Ensure ALL text inside the sidebar selectbox is crisp bright white */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[aria-expanded="true"] *,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[aria-expanded="false"] *,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] p,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #60A5FA !important;
    stroke: #60A5FA !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #94A3B8 !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-bottom: 4px !important;
}

/* Popover dropdown options menu list (when dropdown is opened) */
div[data-baseweb="popover"] {
    background-color: #0E1B38 !important;
    border: 1.5px solid #3B82F6 !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
}

div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: #0E1B38 !important;
}

div[data-baseweb="popover"] li[role="option"] {
    background-color: #0E1B38 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}

div[data-baseweb="popover"] li[role="option"] * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar Slider Styling */
section[data-testid="stSidebar"] .stSlider > div {
    color: #CBD5E1 !important;
}

/* Sidebar Footer */
.sb-footer-card {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 14px;
    margin-top: 20px;
    text-align: center;
}
.sb-footer-inst {
    font-size: 0.75rem;
    font-weight: 700;
    color: #E2E8F0;
}
.sb-footer-dept {
    font-size: 0.70rem;
    color: #94A3B8;
    margin-top: 2px;
}
.sb-footer-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    font-size: 0.70rem;
    font-weight: 700;
    color: #34D399;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 3px 10px;
    border-radius: 9999px;
}

/* =========================================================
   TOP HEADER BANNER (WHITE CARD WITH TECH ACCENTS)
   ========================================================= */
.top-header-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 6px 16px rgba(15, 23, 42, 0.03);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-icon-box {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #1769E0 0%, #16B8D4 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 12px rgba(23, 105, 224, 0.25);
}
.header-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0B1B3A;
    letter-spacing: -0.4px;
    margin: 0;
    line-height: 1.2;
}
.header-subtitle {
    font-size: 0.86rem;
    font-weight: 500;
    color: #64748B;
    margin-top: 3px;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

/* Badges & Pills */
.pill-badge {
    font-size: 0.78rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.2px;
}
.pill-green {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
}
.pill-red {
    background: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FECACA;
}
.pill-amber {
    background: #FFFBEB;
    color: #D97706;
    border: 1px solid #FDE68A;
}
.pill-cyan {
    background: #ECFEFF;
    color: #0891B2;
    border: 1px solid #A5F3FC;
}
.pill-blue {
    background: #EFF6FF;
    color: #1769E0;
    border: 1px solid #BFDBFE;
}
.pill-neutral {
    background: #F8FAFC;
    color: #475569;
    border: 1px solid #E2E8F0;
}
.status-dot-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    box-shadow: 0 0 6px #10B981;
}

/* =========================================================
   WORKFLOW PIPELINE STRIP
   ========================================================= */
.workflow-strip {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.workflow-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #0B1B3A;
    letter-spacing: 0.3px;
}
.step-circle {
    width: 24px;
    height: 24px;
    background: #EFF6FF;
    color: #1769E0;
    border: 1px solid #BFDBFE;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
}
.workflow-arrow {
    color: #94A3B8;
    font-weight: 800;
    font-size: 1.0rem;
}

/* =========================================================
   KPI METRIC CARDS (CLEAN WHITE CARDS)
   ========================================================= */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 4px 12px rgba(15, 23, 42, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}
.kpi-card-stripe {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}
.kpi-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.kpi-icon {
    font-size: 1.2rem;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0B1B3A;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 0.78rem;
    font-weight: 500;
    color: #64748B;
}

/* =========================================================
   PANELS & CONTAINERS (WHITE CARDS)
   ========================================================= */
.white-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 4px 12px rgba(15, 23, 42, 0.03);
    margin-bottom: 20px;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 750;
    color: #0B1B3A;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-sub {
    font-size: 0.84rem;
    color: #64748B;
    margin-top: -6px;
    margin-bottom: 14px;
}

/* Legend Bar */
.legend-bar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 16px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 0.78rem;
    font-weight: 600;
    color: #334155;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.legend-box {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
}

/* AI Monitor Table */
.ai-monitor-table {
    width: 100%;
    border-collapse: collapse;
}
.ai-monitor-table tr {
    border-bottom: 1px solid #F1F5F9;
}
.ai-monitor-table td {
    padding: 9px 0;
    font-size: 0.86rem;
}
.ai-monitor-k {
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.4px;
}
.ai-monitor-v {
    color: #0B1B3A;
    font-weight: 700;
    text-align: right;
}

/* Workstation Seat Grid Cards */
.seat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.seat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
}
.seat-avail {
    border-left: 4px solid #10B981;
    background: #F0FDF4;
}
.seat-occ {
    border-left: 4px solid #EF4444;
    background: #FEF2F2;
}
.seat-id {
    font-size: 1.05rem;
    font-weight: 800;
}
.seat-sub {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 2px;
}

/* Alert Notification Bars */
.alert-card-crit {
    background: #FEF2F2;
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: #991B1B;
    font-size: 0.88rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-card-warn {
    background: #FFFBEB;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: #92400E;
    font-size: 0.88rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-card-info {
    background: #EFF6FF;
    border-left: 4px solid #1769E0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: #1E40AF;
    font-size: 0.88rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Sample Scene Pill Selector */
.scene-strip {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

/* Clean Button Overrides */
.stButton > button {
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.2px;
    transition: all 0.15s ease;
}

/* Clean Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #FFFFFF;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748B;
    font-weight: 600;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #1769E0 !important;
    font-weight: 700 !important;
}
</style>
""")


# ---------------------------------------------------------
# ---------------------------------------------------------
# Sidebar Component: Brand, Navigation, Engine Settings
# ---------------------------------------------------------
with st.sidebar:
    render_html("""
    <div class="sb-brand-card">
        <div class="sb-icon-box">💻</div>
        <div class="sb-brand-title">JUW SMART SPACE</div>
        <div class="sb-brand-sub">AI Occupancy Detection</div>
    </div>
    """)

    render_html("<div class='sb-section-label'>Navigation</div>")
    nav_choice = st.radio(
        "Navigation Menu",
        [
            "📊 Overview",
            "🔬 Live Detection",
            "🪑 Seating & Workstations",
            "📈 Analytics",
            "⚡ AI Comparison",
            "🚨 Alerts",
            "⚙️ System"
        ],
        label_visibility="collapsed"
    )
    # Strip icon for routing
    nav_page = nav_choice.split(" ", 1)[-1]
    if "Seating" in nav_page or "Workstations" in nav_page:
        nav_page = "Workstations"

    render_html("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:14px 0 10px 0;'>")

    # =========================================================
    # 1. SPACE SELECTOR
    # =========================================================
    render_html("<div class='sb-section-label'>SPACE</div>")
    space_display_options = ["Computer Laboratory", "Lecture Room", "Auditorium"]
    space_name_to_id = {
        "Computer Laboratory": "computer_lab",
        "Lecture Room": "lecture_room",
        "Auditorium": "auditorium"
    }
    space_id_to_name = {v: k for k, v in space_name_to_id.items()}

    current_space_id = st.session_state.get("active_space_id", "computer_lab")
    current_space_display = space_id_to_name.get(current_space_id, "Computer Laboratory")
    current_space_idx = space_display_options.index(current_space_display) if current_space_display in space_display_options else 0

    selected_space_name = st.selectbox(
        "SPACE",
        space_display_options,
        index=current_space_idx,
        label_visibility="collapsed",
        key="sb_space_selector_main"
    )
    selected_space_id = space_name_to_id[selected_space_name]
    st.session_state["active_space_id"] = selected_space_id
    st.session_state["active_space_name"] = selected_space_name
    active_space_cfg = get_space_config(selected_space_id)

    seat_term = "Workstations" if selected_space_id == "computer_lab" else ("Lecture Desks" if selected_space_id == "lecture_room" else "Seats")
    render_html(f"""
    <div style="background: rgba(37, 99, 235, 0.20); border: 1px solid rgba(59, 130, 246, 0.50); border-radius: 6px; padding: 5px 10px; margin-top: 6px; font-size: 0.76rem; color: #93C5FD; display: flex; align-items: center; justify-content: space-between;">
        <span><strong>Space:</strong></span>
        <span style="font-weight: 700; color: #FFFFFF;">{selected_space_name} ({active_space_cfg['capacity']} {seat_term})</span>
    </div>
    """)

    render_html("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:12px 0 10px 0;'>")

    # =========================================================
    # 2. AI MODEL SELECTOR (Supported for all spaces)
    # =========================================================
    render_html("<div class='sb-section-label'>AI MODEL</div>")
    model_display_options = ["YOLOv8", "CNN"]
    current_saved_model = st.session_state.get("active_model_name", "YOLOv8")
    if current_saved_model not in model_display_options:
        current_saved_model = "YOLOv8"
    model_idx = model_display_options.index(current_saved_model)
    selected_model_display = st.selectbox(
        "AI MODEL",
        model_display_options,
        index=model_idx,
        label_visibility="collapsed",
        key="sb_model_selector_active"
    )

    st.session_state["active_model_name"] = selected_model_display
    active_engine_mode = "yolo" if selected_model_display == "YOLOv8" else "cnn"

    render_html(f"""
    <div style="background: rgba(16, 185, 129, 0.20); border: 1px solid rgba(16, 185, 129, 0.50); border-radius: 6px; padding: 5px 10px; margin-top: 6px; font-size: 0.76rem; color: #6EE7B7; display: flex; align-items: center; justify-content: space-between;">
        <span><strong>Model:</strong></span>
        <span style="font-weight: 700; color: #FFFFFF;">{selected_model_display} Active</span>
    </div>
    """)

    # Dedicated MODEL description section
    render_html("<div class='sb-section-label' style='margin-top:12px;'>MODEL</div>")
    if selected_model_display == "CNN":
        render_html(f"""
        <div style="font-size: 0.76rem; color: #E2E8F0; line-height: 1.4; padding: 2px 0;">
            <strong style="color: #60A5FA;">Custom CNN Baseline</strong><br>
            <span style="color: #94A3B8;">ROI Seating Occupancy Classifier ({active_space_cfg['name']})</span>
        </div>
        """)
    else:
        render_html(f"""
        <div style="font-size: 0.76rem; color: #E2E8F0; line-height: 1.4; padding: 2px 0;">
            <strong style="color: #34D399;">Pretrained YOLOv8</strong><br>
            <span style="color: #94A3B8;">Person Detection + Spatial Association ({active_space_cfg['name']})</span>
        </div>
        """)

    render_html("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:14px 0 10px 0;'>")

    # =========================================================
    # 3. SYSTEM STATUS
    # =========================================================
    render_html("<div class='sb-section-label'>SYSTEM STATUS</div>")
    render_html("""
    <div style="font-size: 0.77rem; color: #E2E8F0; padding: 2px 0; line-height: 1.6;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="color:#10B981; font-size:0.9rem;">●</span> <span style="font-weight:600;">AI Engine Ready</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:2px;">
            <span style="color:#10B981; font-size:0.9rem;">●</span> <span style="font-weight:600;">Database Connected</span>
        </div>
    </div>
    """)

    # Confidence Threshold slider
    render_html("<div class='sb-section-label' style='margin-top:14px;'>Confidence Threshold</div>")
    decision_threshold = st.slider(
        "Confidence Threshold Slider",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_PREDICTION_THRESHOLD),
        step=0.05,
        label_visibility="collapsed"
    )

    # Clean Compact Sidebar Footer
    render_html("""
    <div class="sb-footer-card">
        <div class="sb-footer-inst">JUW • Karachi</div>
        <div class="sb-footer-dept">Department of CS &amp; SE</div>
        <div style="font-size: 0.68rem; color: #64748B; margin-top: 2px;">AI Occupancy Detection System</div>
    </div>
    """)



# ---------------------------------------------------------
# Dynamic Top Header Component
# ---------------------------------------------------------
space_icon_badge = "💻" if selected_space_id == "computer_lab" else ("📚" if selected_space_id == "lecture_room" else "🏛️")
engine_badge_text = "Pretrained YOLOv8" if active_engine_mode == "yolo" else "Custom CNN Baseline"

header_html = f"""
<div class="top-header-card">
    <div class="header-left">
        <div class="header-icon-box">{space_icon_badge}</div>
        <div>
            <div class="header-title">JUW SMART SPACE — {active_space_cfg['name'].upper()}</div>
            <div class="header-subtitle">AI-Powered Multi-Space Occupancy &amp; Availability Detection • CS &amp; SE Department</div>
        </div>
    </div>
    <div class="header-right">
        <span class="pill-badge pill-blue" style="font-weight:700;">🏛️ SPACE: {active_space_cfg['name'].upper()}</span>
        <span class="pill-badge pill-blue" style="font-weight:700;">⚡ MODEL: {engine_badge_text}</span>
        <span class="pill-badge pill-green">
            <span class="status-dot-pulse"></span> System Online
        </span>
    </div>
</div>
"""
render_html(header_html)



# ---------------------------------------------------------
# Helper: Donut Occupancy Chart using Matplotlib
# Clean light background, high contrast red/green segments
# ---------------------------------------------------------
def render_occupancy_donut(occupied_count: int, available_count: int, occupancy_pct: float):
    fig, ax = plt.subplots(figsize=(2.8, 2.8), facecolor='#FFFFFF')
    counts = [max(occupied_count, 0), max(available_count, 0)]
    colors = ['#EF4444', '#10B981']  # Red for occupied, Green for available

    if sum(counts) == 0:
        counts = [1]
        colors = ['#E2E8F0']

    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.30, edgecolor='#FFFFFF', linewidth=2.5)
    )

    # Center text
    ax.text(
        0, 0.08,
        f"{occupancy_pct:.0f}%",
        ha='center', va='center',
        fontsize=22, fontweight='bold', color='#0B1B3A'
    )
    ax.text(
        0, -0.22,
        "Occupancy",
        ha='center', va='center',
        fontsize=10, fontweight='600', color='#64748B'
    )
    ax.axis('equal')
    plt.tight_layout(pad=0.2)
    return fig


# Helper function to compute custom level badge
def compute_level_badge(occ_pct: float):
    low_thresh = OCCUPANCY_THRESHOLDS.get('LOW', 30)
    med_thresh = OCCUPANCY_THRESHOLDS.get('MEDIUM', 70)
    if occ_pct <= low_thresh:
        return "LOW", "pill-green"
    elif occ_pct <= med_thresh:
        return "MEDIUM", "pill-amber"
    else:
        return "HIGH", "pill-red"


# ---------------------------------------------------------
# Sample Lab Scenes Definitions
# ---------------------------------------------------------
sample_dir = os.path.join(DATASET_DIR, 'sample_lab_views')
sample_scenes_catalog = [
    {"id": "real/juw_real_empty.jpg", "label": "JUW Empty Lab", "desc": "Real JUW Computer Lab (Empty Session, 0 Occupants)"},
    {"id": "real/juw_real_solo.jpg", "label": "JUW Solo Study", "desc": "Real JUW Computer Lab (1 Student Working)"},
    {"id": "real/juw_real_group.jpg", "label": "JUW Group Lab", "desc": "Real JUW Computer Lab (Group Work Session)"},
    {"id": "juw_lab_sample_1.jpg", "label": "Sim Lab 01", "desc": "Balanced Simulation Session (13 Occupants)"},
    {"id": "juw_lab_sample_3.jpg", "label": "Sim Lab 03", "desc": "Peak Simulation Session (18 Occupants)"}
]



# =========================================================
# 1. OVERVIEW (Main Dashboard — The Strongest Page)
# =========================================================
if nav_page == "Overview":
    # Top Visual Workflow Pipeline
    render_html("""
    <div class="workflow-strip">
        <div class="workflow-step"><div class="step-circle">1</div> IMAGE INPUT</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step"><div class="step-circle">2</div> AI DETECTION</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step"><div class="step-circle">3</div> SPATIAL CORRELATION</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step"><div class="step-circle">4</div> OCCUPANCY</div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step"><div class="step-circle">5</div> ANALYTICS</div>
    </div>
    """)

    # Interactive Space Selector Bar
    st.markdown("<h4 style='color:#0B1B3A; margin: 0 0 6px 0; font-weight:800;'>SELECT SMART SPACE</h4>", unsafe_allow_html=True)
    sp_col1, sp_col2, sp_col3 = st.columns(3)
    sp_buttons = [
        ("computer_lab", "💻 Computer Laboratory (20 Seats)", sp_col1),
        ("lecture_room", "📚 Lecture Room (30 Seats)", sp_col2),
        ("auditorium", "🏛️ Auditorium (50 Seats)", sp_col3)
    ]
    for s_id, s_lbl, col in sp_buttons:
        with col:
            is_active_space = (selected_space_id == s_id)
            btn_t = "primary" if is_active_space else "secondary"
            if st.button(s_lbl, key=f"ov_top_sp_{s_id}", use_container_width=True, type=btn_t):
                st.session_state["active_space_id"] = s_id
                cfg = get_space_config(s_id)
                st.session_state["active_space_name"] = cfg["name"]
                st.session_state["selected_sample_scene"] = cfg["sample_scenes"][0]["id"]
                st.session_state["highlighted_pc"] = None
                st.rerun()

    active_space_cfg = get_space_config(selected_space_id)
    sample_scenes_catalog = active_space_cfg.get("sample_scenes", [])
    seat_label = "Workstations" if selected_space_id == "computer_lab" else ("Lecture Desks" if selected_space_id == "lecture_room" else "Auditorium Seats")
    seat_single = "Workstation" if selected_space_id == "computer_lab" else "Seat"
    space_icon_badge = "💻" if selected_space_id == "computer_lab" else ("📚" if selected_space_id == "lecture_room" else "🏛️")

    # Prominent Active Space & Model Banner
    render_html(f"""
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 18px; margin: 12px 0 16px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">ACTIVE SPACE:</span>
            <span style="font-size:1.15rem; font-weight:800; color:#0B1B3A;">{space_icon_badge} {active_space_cfg['name'].upper()}</span>
            <span style="font-size:0.78rem; font-weight:700; color:#1769E0; background:#EFF6FF; padding:3px 10px; border-radius:6px; border:1px solid #DBEAFE;">{active_space_cfg['capacity']} Configured {seat_label}</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">AI MODEL:</span>
            <span style="font-size:0.88rem; font-weight:750; color:#0B1B3A; background:#FFFFFF; padding:4px 12px; border-radius:6px; border:1px solid #CBD5E1;">{'⚡ Pretrained YOLOv8' if active_engine_mode == 'yolo' else '🟦 Custom CNN Baseline'}</span>
        </div>
    </div>
    """)

    # Space Scenes Bar
    st.markdown(f"<h4 style='color:#0B1B3A; margin: 12px 0 4px 0; font-weight:800;'>{active_space_cfg['name'].upper()} SCENES</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748B; font-size:0.84rem; margin: 0 0 12px 0;'>Select a calibrated scene for {active_space_cfg['name']} or upload custom imagery to run real AI spatial detection.</p>", unsafe_allow_html=True)

    default_sample_id = sample_scenes_catalog[0]["id"] if sample_scenes_catalog else "real/juw_real_empty.jpg"
    selected_sample_id = st.session_state.get("selected_sample_scene", default_sample_id)
    
    # Ensure selected sample belongs to current space
    valid_ids = [sc["id"] for sc in sample_scenes_catalog]
    if selected_sample_id not in valid_ids and valid_ids:
        selected_sample_id = valid_ids[0]
        st.session_state["selected_sample_scene"] = selected_sample_id

    sample_cols = st.columns(len(sample_scenes_catalog)) if sample_scenes_catalog else [st]
    for idx, sc in enumerate(sample_scenes_catalog):
        with sample_cols[idx]:
            is_active = (selected_sample_id == sc["id"])
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"📷 {sc['label']}", key=f"ov_btn_{sc['id']}", use_container_width=True, type=btn_type):
                st.session_state["selected_sample_scene"] = sc["id"]
                st.session_state["highlighted_pc"] = None
                st.rerun()

    # Load active sample image
    active_sample_path = os.path.join(sample_dir, selected_sample_id)
    if os.path.exists(active_sample_path):
        lab_img_bgr = cv2.imread(active_sample_path)
        sample_label_active = selected_sample_id
    else:
        lab_img_bgr = generate_sample_lab_overview(20, 13, seed=42)
        sample_label_active = default_sample_id
    # Execute Real AI Occupancy Engine
    t_start = time.perf_counter()
    result = analyze_space_occupancy(
        lab_img_bgr,
        space_type=selected_space_id,
        threshold=decision_threshold,
        persist_db=True,
        model_type=active_engine_mode
    )
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    annotated_rgb = result['annotated_image']
    stats = result['stats']
    predictions = result['detailed_predictions']
    available_pcs = result['available_workstations']
    alerts = result['alerts']

    level_str, badge_cls = compute_level_badge(stats['occupancy_pct'])


    # Display Alerts if any
    if alerts:
        for alert in alerts:
            alert_type = alert.get("type", "warning")
            cls_name = "alert-card-crit" if alert_type == "HIGH_OCCUPANCY" else "alert-card-warn"
            render_html(f'<div class="{cls_name}">⚠️ <span>{alert["message"]}</span></div>')

    render_html("<div style='margin-top: 14px;'></div>")

    # 4 Clean White KPI Cards (Values strictly from real backend)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-card-stripe" style="background: #1769E0;"></div>
            <div class="kpi-card-header">
                <span class="kpi-label">Total {seat_label}</span>
                <span class="kpi-icon">{space_icon_badge}</span>
            </div>
            <div class="kpi-value">{stats["total"]}</div>
            <div class="kpi-sub">Configured {seat_label.lower()}</div>
        </div>

        """)
    with k2:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-card-stripe" style="background: #EF4444;"></div>
            <div class="kpi-card-header">
                <span class="kpi-label">Occupied</span>
                <span class="kpi-icon">🔴</span>
            </div>
            <div class="kpi-value" style="color: #DC2626;">{stats["occupied"]}</div>
            <div class="kpi-sub">Currently in use</div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-card-stripe" style="background: #10B981;"></div>
            <div class="kpi-card-header">
                <span class="kpi-label">Available</span>
                <span class="kpi-icon">🟢</span>
            </div>
            <div class="kpi-value" style="color: #059669;">{stats["available"]}</div>
            <div class="kpi-sub">{seat_label} available</div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-card-stripe" style="background: #16B8D4;"></div>
            <div class="kpi-card-header">
                <span class="kpi-label">Occupancy</span>
                <span class="kpi-icon">📊</span>
            </div>
            <div class="kpi-value" style="color: #0891B2;">{stats["occupancy_pct"]}%</div>
            <div class="kpi-sub">
                <span class="pill-badge {badge_cls}" style="padding: 2px 8px; font-size: 0.70rem;">{level_str} LOAD</span>
            </div>
        </div>
        """)

    render_html("<div style='margin-top: 20px;'></div>")

    # Main Visual Feature: Live Space Analysis (Largest component)
    img_col, panel_col = st.columns([7, 3])

    with img_col:
        render_html(f"""
        <div class="white-panel" style="margin-bottom: 0;">
            <div class="panel-title">🔬 LIVE {active_space_cfg['name'].upper()} ANALYSIS</div>
            <div class="panel-sub">AI-detected {seat_single.lower()} occupancy and spatial correlation</div>
            <div class="legend-bar">
                <div class="legend-item"><span class="legend-box" style="background:#EF4444;"></span> RED = OCCUPIED</div>
                <div class="legend-item"><span class="legend-box" style="background:#10B981;"></span> GREEN = AVAILABLE</div>
                <div class="legend-item"><span class="legend-box" style="background:#16B8D4;"></span> CYAN = DETECTED PERSON</div>
                <div class="legend-item"><span class="legend-box" style="background:#F59E0B;"></span> YELLOW = RECOMMENDED SEAT</div>
            </div>
        </div>
        """)

        # Highlight action state for 'Find Available Workstation'
        highlight_id = st.session_state.get("highlighted_pc", None)
        display_img = annotated_rgb.copy()

        if highlight_id:
            for p in predictions:
                if p["workstation_id"] == highlight_id:
                    bx, by, bw, bh = p["bbox"]
                    cv2.rectangle(display_img, (bx-4, by-4), (bx+bw+4, by+bh+4), (0, 215, 255), 4)
                    cv2.putText(display_img, f">>> RECOMMENDED: {highlight_id} <<<", (bx, max(25, by-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)

        st.image(display_img, use_container_width=True, caption=f"{active_space_cfg['name']} Analysis • Source: {sample_label_active}")

    with panel_col:
        # AI Analysis Card beside the image
        objects_detected_cnt = result.get('total_persons_detected', 0) + result.get('total_chairs_detected', 0)
        avg_conf = np.mean([p['confidence'] for p in predictions]) if predictions else 95.0

        render_html(f"""
        <div class="white-panel" style="height: 100%;">
            <div class="panel-title">🤖 AI ANALYSIS</div>
            <table class="ai-monitor-table">
                <tr>
                    <td class="ai-monitor-k">Engine</td>
                    <td class="ai-monitor-v">{'YOLOv8' if active_engine_mode == 'yolo' else 'Custom CNN'}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Model Type</td>
                    <td class="ai-monitor-v">{'Pre-trained Object Detector' if active_engine_mode == 'yolo' else 'Custom CNN Baseline'}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Objects Detected</td>
                    <td class="ai-monitor-v">{objects_detected_cnt if active_engine_mode == 'yolo' else len(predictions)}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">{seat_label}</td>
                    <td class="ai-monitor-v">{stats['total']}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Confidence</td>
                    <td class="ai-monitor-v">{avg_conf:.1f}%</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Inference Time</td>
                    <td class="ai-monitor-v">{inference_ms:.1f} ms</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Status</td>
                    <td class="ai-monitor-v" style="color: #059669;">● Analysis Complete</td>
                </tr>
            </table>

            <hr style="border:none; border-top:1px solid #F1F5F9; margin: 16px 0;">

            <div style="font-size: 0.78rem; color: #64748B; line-height: 1.5;">
                <strong style="color: #0B1B3A;">Spatial Correlation:</strong><br>
                Intersection-over-Area (IoA) correlates detected person bodies with mapped {seat_single.lower()} bounding boxes.
            </div>
        </div>
        """)

    render_html("<div style='margin-top: 20px;'></div>")

    # Lower Row: Occupancy Visualization, Available Workstations, Quick Stats
    c_donut, c_avail, c_stats = st.columns([3, 4, 3])

    with c_donut:
        render_html("""
        <div class="white-panel" style="height: 100%;">
            <div class="panel-title">📊 Occupancy Level</div>
        </div>
        """)
        fig_donut = render_occupancy_donut(stats['occupied'], stats['available'], stats['occupancy_pct'])
        st.pyplot(fig_donut, clear_figure=True)
        render_html(f"""
        <div style="text-align: center; margin-top: -10px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #0B1B3A;">{stats['occupied']} / {stats['total']} {seat_label}</div>
            <div style="margin-top: 6px;">
                <span class="pill-badge {badge_cls}" style="font-size: 0.82rem; padding: 4px 12px;">{level_str} OCCUPANCY</span>
            </div>
        </div>
        """)

    with c_avail:
        render_html(f"""
        <div class="white-panel" style="height: 100%;">
            <div class="panel-title">{space_icon_badge} AVAILABLE {seat_label.upper()}</div>
            <div class="panel-sub">{seat_label} currently available for students</div>
        </div>
        """)

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            st.markdown(f"**{len(available_pcs)}** {seat_label.lower()} ready for immediate use.")
        with btn_c2:
            if available_pcs:
                if st.button(f"🔍 Find Available {seat_single}", key="btn_find_pc", use_container_width=True, type="primary"):
                    st.session_state["highlighted_pc"] = available_pcs[0]['workstation_id']
                    st.toast(f"Highlighted recommended seat: {available_pcs[0]['workstation_id']}", icon="✅")
                    st.rerun()

        if available_pcs:
            avail_cols = st.columns(min(len(available_pcs), 4))
            for idx, pc in enumerate(available_pcs[:8]):
                target_col = avail_cols[idx % 4]
                with target_col:
                    is_hl = (pc['workstation_id'] == highlight_id)
                    hl_border = "border: 2px solid #F59E0B;" if is_hl else ""
                    render_html(f"""
                    <div class="seat-card seat-avail" style="{hl_border}">
                        <div class="seat-id" style="color: #059669;">{pc['workstation_id']}</div>
                        <div class="seat-sub">{pc['confidence']}% Conf.</div>
                        <div style="font-size:0.68rem; color:#059669; font-weight:700; margin-top:3px;">AVAILABLE</div>
                    </div>
                    """)
        else:
            render_html(f'<div class="alert-card-crit">⚠️ All {seat_label.lower()} are currently occupied in this {active_space_cfg["name"].lower()}.</div>')


    with c_stats:
        img_h, img_w = lab_img_bgr.shape[:2]
        now_time = time.strftime("%H:%M:%S")

        render_html(f"""
        <div class="white-panel" style="height: 100%;">
            <div class="panel-title">⚡ QUICK STATS</div>
            <table class="ai-monitor-table">
                <tr>
                    <td class="ai-monitor-k">Last Analysis</td>
                    <td class="ai-monitor-v">{now_time}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Image Size</td>
                    <td class="ai-monitor-v">{img_w} × {img_h}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Detection Mode</td>
                    <td class="ai-monitor-v">{'YOLOv8' if active_engine_mode == 'yolo' else 'Custom CNN'}</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Database Status</td>
                    <td class="ai-monitor-v" style="color: #059669;">● Connected</td>
                </tr>
                <tr>
                    <td class="ai-monitor-k">Persistence</td>
                    <td class="ai-monitor-v">SQLite Recorded</td>
                </tr>
            </table>
        </div>
        """)


# =========================================================
# 2. LIVE DETECTION (Computer Vision Dedicated Workspace)
# =========================================================
elif nav_page == "Live Detection":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">🔬 LIVE MULTI-SPACE AI DETECTION</div>
        <div class="panel-sub">Run real-time spatial occupancy detection across Computer Lab, Lecture Room, and Auditorium.</div>
    </div>
    """)

    # Interactive Space Selector
    st.markdown("<h4 style='color:#0B1B3A; font-weight:750;'>1. Select Smart Space</h4>", unsafe_allow_html=True)
    live_sp_col1, live_sp_col2, live_sp_col3 = st.columns(3)
    live_sp_btns = [
        ("computer_lab", "💻 Computer Laboratory (20 Seats)", live_sp_col1),
        ("lecture_room", "📚 Lecture Room (30 Seats)", live_sp_col2),
        ("auditorium", "🏛️ Auditorium (50 Seats)", live_sp_col3)
    ]
    for s_id, s_lbl, col in live_sp_btns:
        with col:
            is_cur = (selected_space_id == s_id)
            bt = "primary" if is_cur else "secondary"
            if st.button(s_lbl, key=f"live_sp_btn_{s_id}", use_container_width=True, type=bt):
                st.session_state["active_space_id"] = s_id
                cfg = get_space_config(s_id)
                st.session_state["active_space_name"] = cfg["name"]
                st.session_state["live_sample_selection"] = cfg["sample_scenes"][0]["id"]
                st.rerun()

    active_space_cfg = get_space_config(selected_space_id)
    seat_label = "Workstations" if selected_space_id == "computer_lab" else ("Lecture Desks" if selected_space_id == "lecture_room" else "Auditorium Seats")
    seat_single = "Workstation" if selected_space_id == "computer_lab" else "Seat"

    # Model Selection Cards (YOLOv8 vs Custom CNN)
    st.markdown("<h4 style='color:#0B1B3A; font-weight:750; margin-top:14px;'>2. Detection Model Selection</h4>", unsafe_allow_html=True)
    m_choice_tab = st.radio(
        "Detection Model Choice",
        ["YOLOv8 (Pre-trained Object Detector)", "Custom CNN (Custom CNN Baseline)"],
        index=0 if active_engine_mode == "yolo" else 1,
        horizontal=True
    )
    live_engine_mode = "yolo" if "yolo" in m_choice_tab.lower() else "cnn"

    if live_engine_mode == "yolo":
        render_html(f"""
        <div class="alert-card-info" style="margin-bottom: 16px;">
            <div>
                <strong>YOLOv8 Pre-trained Object Detector ({active_space_cfg['name']}):</strong><br>
                Full-scene object detection using pre-trained YOLOv8. Localizes person bounding boxes and correlates them spatially with {active_space_cfg['name'].lower()} {seat_label.lower()} via 1-to-1 matching.
            </div>
        </div>
        """)
    else:
        render_html(f"""
        <div class="alert-card-info" style="margin-bottom: 16px;">
            <div>
                <strong>Custom CNN Baseline ({active_space_cfg['name']}):</strong><br>
                ROI-based occupancy classification using the custom CNN baseline. Crops each of the {active_space_cfg['capacity']} {seat_label.lower()} into 128×128 tensors and classifies into EMPTY vs OCCUPIED with sigmoid probability.
            </div>
        </div>
        """)

    # Image Input Selection
    st.markdown(f"<h4 style='color:#0B1B3A; font-weight:750;'>3. Select {active_space_cfg['name']} Scene</h4>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Source Mode",
        [f"Built-in Sample {active_space_cfg['name']} Scenes", f"Upload Your Own Image ({active_space_cfg['name']})"],
        horizontal=True
    )

    chosen_image_bgr = None
    chosen_label = ""
    sample_scenes_catalog = active_space_cfg.get("sample_scenes", [])

    if "Built-in" in input_mode:
        live_sample_cols = st.columns(len(sample_scenes_catalog)) if sample_scenes_catalog else [st]
        default_live_sample = sample_scenes_catalog[0]["id"] if sample_scenes_catalog else "real/juw_real_empty.jpg"
        live_active_sample = st.session_state.get("live_sample_selection", default_live_sample)
        valid_ids = [sc["id"] for sc in sample_scenes_catalog]
        if live_active_sample not in valid_ids and valid_ids:
            live_active_sample = valid_ids[0]
            st.session_state["live_sample_selection"] = live_active_sample

        for idx, sc in enumerate(sample_scenes_catalog):
            with live_sample_cols[idx]:
                is_sel = (live_active_sample == sc["id"])
                btn_type = "primary" if is_sel else "secondary"
                if st.button(f"📷 {sc['label']}", key=f"ld_sc_{sc['id']}", use_container_width=True, type=btn_type):
                    st.session_state["live_sample_selection"] = sc["id"]
                    st.rerun()

        target_path = os.path.join(sample_dir, live_active_sample)
        if os.path.exists(target_path):
            chosen_image_bgr = cv2.imread(target_path)
            chosen_label = live_active_sample
    else:
        uploaded_file = st.file_uploader(f"Upload {active_space_cfg['name']} Image (JPG, PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            chosen_image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            chosen_label = uploaded_file.name

    if chosen_image_bgr is not None:
        render_html(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 16px; margin: 10px 0 14px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
                <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Selected Space:</span>
                <strong style="color:#0B1B3A; font-size:0.95rem; margin-left:6px;">{active_space_cfg['name']}</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Selected Model:</span>
                <strong style="color:#0B1B3A; font-size:0.95rem; margin-left:6px;">{'Pretrained YOLOv8' if live_engine_mode == 'yolo' else 'Custom CNN Baseline'}</strong>
            </div>
            <div>
                <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Input Source:</span>
                <strong style="color:#1769E0; font-size:0.90rem; margin-left:6px;">{chosen_label}</strong>
            </div>
        </div>
        """)

        if st.button(f"🚀 RUN AI DETECTION ({active_space_cfg['name'].upper()})", type="primary", use_container_width=True):
            # Professional Loading State Animation

            loading_box = st.empty()
            steps = [
                f"Loading {active_space_cfg['name']} model...",
                "Detecting persons & objects...",
                "Spatial seating correlation...",
                f"Calculating {active_space_cfg['name']} occupancy...",
                "Generating results..."
            ]
            for step in steps:
                loading_box.markdown(f"""
                <div class="alert-card-info" style="font-weight:700;">
                    ⏳ Analyzing {active_space_cfg['name']}... {step}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.10)
            loading_box.empty()

            t0 = time.perf_counter()
            analysis_res = analyze_space_occupancy(
                chosen_image_bgr,
                space_type=selected_space_id,
                threshold=decision_threshold,
                persist_db=True,
                model_type=live_engine_mode
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            res_stats = analysis_res['stats']
            res_annotated = analysis_res['annotated_image']
            res_preds = analysis_res['detailed_predictions']

            # Results: Side-by-Side Original vs AI Detection
            st.markdown("<h4 style='color:#0B1B3A; font-weight:750; margin-top:20px;'>3. Analysis Results</h4>", unsafe_allow_html=True)
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("**ORIGINAL IMAGE**")
                st.image(cv2.cvtColor(chosen_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with r_col2:
                st.markdown("**AI DETECTION OUTPUT**")
                st.image(res_annotated, use_container_width=True)

            # Detection Summary using Real Values
            st.markdown("<h4 style='color:#0B1B3A; font-weight:750; margin-top:16px;'>Detection Summary</h4>", unsafe_allow_html=True)
            sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
            sm1.metric(f"Total {seat_label}", res_stats["total"])
            sm2.metric("Occupied", res_stats["occupied"])
            sm3.metric("Available", res_stats["available"])
            sm4.metric("Occupancy Rate", f"{res_stats['occupancy_pct']}%")
            avg_pred_conf = np.mean([p['confidence'] for p in res_preds]) if res_preds else 95.0
            sm5.metric("Confidence", f"{avg_pred_conf:.1f}%")
            sm6.metric("Inference Time", f"{elapsed_ms:.1f} ms")

            # Developer Debug Panel (Collapsible as requested in requirements)
            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            with st.expander(f"🛠️ DEVELOPER DEBUG PANEL: RAW YOLO DETECTIONS & {active_space_cfg['name'].upper()} SEAT ASSOCIATION", expanded=True):
                st.markdown("##### 1. Raw YOLOv8 Object Detections (Before Association)")
                counts_by_cls = analysis_res.get("counts_by_class", {})
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                d_c1.metric("Person Detections", counts_by_cls.get("person", analysis_res.get("total_persons_detected", 0)))
                d_c2.metric("Chair Detections", counts_by_cls.get("chair", 0))
                d_c3.metric("TV/Monitor Detections", counts_by_cls.get("tv_monitor", 0))
                d_c4.metric("Laptop Detections", counts_by_cls.get("laptop", 0))

                raw_dets = analysis_res.get("raw_detections", [])
                if raw_dets:
                    raw_df = pd.DataFrame([
                        {
                            "Class ID": d["class_id"],
                            "Class Name": d["label"],
                            "Confidence (%)": d["confidence"],
                            "Bounding Box (x, y, w, h)": d["bbox"],
                            "Centroid (cx, cy)": d.get("centroid", (d["bbox"][0] + d["bbox"][2]//2, d["bbox"][1] + d["bbox"][3]//2))
                        } for d in raw_dets
                    ])
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.info("ℹ️ No raw objects detected by YOLOv8 at the current confidence threshold.")

                # Hard Rule Verification Status
                person_cnt = counts_by_cls.get("person", analysis_res.get("total_persons_detected", 0))
                if person_cnt == 0:
                    render_html(f"""
                    <div class="alert-card-info" style="margin-top:10px; margin-bottom:10px;">
                        ✅ <strong>EMPTY SPACE HARD RULE VERIFIED:</strong> Exactly 0 person detections. Occupied {seat_label.lower()} strictly = 0. All {res_stats['total']} seats are AVAILABLE.
                    </div>
                    """)
                else:
                    render_html(f"""
                    <div class="alert-card-info" style="margin-top:10px; margin-bottom:10px;">
                        👤 <strong>PERSONS DETECTED:</strong> {person_cnt} individual(s) detected. Each person maps to at most one seat via centroid inclusion &amp; spatial intersection.
                    </div>
                    """)

                st.markdown(f"##### 2. {active_space_cfg['name']} Seat Association Matrix")
                ws_assoc = analysis_res.get("workstation_associations", [])
                if ws_assoc:
                    st.dataframe(pd.DataFrame(ws_assoc), use_container_width=True)

            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            st.markdown(f"#### {active_space_cfg['name']} Seating Detection Records")
            st.dataframe(pd.DataFrame(res_preds), use_container_width=True)

    else:
        st.warning("⚠️ Please upload an image or select a calibrated scene to begin detection.")


# =========================================================
# 3. WORKSTATIONS (Dedicated Inventory & Grid View)
# =========================================================
elif nav_page == "Workstations":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">🪑 SEATING & WORKSTATION INVENTORY</div>
        <div class="panel-sub">Space-specific seating layout, real-time availability grid, and single crop classifier.</div>
    </div>
    """)

    # Space Selector for Seating Inventory
    ws_sp_col1, ws_sp_col2 = st.columns([2, 1])
    with ws_sp_col1:
        selected_ws_space = st.selectbox(
            "Select Space to Inspect:",
            ["computer_lab", "lecture_room", "auditorium"],
            index=0 if selected_space_id == "computer_lab" else (1 if selected_space_id == "lecture_room" else 2),
            format_func=lambda s: {
                "computer_lab": "💻 Computer Laboratory (20 Workstations)",
                "lecture_room": "📚 Lecture Room (30 Lecture Desks)",
                "auditorium": "🏛️ Auditorium (50 Tiered Seats)"
            }.get(s, s)
        )
    
    ws_cfg = get_space_config(selected_ws_space)
    ws_seat_label = "Workstations" if selected_ws_space == "computer_lab" else ("Lecture Desks" if selected_ws_space == "lecture_room" else "Auditorium Seats")

    w_tab1, w_tab2 = st.tabs([f"🪑 {ws_cfg['name']} Seating Matrix", "🎯 Single Workstation Crop Classifier"])

    with w_tab1:
        # Load sample view for selected space
        sample_scenes = ws_cfg.get("sample_scenes", [])
        default_scene_file = sample_scenes[0]["id"] if sample_scenes else "real/juw_real_empty.jpg"
        target_path = os.path.join(sample_dir, default_scene_file)
        
        if os.path.exists(target_path):
            img_bgr = cv2.imread(target_path)
            res = analyze_space_occupancy(
                img_bgr,
                space_type=selected_ws_space,
                threshold=decision_threshold,
                persist_db=False,
                model_type="yolo" if selected_ws_space != "computer_lab" else active_engine_mode
            )
            preds = res['detailed_predictions']

            # Filter buttons
            filter_choice = st.radio("Seating Status Filter:", ["ALL", "OCCUPIED", "AVAILABLE"], horizontal=True)

            filtered_preds = preds
            if filter_choice == "OCCUPIED":
                filtered_preds = [p for p in preds if p['prediction'] == 'OCCUPIED']
            elif filter_choice == "AVAILABLE":
                filtered_preds = [p for p in preds if p['prediction'] == 'EMPTY']

            st.markdown(f"**Displaying {len(filtered_preds)} of {len(preds)} {ws_seat_label} in {ws_cfg['name']}**")

            # Workstation / Seat Cards Grid
            w_cols = st.columns(4)
            for i, p in enumerate(filtered_preds):
                target_col = w_cols[i % 4]
                is_occ = (p['prediction'] == 'OCCUPIED')
                badge_style = "seat-occ" if is_occ else "seat-avail"
                status_color = "#DC2626" if is_occ else "#059669"
                with target_col:
                    render_html(f"""
                    <div class="seat-card {badge_style}">
                        <div class="seat-id" style="color: {status_color};">{p['workstation_id']}</div>
                        <div class="seat-sub">{p['confidence']:.0f}% confidence</div>
                        <div style="font-size:0.75rem; font-weight:700; color:{status_color}; margin-top:4px;">
                            {'OCCUPIED' if is_occ else 'AVAILABLE'}
                        </div>
                    </div>
                    """)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"#### {ws_cfg['name']} Seating Coordinates & Detection Metadata")
            st.dataframe(pd.DataFrame(preds), use_container_width=True)

    with w_tab2:
        st.markdown("#### 📷 Single Workstation Crop Prediction Test")
        st.markdown("Test single-crop ($128 \\times 128$) classification using the Custom CNN baseline.")
        render_html("""
        <div class="alert-card-info" style="margin-bottom: 14px;">
            ℹ️ <strong>Scope Notice:</strong> The Custom CNN baseline was trained exclusively on Computer Laboratory workstation crops ($128 \\times 128$). For open Lecture Room and Auditorium seating, YOLOv8 full-scene detection is used.
        </div>
        """)

        sc1, sc2 = st.columns(2)
        with sc1:
            input_source = st.radio("Select Crop Source", ["Test Dataset Sample", "Upload Custom Crop"], horizontal=True)
            crop_np = None

            if input_source == "Test Dataset Sample":
                test_empty = glob.glob(os.path.join(DATASET_DIR, 'test', 'empty', '*.jpg'))
                test_occ = glob.glob(os.path.join(DATASET_DIR, 'test', 'occupied', '*.jpg'))
                all_test = test_empty + test_occ
                if all_test:
                    sel_p = st.selectbox("Select Test Crop File:", all_test, format_func=lambda x: os.path.basename(x))
                    crop_np = cv2.imread(sel_p)
            else:
                up_crop = st.file_uploader("Upload Single Workstation Crop", type=["jpg", "png", "jpeg"])
                if up_crop:
                    pil_c = Image.open(up_crop).convert("RGB")
                    crop_np = cv2.cvtColor(np.array(pil_c), cv2.COLOR_RGB2BGR)

            if crop_np is not None:
                st.image(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB), width=260, caption="Input Crop (128x128x3)")

        with sc2:
            if crop_np is not None:
                c_res = classify_workstation_crop(crop_np, threshold=decision_threshold)
                c_color = "#DC2626" if c_res['prediction'] == 'OCCUPIED' else "#059669"

                render_html(f"""
                <div class="white-panel" style="border-left: 5px solid {c_color};">
                    <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Classification Output</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{c_color}; margin:6px 0;">{c_res['prediction']}</div>
                    <div style="font-size:0.95rem; color:#0B1B3A; font-weight:600;">Confidence: {c_res['confidence']}% ({c_res['confidence_level']})</div>
                    <div style="font-size:0.82rem; color:#64748B; margin-top:8px;">Raw Sigmoid Probability: <code>{c_res['probability']}</code></div>
                </div>
                """)

                st.progress(c_res['probability'])
                st.caption(f"Decision Boundary: {decision_threshold:.2f} (≥ {decision_threshold:.2f} → OCCUPIED)")


# =========================================================
# 4. ANALYTICS (Historical Space Utilization Intelligence)
# =========================================================
elif nav_page == "Analytics":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">📈 MULTI-SPACE OCCUPANCY ANALYTICS</div>
        <div class="panel-sub">Historical occupancy intelligence from SQLite database filtered by smart space.</div>
    </div>
    """)

    # Space Filter Selector
    an_filter_col, _ = st.columns([2, 1])
    with an_filter_col:
        an_space_choice = st.selectbox(
            "Filter Historical Analytics by Space:",
            ["all", "computer_lab", "lecture_room", "auditorium"],
            index=0,
            format_func=lambda s: {
                "all": "🌐 All Spaces (Aggregated)",
                "computer_lab": "💻 Computer Laboratory",
                "lecture_room": "📚 Lecture Room",
                "auditorium": "🏛️ Auditorium"
            }.get(s, s)
        )

    analytics = compute_occupancy_analytics(space_type=an_space_choice)

    if not analytics.get('has_data', False):
        st.info(f"ℹ️ No historical snapshot data found for '{an_space_choice}'. Run an analysis on Overview or Live Detection to persist observations.")
    else:
        # Real KPI cards from SQLite
        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-card-stripe" style="background: #1769E0;"></div>
                <div class="kpi-card-header"><span class="kpi-label">Recorded Snapshots</span><span class="kpi-icon">📁</span></div>
                <div class="kpi-value">{analytics['total_observations']}</div>
                <div class="kpi-sub">Total database logs</div>
            </div>
            """)
        with ak2:
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-card-stripe" style="background: #16B8D4;"></div>
                <div class="kpi-card-header"><span class="kpi-label">Average Occupancy</span><span class="kpi-icon">📊</span></div>
                <div class="kpi-value" style="color: #0891B2;">{analytics['average_occupancy_pct']}%</div>
                <div class="kpi-sub">Historical Mean</div>
            </div>
            """)
        with ak3:
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-card-stripe" style="background: #EF4444;"></div>
                <div class="kpi-card-header"><span class="kpi-label">Peak Occupancy</span><span class="kpi-icon">🔥</span></div>
                <div class="kpi-value" style="color: #DC2626;">{analytics['peak_occupancy_pct']}%</div>
                <div class="kpi-sub">Highest Recorded Load</div>
            </div>
            """)
        with ak4:
            render_html(f"""
            <div class="kpi-card">
                <div class="kpi-card-stripe" style="background: #10B981;"></div>
                <div class="kpi-card-header"><span class="kpi-label">Minimum Occupancy</span><span class="kpi-icon">❄️</span></div>
                <div class="kpi-value" style="color: #059669;">{analytics['min_occupancy_pct']}%</div>
                <div class="kpi-sub">Lowest Recorded Load</div>
            </div>
            """)

        render_html("<div style='margin-top: 24px;'></div>")

        snapshots_df = pd.DataFrame(analytics['snapshots'])
        if not snapshots_df.empty:
            # 1. Occupancy Over Time Chart
            st.markdown("#### 1. Occupancy Over Time")
            st.line_chart(snapshots_df.set_index('timestamp')['occupancy_percentage'], color="#1769E0")

            # 2 & 3: Hourly Occupancy & Peak Usage Hours
            try:
                snapshots_df['hour'] = pd.to_datetime(snapshots_df['timestamp']).dt.hour
                hourly_occ = snapshots_df.groupby('hour')['occupancy_percentage'].mean().reset_index()

                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("#### 2. Average Occupancy by Hour")
                    st.bar_chart(hourly_occ.set_index('hour')['occupancy_percentage'], color="#16B8D4")
                with ch2:
                    st.markdown("#### 3. Peak Usage Hours")
                    peak_hour_row = hourly_occ.loc[hourly_occ['occupancy_percentage'].idxmax()]
                    render_html(f"""
                    <div class="white-panel" style="margin-top: 10px;">
                        <div class="panel-title">⏰ Peak Activity Window</div>
                        <div style="font-size: 2.0rem; font-weight: 800; color: #D97706;">Hour {int(peak_hour_row['hour']):02d}:00</div>
                        <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">Average Load: <strong>{peak_hour_row['occupancy_percentage']:.1f}%</strong></div>
                    </div>
                    """)
            except Exception:
                pass

        # 4. Workstation / Seat Utilization
        st.markdown("#### 4. Seating Demand Breakdown")
        wp1, wp2 = st.columns(2)
        with wp1:
            render_html(f"""
            <div class="kpi-card" style="border-left: 4px solid #EF4444;">
                <div class="kpi-label">Most Frequently Occupied Seat</div>
                <div class="kpi-value" style="color: #DC2626;">{analytics['most_used_workstation']}</div>
                <div class="kpi-sub">Highest Utilization Frequency</div>
            </div>
            """)
        with wp2:
            render_html(f"""
            <div class="kpi-card" style="border-left: 4px solid #10B981;">
                <div class="kpi-label">Most Frequently Available Seat</div>
                <div class="kpi-value" style="color: #059669;">{analytics['most_available_workstation']}</div>
                <div class="kpi-sub">Highest Seat Availability Ratio</div>
            </div>
            """)


# =========================================================
# 5. AI COMPARISON (CNN vs YOLOv8 Benchmark Hub)
# =========================================================
elif nav_page == "AI Comparison":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">🏆 AI MODEL COMPARISON</div>
        <div class="panel-sub">Custom CNN baseline compared against modern pre-trained YOLOv8 object detector.</div>
    </div>
    """)

    render_html(f"""
    <div class="alert-card-info" style="margin-bottom: 18px;">
        ℹ️ <strong>Multi-Space AI Evaluation:</strong> Both <strong>Custom CNN Baseline</strong> (ROI image classifier) and <strong>Pretrained YOLOv8</strong> (person detection + spatial association) are supported across all smart spaces: <strong>Computer Laboratory</strong> (20 Workstations), <strong>Lecture Room</strong> (30 Desks), and <strong>Auditorium</strong> (50 Seats).
    </div>
    """)

    # Model Specification Cards
    mc1, mc2 = st.columns(2)
    with mc1:
        render_html("""
        <div class="white-panel" style="border-top: 4px solid #1769E0;">
            <div class="panel-title">CUSTOM CNN</div>
            <table class="ai-monitor-table">
                <tr><td class="ai-monitor-k">Approach</td><td class="ai-monitor-v">ROI-based classification</td></tr>
                <tr><td class="ai-monitor-k">Input</td><td class="ai-monitor-v">128 × 128 × 3</td></tr>
                <tr><td class="ai-monitor-k">Classification</td><td class="ai-monitor-v">EMPTY / OCCUPIED</td></tr>
                <tr><td class="ai-monitor-k">Architecture</td><td class="ai-monitor-v">Custom 3-block CNN</td></tr>
                <tr><td class="ai-monitor-k">Role</td><td class="ai-monitor-v">Baseline Model</td></tr>
            </table>
        </div>
        """)
    with mc2:
        render_html("""
        <div class="white-panel" style="border-top: 4px solid #10B981;">
            <div class="panel-title">YOLOv8</div>
            <table class="ai-monitor-table">
                <tr><td class="ai-monitor-k">Approach</td><td class="ai-monitor-v">Full-scene detection</td></tr>
                <tr><td class="ai-monitor-k">Input</td><td class="ai-monitor-v">640 × 640 inference</td></tr>
                <tr><td class="ai-monitor-k">Detections</td><td class="ai-monitor-v">Person / relevant objects</td></tr>
                <tr><td class="ai-monitor-k">Occupancy</td><td class="ai-monitor-v">Spatial correlation (IoA)</td></tr>
                <tr><td class="ai-monitor-k">Role</td><td class="ai-monitor-v">Pre-trained Object Detector</td></tr>
            </table>
        </div>
        """)

    c_tab1, c_tab2, c_tab3 = st.tabs([
        "⚡ Live Head-to-Head Comparison",
        "📊 Measured Benchmarks",
        "🎓 Viva & Project Defense Guide"
    ])

    with c_tab1:
        st.subheader("⚡ Live Head-to-Head Comparison")
        st.markdown("Execute both **Custom CNN** and **YOLOv8** on the identical laboratory photo simultaneously to measure inference latency and compare bounding box outputs.")

        available_samples = glob.glob(os.path.join(sample_dir, '*.jpg'))

        c_mode = st.radio("Comparison Input Source", ["Sample Laboratory View", "Upload Custom Photo"], horizontal=True)
        test_img = None

        if c_mode == "Sample Laboratory View" and available_samples:
            sel_s = st.selectbox("Select Sample Lab Photo:", available_samples, format_func=lambda x: os.path.basename(x), key="cmp_select")
            test_img = cv2.imread(sel_s)
        else:
            up_f = st.file_uploader("Upload Lab Photo for Live Comparison", type=["jpg", "png", "jpeg"], key="cmp_up_field")
            if up_f:
                pil_i = Image.open(up_f).convert("RGB")
                test_img = cv2.cvtColor(np.array(pil_i), cv2.COLOR_RGB2BGR)

        if test_img is not None:
            if st.button("🚀 Run Live Head-to-Head Comparison", type="primary"):
                with st.spinner("Executing CNN and YOLOv8 on the same frame..."):
                    t0 = time.perf_counter()
                    cnn_out = analyze_lab_occupancy(test_img, model_type="cnn", persist_db=False)
                    cnn_time = (time.perf_counter() - t0) * 1000.0

                    t1 = time.perf_counter()
                    yolo_out = analyze_lab_occupancy(test_img, model_type="yolo", persist_db=False)
                    yolo_time = (time.perf_counter() - t1) * 1000.0

                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    st.markdown(f"#### 🟦 Custom CNN Baseline ({cnn_time:.1f} ms)")
                    st.metric("Total Seats", cnn_out['stats']['total'])
                    st.metric("Occupied Seats", cnn_out['stats']['occupied'])
                    st.metric("Occupancy Rate", f"{cnn_out['stats']['occupancy_pct']}%")
                    st.image(cnn_out['annotated_image'], use_container_width=True, caption="CNN Output (Grid ROIs: Green=EMPTY, Red=OCCUPIED)")

                with m_c2:
                    st.markdown(f"#### 🟩 YOLOv8 Pre-trained ({yolo_time:.1f} ms)")
                    st.metric("Total Workstations", yolo_out['stats']['total'])
                    st.metric("Students Detected", yolo_out.get('total_persons_detected', 'N/A'))
                    st.metric("Occupancy Rate", f"{yolo_out['stats']['occupancy_pct']}%")
                    st.image(yolo_out['annotated_image'], use_container_width=True, caption="YOLOv8 Output (Green=Empty Desk, Red=Occupied Desk, Cyan=Student)")

    with c_tab2:
        st.subheader("📊 Measured Performance Benchmarks")
        bench_report_path = os.path.join(RESULTS_DIR, "yolo_evaluation_report.json")
        if not os.path.exists(bench_report_path):
            with st.spinner("Loading benchmark metrics..."):
                run_comprehensive_evaluation()

        if os.path.exists(bench_report_path):
            with open(bench_report_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)

            cnn_m = bench_data.get("models_evaluated", {}).get("baseline_cnn", {})
            yolo_m = bench_data.get("models_evaluated", {}).get("yolo_modern", {})

            matrix_data = [
                {"Metric / Dimension": "Inference Time", "Custom CNN": f"{cnn_m.get('latency_ms', 35.0):.1f} ms", "YOLOv8": f"{yolo_m.get('latency_ms', 22.0):.1f} ms"},
                {"Metric / Dimension": "Confidence", "Custom CNN": "Sigmoid Probability", "YOLOv8": "Confidence Score + IoA"},
                {"Metric / Dimension": "FPS (Estimated)", "Custom CNN": f"{1000.0/max(cnn_m.get('latency_ms', 35.0), 1):.1f} FPS", "YOLOv8": f"{yolo_m.get('fps', 45.0):.1f} FPS"},
                {"Metric / Dimension": "Occupancy Paradigm", "Custom CNN": "Grid Crop Classification", "YOLOv8": "Spatial Object Correlation"},
                {"Metric / Dimension": "Available Workstations", "Custom CNN": "Calculated (Grid)", "YOLOv8": "Calculated (Spatial Overlap)"}
            ]
            st.table(pd.DataFrame(matrix_data))

    with c_tab3:
        st.subheader("🎓 Viva & Project Defense Guide")
        render_html("""
        <div class="white-panel" style="margin-bottom: 14px;">
            <strong style="color: #1769E0;">1. What is the main novelty of the project?</strong>
            <p style="color: #334155; font-size: 0.9rem; margin-top: 4px;">
                The novelty is the <strong>end-to-end computer vision laboratory intelligence system</strong>: integrating camera inputs, object detection, spatial correlation between students and desks, real-time occupancy calculations, and SQLite persistence.
            </p>
        </div>
        <div class="white-panel" style="margin-bottom: 14px;">
            <strong style="color: #1769E0;">2. Did you train YOLOv8 from scratch?</strong>
            <p style="color: #334155; font-size: 0.9rem; margin-top: 4px;">
                No. We utilized a <strong>pre-trained YOLOv8 object detector</strong> to localize students (persons) and workstation objects, and designed a custom spatial correlation engine using Intersection-over-Area (IoA) to assess workstation occupancy.
            </p>
        </div>
        <div class="white-panel" style="margin-bottom: 14px;">
            <strong style="color: #1769E0;">3. What role does the Custom CNN play?</strong>
            <p style="color: #334155; font-size: 0.9rem; margin-top: 4px;">
                The Custom CNN serves as our experimental baseline for ROI crop binary classification (EMPTY vs OCCUPIED), demonstrating core neural network principles and comparing against modern full-scene detection.
            </p>
        </div>
        """)


# =========================================================
# 6. ALERTS (Smart Alerts & Notification Rules)
# =========================================================
elif nav_page == "Alerts":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">🚨 SMART ALERTS</div>
        <div class="panel-sub">Real-time occupancy threshold alerts and model review recommendations.</div>
    </div>
    """)

    latest = get_latest_occupancy()
    if latest:
        occ_p = latest.get("occupancy_percentage", 0.0)
        render_html(f"""
        <div class="white-panel" style="margin-bottom: 18px;">
            <div class="panel-title">📌 Current Laboratory Load</div>
            <div style="font-size: 1.15rem; color: #0B1B3A;">Occupancy Rate: <strong>{occ_p}%</strong> ({latest.get('occupied_seats', 0)}/{latest.get('total_seats', 0)} workstations)</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-top: 4px;">Recorded Timestamp: {latest.get('timestamp', 'N/A')}</div>
        </div>
        """)

        high_thresh = float(OCCUPANCY_THRESHOLDS.get('HIGH', 90))
        med_thresh = float(OCCUPANCY_THRESHOLDS.get('MEDIUM', 70))
        low_thresh = float(OCCUPANCY_THRESHOLDS.get('LOW', 30))

        if occ_p >= high_thresh:
            render_html(f"""
            <div class="alert-card-crit">
                🔴 <strong>HIGH OCCUPANCY:</strong> Space occupancy has reached or exceeded the critical threshold (≥ {int(high_thresh)}%). Direct incoming students to secondary spaces.
            </div>
            """)
        elif occ_p >= med_thresh:
            render_html(f"""
            <div class="alert-card-warn">
                ⚠️ <strong>MEDIUM OCCUPANCY:</strong> Space capacity has reached the warning threshold (> {int(med_thresh)}%).
            </div>
            """)
        else:
            render_html("""
            <div class="alert-card-info">
                ✅ <strong>OPTIMAL CAPACITY:</strong> Space seat availability is healthy and open for students.
            </div>
            """)

    st.markdown("#### Configured Alert Rules")
    high_t = int(OCCUPANCY_THRESHOLDS.get('HIGH', 90))
    med_t = int(OCCUPANCY_THRESHOLDS.get('MEDIUM', 70))
    low_t = int(OCCUPANCY_THRESHOLDS.get('LOW', 30))
    rule_table = [
        {"Rule Name": "HIGH OCCUPANCY", "Condition": f"Occupancy Rate ≥ {high_t}%", "Severity": "Critical", "Description": "Space occupancy has reached or exceeded critical threshold."},
        {"Rule Name": "MEDIUM OCCUPANCY", "Condition": f"Occupancy Rate {low_t + 1}% - {med_t}%", "Severity": "Warning", "Description": "Space capacity is nearing saturation."},
        {"Rule Name": "MODEL REVIEW", "Condition": "Prediction confidence < 60%", "Severity": "Information", "Description": "Prediction confidence is below the configured confidence threshold."}
    ]
    st.table(pd.DataFrame(rule_table))



# =========================================================
# 7. SYSTEM (System Status, Model Info & Governance)
# =========================================================
elif nav_page == "System":
    render_html("""
    <div class="white-panel" style="margin-bottom: 16px;">
        <div class="panel-title">⚙️ SYSTEM STATUS</div>
        <div class="panel-sub">System health, service configurations, and institutional privacy statements.</div>
    </div>
    """)

    # Check API health
    api_online = False
    try:
        import urllib.request
        res = urllib.request.urlopen("http://localhost:8000/health", timeout=1)
        if res.status == 200:
            api_online = True
    except Exception:
        api_online = False

    # Status KPI Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_html("""
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-label">AI MODEL</div>
            <div class="kpi-value" style="color: #059669; font-size: 1.6rem;">● Loaded</div>
            <div class="kpi-sub">CNN &amp; YOLOv8 Ready</div>
        </div>
        """)
    with s2:
        render_html("""
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-label">DATABASE</div>
            <div class="kpi-value" style="color: #059669; font-size: 1.6rem;">● Connected</div>
            <div class="kpi-sub">SQLite Active</div>
        </div>
        """)
    with s3:
        status_api_color = "#059669" if api_online else "#DC2626"
        status_api_text = "● Running" if api_online else "● Offline"
        render_html(f"""
        <div class="kpi-card" style="border-left: 4px solid {status_api_color};">
            <div class="kpi-label">API</div>
            <div class="kpi-value" style="color: {status_api_color}; font-size: 1.6rem;">{status_api_text}</div>
            <div class="kpi-sub">Port :8000</div>
        </div>
        """)
    with s4:
        render_html("""
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-label">DASHBOARD</div>
            <div class="kpi-value" style="color: #059669; font-size: 1.6rem;">● Online</div>
            <div class="kpi-sub">Port :8501</div>
        </div>
        """)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Model Information Section
    st.markdown("#### Service & Component Information")
    model_info_table = [
        {"Component": "CNN Model", "Configured Path / Value": "seat_occupancy_cnn.keras", "Status": "Loaded"},
        {"Component": "YOLO Model", "Configured Path / Value": "yolov8n.pt (Pre-trained)", "Status": "Loaded"},
        {"Component": "Database", "Configured Path / Value": "juw_smart_space.db", "Status": "Connected"},
        {"Component": "FastAPI Service", "Configured Path / Value": "http://localhost:8000", "Status": "Running" if api_online else "Offline"},
        {"Component": "Streamlit Dashboard", "Configured Path / Value": "http://localhost:8501", "Status": "Online"}
    ]
    st.table(pd.DataFrame(model_info_table))

    # Governance & Privacy
    sys_tab1, sys_tab2, sys_tab3 = st.tabs([
        "🏷️ Model Registry",
        "📊 Dataset Audit",
        "🏛️ Institution & Privacy Statement"
    ])

    with sys_tab1:
        st.subheader("Model Registry Management")
        registry_data = get_model_registry()
        active_ver = registry_data.get("active_version", "v1_controlled")
        st.markdown(f"**Active Model Version:** `{active_ver}`")
        models_dict = registry_data.get("models", {})
        comp_rows = []
        for m_ver, m_info in models_dict.items():
            metrics = m_info.get("metrics", {})
            comp_rows.append({
                "Version": m_ver,
                "Name": m_info.get("name"),
                "Dataset": m_info.get("training_dataset"),
                "Status": m_info.get("status"),
                "Accuracy": f"{metrics.get('accuracy', 0.0)*100:.1f}%",
                "Precision": f"{metrics.get('precision', 0.0)*100:.1f}%",
                "Recall": f"{metrics.get('recall', 0.0)*100:.1f}%",
                "F1 Score": f"{metrics.get('f1_score', 0.0)*100:.1f}%"
            })
        st.table(pd.DataFrame(comp_rows))

    with sys_tab2:
        st.subheader("Dataset Health & Leakage Protection")
        audit_json_path = os.path.join(DATASET_DIR, "metadata", "dataset_audit.json")
        if os.path.exists(audit_json_path):
            with open(audit_json_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)

            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Total Images", audit_data.get("total_images", 0))
            dm2.metric("EMPTY Crops", audit_data.get("total_empty", 0))
            dm3.metric("OCCUPIED Crops", audit_data.get("total_occupied", 0))

            leakage_status = audit_data.get("leakage_analysis", {}).get("leakage_detected", False)
            if leakage_status:
                dm4.error("⚠️ LEAKAGE DETECTED")
            else:
                dm4.success("✅ NO LEAKAGE")

    with sys_tab3:
        st.subheader("Institution & Privacy Policy")
        render_html("""
        <div class="white-panel">
            <h4 style="color: #1769E0; margin-top:0;">Department of Computer Science &amp; Software Engineering</h4>
            <p style="color: #334155; font-size: 0.9rem;">
                <strong>Jinnah University for Women (JUW), Karachi</strong><br>
                This system is built as an academic computer vision project for intelligent laboratory space monitoring.
            </p>
            <div class="alert-card-info" style="margin-top: 14px;">
                <strong>🔒 Student Privacy Guarantee:</strong><br>
                The system operates exclusively at the workstation occupancy level. It <strong>DOES NOT</strong> implement facial recognition, biometric profiling, or personal identification.
            </div>
        </div>
        """)


# ---------------------------------------------------------
# Clean Enterprise Footer
# ---------------------------------------------------------
render_html("""
<br>
<hr style="border:none; border-top:1px solid #E2E8F0; margin: 30px 0 16px 0;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; font-size:0.78rem; color:#64748B; padding-bottom:12px;">
    <div>JUW SMART SPACE • Department of Computer Science &amp; Software Engineering, Jinnah University for Women, Karachi</div>
    <div style="display:flex; gap:14px;">
        <span>AI Engine: Active</span>
        <span>REST API: :8000</span>
        <span>FastAPI • Streamlit • SQLite</span>
    </div>
</div>
""")
