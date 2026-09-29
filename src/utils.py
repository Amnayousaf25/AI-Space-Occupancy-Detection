import os
import cv2
import numpy as np

# ==========================================
# SYSTEM CONFIGURATION & CONSTANTS
# ==========================================
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

CLASS_NAMES = ['EMPTY', 'OCCUPIED']
CLASS_MAP = {0: 'EMPTY', 1: 'OCCUPIED'}

# Thresholds for Occupancy Level Interpretation
try:
    from src.config import OCCUPANCY_THRESHOLDS
except ImportError:
    OCCUPANCY_THRESHOLDS = {
        "LOW": 30,
        "MEDIUM": 70,
        "HIGH": 90
    }

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
MODEL_PATH = os.path.join(MODELS_DIR, 'seat_occupancy_cnn.keras')

def get_occupancy_level(occupancy_pct):
    """
    Categorize occupancy percentage into human-readable levels:
    0–30%   = LOW (green)
    31–70%  = MEDIUM (orange)
    71–100% = HIGH (red)
    """
    low_thresh = OCCUPANCY_THRESHOLDS.get('LOW', 30)
    med_thresh = OCCUPANCY_THRESHOLDS.get('MEDIUM', 70)
    if occupancy_pct <= low_thresh:
        return "LOW", "green"
    elif occupancy_pct <= med_thresh:
        return "MEDIUM", "orange"
    else:
        return "HIGH", "red"

def calculate_occupancy_stats(total_seats, occupied_seats):
    """
    Calculate occupancy metrics based on defined formulas.
    """
    if total_seats == 0:
        return {
            'total': 0,
            'occupied': 0,
            'available': 0,
            'occupancy_pct': 0.0,
            'availability_pct': 0.0,
            'level': 'LOW',
            'level_color': 'green'
        }
    
    available_seats = total_seats - occupied_seats
    occupancy_pct = (occupied_seats / total_seats) * 100.0
    availability_pct = (available_seats / total_seats) * 100.0
    level, color = get_occupancy_level(occupancy_pct)
    
    return {
        'total': total_seats,
        'occupied': occupied_seats,
        'available': available_seats,
        'occupancy_pct': round(occupancy_pct, 1),
        'availability_pct': round(availability_pct, 1),
        'level': level,
        'level_color': color
    }

def get_default_rois(image_shape, is_synthetic=None):
    """
    Generate a realistic grid of workstation ROIs (Region of Interest)
    for a computer laboratory layout given image dimensions (height, width).
    
    Calibration logic:
    In natural university laboratory photographs, the top ~38% of the frame
    contains ceiling lights, fans, air conditioners, and upper walls.
    Desks and computer monitors occupy the lower 60% (y ≈ 0.38*h to 0.96*h).
    For synthetic canvas overviews (e.g. 900x600), the synthetic layout is used.
    """
    h, w = image_shape[:2]
    rois = []
    
    # 4 rows x 5 columns = 20 workstations
    rows = 4
    cols = 5
    
    if is_synthetic is None:
        is_synthetic = (w == 900 and h == 600)
        
    margin_x = int(w * 0.05)
    spacing_x = int((w - 2 * margin_x) / cols)
    box_w = int(spacing_x * 0.82)
    
    if is_synthetic:
        # Synthetic overview layout: canvas begins at y = 0.08*h
        margin_y = int(h * 0.08)
        spacing_y = int((h - 2 * margin_y) / rows)
        box_h = int(spacing_y * 0.75)
        start_y = margin_y
    else:
        # Real laboratory camera perspective: desks are in lower 60% of image
        start_y = int(h * 0.38)
        end_y = int(h * 0.96)
        avail_h = end_y - start_y
        spacing_y = int(avail_h / rows)
        box_h = int(spacing_y * 0.78)
        
    count = 1
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * spacing_x + int(spacing_x * 0.09)
            y = start_y + r * spacing_y + int(spacing_y * 0.11)
            # Ensure boundaries are inside image
            x = max(0, min(x, w - box_w))
            y = max(0, min(y, h - box_h))
            roi_id = f"PC-{count:02d}"
            rois.append((roi_id, (x, y, box_w, box_h)))
            count += 1
            
    return rois


def preprocess_image_crop(crop_img):
    """
    Preprocess image crop for CNN model input (128x128, RGB float32).
    Model contains internal Rescaling(1./255) layer.
    """
    if crop_img is None or crop_img.size == 0:
        raise ValueError("Invalid crop image provided")
        
    # Ensure RGB
    if len(crop_img.shape) == 2:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    elif crop_img.shape[2] == 4:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2RGB)
    else:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
    resized = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
    return resized.astype(np.float32)
