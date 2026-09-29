import os
import yaml
from typing import Dict, List, Tuple, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')

DEFAULT_SPACES = {
    'computer_lab': {
        'id': 'computer_lab',
        'name': 'Computer Laboratory',
        'short_label': 'Computer Lab',
        'description': 'University Computer Lab with desktop PCs, monitors, and ergonomic task chairs.',
        'institution': 'Jinnah University for Women - Department of CS & SE',
        'capacity': 20,
        'layout_type': 'grid',
        'seat_prefix': 'PC',
        'grid': {
            'rows': 4,
            'cols': 5,
            'margin_x_ratio': 0.05,
            'synthetic_margin_y_ratio': 0.08,
            'natural_start_y_ratio': 0.38,
            'natural_end_y_ratio': 0.96,
            'box_width_ratio': 0.82,
            'box_height_ratio': 0.78
        },
        'supported_models': ['yolo', 'cnn'],
        'default_model': 'yolo',
        'accuracy_status': 'Verified with Real JUW Lab Imagery',
        'model_notes': {
            'yolo': 'Primary enterprise engine. Full-scene YOLOv8 object detection with spatial desk association.',
            'cnn': 'Specialized baseline. Workstation crop classifier calibrated for PC monitor/desk regions.'
        },
        'sample_scenes': [
            {'id': 'real/juw_real_empty.jpg', 'label': 'JUW Empty Lab', 'desc': 'Real JUW Computer Lab (Empty Session, 0 Occupants)'},
            {'id': 'real/juw_real_solo.jpg', 'label': 'JUW Solo Study', 'desc': 'Real JUW Computer Lab (1 Student Working)'},
            {'id': 'real/juw_real_group.jpg', 'label': 'JUW Group Lab', 'desc': 'Real JUW Computer Lab (Group Work Session)'},
            {'id': 'juw_lab_sample_1.jpg', 'label': 'Sim Lab 01', 'desc': 'Balanced Simulation Session (13 Occupants)'},
            {'id': 'juw_lab_sample_3.jpg', 'label': 'Sim Lab 03', 'desc': 'Peak Capacity Simulation (18 Occupants)'}
        ]
    },
    'lecture_room': {
        'id': 'lecture_room',
        'name': 'Lecture Room',
        'short_label': 'Lecture Room',
        'description': 'Academic Lecture Hall with tiered student benches, presentation podium, and dual aisles.',
        'institution': 'Jinnah University for Women - Academic Block',
        'capacity': 30,
        'layout_type': 'classroom_rows',
        'seat_prefix': 'LR',
        'grid': {
            'rows': 5,
            'cols': 6,
            'margin_x_ratio': 0.06,
            'natural_start_y_ratio': 0.35,
            'natural_end_y_ratio': 0.95,
            'box_width_ratio': 0.84,
            'box_height_ratio': 0.72
        },
        'supported_models': ['yolo', 'cnn'],
        'default_model': 'yolo',
        'accuracy_status': 'Active Architecture — Calibrated Classroom Desk Geometry',
        'model_notes': {
            'yolo': 'Primary engine. Detects students and associates them with classroom desk positions via 1-to-1 matching.',
            'cnn': 'Custom CNN baseline. Classifies individual lecture desk crops into OCCUPIED vs AVAILABLE.'
        },
        'sample_scenes': [
            {'id': 'lecture/juw_lecture_empty.jpg', 'label': 'Empty Lecture', 'desc': 'Lecture Hall (0 Students, Available for Booking)'},
            {'id': 'lecture/juw_lecture_moderate.jpg', 'label': 'Half Lecture', 'desc': 'Morning Class Session (Moderate Attendance)'},
            {'id': 'lecture/juw_lecture_full.jpg', 'label': 'Full Lecture', 'desc': 'Core Course Lecture (Full Capacity)'}
        ]
    },
    'auditorium': {
        'id': 'auditorium',
        'name': 'Auditorium',
        'short_label': 'Auditorium',
        'description': 'Main Campus Auditorium with stepped theater seating rows, presentation stage, and central walkway.',
        'institution': 'Jinnah University for Women - Main Campus Hall',
        'capacity': 50,
        'layout_type': 'tiered_theater',
        'seat_prefix': 'AUD',
        'grid': {
            'rows': 5,
            'cols': 10,
            'margin_x_ratio': 0.04,
            'natural_start_y_ratio': 0.34,
            'natural_end_y_ratio': 0.96,
            'box_width_ratio': 0.86,
            'box_height_ratio': 0.70
        },
        'supported_models': ['yolo', 'cnn'],
        'default_model': 'yolo',
        'accuracy_status': 'Active Architecture — Stepped Tier Geometry Configured',
        'model_notes': {
            'yolo': 'Primary engine. Detects audience attendees and maps them to tiered theater seating slots with duplicate prevention.',
            'cnn': 'Custom CNN baseline. Classifies individual auditorium tiered seat crops into OCCUPIED vs AVAILABLE.'
        },
        'sample_scenes': [
            {'id': 'auditorium/juw_auditorium_empty.jpg', 'label': 'Empty Hall', 'desc': 'Main Auditorium (Empty Hall, Available)'},
            {'id': 'auditorium/juw_auditorium_session.jpg', 'label': 'Seminar Session', 'desc': 'Department Seminar (Partial Attendance)'},
            {'id': 'auditorium/juw_auditorium_keynote.jpg', 'label': 'Keynote Hall', 'desc': 'Annual Symposium / Keynote (High Attendance)'}
        ]
    }
}


def load_space_config_file(space_id: str) -> Optional[Dict[str, Any]]:
    """Loads space configuration from YAML file if available."""
    yaml_path = os.path.join(CONFIGS_DIR, f"{space_id}.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    return None


def get_space_config(space_id: str) -> Dict[str, Any]:
    """
    Returns space configuration dictionary for given space_id.
    Falls back to 'computer_lab' if unknown space requested.
    """
    norm_id = str(space_id).strip().lower().replace(" ", "_").replace("-", "_")
    
    # Map common aliases
    if norm_id in ('lab', 'computer_laboratory'):
        norm_id = 'computer_lab'
    elif norm_id in ('lecture', 'classroom', 'lecture_hall'):
        norm_id = 'lecture_room'
    elif norm_id in ('hall', 'audi'):
        norm_id = 'auditorium'
        
    cfg = load_space_config_file(norm_id)
    if cfg:
        return cfg
        
    return DEFAULT_SPACES.get(norm_id, DEFAULT_SPACES['computer_lab'])


def get_all_spaces() -> List[Dict[str, Any]]:
    """Returns list of all supported space configurations."""
    order = ['computer_lab', 'lecture_room', 'auditorium']
    return [get_space_config(s_id) for s_id in order]


def get_space_rois(
    space_id: str,
    image_shape: Tuple[int, int],
    is_synthetic: Optional[bool] = None
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Generate space-specific seating/workstation ROIs for a given space and image dimensions.
    Returns list of tuples: [(seat_id, (x, y, w, h)), ...]
    """
    cfg = get_space_config(space_id)
    h, w = image_shape[:2]
    grid_cfg = cfg.get('grid', {})
    
    rows = grid_cfg.get('rows', 4)
    cols = grid_cfg.get('cols', 5)
    prefix = cfg.get('seat_prefix', 'PC')
    
    if is_synthetic is None:
        is_synthetic = (w == 900 and h == 600) and (cfg['id'] == 'computer_lab')
        
    margin_x = int(w * grid_cfg.get('margin_x_ratio', 0.05))
    spacing_x = int((w - 2 * margin_x) / cols)
    box_w = int(spacing_x * grid_cfg.get('box_width_ratio', 0.82))
    
    if is_synthetic:
        # Synthetic overview canvas for lab
        margin_y = int(h * grid_cfg.get('synthetic_margin_y_ratio', 0.08))
        spacing_y = int((h - 2 * margin_y) / rows)
        box_h = int(spacing_y * 0.75)
        start_y = margin_y
    else:
        # Real camera perspective: seats reside in lower part of the camera frame
        start_y = int(h * grid_cfg.get('natural_start_y_ratio', 0.38))
        end_y = int(h * grid_cfg.get('natural_end_y_ratio', 0.96))
        avail_h = end_y - start_y
        spacing_y = int(avail_h / rows)
        box_h = int(spacing_y * grid_cfg.get('box_height_ratio', 0.75))
        
    rois = []
    count = 1
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * spacing_x + int(spacing_x * 0.09)
            y = start_y + r * spacing_y + int(spacing_y * 0.11)
            
            # Boundary checks
            x = max(0, min(x, w - box_w))
            y = max(0, min(y, h - box_h))
            
            seat_id = f"{prefix}-{count:02d}"
            rois.append((seat_id, (x, y, box_w, box_h)))
            count += 1
            
    return rois
