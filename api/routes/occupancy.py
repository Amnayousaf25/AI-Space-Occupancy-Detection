import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List, Dict, Any, Optional

from src.occupancy_engine import analyze_space_occupancy, analyze_lab_occupancy
from src.database import get_latest_occupancy, get_all_snapshots
from src.spaces_config import get_space_config, get_all_spaces

router = APIRouter(prefix="/occupancy", tags=["Occupancy"])

@router.post("/analyze")
async def analyze_space_endpoint(
    file: UploadFile = File(...),
    space: str = Query("computer_lab", description="Supported spaces: computer_lab, lecture_room, auditorium"),
    model: str = Query("yolo", description="Detection model: yolo (recommended) or cnn")
):
    """
    Unified Multi-Space AI Occupancy Detection Endpoint.
    Analyzes uploaded space overview image for Computer Lab, Lecture Room, or Auditorium.
    """
    try:
        norm_space = str(space).strip().lower().replace(" ", "_").replace("-", "_")
        if norm_space not in ("computer_lab", "lecture_room", "auditorium"):
            norm_space = "computer_lab"

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid space image provided")

        result = analyze_space_occupancy(
            img,
            space_type=norm_space,
            model_type=model,
            persist_db=True
        )

        st = result["stats"]
        return {
            "space": result.get("space", norm_space),
            "space_name": result.get("space_name", norm_space.title()),
            "total_seats": st["total"],
            "occupied": st["occupied"],
            "available": st["available"],
            "occupancy_rate": st["occupancy_pct"],
            "availability_rate": st["availability_pct"],
            "occupancy_level": st["level"],
            "total_persons_detected": result.get("total_persons_detected", 0),
            "detections": [
                {
                    "seat_id": p["workstation_id"],
                    "prediction": p["prediction"],
                    "confidence": p["confidence"],
                    "bbox": p["bbox"]
                } for p in result["detailed_predictions"]
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-lab")
async def analyze_lab_endpoint(file: UploadFile = File(...), rows: int = 4, cols: int = 5, model_type: str = "auto"):
    """Legacy backward-compatible lab analysis endpoint."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid laboratory image provided")
            
        h, w = img.shape[:2]
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.08)
        spacing_x = int((w - 2 * margin_x) / cols)
        spacing_y = int((h - 2 * margin_y) / rows)
        box_w = int(spacing_x * 0.8)
        box_h = int(spacing_y * 0.75)
        
        custom_rois = []
        count = 1
        for r in range(rows):
            for c in range(cols):
                x = margin_x + c * spacing_x + int(spacing_x * 0.1)
                y = margin_y + r * spacing_y + int(spacing_y * 0.1)
                custom_rois.append((f"PC-{count:02d}", (x, y, box_w, box_h)))
                count += 1
                
        result = analyze_lab_occupancy(img, custom_rois=custom_rois, persist_db=True, model_type=model_type)
        
        return {
            'stats': result['stats'],
            'available_workstations': result['available_workstations'],
            'alerts': result['alerts'],
            'predictions_summary': [
                {
                    'id': p['workstation_id'],
                    'prediction': p['prediction'],
                    'confidence': p['confidence'],
                    'review_required': p['review_required']
                } for p in result['detailed_predictions']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-lab-yolo")
async def analyze_lab_yolo_endpoint(file: UploadFile = File(...), conf_threshold: float = 0.25):
    """Legacy backward-compatible YOLO endpoint for lab analysis."""
    try:
        from src.yolo_detector import get_yolo_detector
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid laboratory image provided")

        detector = get_yolo_detector()
        result = detector.analyze_lab_occupancy_yolo(img, space_type="computer_lab", conf_threshold=conf_threshold, persist_db=True)

        return {
            'detector': 'YOLOv8',
            'stats': result['stats'],
            'total_persons_detected': result['total_persons_detected'],
            'total_chairs_detected': result['total_chairs_detected'],
            'available_workstations': result['available_workstations'],
            'alerts': result['alerts'],
            'predictions_summary': [
                {
                    'id': p['workstation_id'],
                    'prediction': p['prediction'],
                    'confidence': p['confidence'],
                    'overlap_score': p.get('overlap_score', 0.0)
                } for p in result['detailed_predictions']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/spaces")
def list_supported_spaces():
    """Returns list of supported smart spaces with configurations."""
    return get_all_spaces()

@router.get("/current")
def get_current_occupancy(space: Optional[str] = Query(None, description="Optional space filter: computer_lab, lecture_room, auditorium")):
    latest = get_latest_occupancy(space_type=space)
    if not latest:
        return {"status": "no_data", "message": "No occupancy snapshots recorded yet"}
    return latest

@router.get("/history")
def get_occupancy_history(space: Optional[str] = Query(None, description="Optional space filter: computer_lab, lecture_room, auditorium")):
    return get_all_snapshots(space_type=space)
