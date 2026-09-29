import cv2
import numpy as np

from src.utils import get_default_rois, preprocess_image_crop, calculate_occupancy_stats
from src.model_service import get_model_service
from src.database import save_predictions_batch, save_occupancy_snapshot
from src.alerts import evaluate_business_alerts
from src.config import DEFAULT_PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD

from src.spaces_config import get_space_config, get_space_rois

def analyze_space_occupancy(
    image_input,
    space_type: str = "computer_lab",
    custom_rois=None,
    threshold=DEFAULT_PREDICTION_THRESHOLD,
    persist_db=True,
    model_type="yolo"
):
    """
    Process full space overview image (Computer Lab, Lecture Room, or Auditorium)
    through YOLO object detector (primary) or custom CNN workstation ROI engine (baseline),
    compute occupancy metrics, evaluate alerts, and persist results into SQLite database.
    
    Default model_type is 'yolo' to ensure real person-detection based occupancy.
    """
    space_cfg = get_space_config(space_type)
    actual_space_id = space_cfg["id"]

    # Run YOLO detector if requested
    if model_type in ("yolo", "auto"):
        from src.yolo_detector import get_yolo_detector
        detector = get_yolo_detector(conf_thresh=threshold)
        return detector.analyze_lab_occupancy_yolo(
            image_input,
            space_type=actual_space_id,
            custom_rois=custom_rois,
            conf_threshold=threshold,
            persist_db=persist_db
        )

    # Fallback to Custom CNN Baseline (Calibrated for Computer Lab Workstation Crops)
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        image_source = image_input
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
        image_source = "uploaded_image"
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            pass
    else:
        raise ValueError("Invalid image input provided to occupancy engine")

    if img_bgr is None:
        raise ValueError("Image could not be decoded.")

    h, w = img_bgr.shape[:2]
    is_synthetic = (w == 900 and h == 600) and (actual_space_id == "computer_lab")
    rois = custom_rois if custom_rois else get_space_rois(actual_space_id, (h, w), is_synthetic=is_synthetic)
    
    crop_batch = []
    valid_rois = []
    
    for roi_id, (x, y, bw, bh) in rois:
        x_c = max(0, min(x, w - 1))
        y_c = max(0, min(y, h - 1))
        bw_c = max(10, min(bw, w - x_c))
        bh_c = max(10, min(bh, h - y_c))
        
        crop = img_bgr[y_c:y_c+bh_c, x_c:x_c+bw_c]
        if crop.size == 0:
            continue
            
        norm_crop = preprocess_image_crop(crop)
        crop_batch.append(norm_crop)
        valid_rois.append((roi_id, (x_c, y_c, bw_c, bh_c)))
        
    if not crop_batch:
        raise ValueError("No valid workstation ROIs could be extracted")
        
    batch_tensor = np.array(crop_batch, dtype=np.float32)
    model_service = get_model_service()
    probs = model_service.predict_batch(batch_tensor).flatten()
    
    # On natural photographs, the synthetic-trained CNN has an upward baseline bias (~0.65-0.70).
    # Calibrated decision threshold on natural camera images prevents empty desks from turning red.
    effective_thresh = threshold if is_synthetic else max(threshold, 0.85)

    occupied_count = 0
    detailed_predictions = []
    available_workstations = []
    workstation_associations = []
    annotated_bgr = img_bgr.copy()
    
    for idx, (roi_id, (x, y, bw, bh)) in enumerate(valid_rois):
        prob = float(probs[idx])
        is_occupied = prob >= effective_thresh
        
        prediction = 'OCCUPIED' if is_occupied else 'EMPTY'
        confidence = (prob * 100.0) if is_occupied else ((1.0 - prob) * 100.0)
        conf_ratio = confidence / 100.0
        
        if conf_ratio >= HIGH_CONFIDENCE_THRESHOLD:
            conf_level = "HIGH"
            review_req = False
        elif conf_ratio >= MEDIUM_CONFIDENCE_THRESHOLD:
            conf_level = "MEDIUM"
            review_req = False
        else:
            conf_level = "LOW"
            review_req = True
            
        if is_occupied:
            occupied_count += 1
            box_color = (0, 0, 220)  # Red BGR
            assoc_detail = f"CNN Crop Sigmoid: {prob:.3f} (>= {effective_thresh:.2f})"
        else:
            box_color = (0, 180, 0)  # Green BGR
            assoc_detail = f"CNN Crop Sigmoid: {prob:.3f} (< {effective_thresh:.2f})"
            available_workstations.append({
                'workstation_id': roi_id,
                'confidence': round(confidence, 1),
                'review_required': review_req
            })
            
        pred_record = {
            'workstation_id': roi_id,
            'space_type': actual_space_id,
            'prediction': prediction,
            'probability': round(prob, 4),
            'confidence': round(confidence, 1),
            'confidence_level': conf_level,
            'review_required': review_req,
            'bbox': (x, y, bw, bh),
            'image_source': image_source,
            'detector': 'Custom CNN'
        }
        detailed_predictions.append(pred_record)
        workstation_associations.append({
            'workstation_id': roi_id,
            'status': prediction,
            'details': assoc_detail,
            'confidence': round(confidence, 1)
        })
        
        # Bounding Box Overlay
        cv2.rectangle(annotated_bgr, (x, y), (x + bw, y + bh), box_color, 2)
        label_text = f"{roi_id}: {prediction} ({confidence:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(annotated_bgr, (x, y - th - 6), (x + tw + 6, y), box_color, -1)
        cv2.putText(annotated_bgr, label_text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        
    stats = calculate_occupancy_stats(len(valid_rois), occupied_count)
    alerts = evaluate_business_alerts(stats, detailed_predictions)
    
    if persist_db:
        save_occupancy_snapshot(stats, space_type=actual_space_id)
        save_predictions_batch(detailed_predictions, space_type=actual_space_id)
        
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return {
        'annotated_image': annotated_rgb,
        'space': actual_space_id,
        'space_name': space_cfg["name"],
        'space_config': space_cfg,
        'stats': stats,
        'detailed_predictions': detailed_predictions,
        'available_workstations': available_workstations,
        'alerts': alerts,
        'total_persons_detected': 0,
        'total_chairs_detected': 0,
        'total_screens_detected': 0,
        'counts_by_class': {"person": 0, "chair": 0, "tv_monitor": 0, "laptop": 0},
        'raw_detections': [],
        'person_detections': [],
        'workstation_associations': workstation_associations,
        'model_version': 'seat_occupancy_cnn_v1'
    }

# Backwards compatible alias
analyze_lab_occupancy = analyze_space_occupancy

