import numpy as np
from datetime import datetime
from src.config import (
    DEFAULT_PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
)
from src.utils import preprocess_image_crop
from src.model_service import get_model_service

def classify_workstation_crop(crop_img, workstation_id="PC-01", threshold=DEFAULT_PREDICTION_THRESHOLD):
    """
    Perform confidence-aware classification on a single workstation image crop.
    """
    model_service = get_model_service()
    norm_crop = preprocess_image_crop(crop_img)
    input_tensor = np.expand_dims(norm_crop, axis=0) # Shape: (1, 128, 128, 3)
    
    raw_prob = float(model_service.predict_batch(input_tensor)[0][0])
    
    is_occupied = raw_prob >= threshold
    prediction = 'OCCUPIED' if is_occupied else 'EMPTY'
    
    confidence = (raw_prob * 100.0) if is_occupied else ((1.0 - raw_prob) * 100.0)
    conf_ratio = confidence / 100.0
    
    if conf_ratio >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "HIGH"
        review_required = False
    elif conf_ratio >= MEDIUM_CONFIDENCE_THRESHOLD:
        confidence_level = "MEDIUM"
        review_required = False
    else:
        confidence_level = "LOW"
        review_required = True
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'workstation_id': workstation_id,
        'prediction': prediction,
        'probability': round(raw_prob, 4),
        'confidence': round(confidence, 1),
        'confidence_level': confidence_level,
        'review_required': review_required,
        'threshold': threshold,
        'timestamp': now_str
    }
