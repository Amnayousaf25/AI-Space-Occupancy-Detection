from src.config import OCCUPANCY_HIGH_THRESHOLD
from src.logging_config import logger

def evaluate_business_alerts(stats, detailed_predictions):
    """
    Evaluate system alerts based on business rules:
    1. High Occupancy Alert (>= 90%)
    2. Low Confidence / Model Review Alert
    """
    alerts = []
    
    # Alert Rule 1: Occupancy Threshold
    if stats['occupancy_pct'] >= OCCUPANCY_HIGH_THRESHOLD:
        alerts.append({
            'type': 'HIGH_OCCUPANCY',
            'severity': 'WARNING',
            'message': f"⚠️ High Occupancy Alert: Lab occupancy reached {stats['occupancy_pct']}% ({stats['occupied']}/{stats['total']} PCs occupied)."
        })
        
    # Alert Rule 2: Model Confidence Review
    low_conf_items = [p.get('workstation_id', p.get('id', 'Unknown')) for p in detailed_predictions if p.get('review_required', False)]
    if low_conf_items:
        alerts.append({
            'type': 'MODEL_REVIEW_REQUIRED',
            'severity': 'INFO',
            'message': f"🔍 Model Review Alert: Low prediction confidence on {len(low_conf_items)} workstation(s): {', '.join(low_conf_items)}."
        })
        
    return alerts
