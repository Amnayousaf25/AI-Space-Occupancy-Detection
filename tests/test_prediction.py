import numpy as np
import pytest
from src.prediction_service import classify_workstation_crop

def test_classify_workstation_crop():
    # Create synthetic test image crop (128x128x3 RGB)
    crop_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    
    result = classify_workstation_crop(crop_img, workstation_id="PC-99", threshold=0.50)
    
    assert result['workstation_id'] == "PC-99"
    assert result['prediction'] in ['EMPTY', 'OCCUPIED']
    assert 0.0 <= result['probability'] <= 1.0
    assert 0.0 <= result['confidence'] <= 100.0
    assert result['confidence_level'] in ['HIGH', 'MEDIUM', 'LOW']
    assert isinstance(result['review_required'], bool)
