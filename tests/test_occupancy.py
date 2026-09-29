import numpy as np
import pytest
from src.occupancy_engine import analyze_lab_occupancy

def test_analyze_lab_occupancy():
    # Synthetic lab overview image (600x900x3 BGR)
    lab_img = np.random.randint(0, 256, (600, 900, 3), dtype=np.uint8)
    
    result = analyze_lab_occupancy(lab_img, persist_db=False)
    
    stats = result['stats']
    assert stats['total'] > 0
    assert stats['occupied'] + stats['available'] == stats['total']
    assert abs((stats['occupancy_pct'] + stats['availability_pct']) - 100.0) < 0.1
    assert stats['level'] in ['LOW', 'MEDIUM', 'HIGH']
    assert isinstance(result['available_workstations'], list)
    assert isinstance(result['alerts'], list)
