import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_and_favicon_endpoints():
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    data = root_resp.json()
    assert "JUW SMART SPACE" in data["title"]
    assert "interactive_docs" in data

    fav_resp = client.get("/favicon.ico")
    assert fav_resp.status_code == 204

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_endpoint():
    # Create test image crop
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    response = client.post(
        "/predict",
        files={"file": ("test_crop.jpg", img_bytes, "image/jpeg")},
        data={"workstation_id": "PC-05", "threshold": "0.50"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["workstation_id"] == "PC-05"
    assert data["prediction"] in ["EMPTY", "OCCUPIED"]
    assert 0.0 <= data["probability"] <= 1.0

def test_analytics_summary_endpoint():
    response = client.get("/analytics/summary")
    assert response.status_code == 200

def test_models_registry_endpoints():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    
    active_resp = client.get("/models/active")
    assert active_resp.status_code == 200

def test_dataset_health_and_validation_endpoints():
    health_resp = client.get("/dataset/health")
    assert health_resp.status_code == 200
    
    val_resp = client.get("/validation/summary")
    assert val_resp.status_code == 200

def test_occupancy_analyze_lab_endpoints():
    # Synthetic lab image
    lab_img = np.zeros((400, 600, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", lab_img)
    img_bytes = io.BytesIO(encoded.tobytes())

    # Test baseline CNN / analyze-lab endpoint
    resp = client.post(
        "/occupancy/analyze-lab",
        files={"file": ("lab.jpg", img_bytes, "image/jpeg")},
        params={"rows": 2, "cols": 3, "model_type": "cnn"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "stats" in data
    assert data["stats"]["total"] == 6

    # Test YOLO endpoint
    img_bytes.seek(0)
    yolo_resp = client.post(
        "/occupancy/analyze-lab-yolo",
        files={"file": ("lab.jpg", img_bytes, "image/jpeg")},
        params={"conf_threshold": 0.25}
    )
    assert yolo_resp.status_code == 200
    yolo_data = yolo_resp.json()
    assert yolo_data["detector"] == "YOLOv8"
    assert "stats" in yolo_data
    assert "total_persons_detected" in yolo_data

