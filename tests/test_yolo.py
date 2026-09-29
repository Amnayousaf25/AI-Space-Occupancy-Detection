import os
import cv2
import numpy as np
import pytest

from src.yolo_detector import YOLOOccupancyDetector, get_yolo_detector
from src.occupancy_engine import analyze_lab_occupancy


def test_yolo_box_overlap():
    # Identical boxes
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    ioa_b, ioa_a = YOLOOccupancyDetector._compute_box_overlap(box1, box2)
    assert abs(ioa_b - 1.0) < 1e-4
    assert abs(ioa_a - 1.0) < 1e-4

    # Completely disjoint boxes
    box3 = (100, 100, 20, 20)
    ioa_b, ioa_a = YOLOOccupancyDetector._compute_box_overlap(box1, box3)
    assert ioa_b == 0.0
    assert ioa_a == 0.0

    # Half overlap
    box_half = (10, 10, 25, 50)
    ioa_b, ioa_a = YOLOOccupancyDetector._compute_box_overlap(box1, box_half)
    assert abs(ioa_b - 1.0) < 1e-4  # with respect to box_half area (area_b)


def test_yolo_analyze_lab_occupancy_synthetic():
    detector = get_yolo_detector("yolov8n.pt")
    # Synthetic lab image
    lab_img = np.zeros((600, 900, 3), dtype=np.uint8)
    # Simulate a desk and empty chair
    cv2.rectangle(lab_img, (50, 50), (150, 150), (100, 100, 100), -1)

    custom_rois = [
        ("PC-01", (50, 50, 100, 100)),
        ("PC-02", (200, 50, 100, 100)),
    ]

    res = detector.analyze_lab_occupancy_yolo(lab_img, custom_rois=custom_rois, persist_db=False)

    assert "stats" in res
    assert "detailed_predictions" in res
    assert "available_workstations" in res
    assert "annotated_image" in res
    assert res["stats"]["total"] == 2
    assert res["stats"]["occupied"] + res["stats"]["available"] == 2
    assert res["stats"]["level"] in ["LOW", "MEDIUM", "HIGH"]
    assert res["annotated_image"].shape == (600, 900, 3)


def test_occupancy_engine_with_model_type_yolo():
    lab_img = np.zeros((600, 900, 3), dtype=np.uint8)
    res = analyze_lab_occupancy(lab_img, model_type="yolo", persist_db=False)

    assert "stats" in res
    assert res["stats"]["total"] > 0
    assert len(res["detailed_predictions"]) == res["stats"]["total"]
