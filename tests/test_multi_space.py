import pytest
import numpy as np
from src.spaces_config import get_space_config, get_all_spaces, get_space_rois
from src.occupancy_engine import analyze_space_occupancy
from src.database import get_all_snapshots, save_occupancy_snapshot


def test_space_configurations():
    spaces = get_all_spaces()
    space_ids = [s["id"] for s in spaces]
    assert "computer_lab" in space_ids
    assert "lecture_room" in space_ids
    assert "auditorium" in space_ids

    lab_cfg = get_space_config("computer_lab")
    assert lab_cfg["capacity"] == 20
    assert lab_cfg["seat_prefix"] == "PC"

    lec_cfg = get_space_config("lecture_room")
    assert lec_cfg["capacity"] == 30
    assert lec_cfg["seat_prefix"] == "LR"

    aud_cfg = get_space_config("auditorium")
    assert aud_cfg["capacity"] == 50
    assert aud_cfg["seat_prefix"] == "AUD"


def test_space_rois_generation():
    lab_rois = get_space_rois("computer_lab", (800, 1200))
    assert len(lab_rois) == 20
    assert lab_rois[0][0].startswith("PC-")

    lec_rois = get_space_rois("lecture_room", (800, 1200))
    assert len(lec_rois) == 30
    assert lec_rois[0][0].startswith("LR-")

    aud_rois = get_space_rois("auditorium", (800, 1200))
    assert len(aud_rois) == 50
    assert aud_rois[0][0].startswith("AUD-")


def test_empty_synthetic_frame_hard_rule():
    # Invariant: 0 persons detected -> 0 occupied across all 3 spaces
    empty_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    for space_id in ["computer_lab", "lecture_room", "auditorium"]:
        cfg = get_space_config(space_id)
        res = analyze_space_occupancy(empty_frame, space_type=space_id, persist_db=False, model_type="yolo")
        stats = res["stats"]
        assert stats["total"] == cfg["capacity"]
        assert stats["occupied"] == 0
        assert stats["available"] == cfg["capacity"]
        assert stats["occupancy_pct"] == 0.0
        assert stats["occupied"] + stats["available"] == stats["total"]
