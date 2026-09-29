import os
import pytest
from PIL import Image
from src.dataset_manager import (
    compute_image_hash,
    validate_image_file,
    ingest_image,
    check_data_leakage,
    initialize_dataset_directories
)
from src.dataset_audit import run_dataset_audit

def test_initialize_dataset_directories(tmp_path):
    initialize_dataset_directories()
    from src.config import DATASET_MANIFEST_PATH
    assert os.path.exists(DATASET_MANIFEST_PATH)

def test_validate_and_hash_image(tmp_path):
    # Create sample synthetic image
    img_path = str(tmp_path / "test_sample.jpg")
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(img_path)
    
    is_valid, w, h, err = validate_image_file(img_path)
    assert is_valid is True
    assert w == 100
    assert h == 100
    
    img_hash = compute_image_hash(img_path)
    assert len(img_hash) == 64  # SHA256 length

def test_ingest_image_and_leakage_block(tmp_path):
    img_path = str(tmp_path / "test_lab_seat.jpg")
    img = Image.new('RGB', (128, 128), color='green')
    img.save(img_path)
    
    record = ingest_image(
        file_path=img_path,
        target_purpose="training",
        class_label="empty",
        session_id="test_session_99",
        source_type="UNIT_TEST"
    )
    assert record["label"] == "EMPTY"
    assert record["target_purpose"] == "training"
    
    # Try ingesting identical hash into external_validation -> should raise ValueError for leakage
    with pytest.raises(ValueError, match="Data Leakage Blocked"):
        ingest_image(
            file_path=img_path,
            target_purpose="external_validation",
            class_label="empty",
            session_id="test_session_99",
            source_type="UNIT_TEST"
        )

def test_dataset_audit_run():
    audit_res = run_dataset_audit()
    assert "total_images" in audit_res
    assert "class_balance" in audit_res
    assert "leakage_analysis" in audit_res
