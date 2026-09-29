import os
import csv
import hashlib
import shutil
import uuid
import json
from PIL import Image
from src.config import (
    DATASET_DIR,
    DATASET_RAW_DIR,
    DATASET_PROCESSED_DIR,
    DATASET_EXTERNAL_VAL_DIR,
    DATASET_METADATA_DIR,
    DATASET_MANIFEST_PATH,
)
from src.logging_config import logger

MANIFEST_HEADERS = [
    "image_id",
    "filename",
    "label",
    "source",
    "session_id",
    "target_purpose",
    "original_path",
    "width",
    "height",
    "file_size",
    "hash",
    "split_path"
]

def initialize_dataset_directories():
    """Ensure raw, processed, external validation, and metadata directories exist."""
    directories = [
        os.path.join(DATASET_RAW_DIR, "juw_real", "empty"),
        os.path.join(DATASET_RAW_DIR, "juw_real", "occupied"),
        os.path.join(DATASET_PROCESSED_DIR, "juw_real", "empty"),
        os.path.join(DATASET_PROCESSED_DIR, "juw_real", "occupied"),
        os.path.join(DATASET_EXTERNAL_VAL_DIR, "juw_real", "empty"),
        os.path.join(DATASET_EXTERNAL_VAL_DIR, "juw_real", "occupied"),
        DATASET_METADATA_DIR,
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
    
    if not os.path.exists(DATASET_MANIFEST_PATH):
        with open(DATASET_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(MANIFEST_HEADERS)
        logger.info(f"Initialized dataset manifest CSV at {DATASET_MANIFEST_PATH}")

def compute_image_hash(file_path):
    """Compute SHA256 hash of image file for duplicate & leakage detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_image_file(file_path):
    """Validate image readability, format, and dimensions."""
    if not os.path.exists(file_path):
        return False, 0, 0, "File does not exist"
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            width, height = img.size
            return True, width, height, "Valid"
    except Exception as e:
        return False, 0, 0, f"Corrupted or invalid image: {str(e)}"

def read_manifest():
    """Read dataset manifest CSV as list of dictionaries."""
    initialize_dataset_directories()
    records = []
    if os.path.exists(DATASET_MANIFEST_PATH):
        with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    return records

def check_data_leakage():
    """Check for data leakage: identical image hashes appearing across different split purposes."""
    records = read_manifest()
    hash_map = {}
    conflicts = []

    for r in records:
        img_hash = r.get("hash")
        purpose = r.get("target_purpose")
        filename = r.get("filename")
        if not img_hash:
            continue
        
        if img_hash in hash_map:
            existing_purpose = hash_map[img_hash]["target_purpose"]
            if existing_purpose != purpose:
                conflicts.append({
                    "hash": img_hash,
                    "file_1": hash_map[img_hash]["filename"],
                    "purpose_1": existing_purpose,
                    "file_2": filename,
                    "purpose_2": purpose,
                    "conflict_type": "CROSS_SPLIT_HASH_LEAKAGE"
                })
        else:
            hash_map[img_hash] = {"target_purpose": purpose, "filename": filename}

    leakage_detected = len(conflicts) > 0
    return {
        "leakage_detected": leakage_detected,
        "conflict_count": len(conflicts),
        "conflicts": conflicts
    }

def ingest_image(
    file_path,
    target_purpose="training",
    class_label="empty",
    session_id="session_01",
    source_type="JUW_REAL_LAB"
):
    """
    Ingest a real JUW workstation photograph into the structured dataset hierarchy.
    Enforces image integrity, hash calculation, manifest tracking, and data leakage rules.
    """
    initialize_dataset_directories()
    
    # 1. Validate image
    is_valid, width, height, err_msg = validate_image_file(file_path)
    if not is_valid:
        raise ValueError(f"Image validation failed for {file_path}: {err_msg}")
    
    class_label = class_label.lower().strip()
    if class_label not in ["empty", "occupied"]:
        raise ValueError(f"Invalid class label '{class_label}'. Must be 'empty' or 'occupied'.")

    target_purpose = target_purpose.lower().strip()
    if target_purpose not in ["training", "external_validation"]:
        raise ValueError("Target purpose must be 'training' or 'external_validation'.")

    img_hash = compute_image_hash(file_path)
    file_size = os.path.getsize(file_path)
    
    # 2. Check leakage/duplicate against manifest
    records = read_manifest()
    for r in records:
        if r.get("hash") == img_hash:
            existing_purpose = r.get("target_purpose")
            if existing_purpose != target_purpose:
                raise ValueError(
                    f"Data Leakage Blocked! Image hash {img_hash[:8]} already registered under '{existing_purpose}'. "
                    f"Cannot register identical image into '{target_purpose}'."
                )

    # 3. Target directory selection
    if target_purpose == "training":
        target_dir = os.path.join(DATASET_RAW_DIR, "juw_real", class_label)
    else:
        target_dir = os.path.join(DATASET_EXTERNAL_VAL_DIR, "juw_real", class_label)

    os.makedirs(target_dir, exist_ok=True)
    
    ext = os.path.splitext(file_path)[1].lower()
    if not ext:
        ext = ".jpg"
    image_id = f"JUW_IMG_{uuid.uuid4().hex[:8].upper()}"
    new_filename = f"{image_id}_{class_label}{ext}"
    dest_path = os.path.join(target_dir, new_filename)
    
    shutil.copy2(file_path, dest_path)
    
    rel_dest_path = os.path.relpath(dest_path, DATASET_DIR)
    
    # 4. Append to manifest CSV
    new_record = {
        "image_id": image_id,
        "filename": new_filename,
        "label": class_label.upper(),
        "source": source_type,
        "session_id": session_id,
        "target_purpose": target_purpose,
        "original_path": os.path.basename(file_path),
        "width": width,
        "height": height,
        "file_size": file_size,
        "hash": img_hash,
        "split_path": rel_dest_path
    }
    
    with open(DATASET_MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        writer.writerow(new_record)
        
    logger.info(f"Successfully ingested image {image_id} into {target_purpose} set.")
    return new_record
