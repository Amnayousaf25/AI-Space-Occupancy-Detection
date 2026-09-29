import os
import json
import hashlib
from PIL import Image
from src.config import (
    DATASET_DIR,
    DATASET_RAW_DIR,
    DATASET_EXTERNAL_VAL_DIR,
    DATASET_AUDIT_PATH,
)
from src.dataset_manager import (
    initialize_dataset_directories,
    check_data_leakage,
    read_manifest,
)
from src.logging_config import logger

def run_dataset_audit():
    """
    Perform a complete dataset audit across controlled training/val/test sets
    and real JUW datasets (raw & external validation).
    Generates dataset/metadata/dataset_audit.json.
    """
    initialize_dataset_directories()
    
    splits_to_scan = {
        "controlled_train": os.path.join(DATASET_DIR, "train"),
        "controlled_val": os.path.join(DATASET_DIR, "validation"),
        "controlled_test": os.path.join(DATASET_DIR, "test"),
        "real_juw_raw": os.path.join(DATASET_RAW_DIR, "juw_real"),
        "real_juw_external_val": os.path.join(DATASET_EXTERNAL_VAL_DIR, "juw_real"),
    }
    
    audit_results = {
        "timestamp": os.getenv("AUDIT_TIME", "2026-09-14 16:00:00"),
        "total_images": 0,
        "total_empty": 0,
        "total_occupied": 0,
        "class_balance": {
            "percent_empty": 0.0,
            "percent_occupied": 0.0
        },
        "splits": {},
        "corrupted_images": [],
        "duplicate_hashes": [],
        "leakage_analysis": {}
    }
    
    all_hashes = {}
    total_images = 0
    total_empty = 0
    total_occupied = 0
    
    for split_name, split_path in splits_to_scan.items():
        split_stats = {"total": 0, "empty": 0, "occupied": 0, "corrupted": 0}
        
        if not os.path.exists(split_path):
            audit_results["splits"][split_name] = split_stats
            continue

        for root, _, files in os.walk(split_path):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, DATASET_DIR)
                
                # Determine label from directory or filename
                parent_dir = os.path.basename(root).lower()
                if parent_dir in ["empty", "occupied"]:
                    label = parent_dir.upper()
                elif "empty" in file.lower():
                    label = "EMPTY"
                elif "occupied" in file.lower():
                    label = "OCCUPIED"
                else:
                    label = "UNKNOWN"

                # Image Validation
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    split_stats["corrupted"] += 1
                    audit_results["corrupted_images"].append({
                        "file": rel_path,
                        "error": str(e)
                    })
                    continue

                # Hash Calculation
                sha256 = hashlib.sha256()
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
                img_hash = sha256.hexdigest()
                
                if img_hash in all_hashes:
                    all_hashes[img_hash].append(rel_path)
                else:
                    all_hashes[img_hash] = [rel_path]

                split_stats["total"] += 1
                total_images += 1
                if label == "EMPTY":
                    split_stats["empty"] += 1
                    total_empty += 1
                elif label == "OCCUPIED":
                    split_stats["occupied"] += 1
                    total_occupied += 1

        audit_results["splits"][split_name] = split_stats

    # Calculate class balance
    audit_results["total_images"] = total_images
    audit_results["total_empty"] = total_empty
    audit_results["total_occupied"] = total_occupied
    if total_images > 0:
        audit_results["class_balance"]["percent_empty"] = round((total_empty / total_images) * 100, 2)
        audit_results["class_balance"]["percent_occupied"] = round((total_occupied / total_images) * 100, 2)

    # Calculate duplicate hashes
    duplicates = [paths for img_hash, paths in all_hashes.items() if len(paths) > 1]
    audit_results["duplicate_hashes"] = [
        {"hash": hashlib.sha256(p[0].encode()).hexdigest()[:8], "occurrences": len(p), "files": p}
        for p in duplicates
    ]

    # Data Leakage Check
    audit_results["leakage_analysis"] = check_data_leakage()

    # Save to dataset_audit.json
    os.makedirs(os.path.dirname(DATASET_AUDIT_PATH), exist_ok=True)
    with open(DATASET_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    logger.info(f"Dataset audit completed. Report written to {DATASET_AUDIT_PATH}")
    return audit_results

if __name__ == "__main__":
    audit = run_dataset_audit()
    print("Dataset Audit Summary:")
    print(f"Total Images: {audit['total_images']}")
    print(f"Class Balance: EMPTY={audit['class_balance']['percent_empty']}%, OCCUPIED={audit['class_balance']['percent_occupied']}%")
    print(f"Leakage Detected: {audit['leakage_analysis'].get('leakage_detected', False)}")
