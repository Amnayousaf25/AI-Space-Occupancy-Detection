import os
import json
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.config import (
    DATASET_EXTERNAL_VAL_DIR,
    RESULTS_DIR,
    IMG_SIZE,
)
from src.model_registry import get_active_model_info, get_active_model_path
from src.model_service import get_model_service
from src.logging_config import logger

def evaluate_external_juw_validation(model_version=None):
    """
    Evaluate the active CNN model (or specified version) exclusively on the
    unseen Real JUW External Validation Dataset (dataset/external_validation/juw_real).
    Never fabricates metrics. If no images exist, returns clear status.
    """
    ext_val_base = os.path.join(DATASET_EXTERNAL_VAL_DIR, "juw_real")
    empty_dir = os.path.join(ext_val_base, "empty")
    occupied_dir = os.path.join(ext_val_base, "occupied")

    image_paths = []
    true_labels = []

    if os.path.exists(empty_dir):
        for f in os.listdir(empty_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(empty_dir, f))
                true_labels.append(0)  # EMPTY

    if os.path.exists(occupied_dir):
        for f in os.listdir(occupied_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(occupied_dir, f))
                true_labels.append(1)  # OCCUPIED

    if len(image_paths) == 0:
        logger.info("Real JUW external validation set is empty.")
        return {
            "status": "not_available",
            "message": "Real JUW external validation not available yet.",
            "evaluated_count": 0,
            "metrics": None
        }

    active_info = get_active_model_info()
    model_name = model_version or active_info.get("version", "active_model")

    model_service = get_model_service()
    images_array = []
    
    for path in image_paths:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB").resize(IMG_SIZE)
            arr = np.array(img_rgb, dtype=np.float32) / 255.0
            images_array.append(arr)

    batch_tensor = np.array(images_array)
    raw_preds = model_service.predict_batch(batch_tensor).flatten()
    pred_labels = (raw_preds >= 0.5).astype(int)

    acc = float(accuracy_score(true_labels, pred_labels))
    prec = float(precision_score(true_labels, pred_labels, zero_division=0))
    rec = float(recall_score(true_labels, pred_labels, zero_division=0))
    f1 = float(f1_score(true_labels, pred_labels, zero_division=0))
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).tolist()

    report = {
        "status": "evaluated",
        "model_version": model_name,
        "evaluated_count": len(image_paths),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "tn": cm[0][0], "fp": cm[0][1],
                "fn": cm[1][0], "tp": cm[1][1]
            }
        }
    }

    report_path = os.path.join(RESULTS_DIR, "juw_external_validation_report.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"External JUW validation evaluation complete. Report saved to {report_path}")
    return report
