import os
import json
from PIL import Image
from src.logging_config import logger

def parse_annotation_file(annotation_path):
    """
    Parse ROI annotation file (JSON) for full lab photographs.
    Expected JSON structure:
    [
      {
        "workstation_id": "PC-01",
        "bbox": [x1, y1, x2, y2],
        "label": "EMPTY" | "OCCUPIED"
      }
    ]
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return annotations

def crop_workstations_from_lab_image(lab_image_path, annotations, output_dir):
    """
    Crop workstation ROIs from a full laboratory image based on annotations
    and save them into specified output folder for dataset expansion.
    """
    if not os.path.exists(lab_image_path):
        raise FileNotFoundError(f"Lab image not found at {lab_image_path}")

    os.makedirs(output_dir, exist_ok=True)
    cropped_files = []

    with Image.open(lab_image_path) as img:
        img_w, img_h = img.size
        for idx, ann in enumerate(annotations):
            pc_id = ann.get("workstation_id", f"PC_{idx+1}")
            bbox = ann.get("bbox", [0, 0, img_w, img_h])
            label = ann.get("label", "EMPTY").lower()
            
            x1, y1, x2, y2 = bbox
            crop_box = (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))
            
            cropped = img.crop(crop_box)
            
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            out_filename = f"{os.path.splitext(os.path.basename(lab_image_path))[0]}_{pc_id}.jpg"
            out_path = os.path.join(label_dir, out_filename)
            cropped.save(out_path)
            cropped_files.append(out_path)

    logger.info(f"Cropped {len(cropped_files)} workstation ROIs from {lab_image_path}")
    return cropped_files
