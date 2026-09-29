import os
import cv2
import numpy as np
import tensorflow as tf

try:
    from src.utils import (
        MODEL_PATH, get_default_rois, preprocess_image_crop, calculate_occupancy_stats
    )
except ImportError:
    from utils import (
        MODEL_PATH, get_default_rois, preprocess_image_crop, calculate_occupancy_stats
    )

class SeatCounterEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the CNN model first!")
        self.model = tf.keras.models.load_model(self.model_path)
        
    def analyze_lab_image(self, image_input, custom_rois=None):
        """
        Analyze computer lab workstation occupancy from full room image.
        
        Args:
            image_input: image file path or numpy array (BGR/RGB)
            custom_rois: list of tuples [(roi_id, (x, y, w, h)), ...] or None for default grid
            
        Returns:
            annotated_image: RGB annotated OpenCV image with ROI bounding boxes
            summary_stats: dict containing total, occupied, available, percentages, and level
            detailed_predictions: list of dicts for each workstation
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img_bgr = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                # Assuming incoming array is RGB if passed from Streamlit
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("image_input must be a file path or numpy ndarray")
            
        h, w = img_bgr.shape[:2]
        
        if custom_rois is None or len(custom_rois) == 0:
            rois = get_default_rois((h, w))
        else:
            rois = custom_rois
            
        crop_batch = []
        valid_rois = []
        
        for roi_id, (x, y, bw, bh) in rois:
            # Boundary checks
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(10, min(bw, w - x))
            bh = max(10, min(bh, h - y))
            
            crop = img_bgr[y:y+bh, x:x+bw]
            if crop.size == 0:
                continue
                
            norm_crop = preprocess_image_crop(crop)
            crop_batch.append(norm_crop)
            valid_rois.append((roi_id, (x, y, bw, bh)))
            
        if not crop_batch:
            raise ValueError("No valid ROIs extracted from image")
            
        batch_array = np.array(crop_batch, dtype=np.float32)
        probs = self.model.predict(batch_array, verbose=0).flatten()
        
        occupied_count = 0
        detailed_predictions = []
        annotated_bgr = img_bgr.copy()
        
        for idx, (roi_id, (x, y, bw, bh)) in enumerate(valid_rois):
            prob = float(probs[idx])
            is_occupied = prob >= 0.5
            
            if is_occupied:
                status = 'OCCUPIED'
                confidence = prob * 100.0
                occupied_count += 1
                color = (0, 0, 220) # Bright Red BGR
            else:
                status = 'EMPTY'
                confidence = (1.0 - prob) * 100.0
                color = (0, 180, 0) # Green BGR
                
            detailed_predictions.append({
                'id': roi_id,
                'status': status,
                'confidence': round(confidence, 1),
                'probability': round(prob, 4),
                'bbox': (x, y, bw, bh)
            })
            
            # Draw Bounding Box
            cv2.rectangle(annotated_bgr, (x, y), (x + bw, y + bh), color, 2)
            
            # Draw Header Bar for Label
            label_text = f"{roi_id}: {status}"
            cv2.rectangle(annotated_bgr, (x, y - 22), (x + bw, y), color, -1)
            cv2.putText(annotated_bgr, label_text, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        stats = calculate_occupancy_stats(len(valid_rois), occupied_count)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb, stats, detailed_predictions

def main():
    sample_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'dataset', 'sample_lab_views', 'juw_lab_sample_1.jpg')
    if not os.path.exists(sample_img_path):
        print(f"[INFO] Generating dataset and sample views first...")
        from prepare_dataset import main as prep_main
        prep_main()
        
    print("==================================================")
    print(" JUW SMART SPACE - SEAT & PC OCCUPANCY COUNTER")
    print("==================================================")
    print(f"[INFO] Analyzing lab overview image: {sample_img_path}")
    
    engine = SeatCounterEngine()
    annotated_img, stats, predictions = engine.analyze_lab_image(sample_img_path)
    
    print("\n--------------------------------------------------")
    print(" OCCUPANCY SUMMARY METRICS")
    print("--------------------------------------------------")
    print(f" Total Workstations/PCs : {stats['total']}")
    print(f" Occupied Workstations : {stats['occupied']}")
    print(f" Available Workstations: {stats['available']}")
    print(f" Occupancy Percentage  : {stats['occupancy_pct']}%")
    print(f" Availability Percentage: {stats['availability_pct']}%")
    print(f" Occupancy Level       : {stats['level']}")
    print("--------------------------------------------------")
    
    print("\n[INFO] Detailed Status Per Workstation:")
    for pred in predictions[:8]: # Display first 8 workstations
        print(f"  {pred['id']} -> {pred['status']} (Conf: {pred['confidence']}%)")
    print(f"  ... ({len(predictions)-8} more workstations)")

if __name__ == '__main__':
    main()
