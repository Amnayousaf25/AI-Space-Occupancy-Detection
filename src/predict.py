import sys
import os
import cv2
import numpy as np
import tensorflow as tf

try:
    from src.utils import MODEL_PATH, CLASS_NAMES, preprocess_image_crop
except ImportError:
    from utils import MODEL_PATH, CLASS_NAMES, preprocess_image_crop

def predict_single_image(image_path, model=None):
    """
    Predict occupancy state for a single crop image.
    Returns prediction label ('EMPTY' or 'OCCUPIED') and confidence score.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train model first using src/train.py!")
        model = tf.keras.models.load_model(MODEL_PATH)
        
    # Read Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    normalized = preprocess_image_crop(img_bgr)
    input_tensor = np.expand_dims(normalized, axis=0) # Shape: (1, 128, 128, 3)
    
    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    
    if prob >= 0.5:
        label = 'OCCUPIED'
        confidence = prob * 100.0
    else:
        label = 'EMPTY'
        confidence = (1.0 - prob) * 100.0
        
    return label, confidence, prob

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        print("Example: python src/predict.py dataset/test/occupied/occupied_0001.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    print(f"[INFO] Predicting occupancy for: {image_path}")
    
    label, confidence, raw_prob = predict_single_image(image_path)
    
    print("\n--------------------------------------------------")
    print(f" Prediction : {label}")
    print(f" Confidence : {confidence:.2f}%")
    print(f" Raw Prob   : {raw_prob:.4f}")
    print("--------------------------------------------------")

if __name__ == '__main__':
    main()
