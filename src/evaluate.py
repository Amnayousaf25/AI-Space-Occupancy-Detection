import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

try:
    from src.utils import (
        IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, DATASET_DIR, MODEL_PATH, RESULTS_DIR, CLASS_NAMES
    )
except ImportError:
    from utils import (
        IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, DATASET_DIR, MODEL_PATH, RESULTS_DIR, CLASS_NAMES
    )

def evaluate_model():
    print("==================================================")
    print(" JUW SMART SPACE - CNN MODEL EVALUATION")
    print("==================================================")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run src/train.py first!")
        
    print(f"[INFO] Loading trained CNN model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )
    
    # Collect ground truth labels and predictions
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(probs.flatten())
        
    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate Metrics
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n--------------------------------------------------")
    print(" EMPIRICAL EVALUATION METRICS ON TEST SET")
    print("--------------------------------------------------")
    print(f" Test Loss       : {loss:.4f}")
    print(f" Test Accuracy   : {accuracy * 100:.2f}%")
    print(f" Precision       : {precision * 100:.2f}%")
    print(f" Recall          : {recall * 100:.2f}%")
    print(f" F1-Score        : {f1 * 100:.2f}%")
    print("--------------------------------------------------")
    
    print("\n[INFO] Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # Plot and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, classes):
    """Plot and save clean confusion matrix figure."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Seat Occupancy Detection')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # Format annotations inside grid
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")

if __name__ == '__main__':
    evaluate_model()
