import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import (
    BASE_DIR,
    MODELS_DIR,
    DATASET_DIR,
    DATASET_RAW_DIR,
    IMG_SIZE,
    IMG_WIDTH,
    IMG_HEIGHT,
)
from src.dataset_audit import run_dataset_audit
from src.model_registry import register_model, get_model_registry
from src.logging_config import logger

def train_or_finetune_model(version_name="v1.1_juw_finetuned", epochs=5, batch_size=16):
    """
    Retrain/Fine-tune CNN model on updated dataset (including real JUW photos).
    Performs data leakage check first, trains model, saves weights, and updates model registry.
    """
    logger.info("Initiating dataset audit and leakage check prior to model training...")
    audit = run_dataset_audit()
    
    if audit.get("leakage_analysis", {}).get("leakage_detected", False):
        raise ValueError("Training aborted! Data leakage detected across splits. Resolve leakage before training.")

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "validation")
    juw_raw_dir = os.path.join(DATASET_RAW_DIR, "juw_real")

    # Load dataset using image_dataset_from_directory
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training dataset directory not found at {train_dir}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Base architecture matching baseline CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    save_dir = os.path.join(MODELS_DIR, version_name)
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "seat_occupancy_cnn.keras")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]

    logger.info(f"Starting training run for model version {version_name}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    val_eval = model.evaluate(val_ds, verbose=0)
    acc = float(val_eval[1])
    prec = float(val_eval[2])
    rec = float(val_eval[3])
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    metrics = {
        "dataset_type": "juw_fine_tuned",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }

    rel_path = os.path.relpath(model_save_path, BASE_DIR)
    register_model(
        version_name=version_name,
        rel_file_path=rel_path,
        dataset_type="JUW Fine-Tuned Real Lab Dataset",
        metrics=metrics,
        notes=f"Model fine-tuned with {epochs} epochs on combined lab dataset.",
        make_active=True
    )

    logger.info(f"Successfully completed training for {version_name}. Model saved to {model_save_path}")
    return metrics

if __name__ == "__main__":
    train_or_finetune_model()
