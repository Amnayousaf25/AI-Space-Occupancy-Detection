import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt

try:
    from src.utils import (
        IMG_WIDTH, IMG_HEIGHT, CHANNELS, BATCH_SIZE, EPOCHS, LEARNING_RATE,
        DATASET_DIR, MODELS_DIR, PLOTS_DIR, MODEL_PATH
    )
except ImportError:
    from utils import (
        IMG_WIDTH, IMG_HEIGHT, CHANNELS, BATCH_SIZE, EPOCHS, LEARNING_RATE,
        DATASET_DIR, MODELS_DIR, PLOTS_DIR, MODEL_PATH
    )

def build_cnn_model():
    """
    Build CNN architecture according to university ANN course specifications.
    
    Input Image (128x128x3)
    ↓
    Data Augmentation (Flip, Rotation, Zoom, Brightness)
    ↓
    Rescaling (1./255 Normalization)
    ↓
    Conv2D (32, 3x3, ReLU) -> MaxPooling2D (2x2)
    ↓
    Conv2D (64, 3x3, ReLU) -> MaxPooling2D (2x2)
    ↓
    Conv2D (128, 3x3, ReLU) -> MaxPooling2D (2x2)
    ↓
    Flatten
    ↓
    Dense (128, ReLU)
    ↓
    Dropout (0.5)
    ↓
    Output Dense (1, Sigmoid)
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # Data Augmentation Pipeline
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomBrightness(factor=0.1),
        
        # Normalization Layer
        layers.Rescaling(1./255),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten Feature Maps into 1D Vector
        layers.Flatten(name='flatten'),
        
        # Fully Connected (Dense) Layer
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        
        # Binary Output Layer
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_datasets():
    """Load train, validation, and test datasets using image_dataset_from_directory."""
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'validation')
    test_dir = os.path.join(DATASET_DIR, 'test')
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )
    
    # Prefetch for optimal GPU/CPU throughput
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def plot_training_history(history):
    """Plot and save training & validation accuracy and loss curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-o', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'g-s', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-o', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-s', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training history plot saved to: {save_path}")

def train():
    print("==================================================")
    print(" JUW SMART SPACE - CNN MODEL TRAINING")
    print("==================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    train_ds, val_ds, test_ds = load_datasets()
    
    model = build_cnn_model()
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )
    
    # Save training curves
    plot_training_history(history)
    print(f"\n[SUCCESS] Best CNN model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train()
