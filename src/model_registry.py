import os
import json
from src.config import (
    BASE_DIR,
    MODELS_DIR,
    MODEL_REGISTRY_PATH,
    MODEL_PATH,
)
from src.logging_config import logger

DEFAULT_REGISTRY = {
    "active_version": "v1_controlled",
    "models": {
        "v1_controlled": {
            "version": "v1_controlled",
            "name": "CNN Baseline (Controlled)",
            "file_path": os.path.relpath(MODEL_PATH, BASE_DIR),
            "training_dataset": "Controlled University Lab Dataset",
            "input_shape": [128, 128, 3],
            "status": "active",
            "created_at": "2026-09-14",
            "metrics": {
                "dataset_type": "controlled_test",
                "accuracy": 1.00,
                "precision": 1.00,
                "recall": 1.00,
                "f1_score": 1.00
            },
            "notes": "Original baseline CNN trained on controlled university lab dataset."
        }
    }
}

def initialize_model_registry():
    """Initialize model registry file if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_REGISTRY, f, indent=2)
        logger.info(f"Initialized model registry at {MODEL_REGISTRY_PATH}")

def get_model_registry():
    """Load model registry dictionary."""
    initialize_model_registry()
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading model registry: {e}")
        return DEFAULT_REGISTRY

def save_model_registry(registry_data):
    """Save model registry dictionary."""
    initialize_model_registry()
    with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

def get_active_model_info():
    """Retrieve details of the currently active model."""
    registry = get_model_registry()
    active_version = registry.get("active_version", "v1_controlled")
    models = registry.get("models", {})
    if active_version in models:
        return models[active_version]
    return DEFAULT_REGISTRY["models"]["v1_controlled"]

def get_active_model_path():
    """Retrieve absolute file path of the currently active model."""
    info = get_active_model_info()
    rel_path = info.get("file_path", "models/seat_occupancy_cnn.keras")
    abs_path = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(abs_path):
        return abs_path
    # Fallback to default MODEL_PATH
    return MODEL_PATH

def set_active_model(version_name):
    """Set active model version if model exists."""
    registry = get_model_registry()
    models = registry.get("models", {})
    if version_name not in models:
        raise ValueError(f"Model version '{version_name}' not found in registry.")
    
    target_info = models[version_name]
    abs_path = os.path.join(BASE_DIR, target_info.get("file_path", ""))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file for '{version_name}' does not exist at {abs_path}")

    # Set previous active model status
    for key in models:
        if models[key].get("status") == "active":
            models[key]["status"] = "available"
            
    models[version_name]["status"] = "active"
    registry["active_version"] = version_name
    save_model_registry(registry)
    logger.info(f"Active model switched to {version_name}")
    return models[version_name]

def register_model(version_name, rel_file_path, dataset_type, metrics, notes="", make_active=False):
    """Register or update a model entry in the model registry."""
    registry = get_model_registry()
    models = registry.get("models", {})
    
    models[version_name] = {
        "version": version_name,
        "name": f"CNN ({version_name})",
        "file_path": rel_file_path,
        "training_dataset": dataset_type,
        "input_shape": [128, 128, 3],
        "status": "available",
        "created_at": "2026-09-14",
        "metrics": metrics,
        "notes": notes
    }
    registry["models"] = models
    save_model_registry(registry)
    
    if make_active:
        set_active_model(version_name)
    logger.info(f"Registered model {version_name} in model registry.")
    return models[version_name]
