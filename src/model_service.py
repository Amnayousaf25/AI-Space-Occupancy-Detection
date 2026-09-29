import os
import tensorflow as tf
from src.config import MODEL_PATH
from src.model_registry import get_active_model_path
from src.logging_config import logger

class ModelService:
    _instance = None
    _model = None
    _loaded_model_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        active_path = get_active_model_path()
        if not os.path.exists(active_path):
            logger.error("CNN model file not found at: %s", active_path)
            active_path = MODEL_PATH
            if not os.path.exists(active_path):
                raise FileNotFoundError(f"Model file not found at: {active_path}")
        logger.info("Loading CNN model weights from: %s", active_path)
        self._model = tf.keras.models.load_model(active_path)
        self._loaded_model_path = active_path
        logger.info("CNN Model successfully loaded and ready for inference!")

    def reload_model(self):
        """Force reloading active model from registry."""
        self._load_model()

    def predict_batch(self, batch_tensor):
        """Run batch inference on preprocessed tensor array."""
        active_path = get_active_model_path()
        if self._model is None or self._loaded_model_path != active_path:
            self._load_model()
        return self._model.predict(batch_tensor, verbose=0)

def get_model_service():
    return ModelService()
