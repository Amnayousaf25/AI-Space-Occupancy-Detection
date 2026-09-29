import os
import sys
import warnings

# Suppress verbose TensorFlow and oneDNN runtime logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

DATASET_RAW_DIR = os.path.join(DATASET_DIR, "raw")
DATASET_PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
DATASET_EXTERNAL_VAL_DIR = os.path.join(DATASET_DIR, "external_validation")
DATASET_METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
DATASET_MANIFEST_PATH = os.path.join(DATASET_METADATA_DIR, "dataset_manifest.csv")
DATASET_AUDIT_PATH = os.path.join(DATASET_METADATA_DIR, "dataset_audit.json")

MODEL_REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "seat_occupancy_cnn.keras"))
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3

# Prediction & Confidence Settings
DEFAULT_PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.50"))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.80"))
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.60"))

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "data", "juw_smart_space.db"))

# Business Alert Rules
OCCUPANCY_HIGH_THRESHOLD = float(os.getenv("OCCUPANCY_HIGH_THRESHOLD", "90.0"))
OCCUPANCY_LEVEL_LOW = float(os.getenv("OCCUPANCY_LEVEL_LOW", "30.0"))
OCCUPANCY_LEVEL_MEDIUM = float(os.getenv("OCCUPANCY_LEVEL_MEDIUM", "70.0"))
OCCUPANCY_THRESHOLDS = {
    "LOW": 30,
    "MEDIUM": 70,
    "HIGH": 90
}

# Class Mappings
CLASS_NAMES = ['EMPTY', 'OCCUPIED']
CLASS_MAP = {0: 'EMPTY', 1: 'OCCUPIED'}
