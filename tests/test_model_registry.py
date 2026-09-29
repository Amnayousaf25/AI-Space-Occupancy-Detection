import pytest
from src.model_registry import (
    get_model_registry,
    get_active_model_info,
    get_active_model_path,
    register_model
)

def test_model_registry_operations():
    registry = get_model_registry()
    assert "active_version" in registry
    assert "models" in registry
    
    active_info = get_active_model_info()
    assert "version" in active_info
    assert active_info["version"] == registry["active_version"]

def test_register_new_model():
    metrics = {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1_score": 0.95}
    registered = register_model(
        version_name="test_v1.2",
        rel_file_path="models/test_v1.2/model.keras",
        dataset_type="Test Dataset",
        metrics=metrics,
        notes="Automated test registration"
    )
    assert registered["version"] == "test_v1.2"
    assert registered["metrics"]["accuracy"] == 0.95
