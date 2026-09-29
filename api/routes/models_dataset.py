from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
from src.model_registry import (
    get_model_registry,
    get_active_model_info,
    set_active_model,
)
from src.dataset_audit import run_dataset_audit
from src.external_validation import evaluate_external_juw_validation

router = APIRouter(prefix="", tags=["Models & Dataset Governance"])

class ActiveModelSwitchRequest(BaseModel):
    model_version: str

@router.get("/models", summary="List registered model versions")
def list_models():
    """Retrieve model registry containing baseline and fine-tuned models."""
    return get_model_registry()

@router.get("/models/active", summary="Get currently active CNN model details")
def get_active_model():
    """Return metrics and configuration of the active CNN model."""
    return get_active_model_info()

@router.post("/models/active", summary="Switch active CNN model version")
def switch_active_model(payload: ActiveModelSwitchRequest):
    """Switch system inference engine to specified registered model version."""
    try:
        updated = set_active_model(payload.model_version)
        return {"status": "success", "active_model": updated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset/health", summary="Get dataset audit & data leakage analysis")
def get_dataset_health():
    """Run real-time audit across dataset splits and verify data leakage status."""
    return run_dataset_audit()

@router.get("/validation/summary", summary="Get Real JUW External Validation evaluation metrics")
def get_validation_summary():
    """Evaluate active model on unseen Real JUW External Validation dataset."""
    return evaluate_external_juw_validation()
