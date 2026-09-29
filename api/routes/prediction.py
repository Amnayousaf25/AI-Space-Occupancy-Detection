import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.prediction_service import classify_workstation_crop

router = APIRouter(prefix="/predict", tags=["Prediction"])

class SinglePredictionResponse(BaseModel):
    workstation_id: str
    prediction: str
    probability: float
    confidence: float
    confidence_level: str
    review_required: bool
    threshold: float
    timestamp: str

@router.post("", response_model=SinglePredictionResponse)
async def predict_crop(file: UploadFile = File(...), workstation_id: str = Form("PC-01"), threshold: float = Form(0.50)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file provided")
            
        result = classify_workstation_crop(img, workstation_id=workstation_id, threshold=threshold)
        return SinglePredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
