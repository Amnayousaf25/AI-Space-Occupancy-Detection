import os
import sys
import warnings

# Suppress verbose TensorFlow & oneDNN runtime warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.config import MODEL_PATH
from api.routes import prediction, occupancy, analytics, models_dataset

app = FastAPI(
    title="JUW SMART SPACE API",
    description="RESTful API for AI-Based Smart Space Computer Lab Occupancy & Utilization Detection - Jinnah University for Women (JUW)",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(prediction.router)
app.include_router(occupancy.router)
app.include_router(analytics.router)
app.include_router(models_dataset.router)


@app.get("/", tags=["Root"])
def root():
    """
    Root entrypoint providing service status, campus context, and interactive links.
    """
    return {
        "title": "JUW SMART SPACE: AI-Powered Computer Lab Occupancy API",
        "institution": "Jinnah University for Women (JUW), Karachi",
        "department": "Department of Computer Science & Software Engineering",
        "status": "online",
        "interactive_docs": "/docs",
        "redoc": "/redoc",
        "health_check": "/health",
        "dashboard_ui": "http://localhost:8501",
        "endpoints": {
            "predict_single_crop": "POST /predict",
            "analyze_lab_occupancy": "POST /occupancy/analyze-lab",
            "analyze_lab_yolo": "POST /occupancy/analyze-lab-yolo",
            "current_occupancy": "GET /occupancy/current",
            "occupancy_history": "GET /occupancy/history",
            "analytics_summary": "GET /analytics/summary",
            "active_model": "GET /models/active",
            "dataset_health": "GET /dataset/health"
        }
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Suppresses browser 404s for favicon."""
    return Response(status_code=204)


@app.get("/health", tags=["Health"])
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "privacy_notice": "Workstation occupancy detection ONLY. No facial recognition or individual identity tracking."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
