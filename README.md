# JUW SMART SPACE: Multi-Space AI Occupancy & Availability Intelligence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://ultralytics.com/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-blue.svg)](https://sqlite.org/)
[![Multi-Space AI](https://img.shields.io/badge/Spaces-Lab%20%7C%20Lecture%20%7C%20Auditorium-blueviolet.svg)](#-multi-space-ai-occupancy-detection)

## 📌 Executive Overview

**JUW SMART SPACE** is an enterprise-grade AI computer vision platform developed for the **Department of Computer Science & Software Engineering at Jinnah University for Women (JUW), Karachi**.

The platform provides real, automated AI occupancy and seat availability detection across three distinct campus spaces:
1. **Computer Laboratory**: 20 computer workstations (`PC-01` to `PC-20`) with dual-model support (YOLOv8 & Custom CNN baseline).
2. **Lecture Room**: 30 lecture desks (`LR-01` to `LR-30`) with YOLOv8 person localization and spatial desk association.
3. **Auditorium**: 50 tiered theater seats (`AUD-01` to `AUD-50`) with YOLOv8 attendee detection and row-wise seating matching.

```mermaid
graph TD
    A[Space Camera / Image Input] --> B{Space Selector}
    B -->|Computer Lab| C[Lab Config: 20 Workstations PC-01..20]
    B -->|Lecture Room| D[Lecture Config: 30 Desks LR-01..30]
    B -->|Auditorium| E[Auditorium Config: 50 Seats AUD-01..50]
    C --> F{AI Detection Pipeline}
    D --> F
    E --> F
    F -->|YOLOv8 Full-Scene| G[Person / Object Detection]
    F -->|CNN Baseline (Lab Only)| H[Workstation Crop Classifier]
    G --> I[1-to-1 Spatial Seat Association]
    H --> J[Occupancy Aggregator]
    I --> J
    J --> K[(SQLite Database Persistence)]
    K --> L[Space-Filtered Analytics]
    K --> M[FastAPI REST API: /occupancy/analyze]
    K --> N[Streamlit Interactive Multi-Space Dashboard]
```

---

## 🏛️ Multi-Space AI Occupancy Architecture

| Space | Capacity | Seating Layout | Supported Models | Spatial Association Rules |
| :--- | :---: | :--- | :--- | :--- |
| **Computer Laboratory** | 20 Workstations | 4 rows × 5 columns (`PC-01`..`PC-20`) | **YOLOv8** (Verified) + **Custom CNN** | IoA ≥ 0.20 + centroid containment; 1 person = at most 1 PC; 0 persons = 0 occupied |
| **Lecture Room** | 30 Desks | 5 rows × 6 columns (`LR-01`..`LR-30`) | **YOLOv8** (Active) | Centroid & bounding box intersection; 1 person = at most 1 desk; 0 persons = 0 occupied |
| **Auditorium** | 50 Seats | 5 tiered rows × 10 seats (`AUD-01`..`AUD-50`) | **YOLOv8** (Active) | Tiered row spatial matching; 1 attendee = at most 1 seat; 0 persons = 0 occupied |

> **Strict Non-Fabrication Rule**: The system strictly computes `occupied + available == total_seats`. Chairs, desks, monitors, and laptops never independently trigger occupancy. Zero detected persons unconditionally results in zero occupied seats.

---

## ⚙️ Space Configuration Architecture

Each space's seating geometry is declaratively configured in human-readable YAML files:
* [`configs/computer_lab.yaml`](configs/computer_lab.yaml)
* [`configs/lecture_room.yaml`](configs/lecture_room.yaml)
* [`configs/auditorium.yaml`](configs/auditorium.yaml)

Loaded automatically via `src/spaces_config.py`, eliminating hardcoded coordinates across the codebase.

---

## 📁 System Repository Structure

```
JUW SMART SPACE/
│
├── configs/                       # Declarative Seating Geometry Configurations
│   ├── computer_lab.yaml          # 20 Workstations layout
│   ├── lecture_room.yaml          # 30 Lecture desks layout
│   └── auditorium.yaml            # 50 Tiered auditorium seats layout
│
├── api/                           # FastAPI REST Service Layer
│   ├── __init__.py
│   ├── main.py                    # Server entrypoint & OpenAPI specs
│   └── routes/
│       ├── prediction.py          # POST /predict (Crop classifier)
│       ├── occupancy.py           # POST /occupancy/analyze, GET /occupancy/spaces, GET /current
│       ├── analytics.py           # GET /analytics/summary?space=..., GET /alerts
│       └── models_dataset.py      # Model registry and health checks
│
├── src/                           # Core Business & Domain Service Layer
│   ├── __init__.py
│   ├── config.py                  # Centralized system configurations & thresholds
│   ├── logging_config.py          # Structured logging configuration
│   ├── dataset_manager.py         # Data ingestion, SHA256 hashing, leakage protection
│   ├── dataset_audit.py           # Automated dataset audit & integrity engine
│   ├── model_registry.py          # Model versioning, metadata registry & active switcher
│   ├── external_validation.py     # Unseen Real JUW external validation evaluation
│   ├── train_model.py             # Model retraining & fine-tuning pipeline
│   ├── annotation_manager.py      # Bounding box ROI cropping for lab photos
│   ├── database.py                # SQLite database persistence engine
│   ├── model_service.py           # CNN Model loading singleton service
│   ├── prediction_service.py     # Confidence-aware classification service
│   ├── occupancy_engine.py        # Workstation space processor & aggregator
│   ├── seat_counter.py            # Grid-based ROI extractor
│   ├── analytics.py               # Utilization analytics calculator
│   ├── alerts.py                  # Business alert rules (High Occupancy / Review)
│   └── utils.py                   # Preprocessing & geometry helpers
│
├── models/
│   ├── seat_occupancy_cnn.keras   # Baseline trained CNN model weights
│   ├── model_registry.json        # Centralized JSON Model Registry
│   └── v1.1_juw_finetuned/        # Fine-tuned model directory
│
├── tests/                         # Automated Pytest Suite (14 Tests)
│   ├── test_prediction.py
│   ├── test_occupancy.py
│   ├── test_dataset_manager.py
│   ├── test_model_registry.py
│   ├── test_external_validation.py
│   └── test_api.py
│
├── dataset/                       # Dataset Governance Structure
│   ├── raw/juw_real/              # Real JUW training photos (empty / occupied)
│   ├── processed/juw_real/        # Processed real JUW photos
│   ├── external_validation/       # Unseen Real JUW external validation test set
│   ├── metadata/                  # dataset_manifest.csv & dataset_audit.json
│   ├── train/                     # Controlled training dataset
│   ├── validation/                # Controlled validation dataset
│   └── test/                      # Controlled test dataset
│
├── results/                       # Empirical Evaluation Artifacts
│   ├── plots/training_history.png
│   ├── confusion_matrix.png
│   └── juw_external_validation_report.json
│
├── app.py                         # Production Streamlit Interactive Dashboard
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container deployment specification
├── MODEL_CARD.md                  # Industry-standard model card
└── PROJECT_REPORT.md              # Academic & technical project report
```

---

## ⚡ Setup & Execution Manual

### 1. Environment Setup
```powershell
# Activate Virtual Environment
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Dashboard
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
*Access dashboard at `http://localhost:8501`.*

### 3. Launch FastAPI REST Backend Server
```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```
*Access interactive Swagger API documentation at `http://localhost:8000/docs`.*

### 4. Run Automated Dataset Audit
```powershell
.\venv\Scripts\python.exe src/dataset_audit.py
```

### 5. Run Automated Pytest Suite (17 Tests)
```powershell
$env:OPENBLAS_NUM_THREADS="1" ; $env:OMP_NUM_THREADS="1" ; .\venv\Scripts\python.exe -m pytest tests/ -v
```

### 6. Run Comprehensive YOLO vs CNN Benchmark Evaluation
```powershell
.\venv\Scripts\python.exe src/evaluate_yolo.py
```
*Generates comparison report (`results/yolo_evaluation_report.json`) and high-resolution chart (`results/plots/model_comparison.png`).*

---

## 🔒 Privacy & Security Statement

> **Privacy Notice**: JUW SMART SPACE analyzes workstation space occupancy ONLY. The system is designed to avoid identifying individuals. No facial recognition, identity matching, or individual tracking is implemented.
#   A I - S p a c e - O c c u p a n c y - D e t e c t i o n  
 