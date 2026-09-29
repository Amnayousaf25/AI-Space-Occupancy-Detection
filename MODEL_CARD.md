# Model Card: JUW SMART SPACE Workstation Occupancy Classifier

## 📌 Model Overview
- **Model Name:** JUW Smart Space Workstation Occupancy CNN
- **Active Model Version:** `v1_controlled` (Baseline) / `v1.1_juw_finetuned` (Fine-Tuned)
- **Model Type:** Binary Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input Dimensions:** 128 × 128 × 3 RGB Image Tensor
- **Output:** Class Label (`EMPTY` vs `OCCUPIED`), Sigmoid Probability ($0.0 \dots 1.0$), Confidence Category (`HIGH`, `MEDIUM`, `LOW`, `REVIEW REQUIRED`).

---

## 🎯 Intended Use & Scope
- **Primary Use Case:** Automated detection of computer lab workstation seat occupancy to calculate real-time PC availability.
- **Target Deployment:** Jinnah University for Women (JUW) Computer Science & Software Engineering Labs, Central Library Digital Section, JUBIC Business Incubator.

### 🚫 Non-Intended Uses
- **No Facial Recognition:** The system DOES NOT detect faces, identify students, or match identities.
- **No Individual Behavior Profiling:** The system DOES NOT track student attendance, dwell time, or individual activity.

---

## 📊 Dataset Governance & Leakage Protection
- **Controlled Dataset:** Synthetic/lab workstation crops used for baseline training and architectural validation.
- **Real JUW Dataset:** On-site university lab photographs stored under `dataset/raw/juw_real/`.
- **Real JUW External Validation Set:** Strictly isolated test dataset (`dataset/external_validation/juw_real/`) reserved exclusively for post-training validation. NEVER used during training or hyperparameter tuning.
- **Leakage Prevention:** Enforces SHA256 image hash verification to prevent identical scene samples from appearing across multiple dataset splits.

---

## 📈 Model Performance & Evaluation Metrics

| Model Version | Training Dataset | Evaluation Dataset | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **v1_controlled** | Controlled Lab Dataset | Controlled Test Set | 100.0% | 100.0% | 100.0% | 100.0% |
| **v1.1_juw_finetuned** | JUW Real + Controlled | JUW External Val | *Evaluated upon dataset capture* | *Pending* | *Pending* | *Pending* |

> ⚠️ **Academic Disclaimer:** High performance ($100\%$) recorded on the controlled dataset reflects controlled lighting and consistent camera orientation. Stronger real-world external validation requires ongoing data collection across JUW campus facilities.

---

## ⚙️ Confidence-Aware Decision Policy
- **HIGH CONFIDENCE ($\ge 80\%$):** Automated occupancy logging.
- **MEDIUM CONFIDENCE ($60\% - 79\%$):** Standard processing.
- **LOW CONFIDENCE ($< 60\%$):** Output flagged with **REVIEW REQUIRED** badge for administrative verification.

---

## 🔒 Privacy & Ethical Considerations
1. Images are processed at workstation ROI level to minimize person capturing.
2. No personal identifiable information (PII) or biometric data is stored in the database.
3. Database persistence stores workstation IDs, occupancy counts, timestamps, and probabilities only.
