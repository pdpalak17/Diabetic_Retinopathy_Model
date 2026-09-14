# Multimodal Diabetic Retinopathy Screening & Decision Support System

An end-to-end clinical decision-support AI solution for Diabetic Retinopathy (DR) screening, integrating Computer Vision (CV), Machine Learning (ML), Retrieval-Augmented Generation (RAG), and grounded clinical explanation synthesis.

---

##  Architecture Overview

The system consists of three intelligence pipelines and a reasoning engine:

1. **Computer Vision (CV) Pipeline (`cv_model.py`)**:
   - Analyzes color fundus photographs using **Deep Transfer Learning**.
   - Integrates **PyTorch** and **MobileNetV2** alongside classical image processing to extract a rich 1298-dimensional semantic and statistical feature vector.
   - Classifies eye diseases (e.g., Cataract, Diabetic Retinopathy, Glaucoma, Normal).
   - Generates CLAHE contrast-enhanced views and AI saliency heatmap overlays for lesion localization.

2. **Clinical ML Risk Pipeline (`clinical_risk_model.py`)**:
   - Tabular model (Gradient Boosting / Random Forest) trained on patient biomarkers: `age`, `diabetes_duration_years`, `hba1c`, `systolic_bp`, `diastolic_bp`, `bmi`, `insulin_use`, `prior_eye_exam_months`.
   - Computes continuous risk score (0-100%) and risk category (Low, Moderate, High, Severe).
   - Identifies patient-specific risk drivers and feature importance.

3. **Retrieval-Augmented Generation (RAG) Pipeline (`rag_pipeline.py`)**:
   - Vector store retriever indexing clinical practice guidelines (ADA Standards of Care, AAO Preferred Practice Patterns, ICO Guidelines, ETDRS literature).
   - Retrieves top-K relevant clinical passages with similarity scores and exact citations.

4. **LLM Synthesis & Reasoning Engine (`llm_synthesis.py`)**:
   - Fuses visual classification, clinical risk estimation, and retrieved RAG evidence into source-grounded clinical decision-support reports.

5. **Clinical Dashboard (`app.py`)**:
   - Interactive Streamlit application designed according to clinical UX principles.

---

##  Quick Start Instructions

### 1. Train and Evaluate Models
Execute the training script to generate synthetic datasets, fit both CV and Clinical ML models, and output performance metrics:
```bash
python train_and_evaluate.py
```

### 2. Launch Clinical Web Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

---

##  Evaluation Metrics

### Computer Vision Performance (Kaggle Dataset)
The CV pipeline was rigorously trained and evaluated on **4,145 real retinal images** across 4 classes (Normal, Cataract, Glaucoma, Diabetic Retinopathy).

- **Overall Accuracy**: ~90% (89.87%)
- **Diabetic Retinopathy F1-Score**: 98%
- **ROC-AUC**: 0.98

### Full System Evaluation
| Module | Metrics Evaluated |
| :--- | :--- |
| **Computer Vision** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Clinical ML Risk** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **RAG Retrieval** | Vector Cosine Similarity Score, Citation Correctness |
| **LLM Synthesis** | Factuality, Groundedness, Source Attribution |

---

##  Safety & Scope Disclaimer
This system is designed as a research and decision-support prototype to assist qualified eye-care professionals and is not an autonomous diagnostic device.
