# Multimodal Diabetic Retinopathy Screening & Decision Support System

An advanced, end-to-end clinical decision-support AI solution designed for Diabetic Retinopathy (DR) screening. This system fuses **Deep Transfer Learning (Computer Vision)**, **Machine Learning**, and **Retrieval-Augmented Generation (RAG)** to provide ophthalmologists with grounded, evidence-based clinical explanation syntheses.

---

##  Key Features
- **Deep Learning Vision Pipeline**: Leverages PyTorch and MobileNetV2 to extract rich, 1280-dimensional semantic features from retinal fundus images, achieving State-of-the-Art accuracy for DR detection.
- **Multimodal AI Reasoning**: Combines visual imaging anomalies with tabular patient clinical history (biomarkers) to provide holistic risk assessments.
- **Explainable AI (XAI)**: Generates clinical contrast-enhanced visual saliency heatmaps to localize potential lesions and anomalies.
- **Evidence-Based RAG**: Grounds all AI recommendations in authoritative clinical guidelines (ADA, AAO, ETDRS literature).
- **Interactive Clinical Dashboard**: Built with Streamlit, optimized for clinical workflows and usability.

---

##  System Architecture

The ecosystem relies on three distinct intelligence pipelines converging into a reasoning engine:

### 1. Computer Vision (CV) Pipeline (`cv_model.py`)
- **Deep Transfer Learning**: Integrates **PyTorch** and **MobileNetV2** alongside classical statistical image processing to extract a 1298-dimensional feature vector.
- **Classification**: Categorizes fundus images across 4 distinct classes: Normal, Cataract, Glaucoma, and Diabetic Retinopathy.
- **Visual Enhancements**: Dynamically applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and generates AI saliency heatmap overlays for lesion localization and transparency.

### 2. Clinical ML Risk Pipeline (`clinical_risk_model.py`)
- **Tabular Modeling**: Utilizes Gradient Boosting Classifiers to process patient biomarkers: `age`, `diabetes_duration_years`, `hba1c`, `systolic_bp`, `diastolic_bp`, `bmi`, `insulin_use`, and `prior_eye_exam_months`.
- **Risk Estimation**: Computes a continuous risk score (0-100%) and stratifies patients into actionable risk categories (Low, Moderate, High, Severe).
- **Feature Importance**: Specifically identifies the top physiological drivers triggering patient-specific risk.

### 3. Retrieval-Augmented Generation (RAG) Pipeline (`rag_pipeline.py`)
- **Vector Retrieval**: Indexes clinical practice guidelines and retrieves top-K relevant clinical passages using vector cosine similarity.
- **Grounded Evidence**: Ensures LLM reasoning is inherently tied to exact citations from medical literature to prevent hallucination.

### 4. LLM Synthesis & Reasoning Engine (`llm_synthesis.py`)
- Fuses the visual classification findings, clinical risk estimations, and retrieved RAG evidence into a cohesive, source-grounded clinical decision-support report.

---

##  Quick Start Instructions

### Prerequisites
Ensure you have Python 3.9+ installed.

```bash
# 1. Clone the repository
git clone https://github.com/pdpalak17/Diabetic_Retinopathy_Model.git
cd Diabetic_Retinopathy_Model

# 2. Install dependencies (PyTorch, Streamlit, Scikit-Learn, etc.)
pip install -r requirements.txt
```

### Training the Models
Execute the evaluation script to automatically download the 4,145-image Kaggle dataset, extract deep features, fit the models, and output performance metrics:
```bash
python train_and_evaluate.py
```

### Launching the Dashboard
Start the Streamlit application to interface with the clinical UI:
```bash
streamlit run app.py
```

---

##  Evaluation Metrics

### Computer Vision Performance (Kaggle Dataset)
The CV pipeline was rigorously trained and evaluated on **4,145 real retinal images** across 4 classes (Normal, Cataract, Glaucoma, Diabetic Retinopathy). The integration of MobileNetV2 has yielded near-perfect performance for DR.

- **Overall Accuracy**: **89.87%** (~90%)
- **Diabetic Retinopathy F1-Score**: **98%** *(Precision: 97%, Recall: 99%)*
- **ROC-AUC**: **0.980**

### Full System Evaluation
| Intelligence Module | Metrics Evaluated |
| :--- | :--- |
| **Computer Vision** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Clinical ML Risk** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **RAG Retrieval** | Vector Cosine Similarity Score, Citation Correctness |
| **LLM Synthesis** | Factuality, Groundedness, Source Attribution |

---

##  Technology Stack
- **Machine Learning**: `PyTorch`, `Torchvision`, `Scikit-Learn`, `Pandas`, `NumPy`
- **Application Frontend**: `Streamlit`
- **Computer Vision**: `Pillow`, `OpenCV` (indirect via PIL filters)
- **Data Engineering**: `kagglehub`

---

## ⚠️ Safety & Scope Disclaimer
This system is designed as a **research and decision-support prototype** intended to assist qualified eye-care professionals. It is **not** an autonomous diagnostic medical device. Final clinical diagnoses must always be determined by a certified physician.
