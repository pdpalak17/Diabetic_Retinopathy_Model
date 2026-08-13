import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from cv_model import DiabeticRetinopathyCVModel, extract_visual_features, MODEL_DIR
from clinical_risk_model import DiabeticRetinopathyClinicalModel, FEATURE_NAMES
from dataset_generator import generate_sample_dataset

KAGGLE_DATASET_DIR = r"C:\Users\Cycle\.cache\kagglehub\datasets\falahgatea\eye-diseases-classification\versions\1\eye_diseases_classification"
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_images")

def load_kaggle_dataset(dataset_dir: str, samples_per_class: int = 250):
    """
    Loads images from Kaggle Eye Diseases dataset folders, extracts visual features, and builds feature matrices.
    """
    classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
    X, y = [], []
    sample_files_map = {}

    print(f"\n[Dataset Loader] Loading Kaggle Eye Diseases dataset from {dataset_dir}...")
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder {class_folder} not found.")
            continue

        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f" -> Found {len(image_files)} images in class '{class_name}'. Processing top {samples_per_class}...")

        # Copy sample image for UI dashboard
        if image_files:
            sample_src = os.path.join(class_folder, image_files[0])
            sample_dst = os.path.join(SAMPLE_IMAGES_DIR, f"kaggle_{class_name}.jpg")
            shutil.copy(sample_src, sample_dst)
            sample_files_map[class_name] = sample_dst

        selected_files = image_files[:samples_per_class]
        for fname in selected_files:
            img_path = os.path.join(class_folder, fname)
            try:
                with Image.open(img_path) as img:
                    feats = extract_visual_features(img)
                    X.append(feats)
                    y.append(class_idx)
            except Exception as e:
                continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[Dataset Loader] Loaded total {len(X)} samples across {len(classes)} classes.")
    return X, y, classes, sample_files_map

def train_and_evaluate_all():
    print("=" * 70)
    print("     DIABETIC RETINOPATHY & EYE DISEASES MODEL TRAINING (KAGGLE)     ")
    print("=" * 70)

    # 1. Dataset Generation for Tabular Clinical Data
    img_paths_synth, csv_path = generate_sample_dataset()

    # 2. Load & Train CV Model on Kaggle Dataset
    if os.path.exists(KAGGLE_DATASET_DIR):
        X_kaggle, y_kaggle, class_names, sample_map = load_kaggle_dataset(KAGGLE_DATASET_DIR, samples_per_class=250)
        
        # Train / Test split on Kaggle dataset (80% train, 20% test)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
            X_kaggle, y_kaggle, test_size=0.20, random_state=42, stratify=y_kaggle
        )

        print(f"\n[1/2] Training CV Model on Kaggle dataset ({len(X_train_cv)} train samples)...")
        cv_model = DiabeticRetinopathyCVModel()
        cv_model.train_on_kaggle_dataset(X_train_cv, y_train_cv, class_names=class_names)

        # Test evaluation on Kaggle test set
        X_scaled_test = cv_model.scaler.transform(X_test_cv)
        y_pred_cv = cv_model.clf.predict(X_scaled_test)
        y_proba_cv = cv_model.clf.predict_proba(X_scaled_test)

        cv_acc = accuracy_score(y_test_cv, y_pred_cv)
        cv_prec = precision_score(y_test_cv, y_pred_cv, average="weighted")
        cv_rec = recall_score(y_test_cv, y_pred_cv, average="weighted")
        cv_f1 = f1_score(y_test_cv, y_pred_cv, average="weighted")
        cv_auc = roc_auc_score(pd.get_dummies(y_test_cv), y_proba_cv, multi_class="ovr")

        print("\n" + "=" * 70)
        print("         KAGGLE EYE DISEASES CV EVALUATION METRICS (TEST SET)         ")
        print("=" * 70)
        print(f"  Test Accuracy:  {cv_acc * 100:.2f}%")
        print(f"  Test Precision: {cv_prec:.4f}")
        print(f"  Test Recall:    {cv_rec:.4f}")
        print(f"  Test F1-Score:  {cv_f1:.4f}")
        print(f"  Test ROC-AUC:   {cv_auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_cv, y_pred_cv))
        print("\nClassification Report:")
        print(classification_report(y_test_cv, y_pred_cv, target_names=class_names))
    else:
        print("Kaggle dataset directory not found, fallback training...")
        cv_model = DiabeticRetinopathyCVModel()
        cv_model.train_on_samples(img_paths_synth)

    # 3. Train Clinical Risk ML Model
    print("\n[2/2] Training Clinical Risk ML Model...")
    clinical_model = DiabeticRetinopathyClinicalModel()
    clinical_model.fit(csv_path)

    df = pd.read_csv(csv_path)
    X_clin = df[FEATURE_NAMES]
    y_clin = df["dr_risk_label"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clin, y_clin, test_size=0.25, random_state=42, stratify=y_clin)
    
    test_scaled_c = clinical_model.scaler.transform(X_test_c)
    y_pred_c = clinical_model.clf.predict(test_scaled_c)
    y_proba_c = clinical_model.clf.predict_proba(test_scaled_c)

    c_acc = accuracy_score(y_test_c, y_pred_c)
    c_f1 = f1_score(y_test_c, y_pred_c, average="weighted")
    c_auc = roc_auc_score(pd.get_dummies(y_test_c), y_proba_c, multi_class="ovr")

    print("\n" + "=" * 70)
    print("               CLINICAL RISK ML EVALUATION METRICS               ")
    print("=" * 70)
    print(f"  Accuracy: {c_acc * 100:.2f}% | F1-Score: {c_f1:.4f} | ROC-AUC: {c_auc:.4f}")

    print("=" * 70)
    print("  [SUCCESS] All Models Successfully Trained on Kaggle Dataset!   ")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_all()
