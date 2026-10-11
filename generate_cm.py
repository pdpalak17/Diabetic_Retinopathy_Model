import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from clinical_risk_model import DiabeticRetinopathyClinicalModel, FEATURE_NAMES

def main():
    csv_path = os.path.join("data", "clinical_dataset.csv")
    if not os.path.exists(csv_path):
        from dataset_generator import generate_sample_dataset
        _, csv_path = generate_sample_dataset()

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = df["dr_risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = DiabeticRetinopathyClinicalModel()
    # Fit strictly on the training split to avoid data leakage on held-out test data
    model.fit_arrays(X_train, y_train, save=False)

    X_test_scaled = model.scaler.transform(X_test)
    y_pred = model.clf.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    print("Clinical Risk Model Confusion Matrix:")
    print(cm)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High', 'Severe'],
                yticklabels=['Low', 'Moderate', 'High', 'Severe'])
    plt.xlabel('Predicted Risk Level')
    plt.ylabel('True Risk Level')
    plt.title('Clinical Risk Model Confusion Matrix')
    
    # Ensure artifacts directory exists or save in root
    out_path = "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix plot to {out_path}")

if __name__ == "__main__":
    main()
