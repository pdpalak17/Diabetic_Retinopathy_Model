import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "age",
    "diabetes_duration_years",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "insulin_use",
    "prior_eye_exam_months"
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLINICAL_MODEL_PATH = os.path.join(MODEL_DIR, "clinical_risk_model.pkl")

RISK_LEVELS = {
    0: {"name": "Low Clinical Risk", "code": "Low", "color": "#10B981"},
    1: {"name": "Moderate Clinical Risk", "code": "Moderate", "color": "#F59E0B"},
    2: {"name": "Elevated Clinical Risk", "code": "High", "color": "#F97316"},
    3: {"name": "Severe Clinical Risk", "code": "Severe", "color": "#EF4444"}
}

class DiabeticRetinopathyClinicalModel:
    """
    Machine Learning Model for Clinical Risk Estimation based on patient biomarkers.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        self.is_fitted = False

    def fit(self, csv_path: str):
        """
        Trains the clinical risk model on the tabular dataset.
        """
        df = pd.read_csv(csv_path)
        X = df[FEATURE_NAMES]
        y = df["dr_risk_label"]

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CLINICAL_MODEL_PATH, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf}, f)
        print(f"[Clinical ML Model] Model trained and saved to {CLINICAL_MODEL_PATH}")

    def load_model(self) -> bool:
        if os.path.exists(CLINICAL_MODEL_PATH):
            with open(CLINICAL_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.scaler = data["scaler"]
                self.clf = data["clf"]
                self.is_fitted = True
            return True
        return False

    def predict(self, patient_dict: dict) -> dict:
        """
        Calculates patient risk score, risk level, feature contributions, and key risk drivers.
        """
        if not self.is_fitted:
            if not self.load_model():
                # Fallback rule-based formula if model is not yet loaded
                hba1c = patient_dict.get("hba1c", 7.5)
                duration = patient_dict.get("diabetes_duration_years", 8.0)
                sys_bp = patient_dict.get("systolic_bp", 130)
                score = (hba1c - 5.0) * 8.5 + duration * 1.5 + (sys_bp - 120) * 0.4
                score = float(np.clip(score, 5.0, 98.0))
                risk_cat = 0 if score < 25 else (1 if score < 50 else (2 if score < 75 else 3))
                risk_probs = [0.25, 0.25, 0.25, 0.25]
                risk_probs[risk_cat] = 0.70
            else:
                input_df = pd.DataFrame([[patient_dict[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)
                X_scaled = self.scaler.transform(input_df)
                risk_cat = int(self.clf.predict(X_scaled)[0])
                probs = self.clf.predict_proba(X_scaled)[0]
                # Calculate weighted risk percentage score
                weights = np.array([12.5, 37.5, 62.5, 87.5])
                score = float(np.sum(probs * weights))
                risk_probs = list(probs)
        else:
            input_df = pd.DataFrame([[patient_dict[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)
            X_scaled = self.scaler.transform(input_df)
            risk_cat = int(self.clf.predict(X_scaled)[0])
            probs = self.clf.predict_proba(X_scaled)[0]
            weights = np.array([12.5, 37.5, 62.5, 87.5])
            score = float(np.sum(probs * weights))
            risk_probs = list(probs)

        risk_info = RISK_LEVELS[risk_cat]

        # Analyze patient-specific risk drivers
        risk_drivers = []
        hba1c = patient_dict.get("hba1c", 7.0)
        if hba1c >= 8.5:
            risk_drivers.append(f"Severely elevated HbA1c ({hba1c}%) increases microvascular damage risk exponentially.")
        elif hba1c >= 7.5:
            risk_drivers.append(f"Elevated HbA1c ({hba1c}%) above the ADA recommended target of 7.0%.")

        sys_bp = patient_dict.get("systolic_bp", 120)
        if sys_bp >= 140:
            risk_drivers.append(f"Hypertension (Systolic BP {sys_bp} mmHg) elevates retinal capillary hydrostatic pressure.")

        duration = patient_dict.get("diabetes_duration_years", 5.0)
        if duration >= 15:
            risk_drivers.append(f"Long diabetes duration ({duration} years) is a primary non-modifiable risk factor for DR progression.")
        elif duration >= 10:
            risk_drivers.append(f"Moderate diabetes duration ({duration} years) requires annual screening protocol.")

        last_exam = patient_dict.get("prior_eye_exam_months", 12)
        if last_exam >= 24:
            risk_drivers.append(f"Extended interval since last eye examination ({last_exam} months) increases risk of unmonitored progression.")

        if not risk_drivers:
            risk_drivers.append("Glycemic control and blood pressure are within low-risk target thresholds.")

        # Feature importance breakdown
        feature_importance = {}
        if hasattr(self.clf, "feature_importances_"):
            for name, imp in zip(FEATURE_NAMES, self.clf.feature_importances_):
                feature_importance[name] = round(float(imp) * 100, 1)

        return {
            "risk_category_code": risk_cat,
            "risk_name": risk_info["name"],
            "risk_code": risk_info["code"],
            "risk_color": risk_info["color"],
            "risk_score_percent": round(score, 1),
            "risk_probabilities": [round(float(p) * 100, 1) for p in risk_probs],
            "risk_drivers": risk_drivers,
            "feature_importance": feature_importance
        }
