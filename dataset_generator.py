import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")

def create_synthetic_fundus(severity_class: int, filename: str) -> str:
    """
    Generates a realistic synthetic fundus photograph representing a specific DR severity stage:
    0: No DR
    1: Mild DR (microaneurysms)
    2: Moderate DR (hemorrhages, hard exudates)
    3: Severe DR (extensive hemorrhages, cotton wool spots)
    4: Proliferative DR (neovascularization, preretinal hemorrhage)
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    filepath = os.path.join(IMAGES_DIR, filename)

    img_size = (512, 512)
    center = (256, 256)
    radius = 230

    # Create dark background
    img = Image.new("RGB", img_size, (15, 10, 8))
    draw = ImageDraw.Draw(img)

    # Fundus base circle with radial color gradient (deep red-orange fundus color)
    y, x = np.ogrid[:img_size[1], :img_size[0]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fundus_mask = dist_from_center <= radius

    # Base orange-red fundus background
    fundus_array = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Gradient fundus background simulation
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    r_channel = (180 - norm_dist * 50).astype(np.uint8)
    g_channel = (75 - norm_dist * 35).astype(np.uint8)
    b_channel = (20 - norm_dist * 10).astype(np.uint8)

    fundus_array[:, :, 0] = r_channel
    fundus_array[:, :, 1] = g_channel
    fundus_array[:, :, 2] = b_channel

    # Apply circular mask
    fundus_array[~fundus_mask] = [10, 8, 6]
    img = Image.fromarray(fundus_array)
    draw = ImageDraw.Draw(img)

    # Optic Disc (yellowish-pink oval on nasal side, e.g. x=160, y=256)
    optic_disc_center = (160, 256)
    draw.ellipse(
        [optic_disc_center[0] - 30, optic_disc_center[1] - 35,
         optic_disc_center[0] + 30, optic_disc_center[1] + 35],
        fill=(250, 220, 140), outline=(240, 190, 110), width=2
    )
    # Cup inside disc
    draw.ellipse(
        [optic_disc_center[0] - 12, optic_disc_center[1] - 15,
         optic_disc_center[0] + 12, optic_disc_center[1] + 15],
        fill=(255, 245, 190)
    )

    # Macula (dark oval on temporal side, e.g. x=320, y=256)
    draw.ellipse(
        [300, 240, 340, 272],
        fill=(110, 45, 15)
    )

    # Major Retinal Blood Vessels originating from Optic Disc
    vessel_color = (110, 15, 15)
    np.random.seed(42 + severity_class)
    
    # Superior & Inferior arches
    branches = [
        [(160, 256), (180, 200), (220, 150), (280, 120), (360, 110), (430, 120)],
        [(160, 256), (180, 190), (240, 160), (330, 150), (410, 165)],
        [(160, 256), (180, 310), (220, 360), (280, 390), (360, 400), (430, 390)],
        [(160, 256), (180, 320), (240, 350), (330, 360), (410, 345)],
    ]
    for b in branches:
        draw.line(b, fill=vessel_color, width=4)
        # Small side branches
        for idx in range(len(b)-1):
            p1, p2 = b[idx], b[idx+1]
            if np.random.rand() > 0.4:
                sub_end = (p1[0] + np.random.randint(-30, 30), p1[1] + np.random.randint(-30, 30))
                draw.line([p1, sub_end], fill=vessel_color, width=2)

    # Add lesions based on DR Severity Stage
    if severity_class >= 1:
        # Mild DR: Isolated Microaneurysms (small red dots, 2-4px)
        num_ma = 12 if severity_class == 1 else 35
        for _ in range(num_ma):
            mx = np.random.randint(180, 420)
            my = np.random.randint(120, 390)
            if np.sqrt((mx - 320)**2 + (my - 256)**2) > 30: # avoid inside fovea center
                draw.ellipse([mx-2, my-2, mx+2, my+2], fill=(140, 10, 10))

    if severity_class >= 2:
        # Moderate DR: Blot Hemorrhages & Hard Exudates (bright yellow dots)
        num_hem = 15
        for _ in range(num_hem):
            hx = np.random.randint(170, 430)
            hy = np.random.randint(110, 400)
            hr = np.random.randint(4, 9)
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=(130, 5, 5))

        num_exudates = 20
        for _ in range(num_exudates):
            ex = np.random.randint(220, 380)
            ey = np.random.randint(160, 340)
            er = np.random.randint(2, 6)
            draw.ellipse([ex-er, ey-er, ex+er, ey+er], fill=(245, 235, 140))

    if severity_class >= 3:
        # Severe DR: Large Hemorrhages, Cotton Wool Spots (fluffy white/grey lesions)
        num_cw = 8
        for _ in range(num_cw):
            cx = np.random.randint(190, 410)
            cy = np.random.randint(130, 370)
            cr = np.random.randint(8, 16)
            draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], fill=(225, 220, 210))

        # Extensive blot hemorrhages
        for _ in range(25):
            hx = np.random.randint(150, 440)
            hy = np.random.randint(100, 410)
            hr = np.random.randint(6, 14)
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=(110, 0, 0))

    if severity_class >= 4:
        # Proliferative DR: Neovascularization (tangled thin vessels) & Preretinal Hemorrhage
        for _ in range(6):
            start_x = np.random.randint(160, 300)
            start_y = np.random.randint(180, 320)
            points = [(start_x, start_y)]
            for step in range(5):
                prev = points[-1]
                points.append((prev[0] + np.random.randint(-15, 15), prev[1] + np.random.randint(-15, 15)))
            draw.line(points, fill=(180, 20, 20), width=2)
        
        # Boat-shaped preretinal hemorrhage
        draw.chord([260, 290, 340, 350], start=0, end=180, fill=(100, 0, 0))

    # Apply slight blur to make it look smooth and realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img.save(filepath, "JPEG", quality=92)
    return filepath

def generate_sample_dataset():
    """
    Generates representative sample images for all 5 DR stages and a synthetic clinical CSV dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    stage_names = {
        0: "0_No_DR.jpg",
        1: "1_Mild_DR.jpg",
        2: "2_Moderate_DR.jpg",
        3: "3_Severe_DR.jpg",
        4: "4_Proliferative_DR.jpg"
    }

    generated_paths = {}
    for stage, fname in stage_names.items():
        path = create_synthetic_fundus(stage, fname)
        generated_paths[stage] = path

    # Generate Clinical Dataset for ML Risk Model
    np.random.seed(42)
    n_samples = 800

    age = np.random.randint(35, 82, size=n_samples)
    diabetes_duration_years = np.clip(np.random.normal(11, 6, size=n_samples), 1, 40)
    hba1c = np.clip(np.random.normal(8.0, 1.6, size=n_samples), 5.2, 14.5)
    systolic_bp = np.clip(np.random.normal(136, 18, size=n_samples), 95, 200)
    diastolic_bp = np.clip(np.random.normal(84, 11, size=n_samples), 60, 120)
    bmi = np.clip(np.random.normal(28.5, 4.8, size=n_samples), 18.0, 48.0)
    insulin_use = np.random.binomial(1, 0.45, size=n_samples)
    prior_eye_exam_months = np.random.randint(1, 36, size=n_samples)

    # Compute risk score formulation grounded in medical epidemiology
    risk_score = (
        0.02 * age +
        0.05 * diabetes_duration_years +
        0.18 * (hba1c - 6.0) +
        0.015 * (systolic_bp - 120) +
        0.025 * (bmi - 25.0) +
        0.35 * insulin_use +
        0.02 * (prior_eye_exam_months - 12) +
        np.random.normal(0, 0.3, size=n_samples)
    )

    # Convert to 4 risk categories (0: Low, 1: Moderate, 2: High, 3: Severe)
    risk_labels = pd.qcut(risk_score, q=4, labels=[0, 1, 2, 3]).astype(int)

    df_clinical = pd.DataFrame({
        "patient_id": [f"PT-{1000+i}" for i in range(n_samples)],
        "age": np.round(age, 1),
        "diabetes_duration_years": np.round(diabetes_duration_years, 1),
        "hba1c": np.round(hba1c, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "bmi": np.round(bmi, 1),
        "insulin_use": insulin_use,
        "prior_eye_exam_months": prior_eye_exam_months,
        "dr_risk_label": risk_labels
    })

    clinical_csv_path = os.path.join(DATA_DIR, "clinical_dataset.csv")
    df_clinical.to_csv(clinical_csv_path, index=False)
    
    print(f"[Dataset Generator] Sample fundus images created in {IMAGES_DIR}")
    print(f"[Dataset Generator] Clinical dataset created at {clinical_csv_path}")
    return generated_paths, clinical_csv_path

if __name__ == "__main__":
    generate_sample_dataset()
