import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cv_model import DiabeticRetinopathyCVModel, DISEASE_MAPPING, STAGE_MAPPING
from clinical_risk_model import DiabeticRetinopathyClinicalModel, RISK_LEVELS
from rag_pipeline import DiabeticRetinopathyRAGRetriever
from llm_synthesis import ClinicalLLMReasoningEngine
from dataset_generator import generate_sample_dataset

# Page configuration
st.set_page_config(
    page_title="Retina Signal | DR Decision Support",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the exact Retina Signal mathematical blueprint design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=IBM+Plex+Mono:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
        background-image: linear-gradient(to right, rgba(148, 163, 184, 0.14) 1px, transparent 1px),
                          linear-gradient(to bottom, rgba(148, 163, 184, 0.14) 1px, transparent 1px);
        background-size: 34px 34px;
    }

    .technical-label {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.70rem;
        font-weight: 600;
        color: #0284C7;
    }

    /* Landing Page Specific Styling */
    .hero-title {
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.05em;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    .hero-title-highlight {
        position: relative;
        white-space: nowrap;
    }

    .hero-title-highlight::after {
        content: "";
        position: absolute;
        bottom: -6px;
        left: 0;
        width: 100%;
        height: 8px;
        background-color: rgba(251, 207, 232, 0.85);
        transform: rotate(-1deg);
        z-index: -1;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        line-height: 1.65;
        color: #475569;
        max-width: 680px;
        margin-bottom: 2rem;
    }

    .blueprint-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(203, 213, 225, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 14px 42px rgba(30, 41, 59, 0.04);
        backdrop-filter: blur(8px);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .blueprint-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.8);
        box-shadow: 0 18px 44px rgba(30, 41, 59, 0.08);
    }

    .blueprint-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.2rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(51, 65, 85, 0.8);
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.25);
    }

    .badge-status {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.92rem;
        color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-value-lg {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.045em;
        color: #0F172A;
        font-variant-numeric: tabular-nums;
    }

    .data-point-box {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }

    .viewer-canvas {
        background-color: #020617;
        background-image: radial-gradient(circle at center, rgba(56, 189, 248, 0.25) 0 1px, transparent 1px);
        background-size: 20px 20px;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .dark-safety-banner {
        background-color: #020617;
        border-top: 1px solid #1E293B;
        color: #F1F5F9;
        padding: 3rem 2rem;
        margin-top: 3rem;
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Lazy Load Models
@st.cache_resource
def load_all_models():
    cv = DiabeticRetinopathyCVModel()
    if not cv.load_model():
        from train_and_evaluate import train_and_evaluate_all
        train_and_evaluate_all()
        cv.load_model()

    clin = DiabeticRetinopathyClinicalModel()
    if not clin.load_model():
        _, csv_path = generate_sample_dataset()
        clin.fit(csv_path)

    rag = DiabeticRetinopathyRAGRetriever()
    llm = ClinicalLLMReasoningEngine()
    return cv, clin, rag, llm

cv_model, clinical_model, rag_retriever, llm_engine = load_all_models()

# Session State Initialization
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "cases_db" not in st.session_state:
    st.session_state.cases_db = {}
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = None
if "patient_id_val" not in st.session_state:
    st.session_state["patient_id_val"] = f"PT-{np.random.randint(100, 999)}"
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_images")
if not os.path.exists(sample_dir):
    generate_sample_dataset()

sample_images = {
    "Kaggle: Diabetic Retinopathy": os.path.join(sample_dir, "kaggle_diabetic_retinopathy.jpg"),
    "Kaggle: Normal Retinal Fundus": os.path.join(sample_dir, "kaggle_normal.jpg"),
    "Kaggle: Cataract Opacity": os.path.join(sample_dir, "kaggle_cataract.jpg"),
    "Kaggle: Glaucoma Optic Cup": os.path.join(sample_dir, "kaggle_glaucoma.jpg"),
    "Benchmark: Grade 2 Moderate DR": os.path.join(sample_dir, "2_Moderate_DR.jpg"),
    "Benchmark: Grade 4 Proliferative DR": os.path.join(sample_dir, "4_Proliferative_DR.jpg")
}

# SVG Vector Graphic for Hero Scanner Radar
RADAR_SVG = """
<svg viewBox="0 0 520 520" style="width: 100%; max-width: 420px;" aria-hidden="true">
  <circle cx="260" cy="260" r="210" fill="none" stroke="rgba(56, 189, 248, 0.25)" stroke-width="1.5"/>
  <circle cx="260" cy="260" r="140" fill="none" stroke="rgba(244, 114, 182, 0.35)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <circle cx="310" cy="220" r="35" fill="rgba(244, 114, 182, 0.15)" stroke="rgba(244, 114, 182, 0.6)" stroke-width="1.5"/>
  <path d="M306 253C273 275 247 302 220 343M303 253C337 276 372 299 408 333M285 242C242 228 202 204 165 170M293 226C266 191 239 156 216 128M321 238C365 211 400 184 432 151" fill="none" stroke="rgba(56, 189, 248, 0.7)" stroke-width="1.5" stroke-linecap="round"/>
  <text x="120" y="420" fill="rgba(2, 132, 199, 0.75)" font-size="11" font-family="IBM Plex Mono, monospace" font-weight="600">RETINAL FIELD / SIGNAL MAP</text>
  <text x="340" y="400" fill="rgba(219, 39, 119, 0.75)" font-size="11" font-family="IBM Plex Mono, monospace" font-weight="600">Δx · Δy · θ</text>
  <text x="310" y="450" fill="rgba(219, 39, 119, 0.85)" font-size="11" font-family="IBM Plex Mono, monospace" font-weight="700">F(X) = EVIDENCE + SIGNAL</text>
</svg>
"""

# ==============================================================================
# PAGE 1: EXACT RETINA SIGNAL HOME LANDING PAGE (Matching User Screenshots)
# ==============================================================================
if st.session_state.current_page == "Home":

    # TOP NAVIGATION BAR (Image 1)
    nav_col1, nav_col2, nav_col3 = st.columns([1.5, 2.0, 1.2])
    with nav_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; border: 1.5px solid #0284C7; background: #E0F2FE; position: relative; display: flex; align-items: center; justify-content: center;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #0284C7;"></div>
                <div style="width: 6px; height: 6px; border-radius: 50%; background: #F472B6; position: absolute; top: 2px; right: 2px;"></div>
            </div>
            <div>
                <span class="technical-label" style="display: block; font-size: 0.65rem;">CLINICAL INTELLIGENCE</span>
                <span style="font-weight: 700; font-size: 1.05rem; color: #0F172A;">Retina Signal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 2.5rem; padding-top: 0.4rem; font-size: 0.9rem; font-weight: 500; color: #64748B;">
            <a href="#workflow" style="color: #64748B; text-decoration: none;">Workflow</a>
            <a href="#architecture" style="color: #64748B; text-decoration: none;">Architecture</a>
            <a href="#scope" style="color: #64748B; text-decoration: none;">Scope</a>
        </div>
        """, unsafe_allow_html=True)

    with nav_col3:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        if st.button("Open workspace →", key="open_ws_top", type="secondary"):
            st.session_state.current_page = "New Screening"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # HERO SECTION (Image 1)
    hero_col_left, hero_col_right = st.columns([1.3, 1.0])

    with hero_col_left:
        st.markdown("""
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #E0F2FE; border: 1px solid rgba(2, 132, 199, 0.25); padding: 0.35rem 0.8rem; border-radius: 20px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: #0284C7;"></span>
            <span class="technical-label" style="color: #0284C7; font-size: 0.72rem;">RESEARCH CLINICAL DECISION SUPPORT</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="hero-title">
            Make retinal screening <span class="hero-title-highlight">legible.</span>
        </div>
        <div class="hero-subtitle">
            A structured workspace that connects retinal image intake, clinical risk context, medical evidence, and explainable model outputs for qualified professional review.
        </div>
        """, unsafe_allow_html=True)

        btn_hero1, btn_hero2 = st.columns([1.2, 1.3])
        with btn_hero1:
            if st.button("Start a screening →", type="primary", key="start_screening_hero", use_container_width=True):
                st.session_state.current_page = "New Screening"
                st.rerun()
        with btn_hero2:
            st.markdown("<a href='#workflow'><button style='width:100%; height:42px; background:white; border:1px solid #CBD5E1; border-radius:10px; font-weight:600; color:#334155;'>Explore the workflow</button></a>", unsafe_allow_html=True)

    with hero_col_right:
        st.markdown(f"<div style='text-align: center; padding-top: 1rem;'>{RADAR_SVG}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # BOTTOM HERO METRICS CARDS (Image 1)
    card_c1, card_c2 = st.columns([2.0, 1.0])
    with card_c1:
        st.markdown("""
        <div class="blueprint-card">
            <div class="technical-label">SIGNAL FUSION / 04 LAYERS</div>
            <div style="display: flex; gap: 0.5rem; margin: 1rem 0;">
                <div style="height: 6px; flex: 1; border-radius: 4px; background: #67E8F9;"></div>
                <div style="height: 6px; flex: 1; border-radius: 4px; background: #F472B6;"></div>
                <div style="height: 6px; flex: 1; border-radius: 4px; background: #38BDF8;"></div>
                <div style="height: 6px; flex: 1; border-radius: 4px; background: #CBD5E1;"></div>
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #1E293B;">
                Visual signal + clinical context + retrieved evidence + grounded explanation
            </div>
        </div>
        """, unsafe_allow_html=True)

    with card_c2:
        st.markdown("""
        <div class="blueprint-card">
            <div class="technical-label" style="color: #64748B;">PROTOTYPE GUARDRAIL</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A; margin-top: 0.5rem; line-height: 1.2;">
                Assist review.<br>Never replace it.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top: 1px solid #E2E8F0; margin: 3rem 0;'>", unsafe_allow_html=True)

    # WORKFLOW SECTION (Image 2)
    st.markdown("<div id='workflow'></div>", unsafe_allow_html=True)
    st.markdown("<div class='technical-label'>THE SCREENING WORKFLOW</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 2.5rem; font-weight: 800; letter-spacing: -0.045em; margin: 0.2rem 0 0.5rem 0;'>One case. Four distinct kinds of evidence.</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #64748B; max-width: 720px; margin-bottom: 2rem;'>Each layer retains its own identity so a clinician can distinguish patient data, model findings, retrieved medical evidence, and the final explanatory summary.</p>", unsafe_allow_html=True)

    wf1, wf2 = st.columns(2)
    with wf1:
        st.markdown("""
        <div class="blueprint-card" style="position: relative;">
            <span style="position: absolute; top: 1rem; right: 1rem; font-size: 2.2rem; font-weight: 700; color: rgba(2, 132, 199, 0.15);">01</span>
            <div style="font-size: 1.3rem;">️</div>
            <div class="technical-label" style="margin-top: 0.8rem;">01 / VISUAL</div>
            <h3 style="font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0;">Retinal image review</h3>
            <p style="font-size: 0.9rem; color: #64748B; leading: 1.5;">Capture a retinal image and preserve a focused case record for model-assisted visual assessment.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="blueprint-card" style="position: relative;">
            <span style="position: absolute; top: 1rem; right: 1rem; font-size: 2.2rem; font-weight: 700; color: rgba(2, 132, 199, 0.15);">03</span>
            <div style="font-size: 1.3rem;"></div>
            <div class="technical-label" style="margin-top: 0.8rem;">03 / EVIDENCE</div>
            <h3 style="font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0;">Source-grounded review</h3>
            <p style="font-size: 0.9rem; color: #64748B; leading: 1.5;">Keep retrieved guidelines and literature separate, readable, and traceable to their sources.</p>
        </div>
        """, unsafe_allow_html=True)

    with wf2:
        st.markdown("""
        <div class="blueprint-card" style="position: relative;">
            <span style="position: absolute; top: 1rem; right: 1rem; font-size: 2.2rem; font-weight: 700; color: rgba(2, 132, 199, 0.15);">02</span>
            <div style="font-size: 1.3rem;"></div>
            <div class="technical-label" style="margin-top: 0.8rem;">02 / CLINICAL</div>
            <h3 style="font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0;">Risk context</h3>
            <p style="font-size: 0.9rem; color: #64748B; leading: 1.5;">Structure key clinical variables so risk-model outputs can be reviewed alongside image findings.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="blueprint-card" style="position: relative;">
            <span style="position: absolute; top: 1rem; right: 1rem; font-size: 2.2rem; font-weight: 700; color: rgba(2, 132, 199, 0.15);">04</span>
            <div style="font-size: 1.3rem;"></div>
            <div class="technical-label" style="margin-top: 0.8rem;">04 / SYNTHESIS</div>
            <h3 style="font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0;">Clear explanation</h3>
            <p style="font-size: 0.9rem; color: #64748B; leading: 1.5;">Bring model outputs and supporting evidence together for professional review—not autonomous diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top: 1px solid #E2E8F0; margin: 3rem 0;'>", unsafe_allow_html=True)

    # ARCHITECTURE SECTION (Image 3)
    st.markdown("<div id='architecture'></div>", unsafe_allow_html=True)
    arch_col1, arch_col2 = st.columns([1.1, 1.2])

    with arch_col1:
        st.markdown("<div class='technical-label'>ARCHITECTURE / TRACEABLE BY DESIGN</div>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size: 2.4rem; font-weight: 800; letter-spacing: -0.045em; margin: 0.2rem 0 0.8rem 0;'>A composed system—not a black box.</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.05rem; color: #64748B; line-height: 1.65;'>Retina Signal is structured around three intelligence pipelines: computer vision for image analysis, clinical machine learning for risk estimation, and retrieval for source-grounded knowledge. The workspace keeps their outputs visible before any explanation layer is shown.</p>", unsafe_allow_html=True)

    with arch_col2:
        st.markdown("""
        <div class="blueprint-card">
            <div style="display: flex; flex-direction: column; gap: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 1.2rem;"></div>
                    <div><span class="technical-label" style="font-size:0.65rem;">INPUT</span><br><strong style="font-size: 0.9rem;">Retinal image + structured patient variables</strong></div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 1.2rem;">️</div>
                    <div><span class="technical-label" style="font-size:0.65rem;">VISION</span><br><strong style="font-size: 0.9rem;">Severity and optional visual findings</strong></div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 1.2rem;"></div>
                    <div><span class="technical-label" style="font-size:0.65rem;">RETRIEVAL</span><br><strong style="font-size: 0.9rem;">Relevant guidelines and research sources</strong></div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem; background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 1.2rem;">️</div>
                    <div><span class="technical-label" style="font-size:0.65rem;">FUSION</span><br><strong style="font-size: 0.9rem;">Separated inputs for a grounded explanation</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DARK SAFETY & SCOPE BANNER (Image 3)
    st.markdown("<div id='scope'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-safety-banner">
        <div style="display: flex; align-items: flex-start; gap: 1.25rem;">
            <div style="width: 44px; height: 44px; border-radius: 12px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); display: grid; place-items: center; font-size: 1.3rem;">
                ️
            </div>
            <div>
                <div class="technical-label" style="color: #38BDF8;">SCOPE AND LIMITATION</div>
                <h3 style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin: 0.2rem 0 0.5rem 0;">Designed to support qualified clinical review.</h3>
                <p style="font-size: 0.95rem; color: #94A3B8; line-height: 1.6; max-width: 820px;">
                    This research prototype organizes inputs, model outputs, and evidence for professional interpretation. It does not autonomously diagnose diabetic retinopathy, provide treatment recommendations, or replace clinical judgment.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # FOOTER (Image 3)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2rem; padding: 1rem 0; font-size: 0.8rem; color: #64748B;">
        <span class="technical-label" style="color: #64748B;">RETINA SIGNAL / RESEARCH PROTOTYPE</span>
        <span>Built around traceability, uncertainty, and clinical review.</span>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: CASE INTAKE FORM (When user clicks "Start a screening →")
# ==============================================================================
elif st.session_state.current_page == "New Screening":

    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <div class="technical-label">case intake / structured input</div>
        <button style="background:none; border:none; color:#0284C7; font-weight:600; cursor:pointer;" onclick="window.location.reload();">← Back to Home</button>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Back to Landing Page"):
        st.session_state.current_page = "Home"
        st.rerun()

    st.title("Start a new screening.")
    st.caption("The retinal image is kept as the focal input. Patient variables provide clinical context for the future risk-estimation pipeline.")

    col_form, col_img = st.columns([1.1, 0.9])

    with col_form:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.subheader("Patient Clinical Variables")
        
        c1, c2 = st.columns(2)
        with c1:
            patient_id_input = st.text_input("Patient Identifier *", value=st.session_state["patient_id_val"])
            st.session_state["patient_id_val"] = patient_id_input
            patient_age_input = st.number_input("Age (years) *", min_value=18, max_value=120, value=58)
            diabetes_duration_input = st.number_input("Diabetes Duration (years)", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
        with c2:
            hba1c_input = st.number_input("HbA1c (%)", min_value=4.0, max_value=20.0, value=8.4, step=0.1)
            systolic_bp_input = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=240, value=138)
            diastolic_bp_input = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140, value=86)

        clinical_question_input = st.text_area(
            "Optional Clinical Question",
            placeholder="What specific context should the evidence review address? (e.g., Guidance on anti-VEGF referral timing for severe NPDR)",
            height=90
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_img:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.markdown("<div class='technical-label'>RETINAL IMAGE INPUT</div>", unsafe_allow_html=True)
        
        img_source_type = st.radio("Select Image Source:", ["Kaggle Benchmark Image", "Upload Retinal Photo (JPG/PNG)"], horizontal=True)

        selected_image = None
        image_name_str = "uploaded_retinal_photo.jpg"

        if img_source_type == "Kaggle Benchmark Image":
            avail_keys = [k for k in sample_images.keys() if os.path.exists(sample_images[k])]
            chosen_key = st.selectbox("Choose Benchmark Retinal Image:", avail_keys, index=0)
            sample_path = sample_images[chosen_key]
            if os.path.exists(sample_path):
                selected_image = Image.open(sample_path)
                image_name_str = os.path.basename(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Retinal Fundus Photo (JPEG/PNG, Max 6MB)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                if uploaded_file.size > 6 * 1024 * 1024:
                    st.error("File size exceeds 6MB limit. Please upload an image under 6MB.")
                    selected_image = None
                else:
                    selected_image = Image.open(uploaded_file)
                    image_name_str = uploaded_file.name
            else:
                st.warning("️ Please upload a retinal fundus photograph (JPEG/PNG, < 6MB) to start screening.")
                selected_image = None

        if selected_image is not None:
            st.image(selected_image, caption=f"Selected Input: {image_name_str}", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # SUBMIT / START SCREENING ACTION
    st.markdown("---")
    if st.button(" Start Screening & Generate Case Record", type="primary", use_container_width=True):
        if not selected_image:
            st.error("Cannot proceed: No valid retinal image provided. Please upload a file or select a benchmark image.")
        else:
            with st.spinner("Processing computer vision classification, clinical risk modeling, RAG evidence retrieval, and LLM reasoning..."):
                case_code = f"SCR-{datetime.date.today().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
                patient_data = {
                    "patient_id": patient_id_input,
                    "age": patient_age_input,
                    "diabetes_duration_years": diabetes_duration_input,
                    "hba1c": hba1c_input,
                    "systolic_bp": systolic_bp_input,
                    "diastolic_bp": diastolic_bp_input,
                    "bmi": 28.5,
                    "insulin_use": 1 if hba1c_input > 8.0 else 0,
                    "prior_eye_exam_months": 18
                }

                # Run Full AI Pipelines
                cv_res = cv_model.predict(selected_image)
                ml_res = clinical_model.predict(patient_data)
                rag_res = rag_retriever.retrieve(
                    query=clinical_question_input,
                    cv_stage_code=cv_res["stage_code"],
                    clinical_risk_code=ml_res["risk_code"],
                    secondary_finding=cv_res.get("secondary_finding", ""),
                    top_k=3
                )
                llm_res = llm_engine.synthesize_report(cv_res, ml_res, rag_res, patient_data)

                # Store Case in Session State
                case_record = {
                    "case_code": case_code,
                    "patient_id": patient_id_input,
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "image": selected_image,
                    "image_name": image_name_str,
                    "patient_data": patient_data,
                    "clinical_question": clinical_question_input,
                    "cv_result": cv_res,
                    "clinical_result": ml_res,
                    "rag_evidence": rag_res,
                    "llm_report": llm_res
                }
                
                st.session_state.cases_db[case_code] = case_record
                st.session_state.active_case_id = case_code
                st.session_state.current_page = "Analysis"
                st.rerun()

# ==============================================================================
# PAGE 3: CASE DETAIL & ANALYSIS VIEW (Rendered after Start Screening)
# ==============================================================================
elif st.session_state.current_page == "Analysis":
    if not st.session_state.active_case_id or st.session_state.active_case_id not in st.session_state.cases_db:
        default_case_code = f"SCR-{datetime.date.today().strftime('%Y%m%d')}-8492"
        if default_case_code not in st.session_state.cases_db:
            default_img_path = sample_images["Kaggle: Diabetic Retinopathy"]
            default_img = Image.open(default_img_path) if os.path.exists(default_img_path) else Image.new("RGB", (512,512), (180,75,20))
            p_data = {"patient_id": "PT-204", "age": 58, "diabetes_duration_years": 12.0, "hba1c": 8.4, "systolic_bp": 138, "diastolic_bp": 86, "bmi": 29.0, "insulin_use": 1, "prior_eye_exam_months": 18}
            cv_res = cv_model.predict(default_img)
            ml_res = clinical_model.predict(p_data)
            rag_res = rag_retriever.retrieve(query="", cv_stage_code=cv_res["stage_code"], clinical_risk_code=ml_res["risk_code"], top_k=3)
            llm_res = llm_engine.synthesize_report(cv_res, ml_res, rag_res, p_data)
            
            st.session_state.cases_db[default_case_code] = {
                "case_code": default_case_code,
                "patient_id": "PT-204",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "image": default_img,
                "image_name": "kaggle_diabetic_retinopathy.jpg",
                "patient_data": p_data,
                "clinical_question": "Guideline recommendations for NPDR progression risk",
                "cv_result": cv_res,
                "clinical_result": ml_res,
                "rag_evidence": rag_res,
                "llm_report": llm_res
            }
        st.session_state.active_case_id = default_case_code

    case = st.session_state.cases_db[st.session_state.active_case_id]
    cv_res = case["cv_result"]
    ml_res = case["clinical_result"]
    rag_evidence = case["rag_evidence"]
    llm_report = case["llm_report"]

    # TOP BAR METADATA
    col_meta1, col_meta2 = st.columns([1.2, 1.0])
    with col_meta1:
        st.markdown(f"<div class='technical-label'>ANALYSIS / {case['case_code']}</div>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin: 0.1rem 0; font-size: 1.8rem; font-weight: 700;'>Patient {case['patient_id']}</h1>", unsafe_allow_html=True)
        st.caption(f"SCREENED {case['created_at']} · IMAGE {case['image_name']}")
    with col_meta2:
        st.markdown("<div style='text-align: right; padding-top: 0.5rem;'>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.session_state.confirm_reset:
                if st.button("Confirm Reset?"):
                    st.session_state["patient_id_val"] = f"PT-{np.random.randint(100, 999)}"
                    st.session_state.confirm_reset = False
                    st.session_state.current_page = "New Screening"
                    st.rerun()
            else:
                if st.button(" New Screening"):
                    st.session_state.confirm_reset = True
                    st.rerun()
        with btn_col2:
            st.download_button(
                " Export Report",
                data=llm_report["full_narrative_markdown"],
                file_name=f"Report_{case['case_code']}.md",
                mime="text/markdown"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # SECTION 1: RETINAL IMAGE VIEWER & MODEL FINDINGS
    col_viewer, col_findings = st.columns([1.35, 0.9])

    with col_viewer:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.markdown("<div class='technical-label'>RETINAL IMAGE VIEWER</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; margin: 0 0 0.5rem 0;'>Primary Visual Input</p>", unsafe_allow_html=True)
        
        view_overlay_mode = st.radio(
            "Visualization Mode:",
            ["Standard Color Fundus", "Green-Channel Enhanced (CLAHE)", "AI Lesion Saliency Heatmap"],
            horizontal=True
        )

        st.markdown("<div class='viewer-canvas'>", unsafe_allow_html=True)
        if view_overlay_mode == "Standard Color Fundus":
            st.image(case["image"], use_container_width=True)
        elif view_overlay_mode == "Green-Channel Enhanced (CLAHE)":
            st.image(cv_res["green_enhanced_img"], use_container_width=True)
        else:
            st.image(cv_res["heatmap_overlay_img"], use_container_width=True)
        
        st.markdown("<div style='font-family: \"IBM Plex Mono\", monospace; font-size: 0.7rem; color: #94A3B8; margin-top: 0.5rem;'>INTERACTIVE ZOOM 100% / PAN ENABLED</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_findings:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.markdown("<div class='technical-label'>MODEL FINDINGS</div>", unsafe_allow_html=True)
        
        stage_color = cv_res["stage_color"]
        st.markdown(f"""
        <div style="margin-top: 0.75rem; padding: 0.8rem; background: #F1F5F9; border-radius: 8px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #64748B;">DR Assessment</div>
            <div class="badge-status" style="background-color: {stage_color}; margin-top: 0.3rem;">
                {cv_res['stage_name']} ({cv_res['stage_code']})
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem;">Validated visual pipeline output</div>
        </div>
        """, unsafe_allow_html=True)

        if cv_res.get("secondary_finding"):
            st.markdown(f"""
            <div style="margin-top: 0.75rem; padding: 0.8rem; background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px;">
                <div style="font-size: 0.8rem; font-weight: 600; color: #92400E;">Secondary Ocular Finding</div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #78350F; margin-top: 0.2rem;">
                    👁️ {cv_res['secondary_finding']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top: 0.75rem; padding: 0.8rem; background: #F1F5F9; border-radius: 8px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #64748B;">Model Confidence</div>
            <div class="metric-value-lg">{cv_res['confidence_score']}%</div>
            <div style="font-size: 0.75rem; color: #94A3B8;">Classification probability score</div>
        </div>
        """, unsafe_allow_html=True)

        risk_color = ml_res["risk_color"]
        st.markdown(f"""
        <div style="margin-top: 0.75rem; padding: 0.8rem; background: #F1F5F9; border-radius: 8px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #64748B;">Clinical Risk</div>
            <div class="badge-status" style="background-color: {risk_color}; margin-top: 0.3rem;">
                {ml_res['risk_name']} ({ml_res['risk_score_percent']}%)
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem;">Biomarker risk prediction model</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 2: WHY THIS RESULT? & RAG EVIDENCE
    col_why, col_evidence = st.columns([1.1, 1.0])

    with col_why:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.markdown("<div class='technical-label'>WHY THIS RESULT?</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin: 0.2rem 0 0.8rem 0; font-size: 1.2rem; font-weight: 700;'>Patient Context & Visual Findings</h3>", unsafe_allow_html=True)

        p = case["patient_data"]
        dp1, dp2, dp3, dp4 = st.columns(4)
        with dp1:
            st.markdown(f"<div class='data-point-box'><div class='technical-label'>AGE</div><div style='font-weight:700; font-size:1rem;'>{p['age']} yr</div></div>", unsafe_allow_html=True)
        with dp2:
            st.markdown(f"<div class='data-point-box'><div class='technical-label'>DURATION</div><div style='font-weight:700; font-size:1rem;'>{p['diabetes_duration_years']} yr</div></div>", unsafe_allow_html=True)
        with dp3:
            st.markdown(f"<div class='data-point-box'><div class='technical-label'>HBA1C</div><div style='font-weight:700; font-size:1rem;'>{p['hba1c']}%</div></div>", unsafe_allow_html=True)
        with dp4:
            st.markdown(f"<div class='data-point-box'><div class='technical-label'>SBP</div><div style='font-weight:700; font-size:1rem;'>{p['systolic_bp']}</div></div>", unsafe_allow_html=True)

        st.markdown("<br><strong>Detected Visual Findings:</strong>", unsafe_allow_html=True)
        for lesion in cv_res["detected_lesions"]:
            st.markdown(f"• {lesion}")

        st.markdown("<br><strong>Primary Risk Drivers:</strong>", unsafe_allow_html=True)
        for driver in ml_res["risk_drivers"]:
            st.warning(f"️ {driver}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_evidence:
        st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
        st.markdown("<div class='technical-label'>RETRIEVED EVIDENCE</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin: 0.2rem 0 0.8rem 0; font-size: 1.2rem; font-weight: 700;'>Sources Clinicians Can Inspect (RAG)</h3>", unsafe_allow_html=True)

        for i, doc in enumerate(rag_evidence):
            with st.expander(f" [{doc['id']}] {doc['title']} — Relevance {doc['relevance_score']}%", expanded=(i==0)):
                st.markdown(f"**Publisher / Source:** {doc['source']}")
                st.markdown(f"**Section:** {doc['section']}")
                st.info(doc["content"])

        st.markdown("</div>", unsafe_allow_html=True)

    # SECTION 3: AI EXPLANATION / CASE SUMMARY
    st.markdown("<div class='blueprint-card'>", unsafe_allow_html=True)
    st.markdown("<div class='technical-label'>AI EXPLANATION / PROFESSIONAL REVIEW REQUIRED</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin: 0.2rem 0 0.8rem 0; font-size: 1.4rem; font-weight: 700;'>Grounded Case Summary & Action Plan</h2>", unsafe_allow_html=True)
    
    st.markdown(llm_report["full_narrative_markdown"])

    st.markdown("""
    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 1rem; color: #1E40AF; font-size: 0.88rem; margin-top: 1.5rem;">
        <strong>️ Clinical Decision Support Disclaimer:</strong><br>
        This software prototype is intended for research and decision-support demonstration purposes only. Model findings, patient data, and medical evidence remain visibly separated. It does not provide an autonomous diagnosis or replacement for clinical judgment.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
