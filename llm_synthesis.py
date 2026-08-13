import json

class ClinicalLLMReasoningEngine:
    """
    Multimodal Reasoning & Explanation Synthesis Layer.
    Combines CV predictions, Clinical ML risk, and RAG medical evidence into source-grounded clinical reports.
    """
    def synthesize_report(
        self,
        cv_result: dict,
        clinical_result: dict,
        rag_evidence: list,
        patient_info: dict = None
    ) -> dict:
        """
        Synthesizes structured clinical decision-support report.
        """
        patient_id = patient_info.get("patient_id", "PT-1001") if patient_info else "PT-1001"
        age = patient_info.get("age", 58) if patient_info else 58
        hba1c = patient_info.get("hba1c", 8.2) if patient_info else 8.2
        duration = patient_info.get("diabetes_duration_years", 12) if patient_info else 12

        cv_stage = cv_result.get("stage_name", "No Diabetic Retinopathy")
        cv_conf = cv_result.get("confidence_score", 91.5)
        lesions = cv_result.get("detected_lesions", [])
        secondary_finding = cv_result.get("secondary_finding", None)

        ml_risk_name = clinical_result.get("risk_name", "Elevated Clinical Risk")
        ml_risk_score = clinical_result.get("risk_score_percent", 64.5)
        risk_drivers = clinical_result.get("risk_drivers", [])

        # Format retrieved RAG sources for grounded citations
        citations = []
        evidence_summary_lines = []
        for doc in rag_evidence:
            citations.append(f"[{doc['id']}] {doc['title']} ({doc['source']})")
            evidence_summary_lines.append(f"- **[{doc['id']}] {doc['title']}**: {doc['content']}")

        # Determine referral urgency by synthesizing BOTH CV DR Stage AND Tabular Clinical Risk Score
        stage_num = cv_result.get("stage_code_num", 0)

        if stage_num >= 4 or ml_risk_score >= 80.0:
            referral_urgency = "URGENT / IMMEDIATE (Retina Specialist evaluation within 1-2 weeks)"
            action_plan = (
                "Prompt referral to a Retina Specialist for comprehensive evaluation (including OCT and Fluorescein Angiography). "
                "Evaluate for immediate anti-VEGF therapy or panretinal photocoagulation (PRP). "
                "Intensify systemic management: target HbA1c < 7.0% and blood pressure < 130/80 mmHg."
            )
        elif stage_num == 3 or ml_risk_score >= 60.0:
            referral_urgency = "HIGH URGENCY (Ophthalmology specialist evaluation within 2-4 weeks)"
            action_plan = (
                "Refer to Ophthalmology within 2 to 4 weeks. Perform dilated fundus examination and Macular OCT. "
                "Tighten glycemic control (HbA1c target < 7.0%) and monitor blood pressure closely to prevent rapid disease progression."
            )
        elif stage_num == 2 or ml_risk_score >= 35.0:
            referral_urgency = "SEMI-URGENT (Eye care specialist follow-up within 2-3 months)"
            action_plan = (
                "Schedule a comprehensive dilated eye examination within 2 to 3 months. "
                "Optimize diabetes self-management, review HbA1c every 3 months, and control blood pressure."
            )
        elif stage_num == 1 or ml_risk_score >= 20.0:
            referral_urgency = "EARLY FOLLOW-UP (Dilated eye examination within 6 months)"
            action_plan = (
                "Schedule follow-up dilated eye examination within 6 months. "
                "Reinforce lifestyle modifications, blood glucose monitoring, and annual microvascular risk assessments."
            )
        else:
            referral_urgency = "ROUTINE (Annual screening protocol every 12-24 months)"
            action_plan = (
                "Continue routine annual dilated eye screening per ADA guidelines. "
                "Maintain optimal glycemic (HbA1c < 7.0%) and blood pressure targets."
            )

        # Executive Clinical Narrative Generation
        narrative = f"""
### Executive Clinical Impression
Patient **{patient_id}** ({age} years old, {duration} yrs T2D, HbA1c {hba1c}%) was evaluated using the Multimodal Diabetic Retinopathy Screening System.
- **Diabetic Retinopathy Assessment**: Classed as **{cv_stage}** with **{cv_conf}%** model confidence.
"""
        if secondary_finding:
            narrative += f"- **Secondary Ocular Finding**: **{secondary_finding}**.\n"

        narrative += f"""- **Clinical Risk Model**: Evaluated at **{ml_risk_name}** ({ml_risk_score}% cumulative risk score).
- **Harmonized Referral Urgency**: **{referral_urgency}**.

### 1. Computer Vision & Lesion Analysis
The deep visual feature analysis identified the following retinal hallmarks:
"""
        for lesion in lesions:
            narrative += f"- {lesion}\n"

        narrative += f"""
### 2. Clinical Biomarker & Risk Factors
The tabular clinical risk pipeline highlighted the following key patient risk drivers:
"""
        for driver in risk_drivers:
            narrative += f"- {driver}\n"

        narrative += f"""
### 3. Source-Grounded Medical Evidence (RAG)
The following clinical practice guidelines and peer-reviewed literature support this case evaluation:
"""
        for line in evidence_summary_lines:
            narrative += f"{line}\n"

        narrative += f"""
### 4. Suggested Action Plan & Next Steps
{action_plan}

> **Clinical Disclaimer & Scope of Use**:
> This automated report is generated by an AI decision-support system designed solely to assist qualified healthcare professionals. It does not replace professional clinical judgement, formal diagnostic examination, or specialist consultation.
"""

        return {
            "executive_summary": f"Case {patient_id}: {cv_stage} ({cv_conf}% conf) | {ml_risk_name} ({ml_risk_score}%)",
            "referral_urgency": referral_urgency,
            "action_plan": action_plan,
            "full_narrative_markdown": narrative.strip(),
            "citations": citations
        }
