# QA Test Report — "Retina Signal" Diabetic Retinopathy Screening App
**URL:** https://diabetic-retinopathy-model.streamlit.app/
**Tested via:** Screen recording walkthrough (landing page → screening intake → results → new screening)
**Tester role:** Functional / clinical-safety QA pass

---

## 1. Summary Verdict

The app has a polished, professional front end and a genuinely well-designed *information architecture* (Visual → Clinical → Evidence → Synthesis). However, testing surfaced **one critical correctness bug** and several functional/UX bugs that would block release for anything beyond an internal demo — especially given this is framed as clinical decision support.

| Severity | Count |
|---|---|
| 🔴 Critical | 1 |
| 🟠 High | 3 |
| 🟡 Medium | 4 |
| 🟢 Low / Polish | 4 |

---

## 2. 🔴 Critical Bug — Model output is domain-mismatched and internally contradictory

On the results page, for a tool explicitly named a **"Multimodal Diabetic Retinopathy Screening System"**:

- **DR Assessment** field reads: `Glaucoma (Glaucoma)` — not a DR grade (e.g., No DR / Mild / Moderate / Severe NPDR / PDR).
- **Detected Visual Findings** lists: *"Increased optic cup-to-disc ratio detected"* and *"Neuroretinal rim thinning and temporal nerve fiber layer alterations"* — these are classic **glaucoma** findings, not diabetic retinopathy findings (DR findings would be microaneurysms, hemorrhages, exudates, IRMA, venous beading, neovascularization).
- Meanwhile, the **Retrieved Evidence (RAG)** panel pulls DR-specific citations (AAO PPP 2023 "4-2-1 rule," UKPDS-33, ICO diabetic eye care guidelines) — correct for DR, but now cited in support of a glaucoma finding.
- The **Executive Clinical Impression** auto-generated text then confidently synthesizes all of this into one paragraph, presenting a glaucoma classification, DR-specific guideline citations, and a diabetes risk-factor discussion as one coherent case — without flagging the mismatch.
- **Referral Urgency: ROUTINE (Annual screening protocol)** is stated even though Clinical Risk is separately labeled **"Elevated Clinical Risk (57.5%)"** — the urgency label and the risk label contradict each other.

**Why this matters:** this isn't a cosmetic bug — it's a decision-support tool confidently stitching together mismatched classifier output, hallucination-prone narrative text, and real citations into something that *reads* as authoritative. That's the most dangerous failure mode for this category of app. I'd treat this as a **stop-ship** issue.

**Suggested root cause to check:** likely an index/label-mapping bug where the vision classifier's class list (e.g., trained on a different/multi-disease dataset) isn't aligned with the label array the UI displays, and the LLM summarizer isn't validating the classifier label against the domain (DR) before generating "findings" text.

---

## 3. 🟠 High-Severity Bugs

### 3.1 Upload path silently falls back to the old benchmark image
When you switch **Image Source** from "Kaggle Benchmark Image" to "Upload Retinal Photo (JPG/PNG)" but don't actually attach a file, the app:
- Still displays the previously-selected Kaggle image underneath, unchanged.
- Runs the screening anyway and labels the result `Selected Input: kaggle_diabetic_retinopathy.jpg` — with **no warning, no validation error, no disabled submit button**.

A clinician could easily believe they screened their uploaded photo when the model actually scored a stock benchmark image. This needs a hard validation: disable "Start Screening" until a file is actually present when "Upload" is selected.

### 3.2 Patient Identifier field auto-regenerates on unrelated interactions
The Patient Identifier defaults to a random `PT-XXX` value, and **that value changes every time any other widget on the page is touched** (e.g., clicking the HbA1c +/- stepper), even though the user never edited it. Observed sequence in one session: `PT-601 → PT-94 → PT-919 → PT-958 → PT-123 → PT-188 → PT-675 (final) → PT-277 (after reset)`.
This indicates the field's default isn't pinned in session state and is being regenerated on every Streamlit rerun. In a clinical record-keeping context, an ID that silently changes underneath the user is a serious data-integrity risk.

### 3.3 File-size limit is displayed inconsistently
The field label says **"Upload Retinal Fundus Photo (JPEG/PNG < 6MB)"**, but the uploader widget itself displays **"200MB per file"**. One of these is wrong; as written, a user has no reliable idea what the actual limit is, and either message could cause a rejected upload or an oversized/slow upload.

---

## 4. 🟡 Medium-Severity / UX Issues

### 4.1 Mystery empty input boxes
Multiple screens (intake form, results page, evidence panel) show **two blank, unlabeled bordered boxes** sitting above the main content with no visible text, placeholder, or purpose. These look like leftover/empty Streamlit columns or components and should either be removed or given content — as-is they read as broken UI.

### 4.2 "New Screening" doesn't clearly confirm a full reset
Clicking **New Screening** wipes all form fields (including the free-text clinical question) and jumps back toward the landing/intake page without a confirmation step. If a clinician clicked this by accident after entering a real case, the data is gone with no undo.

### 4.3 AI Lesion Saliency Heatmap doesn't clearly localize lesions
The heatmap toggle renders correctly (visually distinct from the CLAHE and standard views), but the "hot" (yellow/red) region tracks the optic disc — which is *always* bright in a normal fundus photo — rather than any specific pathological lesion. For a feature explicitly named a *saliency* map meant to explain *why* the model flagged a finding, this needs validation against known lesion locations, or it risks giving false clinical confidence in the explanation.

### 4.4 Action Plan text is generic and disconnected from the stated risk
Despite an "Elevated Clinical Risk" label, the generated **Suggested Action Plan** just says *"Continue annual dilated eye examination. Reinforce lifestyle and glycemic adherence"* — the same boilerplate you'd expect for a low-risk case. The synthesis layer should scale its recommendation language to the computed risk/urgency tier.

---

## 5. 🟢 Low-Severity / Polish

- **Browser tab title** reads "Retina Signal — Clinical Decis..." (truncated) while the on-page hero says "Retina Signal / Make retinal screening legible" — fine, but worth a shorter tab title for clarity.
- **"Sign in"** button in the top-right of the Streamlit chrome — unclear if this is a real auth feature or just the default Streamlit Cloud chrome; if the app is meant to be identity-gated for clinician use, this should be wired up and tested.
- The landing page's architecture/marketing sections ("Signal Fusion / 04 Layers," "A composed system—not a black box") are well-written but slightly oversell the reliability of a prototype that mislabels its own core output — consider softening claims like *"Validated visual pipeline output"* next to the DR Assessment badge until the bug in §2 is fixed.
- No visible way to view/download prior screenings (only "Export Report" for the current one) — a case history/list view would be a natural next feature given the "structured workspace" positioning.

---

## 6. What Worked Well

- **Layered results structure** (Model Findings → Why This Result → Retrieved Evidence → Grounded Summary) is genuinely good design for clinical explainability — it keeps raw model output, patient context, and citations visibly separate, which is the right instinct for a decision-support tool.
- **Green-Channel Enhanced (CLAHE)** visualization mode renders correctly and is a legitimate, useful fundus-image enhancement.
- **RAG citations** are sourced from real, appropriately authoritative bodies (AAO, UKPDS, ICO) with visible publisher/section metadata and relevance scores — good traceability pattern, when the underlying classification is correct.
- Clear **"assist review, never replace it"** framing and a repeated clinical disclaimer at the bottom of the report.
- Form inputs (steppers, radio toggles, expandable evidence cards) are all responsive with no crashes observed across the full session.

---

## 7. Recommended Priority Order

1. **Fix the classifier/label mapping bug** (§2) — do not ship clinical-facing copy until DR Assessment reliably reflects DR classes, and add a sanity check that flags when vision output, RAG evidence, and narrative summary reference different conditions.
2. **Block submission on unvalidated uploads** (§3.1) and reconcile the two file-size messages (§3.3).
3. **Persist Patient Identifier in session state** (§3.2) so it doesn't drift on unrelated reruns.
4. Clean up the empty UI boxes (§4.1) and add a confirm step to "New Screening" (§4.2).
5. Revisit heatmap fidelity (§4.3) and make the action-plan text risk-aware (§4.4).
