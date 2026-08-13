# Retina Signal — Requirements Traceability Summary

## Product Scope

Retina Signal is a **research and clinical decision-support prototype** for diabetic retinopathy screening. It supports qualified professional review by structuring retinal images, patient variables, model outputs, evidence retrieved from trusted sources, and a grounded explanation. It is not an autonomous diagnostic, triage, treatment, or replacement for clinical judgment system.

## Required End-to-End Workflow

| Step | User action | System responsibility | User-facing output |
|---|---|---|---|
| 1. Case intake | Upload a retinal fundus image, enter patient data, and optionally pose a clinical question. | Validate the image, prepare structured clinical inputs, and create a protected case record. | A saved case with a visible pending status. |
| 2. Visual analysis | Initiate or await the image-analysis pipeline. | Apply preprocessing and a CNN, Vision Transformer, or equivalent visual model. | DR severity, confidence, and optional visual/lesion findings. |
| 3. Clinical risk | Supply available health indicators. | Clean and feature-engineer patient data for a clinical ML model. | Risk estimate, risk score, and attributable patient factors. |
| 4. Evidence retrieval | Provide an optional question or review case context. | Retrieve relevant chunks from curated guidelines, research, and institutional documents. | Expandable evidence cards with sources, passages, and relevance scores. |
| 5. Fusion and explanation | Review the assembled case. | Keep model outputs and retrieved evidence distinct while passing structured context to an explanation layer. | A plain-language, source-grounded summary with limitations. |
| 6. Report review | Inspect or export the completed case. | Present a traceable case summary for professional assessment. | Image findings, risk assessment, evidence, uncertainty, and explicit limitations. |

## Architecture and Data Model

The frontend is a React workspace protected by **Manus OAuth**. The backend uses tRPC procedures and a relational schema designed around the documented architecture.

| Entity | Primary purpose | Important relationships |
|---|---|---|
| `users` | Manus-authenticated account and ownership boundary. | One user owns many screening cases. |
| `screeningCases` | Source record for the retinal image, patient information, optional question, and case state. | Belongs to one user; has one assessment and many evidence records. |
| `screeningAssessments` | Stores fused status and eventual visual, clinical, retrieval, and explanation outputs. | One-to-one with a screening case. |
| `screeningEvidence` | Stores source-attributed retrieval passages, source metadata, type, and relevance. | Many-to-one with a screening case. |

The designed integration boundary preserves the three intelligence pipelines specified in the architecture: **computer vision**, **clinical risk prediction**, and **retrieval-augmented generation**. A fusion/explanation layer can only display grounded content after attributable upstream outputs become available.

## Required Screens and Navigation

| Route | Screen | Required content and interaction |
|---|---|---|
| `/` | Public landing page | Problem framing, proposed multimodal solution, architecture overview, Manus OAuth entry point, and prototype limitation. |
| `/app` | Dashboard / Home | Recent cases, case status, quick navigation, and a new-screening action. |
| `/app/new` | New Screening | Retinal-image upload, patient information form, optional clinical question, validation, and protected case creation. |
| `/app/cases/:id` | Analysis / Case Report | Top bar case metadata, image viewer with zoom/pan, model findings, risk, “why this result,” evidence panel, AI explanation, limitation, and export action. |

The authenticated workspace uses a compact sidebar for **Workspace** and **New screening**, maintains responsive mobile navigation, and provides clear routes back to the case queue from every protected screen.

## UX and Visual System

The interface pairs the specified **light clinical decision-support layout** with the requested mathematical-blueprint aesthetic. Global tokens define white/soft-neutral surfaces, readable dark text, fine grid lines, pastel cyan and soft-pink wireframe accents, and restrained status colors. Large sans-serif headings are paired with monospaced technical labels.

The retinal image remains the visual focal point on the analysis screen. Model predictions, patient data, and medical evidence are always presented as separate sections. Confidence and uncertainty are retained rather than hidden, retrieved evidence is expandable and source-linked, and simple summaries are shown before technical detail. Interaction feedback relies on short, reduced-motion-aware transitions, accessible focus treatment, responsive spacing, and visible loading/error states.
