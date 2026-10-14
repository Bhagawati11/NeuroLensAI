"""
Report Synthesizer Agent.

Takes the model's prediction JSON + retrieved WHO-guideline chunks and drafts
the report text. Two distinct output modes per your locked two-tier design:
  - ACCEPTED  -> full diagnostic report
  - DEFERRED  -> Clinician Triage Packet (NOT a diagnosis) with mandatory
                 MD sign-off banner

Uses the GemiNI API. Set GEMINI_API_KEY as an environment variable
before running (export GEMINI_API_KEY="").
"""

import os
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite" 
HF_MODEL_REPO = "Bhagawati08/NeuroLensAI"
HF_MODEL_FILENAME = "best_efficientnet_b3.pth"

SYSTEM_PROMPT = """You are a report-drafting assistant for a research prototype brain-tumor MRI \
classification system. You are NOT a doctor and this is NOT a clinical product — it is an academic \
capstone project. Draft report text strictly grounded in the retrieved guideline excerpts provided \
to you. Do not invent facts, statistics, or claims that are not either (a) directly stated in the \
retrieved excerpts, or (b) directly stated in the structured prediction data you are given. Every \
factual claim about tumor grading, classification criteria, or prognosis must be traceable to the \
retrieved text. If the retrieved excerpts don't cover something, say so explicitly rather than filling \
the gap from general knowledge."""

ACCEPTED_TEMPLATE = """Prediction data (from a calibrated EfficientNet-B3 classifier with MC Dropout \
uncertainty estimation):
- Predicted class: {predicted_class}
- Calibrated confidence: {calibrated_confidence}
- Predictive entropy: {predictive_entropy}
- Spatial features (derived from Grad-CAM++ heatmap via Otsu thresholding + contour moments): {spatial_features}

IMPORTANT CAVEAT for spatial_features: these describe WHERE THE HEATMAP ACTIVATED, not a verified \
tumor segmentation. For the pituitary class specifically, this project's own evaluation found the \
model's Grad-CAM++ attention rarely lands on the true tumor region (quantified Pointing Game strict \
accuracy: 0.57%) even though classification accuracy for pituitary is ~99.8% — the model appears to \
use broader scan geometry cues rather than tumor-local texture for this class. If predicted_class is \
"pituitary", explicitly flag the spatial_features as low-confidence/unreliable localization in the \
report rather than presenting them as a verified tumor location.

Retrieved WHO 2021 CNS classification excerpts (ground every claim in these):
{retrieved_chunks}

Draft a structured diagnostic report with these sections: Finding, Classification Basis (grounded in \
the retrieved excerpts), Confidence & Reliability Notes, and a closing statement that this is an \
AI-assisted research prototype output requiring clinician review, not a standalone diagnosis."""

DEFERRED_TEMPLATE = """This scan was DEFERRED by the system's uncertainty-based selective prediction \
gate (predictive entropy {predictive_entropy} exceeded the validation-tuned threshold).

Top-2 differential classes: {top_classes}

Retrieved WHO 2021 CNS classification excerpts for these differential classes (ground every claim in \
these):
{retrieved_chunks}

Draft a Clinician Triage Packet (NOT a diagnosis) with these sections: Deferral Reason, Differential \
Context (brief, grounded background on the top-2 candidate classes from the retrieved excerpts), and \
a mandatory closing banner in capital letters: "THIS SCAN WAS NOT CONFIDENTLY CLASSIFIED BY THE MODEL \
AND REQUIRES DIRECT CLINICIAN REVIEW. NO AI-GENERATED DIAGNOSIS IS PROVIDED.\""""


class SynthesizerAgent:
    def __init__(self, api_key: str | None = None, model: str = MODEL):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Configure it before running."
            )

        self.client = genai.Client(api_key=key)
        self.model = model

    def synthesize(
        self,
        prediction_json: dict,
        retrieved_chunks: list[str],
    ) -> str:
        chunks_text = "\n\n---\n\n".join(retrieved_chunks)

        if prediction_json["status"] == "ACCEPTED":
            user_prompt = ACCEPTED_TEMPLATE.format(
                predicted_class=prediction_json["predicted_class"],
                calibrated_confidence=prediction_json["calibrated_confidence"],
                predictive_entropy=prediction_json["predictive_entropy"],
                spatial_features=prediction_json.get("spatial_features"),
                retrieved_chunks=chunks_text,
            )

        elif prediction_json["status"] == "DEFERRED":
            user_prompt = DEFERRED_TEMPLATE.format(
                predictive_entropy=prediction_json["predictive_entropy"],
                top_classes=prediction_json.get("top_classes"),
                retrieved_chunks=chunks_text,
            )

        else:
            raise ValueError(
                f"Unknown status: {prediction_json['status']}"
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
                temperature=0.2,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text
