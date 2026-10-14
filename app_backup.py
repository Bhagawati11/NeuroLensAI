"""
Minimal Streamlit demo UI — upload a scan, see predicted class, Grad-CAM++
overlay, the synthesized report, the Grounding Score, and ask follow-up
questions.

This is for demoing the pipeline, not a clinical interface — same
"research prototype, not a diagnosis" framing as everywhere else in this
project.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image

from model_inference import load_model, predict_single_image, _preprocess, _device
from gradcam import GradCAMPlusPlus, overlay_heatmap
from graph import run_pipeline
from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent
from agents.auditor import AuditorAgent
from agents.chat import ChatAgent


st.set_page_config(page_title="Brain Tumor Agentic AI — Demo", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_agents():
    return RetrieverAgent(), SynthesizerAgent(), AuditorAgent(), ChatAgent()


st.title("Brain Tumor Agentic AI — Research Prototype")
st.caption(
    "Calibrated EfficientNet-B3 classifier + uncertainty-based deferral + "
    "RAG-grounded report synthesis + claim-level auditing. "
    "Research/academic demo only — not a diagnostic tool."
)

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded scan", use_container_width=True)

    with st.spinner("Loading model..."):
        model = get_model()

    with st.spinner("Running inference (MC Dropout, 20 passes)..."):
        # predict_single_image expects a file path; save the upload to a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            prediction = predict_single_image(model, tmp.name, scan_id=uploaded_file.name)

    st.subheader("Model prediction")
    if prediction["status"] == "ACCEPTED":
        st.success(
            f"**{prediction['predicted_class'].upper()}** — "
            f"confidence {prediction['calibrated_confidence']:.1%}, "
            f"entropy {prediction['predictive_entropy']:.4f}"
        )
        if prediction.get("spatial_features"):
            st.json(prediction["spatial_features"])

        # Grad-CAM overlay
        with st.spinner("Generating Grad-CAM++ overlay..."):
            input_tensor = _preprocess(image).unsqueeze(0).to(_device)
            class_idx = ["glioma", "meningioma", "notumor", "pituitary"].index(prediction["predicted_class"])
            cam = GradCAMPlusPlus(model)
            heatmap, _ = cam.generate(input_tensor, class_index=class_idx)
            overlay = overlay_heatmap(np.array(image.resize((300, 300))), heatmap)
        with col2:
            st.image(overlay, caption="Grad-CAM++ attention overlay", use_container_width=True)
    else:
        st.warning(
            f"**DEFERRED FOR CLINICIAN REVIEW** — entropy {prediction['predictive_entropy']:.4f} "
            f"exceeded threshold. Top candidates: {', '.join(prediction['top_classes'])}"
        )

    st.divider()
    st.subheader("Agentic report")

    with st.spinner("Retrieving evidence, synthesizing report, auditing claims..."):
        retriever, synthesizer, auditor, chat_agent = get_agents()
        result = run_pipeline(prediction, retriever=retriever, synthesizer=synthesizer, auditor=auditor)

    st.markdown(result["report_text"])

    score = result["audit_result"]["grounding_score"]
    color = "green" if score >= 80 else ("orange" if score >= 60 else "red")
    st.markdown(f"**Grounding Score:** :{color}[{score}%]  "
                f"({result['audit_result']['supported']} supported / "
                f"{result['audit_result']['contradicted']} contradicted / "
                f"{result['audit_result']['unverified']} unverified claims)")

    with st.expander("Per-claim audit detail"):
        st.json(result["audit_result"]["claim_details"])

    with st.expander("Retrieved source chunks"):
        for i, chunk in enumerate(result["retrieved_chunks"]):
            st.markdown(f"**Chunk {i+1}:** {chunk}")

    st.divider()
    st.subheader("Ask a follow-up question")
    question = st.text_input("Question about this report")
    if question:
        with st.spinner("Answering..."):
            answer = chat_agent.ask(question, result["report_text"], result["retrieved_chunks"])
        st.markdown(answer)
