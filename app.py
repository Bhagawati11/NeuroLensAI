
import hashlib
import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from config import CLASS_NAMES
from model_inference import load_model, predict_single_image, _preprocess, _device
from gradcam import GradCAMPlusPlus, overlay_heatmap
from graph import run_pipeline
from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent
from agents.auditor import AuditorAgent
from agents.chat import ChatAgent


# ---------------------------------------------------------
# PAGE CONFIG + DESIGN
# ---------------------------------------------------------

st.set_page_config(
    page_title="NeuroLens AI | Brain MRI Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0a1020;
        --panel: #111a2e;
        --panel2: #151f36;
        --border: #273653;
        --text: #e8efff;
        --muted: #96a7c5;
        --teal: #42e0c1;
        --purple: #a99bff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #172846 0%, transparent 38%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #0d1628;
        border-right: 1px solid var(--border);
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #304362;
        border-radius: 22px;
        background: linear-gradient(120deg, #172b43, #12192d 65%, #202044);
        margin-bottom: 25px;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        line-height: 1.12;
        margin-bottom: 12px;
        color: #f3f6ff;
    }

    .hero-copy {
        color: #aabbd5;
        max-width: 780px;
        line-height: 1.7;
    }

    .section-label {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 18px 0 8px;
    }

    .metric-card {
        background: linear-gradient(145deg, #151f35, #10182a);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 10px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #f3f6ff;
        overflow-wrap: anywhere;
    }

    .metric-note {
        font-size: 0.76rem;
        color: var(--muted);
        margin-top: 5px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .pill {
        display: inline-block;
        border-radius: 30px;
        padding: 5px 11px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    .pill-ok {
        background: #123c39;
        color: #62f2d1;
        border: 1px solid #216b61;
    }

    .pill-warn {
        background: #49351b;
        color: #ffd58b;
        border: 1px solid #795b2c;
    }

    .disclaimer {
        padding: 14px 16px;
        border: 1px solid #66532b;
        background: #241f18;
        color: #e9d8b4;
        border-radius: 12px;
        font-size: 0.84rem;
        line-height: 1.6;
        margin-top: 20px;
    }

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #3d796f;
        background: linear-gradient(100deg, #42e0c1, #76e9d4);
        color: #071923;
        font-weight: 700;
        min-height: 44px;
    }

    div.stButton > button:hover {
        border-color: #83ffe7;
        color: #071923;
        filter: brightness(1.06);
    }

    div[data-testid="stFileUploader"] {
        background: #111a2e;
        border: 1px dashed #42617f;
        border-radius: 14px;
        padding: 10px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    hr {
        border-color: var(--border);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_agents():
    return (
        RetrieverAgent(),
        SynthesizerAgent(),
        AuditorAgent(),
        ChatAgent(),
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 NeuroLens AI")
    st.caption("Brain MRI research workspace")
    st.divider()

    st.markdown("### Analysis workflow")
    st.markdown("01 · Image upload")
    st.markdown("02 · Model inference")
    st.markdown("03 · Evidence retrieval")
    st.markdown("04 · Report synthesis")
    st.markdown("05 · Claim-level audit")

    st.divider()
    st.markdown("### System")
    st.caption("EfficientNet-B3")
    st.caption("MC Dropout + temperature scaling")
    st.caption("WHO CNS 2021 retrieval")
    st.caption("Gemini-powered report synthesis")

    st.divider()
    st.caption("Academic research prototype · Not a diagnostic tool")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-assisted neuroimaging research</div>
        <div class="hero-title">Brain MRI Analysis<br>with Evidence-Grounded AI</div>
        <div class="hero-copy">
            Explore model predictions, visualize Grad-CAM++ attention,
            retrieve relevant WHO classification excerpts, and inspect
            the evidence behind a generated research report.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">01 / Scan input</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an MRI image to begin",
    type=["jpg", "jpeg", "png"],
    help="Upload one MRI image. JPG, JPEG and PNG are supported.",
)

if uploaded_file is None:
    st.info("Upload an MRI image to begin your analysis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">01 · CLASSIFICATION</div>'
            '<div class="metric-value">EfficientNet-B3</div>'
            '<div class="metric-note">Calibrated image inference</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">02 · EXPLAINABILITY</div>'
            '<div class="metric-value">Grad-CAM++</div>'
            '<div class="metric-note">Visual attention overlay</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">03 · GROUNDED REPORTING</div>'
            '<div class="metric-value">RAG + Audit</div>'
            '<div class="metric-note">Retrieved evidence and claim checks</div></div>',
            unsafe_allow_html=True,
        )

    st.stop()


image_bytes = uploaded_file.getvalue()
image_hash = hashlib.sha256(image_bytes).hexdigest()
image = Image.open(uploaded_file).convert("RGB")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### Original MRI")
    st.image(image, use_container_width=True)

with right:
    st.markdown("#### Analysis")
    st.markdown(
        """
        <div class="panel">
            <div class="metric-label">INPUT FILE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Dimensions:** {image.width} × {image.height} pixels")
    st.write("**Pipeline:** Inference → Retrieval → Synthesis → Audit")

    analyze_clicked = st.button(
        "Run complete analysis",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# ANALYSIS: RUN ONLY ON BUTTON CLICK
# ---------------------------------------------------------

if analyze_clicked:
    try:
        with st.status("Running the analysis pipeline...", expanded=True) as status:
            st.write("Loading the trained model...")
            model = get_model()

            st.write("Loading retrieval, synthesis and audit agents...")
            retriever, synthesizer, auditor, chat_agent = get_agents()

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    temp_file.write(image_bytes)
                    temp_path = temp_file.name

                st.write("Running calibrated model inference...")
                prediction = predict_single_image(
                    model, temp_path, scan_id=uploaded_file.name
                )

                overlay = None
                if prediction["status"] == "ACCEPTED":
                    st.write("Generating Grad-CAM++ visualization...")
                    input_tensor = _preprocess(image).unsqueeze(0).to(_device)

                    cam = GradCAMPlusPlus(model)
                    heatmap, _ = cam.generate(
                        input_tensor,
                        class_index=CLASS_NAMES.index(
                            prediction["predicted_class"]
                        ),
                    )
                    overlay = overlay_heatmap(
                        np.array(image.resize((300, 300))), heatmap
                    )

                st.write("Retrieving WHO evidence and generating the report...")
                result = run_pipeline(
                    prediction,
                    retriever=retriever,
                    synthesizer=synthesizer,
                    auditor=auditor,
                )

                st.session_state["analysis_data"] = {
                    "image_hash": image_hash,
                    "prediction": prediction,
                    "overlay": overlay,
                    "result": result,
                }

                status.update(
                    label="Analysis pipeline completed",
                    state="complete",
                    expanded=False,
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    except Exception as exc:
        st.error("The analysis could not be completed.")
        st.exception(exc)


# Only show results for the currently uploaded image.
data = st.session_state.get("analysis_data")

if not data or data["image_hash"] != image_hash:
    st.info("Click **Run complete analysis** to generate the prediction and report.")
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Research-use notice:</strong> This application is an
            academic research prototype. It is not a medical device and must
            not be used to diagnose, treat, or make clinical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------
# PREDICTION SUMMARY
# ---------------------------------------------------------

prediction = data["prediction"]
result = data["result"]
overlay = data["overlay"]

st.markdown('<div class="section-label">02 / Model results</div>', unsafe_allow_html=True)
st.subheader("Analysis overview")

status_value = prediction.get("status", "UNKNOWN")
is_accepted = status_value == "ACCEPTED"

if is_accepted:
    predicted_class = prediction.get("predicted_class", "Unknown")
    st.success(f"Model status: ACCEPTED · Predicted class: {predicted_class.upper()}")
else:
    st.warning(
        "DEFERRED FOR CLINICIAN REVIEW — the model did not confidently "
        "classify this scan. No AI-generated diagnosis should be inferred."
    )

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">MODEL STATUS</div>'
        f'<div class="metric-value">{status_value}</div>'
        f'<div class="metric-note">Inference decision</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    confidence = prediction.get("calibrated_confidence")
    confidence_text = f"{confidence:.1%}" if confidence is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">CONFIDENCE</div>'
        f'<div class="metric-value">{confidence_text}</div>'
        f'<div class="metric-note">Model output, not clinical certainty</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    entropy = prediction.get("predictive_entropy")
    entropy_text = f"{entropy:.4f}" if entropy is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">PREDICTIVE ENTROPY</div>'
        f'<div class="metric-value">{entropy_text}</div>'
        f'<div class="metric-note">Uncertainty measure</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    score = result.get("audit_result", {}).get("grounding_score")
    score_text = f"{score}%" if score is not None else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">GROUNDING SCORE</div>'
        f'<div class="metric-value">{score_text}</div>'
        f'<div class="metric-note">Automated report audit</div></div>',
        unsafe_allow_html=True,
    )

if not is_accepted:
    st.error(
        "THIS SCAN WAS NOT CONFIDENTLY CLASSIFIED BY THE MODEL AND REQUIRES "
        "DIRECT CLINICIAN REVIEW. NO AI-GENERATED DIAGNOSIS IS PROVIDED."
    )

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

st.markdown('<div class="section-label">03 / Research workspace</div>', unsafe_allow_html=True)

overview_tab, report_tab, evidence_tab, chat_tab = st.tabs(
    ["🧠 Overview", "📄 AI Report", "🔎 Evidence & Audit", "💬 Ask AI"]
)

with overview_tab:
    if is_accepted:
        image_col, detail_col = st.columns([1.1, 0.9], gap="large")

        with image_col:
            st.markdown("#### Grad-CAM++ visualization")
            if overlay is not None:
                st.image(
                    overlay,
                    caption="Model attention visualization — not a tumor segmentation mask",
                    use_container_width=True,
                )
            else:
                st.info("Grad-CAM++ visualization is unavailable for this scan.")

        with detail_col:
            st.markdown("#### Prediction details")
            st.write(f"**Predicted class:** {prediction.get('predicted_class', 'N/A')}")
            st.write(f"**Confidence:** {confidence_text}")
            st.write(f"**Predictive entropy:** {entropy_text}")

            spatial = prediction.get("spatial_features")
            if spatial:
                st.markdown("#### Extracted spatial features")
                st.json(spatial)
            else:
                st.info("Spatial features were not available.")

    else:
        st.markdown("#### Deferred prediction")
        st.write("Candidate classes:", prediction.get("top_classes", []))
        st.write("Deferral reason:", prediction.get("message", "Not provided"))
        st.info(
            "The model's candidate classes are not a confirmed classification. "
            "Review by a qualified clinician is required."
        )

    st.caption(
        "Grad-CAM++ highlights regions influencing the model. It does not establish "
        "tumor boundaries, pathology, or clinical localization."
    )


with report_tab:
    st.markdown("#### Synthesized research report")
    st.markdown(result.get("report_text", "No report text was returned."))

    st.download_button(
        "Download report as Markdown",
        data=result.get("report_text", ""),
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_report.md",
        mime="text/markdown",
        use_container_width=False,
    )

    with st.expander("View complete prediction and pipeline output"):
        st.json(
            {
                "prediction": prediction,
                "retrieval_query": result.get("retrieval_query"),
                "retry_count": result.get("retry_count", 0),
            }
        )


with evidence_tab:
    st.markdown("#### Automated claim-level audit")

    audit = result.get("audit_result", {})
    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Supported claims", audit.get("supported", "N/A"))
    with a2:
        st.metric("Contradicted claims", audit.get("contradicted", "N/A"))
    with a3:
        st.metric("Unverified claims", audit.get("unverified", "N/A"))

    if score is not None:
        st.progress(max(0.0, min(1.0, float(score) / 100.0)))
        st.caption(
            "Grounding score measures automated evidence alignment. "
            "It does not measure diagnostic accuracy."
        )

    with st.expander("Claim details", expanded=True):
        st.json(audit.get("claim_details", []))

    st.markdown("#### Retrieved WHO excerpts")
    chunks = result.get("retrieved_chunks") or []

    if chunks:
        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"Evidence excerpt {index}"):
                st.write(chunk)
    else:
        st.info("No evidence excerpts were returned.")


with chat_tab:
    st.markdown("#### Ask a follow-up question")
    st.caption(
        "Ask about the generated report or retrieved evidence. "
        "Answers are constrained by the available report context."
    )

    with st.form("follow_up_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="Which retrieved excerpt supports a statement in the report?",
            height=100,
        )
        ask_clicked = st.form_submit_button(
            "Submit question",
            use_container_width=True,
        )

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                _, _, _, chat_agent = get_agents()
                with st.spinner("Generating an evidence-grounded answer..."):
                    answer = chat_agent.ask(
                        question.strip(),
                        result.get("report_text", ""),
                        result.get("retrieved_chunks", []),
                    )
                st.markdown("#### Answer")
                st.markdown(answer)
            except Exception as exc:
                st.error("The follow-up answer could not be generated.")
                st.exception(exc)


st.markdown(
    """
    <div class="disclaimer">
        <strong>Research prototype — not for clinical use.</strong>
        Predictions, confidence estimates, Grad-CAM++ visualizations, generated
        reports, and automated grounding scores require independent validation.
        Do not use this application as a substitute for professional medical review.
    </div>
    """,
    unsafe_allow_html=True,
)
