"""
Main entry point for the agentic layer.

Two ways to use this:

(A) You already have prediction JSONs saved (e.g. from predict_single_image()
    in your training notebook, following the team's agreed contract). Point
    this script at a folder of .json files and it runs each through the
    Retriever -> Synthesizer -> Auditor pipeline.

(B) Run inference straight from a trained checkpoint. `load_model()` and
    `predict_single_image()` now point at model_inference.py, which builds a
    standard torchvision EfficientNet-B3, loads your state_dict, runs MC
    Dropout (N=20) + Temperature Scaling, and applies your frozen deferral
    threshold theta* — all the constants (checkpoint path, T, theta*) live in
    config.py. If your actual training code built the model differently
    (different backbone wrapper, timm, a custom head), edit
    model_inference.build_model() to match — everything else stays the same.

NOTE on spatial_features: if teammate 3's Grad-CAM++ -> Otsu -> contour
moments step isn't finished yet, spatial_features stays null (model_inference.py
already returns it that way for ACCEPTED scans). The retriever gracefully
falls back to querying on predicted_class alone — it just won't get the
location-conditioned retrieval benefit (your novelty claim #1) until that
field is populated.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Option A: pre-computed JSON predictions
    python run_pipeline.py --json_dir ./sample_predictions --out_dir ./results

    # Option B: run inference directly from your checkpoint
    python run_pipeline.py --image_dir ./scans --out_dir ./results
"""

import argparse
import json
import os
import sys

from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent
from agents.auditor import AuditorAgent
from graph import build_graph


def process_image_dir(image_dir: str, out_dir: str):
    from model_inference import load_model, predict_single_image

    os.makedirs(out_dir, exist_ok=True)

    print("Loading trained checkpoint...")
    model = load_model()

    print("Loading agents (one-time model load: embedder, LLM client, NLI cross-encoder)...")
    retriever = RetrieverAgent()
    synthesizer = SynthesizerAgent()
    auditor = AuditorAgent()
    graph = build_graph(retriever, synthesizer, auditor)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print(f"No image files found in {image_dir}")
        sys.exit(1)

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        scan_id = fname.rsplit(".", 1)[0]

        print(f"\nRunning inference on {scan_id}...")
        prediction_json = predict_single_image(model, image_path, scan_id=scan_id)
        print(f"  -> status={prediction_json['status']}, predicted_class={prediction_json.get('predicted_class')}")

        _run_agentic_layer_and_save(graph, prediction_json, scan_id, out_dir)


# ---------------------------------------------------------------------------


def _run_agentic_layer_and_save(graph, prediction_json, scan_id, out_dir):
    final_state = graph.invoke({"prediction_json": prediction_json})

    output = {
        "scan_id": scan_id,
        "prediction_json": prediction_json,
        "retrieval_query": final_state["retrieval_query"],
        "retrieved_chunks": final_state["retrieved_chunks"],  # saved so the Chat Agent can reuse the same grounding context
        "report_text": final_state["report_text"],
        "grounding_score": final_state["audit_result"]["grounding_score"],
        "audit_result": final_state["audit_result"],
        "retry_count": final_state.get("retry_count", 0),
    }

    out_path = os.path.join(out_dir, f"{scan_id}_report.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  -> Grounding Score: {output['grounding_score']}%"
          + (f" (after {output['retry_count']} re-retrieval retry)" if output["retry_count"] else ""))
    print(f"  -> saved to {out_path}")
    return output


def process_json_dir(json_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading agents (one-time model load: embedder, LLM client, NLI cross-encoder)...")
    retriever = RetrieverAgent()
    synthesizer = SynthesizerAgent()
    auditor = AuditorAgent()
    graph = build_graph(retriever, synthesizer, auditor)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No .json files found in {json_dir}")
        sys.exit(1)

    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path) as f:
            prediction_json = json.load(f)

        scan_id = prediction_json.get("scan_id", fname.replace(".json", ""))
        print(f"\nProcessing {scan_id} (status={prediction_json.get('status')})...")

        _run_agentic_layer_and_save(graph, prediction_json, scan_id, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json_dir", help="Folder of pre-computed prediction JSON files")
    group.add_argument("--image_dir", help="Folder of MRI images — runs inference via your checkpoint first")
    parser.add_argument("--out_dir", default="./results", help="Folder to write report JSONs to")
    args = parser.parse_args()

    if args.json_dir:
        process_json_dir(args.json_dir, args.out_dir)
    else:
        process_image_dir(args.image_dir, args.out_dir)
