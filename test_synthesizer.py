
from agents.synthesizer import SynthesizerAgent

agent = SynthesizerAgent()

accepted_prediction = {
    "status": "ACCEPTED",
    "predicted_class": "glioma",
    "calibrated_confidence": 0.95,
    "predictive_entropy": 0.12,
    "spatial_features": {"test": "sample features"}
}

deferred_prediction = {
    "status": "DEFERRED",
    "predictive_entropy": 0.85,
    "top_classes": ["glioma", "meningioma"]
}

sample_chunks = [
    "Sample guideline excerpt for testing only. "
    "Do not use this as actual WHO classification evidence."
]

for prediction in (accepted_prediction, deferred_prediction):
    print("\n" + "=" * 60)
    print("Testing:", prediction["status"])

    try:
        report = agent.synthesize(prediction, sample_chunks)
        print(report)
    except Exception as exc:
        print(f"Test failed: {type(exc).__name__}: {exc}")
