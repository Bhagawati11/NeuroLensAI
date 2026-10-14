"""
Auditor / Reviewer Agent.

Splits the synthesized report into sentences, checks each one against the
retrieved source text with cross-encoder/nli-deberta-v3-base, and computes
the Grounding Score = % of claims marked Supported.

Thresholds (as locked in your architecture):
  entailment >= 0.70   -> Supported
  contradiction >= 0.50 -> Contradicted
  else                  -> Unverified
"""

import re
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# label order for this checkpoint's output logits
# (cross-encoder/nli-deberta-v3-base outputs [contradiction, entailment, neutral])
LABELS = ["contradiction", "entailment", "neutral"]


def split_sentences(text: str) -> list[str]:
    # simple sentence splitter; swap for nltk/spacy if you want more robustness
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


class AuditorAgent:
    def __init__(self):
        self.model = CrossEncoder(NLI_MODEL)

    def _classify_claim(self, claim: str, source_chunks: list[str]) -> dict:
        """Checks one claim against every retrieved chunk, keeps the strongest signal."""
        import numpy as np

        pairs = [(chunk, claim) for chunk in source_chunks]
        scores = self.model.predict(pairs)  # shape (n_chunks, 3) -> [contra, entail, neutral]
        probs = np.array(scores)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        best_chunk_idx = int(np.argmax(probs[:, LABELS.index("entailment")]))
        best_entail = float(probs[best_chunk_idx, LABELS.index("entailment")])
        best_contra_idx = int(np.argmax(probs[:, LABELS.index("contradiction")]))
        best_contra = float(probs[best_contra_idx, LABELS.index("contradiction")])

        if best_entail >= 0.70:
            verdict = "Supported"
        elif best_contra >= 0.50:
            verdict = "Contradicted"
        else:
            verdict = "Unverified"

        return {
            "claim": claim,
            "verdict": verdict,
            "entailment_score": round(best_entail, 4),
            "contradiction_score": round(best_contra, 4),
            "best_matching_chunk": source_chunks[best_chunk_idx][:200] + "...",
        }

    def audit(self, report_text: str, source_chunks: list[str]) -> dict:
        claims = split_sentences(report_text)
        results = [self._classify_claim(c, source_chunks) for c in claims]

        supported = sum(1 for r in results if r["verdict"] == "Supported")
        grounding_score = round(100.0 * supported / len(results), 2) if results else 0.0

        return {
            "grounding_score": grounding_score,
            "total_claims": len(results),
            "supported": supported,
            "contradicted": sum(1 for r in results if r["verdict"] == "Contradicted"),
            "unverified": sum(1 for r in results if r["verdict"] == "Unverified"),
            "claim_details": results,
        }
