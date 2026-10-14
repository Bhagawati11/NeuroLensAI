
"""
Retriever Agent.

Core novelty claim #1: retrieval queries use predicted class and spatial
features for accepted scans. Deferred scans use their candidate classes and
deferral context without inventing a predicted class.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "who_cns_2021"
EMBED_MODEL = "all-MiniLM-L6-v2"


class RetrieverAgent:
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def build_query(self, prediction_json: dict) -> str:
        status = prediction_json.get("status", "UNKNOWN").upper()
        predicted_class = prediction_json.get("predicted_class")
        spatial = prediction_json.get("spatial_features") or {}

        if predicted_class:
            # Preserve the existing accepted-scan retrieval behavior.
            parts = [predicted_class.replace("_", " ")]
        else:
            # Deferred scans have candidate classes, not a confirmed class.
            top_classes = prediction_json.get("top_classes") or []
            parts = [
                "WHO 2021 CNS tumor classification",
                "deferred prediction clinician review",
            ]

            if top_classes:
                parts.append(
                    "candidate classes " + " ".join(
                        str(cls).replace("_", " ") for cls in top_classes
                    )
                )

        if spatial.get("anatomical_zone"):
            parts.append(spatial["anatomical_zone"])

        if spatial.get("laterality"):
            parts.append(spatial["laterality"])

        if spatial.get("boundary_type"):
            parts.append(f"{spatial['boundary_type']} margin")

        # Add deferral context without treating it as a diagnosis.
        if status == "DEFERRED" and prediction_json.get("message"):
            parts.append("classification uncertainty")

        return " ".join(parts)

    def retrieve(self, prediction_json: dict, k: int = 5) -> dict:
        query = self.build_query(prediction_json)
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        return {
            "query": query,
            "chunks": results["documents"][0],
            "distances": (
                results["distances"][0] if "distances" in results else None
            ),
        }
