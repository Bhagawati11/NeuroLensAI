"""
LangGraph state machine wiring: Retriever -> Synthesizer -> Auditor.

This is mostly a deterministic pipeline, not a free-roaming agent loop — see
the README section "How these agents actually work" for why that's the
right call for a reliability-focused project. The ONE real decision point is
in audit_route(): if the Grounding Score comes back low, the graph loops
back to Retriever with a refined query instead of shipping an ungrounded
report. That's the part worth calling "agentic" in your paper/defense —
it's a conditional edge, a real decision, not just a fixed sequence.

Input:  your predict_single_image() JSON output (ACCEPTED or DEFERRED)
Output: final report text + Grounding Score + full audit trail
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent
from agents.auditor import AuditorAgent

GROUNDING_SCORE_RETRY_THRESHOLD = 60.0  # below this, retry retrieval once with a wider query
MAX_RETRIES = 1


class PipelineState(TypedDict):
    prediction_json: dict
    retrieval_query: Optional[str]
    retrieved_chunks: Optional[list]
    report_text: Optional[str]
    audit_result: Optional[dict]
    retry_count: int


def build_graph(retriever: RetrieverAgent, synthesizer: SynthesizerAgent, auditor: AuditorAgent):
    def retrieve_node(state: PipelineState) -> PipelineState:
        retry_count = state.get("retry_count", 0)
        # on a retry, widen the net (more chunks) instead of just asking again
        k = 5 if retry_count == 0 else 10
        result = retriever.retrieve(state["prediction_json"], k=k)
        return {**state, "retrieval_query": result["query"], "retrieved_chunks": result["chunks"]}

    def synthesize_node(state: PipelineState) -> PipelineState:
        report = synthesizer.synthesize(state["prediction_json"], state["retrieved_chunks"])
        return {**state, "report_text": report}

    def audit_node(state: PipelineState) -> PipelineState:
        audit = auditor.audit(state["report_text"], state["retrieved_chunks"])
        return {**state, "audit_result": audit}

    def audit_route(state: PipelineState) -> str:
        """The one real decision point: retry retrieval if grounding is weak,
        otherwise end. Bounded by MAX_RETRIES so it can't loop forever."""
        score = state["audit_result"]["grounding_score"]
        retry_count = state.get("retry_count", 0)
        if score < GROUNDING_SCORE_RETRY_THRESHOLD and retry_count < MAX_RETRIES:
            return "retry"
        return "end"

    def increment_retry(state: PipelineState) -> PipelineState:
        return {**state, "retry_count": state.get("retry_count", 0) + 1}

    workflow = StateGraph(PipelineState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("increment_retry", increment_retry)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", "audit")
    workflow.add_conditional_edges("audit", audit_route, {"retry": "increment_retry", "end": END})
    workflow.add_edge("increment_retry", "retrieve")

    return workflow.compile()


def run_pipeline(prediction_json: dict, retriever=None, synthesizer=None, auditor=None) -> dict:
    """Convenience entry point. Pass pre-built agents to avoid reloading models repeatedly
    when processing many scans (see run_pipeline.py)."""
    retriever = retriever or RetrieverAgent()
    synthesizer = synthesizer or SynthesizerAgent()
    auditor = auditor or AuditorAgent()

    graph = build_graph(retriever, synthesizer, auditor)
    final_state = graph.invoke({"prediction_json": prediction_json, "retry_count": 0})
    return final_state
