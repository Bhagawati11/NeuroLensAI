"""
Chat Agent — answers a clinician's follow-up question about a specific
report, grounded in the same retrieved WHO-guideline chunks that produced
it (not a fresh free-form LLM answer).

This is reactive (one question in, one answer out) rather than a pipeline
step, so it isn't a LangGraph node like the other three — it's called
directly after run_pipeline() has produced a report, reusing that report's
retrieved context.
"""

import os
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """You are answering a clinician's follow-up question about an AI-assisted brain-tumor \
MRI research report. Answer only from the retrieved guideline excerpts and the report already given to \
you. If the retrieved material doesn't cover the question, say so plainly rather than answering from \
general knowledge. This is a research prototype, not a diagnostic tool — never present an answer as a \
confirmed clinical fact."""

USER_TEMPLATE = """Original report:
{report_text}

Retrieved WHO 2021 CNS classification excerpts (the only source you may draw factual claims from):
{retrieved_chunks}

Clinician's follow-up question:
{question}"""



import os
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"


class ChatAgent:
    def __init__(self, api_key: str | None = None, model: str = MODEL):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=key)
        self.model = model

    def ask(
        self,
        question: str,
        report_text: str,
        retrieved_chunks: list[str],
    ) -> str:
        chunks_text = "\n\n---\n\n".join(retrieved_chunks)

        user_prompt = USER_TEMPLATE.format(
            report_text=report_text,
            retrieved_chunks=chunks_text,
            question=question,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=512,
                temperature=0.2,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text


if __name__ == "__main__":
    # quick manual test: python agents/chat.py <path_to_a_results/*_report.json>
    import sys
    import json

    with open(sys.argv[1]) as f:
        result = json.load(f)

    agent = ChatAgent()
    question = input("Ask a follow-up question about this report: ")
    answer = agent.ask(question, result["report_text"], result["retrieved_chunks"])
    print(f"\n{answer}")
