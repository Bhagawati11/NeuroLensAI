"""
Step 1 — Build the ChromaDB knowledge base from Louis et al. 2021
"The 2021 WHO Classification of Tumors of the Central Nervous System: a summary"
(Neuro-Oncology, 23(8):1231-1251, DOI 10.1093/neuonc/noab106, PMC8328013)

Run this ONCE. It produces a persistent ./chroma_db/ folder that the
retriever agent (agents/retriever.py) reads from.

IMPORTANT: this script fetches the paper automatically via the Europe PMC
full-text API. SPOT-CHECK the printed preview against the real paper before
trusting it — if the fetch returns an abstract-only stub, use the PDF
fallback path below instead (uncomment and point at a manually downloaded
PDF — grab it from https://pmc.ncbi.nlm.nih.gov/articles/PMC8328013/ or
https://doi.org/10.1093/neuonc/noab106).
"""

import re
import sys
import requests

PMCID = "PMC8328013"
OUT_DB_PATH = "./chroma_db"
PDF_FALLBACK_PATH = "who_cns_2021.pdf"  # only used if automatic fetch fails


def fetch_paper_text():
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        raw = r.text
        raw = re.sub(r"<title[^>]*>", "\n\n## ", raw)
        raw = re.sub(r"</title>", "\n", raw)
        raw = re.sub(r"<p[^>]*>", "", raw)
        raw = re.sub(r"</p>", "\n\n", raw)
        text = re.sub(r"<[^>]+>", "", raw)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 3000:
            print(f"Fetched via Europe PMC full-text API — {len(text)} chars")
            return text
        print(f"Europe PMC returned only {len(text)} chars — likely abstract-only, not full text.")
    except Exception as e:
        print(f"Europe PMC fetch failed: {e}")
    return None


def fetch_paper_text_pdf_fallback(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"Extracted {len(text)} chars from {pdf_path}")
    return text


def chunk_text(text, max_chars=1000, overlap=150):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = (current[-overlap:] + " " if current else "") + para
    if current:
        chunks.append(current)
    return chunks


def main():
    paper_text = fetch_paper_text()

    if paper_text is None:
        import os
        if os.path.exists(PDF_FALLBACK_PATH):
            paper_text = fetch_paper_text_pdf_fallback(PDF_FALLBACK_PATH)
        else:
            print(
                f"\nAutomatic fetch failed and no fallback PDF found at '{PDF_FALLBACK_PATH}'.\n"
                f"Download the paper manually (https://doi.org/10.1093/neuonc/noab106), "
                f"save it as '{PDF_FALLBACK_PATH}' next to this script, and rerun."
            )
            sys.exit(1)

    print("\n--- SPOT CHECK: first 400 chars ---")
    print(paper_text[:400])
    print("--- SPOT CHECK: middle 400 chars ---")
    mid = len(paper_text) // 2
    print(paper_text[mid:mid + 400])
    print("--- end spot check ---\n")

    chunks = chunk_text(paper_text)
    print(f"{len(chunks)} chunks, avg {sum(len(c) for c in chunks)//len(chunks)} chars each")

    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=OUT_DB_PATH)
    collection = client.get_or_create_collection(
        name="who_cns_2021",
        metadata={"source": "Louis et al. 2021, Neuro-Oncology 23(8):1231-1251, DOI 10.1093/neuonc/noab106"},
    )

    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
    )
    print(f"Indexed {collection.count()} chunks into ChromaDB at {OUT_DB_PATH}")


if __name__ == "__main__":
    main()
