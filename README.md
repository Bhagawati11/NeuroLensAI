# Agentic layer — Retriever → Synthesizer → Auditor (→ Chat)

This is the multi-agent system that sits on top of your trained EfficientNet-B3
classifier. `model_inference.py` now loads your actual PyTorch checkpoint and
runs full inference (MC Dropout + Temperature Scaling + deferral) — you don't
need to paste anything in yourself unless your training code built the model
differently than assumed (see step 0b).

## 0. Install

```bash
pip install -r requirements.txt
```

## 0b. Point this at your checkpoint

Open `config.py` and set:
- `CHECKPOINT_PATH` — path to your `.pth` file
- `TEMPERATURE` / `THETA_STAR` — your frozen values from the measured run (already filled in with 1.2402 / 0.0669 from your last recorded run — update if you retrained since, since training wasn't seeded until you added `set_full_seed()`)
- `CLASS_NAMES` — must be in the exact order your training label encoding used

`model_inference.py` assumes a **standard torchvision `efficientnet_b3`** with
the classifier head swapped to `Linear(in_features, 4)`, and a checkpoint
saved as a plain `state_dict`. If your actual training notebook built the
model differently (timm instead of torchvision, a custom head, etc.), the
only thing you need to change is `build_model()` in that file — everything
downstream just needs a model that takes `[1, 3, 300, 300]` and returns
`[1, 4]` logits.

Quick sanity check that loading actually works:
```bash
python model_inference.py path/to/one_scan.jpg
```
This should print a single prediction JSON (ACCEPTED or DEFERRED). If it
errors on `load_state_dict`, your checkpoint's architecture doesn't match
`build_model()` — that's the thing to fix before running the full pipeline.

## 1. Build the knowledge base (run once)

```bash
python 01_index_who_cns.py
```

This downloads Louis et al. 2021 (WHO CNS tumor classification summary),
chunks it, embeds it, and saves a persistent ChromaDB at `./chroma_db/`.

**Do not skip the printed spot-check.** It prints the first 400 chars and a
middle 400-char slice of what it fetched — read them and confirm it's really
the WHO paper, not a stub. If the automatic fetch comes back thin, it tells
you to drop a manually-downloaded PDF (`who_cns_2021.pdf`) next to the script
and rerun — the PDF fallback path is already wired in.

## 2. Set your LLM API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

(Or swap `agents/synthesizer.py` and `agents/chat.py` to use whatever
provider you have credits for — the prompt templates don't need to change,
just the client call.)

## 3. Run the pipeline end-to-end on your real scans

```bash
python run_pipeline.py --image_dir ./scans --out_dir ./results
```

This runs your checkpoint on every image in `./scans`, then feeds each
prediction through Retriever → Synthesizer → Auditor. Each output file in
`./results/` has: the raw prediction, the retrieval query used, the full
retrieved context, the report text, the Grounding Score, and the per-claim
audit breakdown.

If you already have prediction JSONs saved from elsewhere instead of raw
images, use `--json_dir ./your_predictions` instead — same output either way.

**Sanity-check first with the 3 included sample JSONs** before running real
scans, so a bug surfaces on fake data instead of eating your remaining time:
```bash
python run_pipeline.py --json_dir ./sample_predictions --out_dir ./results
```
- `scan_001_accepted.json` — meningioma, full spatial features
- `scan_002_deferred.json` — deferred case, triggers the Clinician Triage Packet path
- `scan_003_pituitary_accepted.json` — confirms the synthesizer flags the
  spatial-feature caveat for pituitary (your own finding: Pointing Game
  strict accuracy 0.57% for this class)

## 4. Follow-up Q&A (Chat Agent)

```bash
python agents/chat.py results/scan_001_report.json
```
Answers a question about that specific report, grounded in the same
retrieved chunks that produced it — not a fresh free-form answer. To use it
inside your own UI instead of the CLI:
```python
from agents.chat import ChatAgent
agent = ChatAgent()
answer = agent.ask("Why wasn't this deferred?", result["report_text"], result["retrieved_chunks"])
```

## How these agents actually work (the thing the team hasn't discussed yet)

There are two fundamentally different ways to build "agents" here, and it's
worth being explicit about which one this is before it goes in the paper:

**1. Deterministic pipeline (what this is, mostly).** Each agent is a plain
function with one fixed job — Retriever always retrieves, Synthesizer always
drafts from exactly what it's given, Auditor always checks claims against
exactly those retrieved chunks. LangGraph just wires them in a fixed order.
No agent decides what to do next, no agent calls its own tools, nothing can
go off-script. This is the right default for your project specifically,
because your entire pitch is *groundedness and reliability* — the auditor's
job is to catch hallucination, and a pipeline where every step's input/output
is fixed and inspectable is much easier to defend ("here's exactly what went
into this report, here's exactly what was checked") than a system where an
LLM is freely deciding what to retrieve or when it's "done."

**2. Autonomous agent loop (ReAct-style — NOT what this is, and deliberately
so).** The LLM gets tools (e.g. `retrieve(query)`, `check_grounding(claim)`)
and decides for itself what to call, in what order, how many times, when to
stop. More flexible, genuinely more "agentic" in the strict sense, but much
harder to bound — the model can loop unpredictably, retrieve badly and not
notice, or decide it's "done" prematurely. For a system whose core claim is
trustworthiness, that unpredictability is a cost, not a feature. It's also
much harder to defend to your professor ("why should I trust this report")
than "every claim is checked against a specific retrieved source, and if
grounding is weak the system automatically retries before it ever reaches a
human."

**What's actually in this build is a hybrid, closer to (1) with one real
decision point from (2):** `graph.py`'s `audit_route()` is a genuine
conditional edge — if the Grounding Score comes back below 60%, the graph
loops back to Retriever with a wider query (more chunks) before accepting
the report, capped at one retry so it can't loop forever. That's the part
honestly worth calling "agentic" in your writeup: a real decision made from
the system's own output, not just a fixed sequence of LLM calls. If your
professor pushes on "why is this agentic and not just a script," this retry
loop plus the three distinct, specialized roles (retrieve/synthesize/audit)
is the answer — not "it uses LangGraph."

If the team wants a fuller autonomous-agent version later (closer to your
friend's repo's framing, which gives the LLM more free rein across
knowledge/report/chat agents), that's a bigger, riskier change this close to
a moved-up deadline — worth a deliberate team decision, not a default.

## Status of everything, honestly

Tested here means: I ran it myself in this sandbox against synthetic/dummy
inputs (a random-weight model, a fake image, mocked agents) and confirmed
the logic actually executes correctly — not just that it compiles. I do
NOT have your real checkpoint, your real dataset, or a GPU here, so nothing
below has been validated against your actual trained model or real scans —
that step is still yours to do.

| Piece | Code status | What's left for you |
|---|---|---|
| ChromaDB / WHO paper indexing | written | run it, read the spot-check output |
| Retriever / Synthesizer / Auditor pipeline | written + logic-tested with mocks | run against real predictions |
| Conditional retry edge (low-grounding re-retrieval) | written + tested (forced a low score, confirmed exactly one retry) | nothing — this one's solid |
| Chat Agent | written | try it against a real report |
| Model loading + MC Dropout + calibration + deferral | written + tested with a dummy model (both ACCEPTED and DEFERRED branches confirmed to fire correctly) | point `config.py` at your real checkpoint, confirm `load_state_dict` succeeds |
| Grad-CAM++ (`gradcam.py`) | written + tested end-to-end (ran on a dummy model, produced a valid heatmap with no NaNs) | spot-check it against your training notebook's own verified Grad-CAM++ (the one you ran the Pointing Game against) — if they disagree, the notebook's is the source of truth for the paper |
| `spatial_features.py` (Otsu + contour moments) | written + tested on a synthetic heatmap, correctly recovered the blob's location/size/shape | **read the caveats at the top of the file before trusting this in the paper** — laterality is image-side not clinical side, anatomical_zone is a coarse quadrant not a real atlas lobe. Good enough for a demo, likely not rigorous enough for the novelty-claim writeup without more work |
| `scripts/derive_no_tumor_threshold.py` | method written, cannot produce a real number | needs to run against your actual val set + model in Kaggle — I only wrote the approach (95th percentile of the no_tumor activation-area distribution), the actual threshold number has to come from your data |
| Streamlit UI (`app.py`) | written, not run (no streamlit runtime here) | run `streamlit run app.py` yourself and see what breaks — this is the single least-tested file in this zip |

## Still genuinely open

- **This is teammate 4's locked responsibility** per your team split (RAG &
  multi-agent) — if they've already started building this independently,
  diff against what's here before replacing their work, so you don't end up
  with two conflicting "final" implementations going into the paper.
- Nothing here has touched real MRI images or your real checkpoint. Treat
  every "tested" claim above as "the logic is sound," not "this will work
  on the first try against your actual data" — budget time for at least one
  debugging pass once you plug in the real checkpoint.

## File map

```
config.py                      # YOUR checkpoint path, temperature, theta*, class names — edit this first
model_inference.py             # loads your PyTorch checkpoint, runs MC Dropout + calibration + deferral + spatial features
gradcam.py                     # Grad-CAM++ for the torchvision EfficientNet-B3 model
spatial_features.py            # Otsu + contour moments -> laterality/zone/area/boundary (read its caveats)
01_index_who_cns.py            # one-time: build ChromaDB from the WHO paper
agents/retriever.py            # builds the location-conditioned query, queries ChromaDB
agents/synthesizer.py          # LLM call -> drafts the report (ACCEPTED or DEFERRED template)
agents/auditor.py              # NLI cross-encoder -> per-claim Supported/Contradicted/Unverified + Grounding Score
agents/chat.py                 # follow-up Q&A grounded in a report's retrieved context
graph.py                       # LangGraph wiring: retrieve -> synthesize -> audit -> (retry or end)
run_pipeline.py                # CLI entry point — --image_dir (real inference) or --json_dir (pre-computed)
app.py                         # Streamlit demo UI — streamlit run app.py
scripts/derive_no_tumor_threshold.py  # method for an empirical no_tumor threshold — needs your val set to run
sample_predictions/            # 3 example JSONs to sanity-check the pipeline end-to-end
```
