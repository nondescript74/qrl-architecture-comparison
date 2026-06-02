# Integration Classification — qrl-architecture-comparison

**Purpose:** Per-file classification governing how this repo integrates
with the rest of the system going forward. This is an *integration*
pass, not a refactor. Nothing here promotes code out of this repo; it
only names what each file IS so later decisions (exposure surface,
Phoenix evidence citation, dependency hygiene) have a reference.

**Date:** 2026-04-17
**Scope:** All files under this repo.
**Primary system:** MyControl (see `SYSTEM_PLACEMENT.md`).
**Secondary linkage:** Phoenix (evidence, not product coupling).

## Legend

- **KEEP_AS_LAB** — remains in this repo as-is. The comparison surface
  lives here; this file IS that surface.
- **PORT_TO_MYCONTROL** — idea or contract is reusable by the MyControl
  system at large. Classification only — the actual port is a later
  decision, not part of this pass.
- **SUPPORTS_PHOENIX** — backs a Phoenix product-system claim (MHT /
  FAISS / Bayesian / confidence over collapse). Cite-from-Phoenix
  rather than depend-on-Phoenix.
- **DISCARD** — not useful going forward.

Secondary attribute:

- **architectural** — encodes a load-bearing architectural idea
- **reusable logic** — a piece of code/model other surfaces could use
- **demo-only** — exists to make the animated comparison render

## File-by-file classification

### Repo root

| File | Classification | Type | Rationale |
|---|---|---|---|
| `README.md` | KEEP_AS_LAB | demo-only | Deployment + endpoint overview for the comparison demo. Will be updated this pass with MyControl/Phoenix positioning, but remains the lab README. |
| `Procfile` | KEEP_AS_LAB | demo-only | Railway entry. Binds to the single-app process model of this repo. Not useful outside the comparison demo. |
| `requirements.txt` | KEEP_AS_LAB | demo-only | Python dependencies for the comparison demo's FastAPI + numpy + optional FAISS + optional Databento stack. Scope-local. |
| `LICENSE` | KEEP_AS_LAB | demo-only | Repo-level license. No integration implication. |

### app/ entry and routing

| File | Classification | Type | Rationale |
|---|---|---|---|
| `app/main.py` | KEEP_AS_LAB | demo-only | FastAPI app + static mount. Exists only to host the comparison demo. Not reusable outside this repo. |
| `app/__init__.py` | KEEP_AS_LAB | demo-only | Package marker. |
| `app/routers/__init__.py` | KEEP_AS_LAB | demo-only | Package marker. |
| `app/routers/comparison.py` | **KEEP_AS_LAB** (with a caveat) | architectural | The SSE streaming surface is the *product* of this repo — the animated side-by-side. **Caveat:** the `_compute_divergence()` helper (lines 77–103) encodes a reusable architectural contract — *when do two pipelines meaningfully disagree?* That specific function is tagged **PORT_TO_MYCONTROL** as a reusable definition, though the router itself stays here. |

### app/models/

| File | Classification | Type | Rationale |
|---|---|---|---|
| `app/models/__init__.py` | KEEP_AS_LAB | demo-only | Package marker. |
| `app/models/comparison_models.py` | **SUPPORTS_PHOENIX** + PORT_TO_MYCONTROL | architectural | This is the most reusable file in the repo. `LLMPipelineState`, `DecisionPipelineState`, `DivergenceEvent`, `RankedAction`, `SimilarityMatch`, and the three *failure-mode indicator* booleans (`uncertainty_collapsed`, `traceable`, `uncertainty_preserved`, `hypothesis_count`) are a Pydantic statement of the architectural thesis: decision architecture preserves uncertainty and traceability; LLM-only collapses both. Phoenix's `ConditionMatchResponse.matching_pillars / pillar_details` follows the same "show the work, don't collapse" pattern. Keep here; **cite from Phoenix** docs as methodology evidence; **port the indicator vocabulary to MyControl** if MyControl builds an analogous surface. |

### app/services/ — the core architectural differentiator

| File | Classification | Type | Rationale |
|---|---|---|---|
| `app/services/__init__.py` | KEEP_AS_LAB | demo-only | Package marker. |
| `app/services/hypothesis_tracker.py` | **SUPPORTS_PHOENIX** (primary) + KEEP_AS_LAB (runtime) | architectural | Contains `LatentEncoder`, `RegimeIndex`, `HypothesisUpdater`, `HypothesisManager`, `HypothesisTracker`. This IS the MHT-FAISS thesis in code: feature → latent → ANN retrieval → Bayesian posterior update → entropy → prune + rehydrate. Doctrine value: demonstrates that regime estimation under uncertainty is an *algorithm*, not a prompt. **This is the single strongest evidence artifact in the repo for Phoenix's "confidence ranking vs single-answer collapse" claim.** The runtime stays here; cite-not-depend from Phoenix. |
| `app/services/decision_pipeline.py` | **SUPPORTS_PHOENIX** + KEEP_AS_LAB | architectural | `DecisionEvaluator` + `DecisionPipeline` compose the MHT tracker into a ranked-action output with confidence intervals. Preserves the sequence: Latent Encoder → Similarity Retrieval → Hypothesis Tracking → Decision Evaluation → Ranked Action. Citation target for Phoenix when explaining why the Condition Match surface returns pillar-wise detail instead of a single bias label. |
| `app/services/llm_pipeline.py` | **SUPPORTS_PHOENIX** (as negative control) + KEEP_AS_LAB | architectural | Not a negative result — a deliberate negative control. `PromptBuilder` + `ResponseParser` + `LLMPipeline` show what "one Claude call per tick" does: single label, no state, no traceable decision path, `uncertainty_collapsed = True`. This is what Phoenix's Condition Match engine and IASG matcher explicitly reject. Valuable precisely because it's the "other side" of the comparison. |
| `app/services/signal_service.py` | KEEP_AS_LAB | reusable logic | `FinancialRegime`, `SensorRegime`, `FeatureEnricher`, `DatabentoFeed`, `SignalService`. The synthetic HMM regime simulator + Databento fallback is lab-grade signal generation. Nice pedagogically but not architecturally load-bearing — any stream of typed ticks would do. |

### app/static/ — frontend animation

| File | Classification | Type | Rationale |
|---|---|---|---|
| `app/static/index.html` | KEEP_AS_LAB | demo-only | The "MHT-FAISS Live Research Console" shell. Self-contained animated frontend. Entry point users actually see. |
| `app/static/styles.css` | KEEP_AS_LAB | demo-only | Styling for the animated shell. |
| `app/static/demoController.js` | KEEP_AS_LAB | demo-only | Drives the SSE consumption + side-by-side rendering. |
| `app/static/hypothesisEngine.js` | KEEP_AS_LAB | demo-only | Client-side rendering of the hypothesis set animation. Not a second implementation of the tracker — it's a visualization of the tracker's emitted frames. |
| `app/static/historyChart.js` | KEEP_AS_LAB | demo-only | Time-series chart of divergence and entropy. |
| `app/static/tooltip.js` | KEEP_AS_LAB | demo-only | `data-tip-*` tooltip rendering for help-target elements. |

### Status summary

| Bucket | Count | Files |
|---|---|---|
| KEEP_AS_LAB | 14 | everything not listed below + shared with other buckets |
| SUPPORTS_PHOENIX | 4 | `comparison_models.py`, `hypothesis_tracker.py`, `decision_pipeline.py`, `llm_pipeline.py` |
| PORT_TO_MYCONTROL | 2 | `comparison_models.py` (indicator vocabulary), `routers/comparison.py::_compute_divergence` (contract only) |
| DISCARD | 0 | — |

## Notes on the dual classifications

Four files carry two tags. That is deliberate and means:

- The file **stays here** as part of the runnable comparison surface
  (`KEEP_AS_LAB` as residency).
- The file **is cited** from Phoenix docs (`SUPPORTS_PHOENIX`) as
  methodology evidence for why Phoenix returns structured, traceable
  decision output instead of a single LLM label.

There is no code copy-paste. Nothing is forked. Phoenix does not import
from this repo. This is an evidence-citation relationship, not a build
dependency.

## Confirmed non-actions (boundary)

- No code is being moved out of this repo.
- No code is being refactored.
- No TypeScript or Swift port is being produced.
- No new endpoints.
- No UI changes beyond the README positioning update.
- No dependency change to Phoenix or to MyControl.

Classification + citation contract only.
