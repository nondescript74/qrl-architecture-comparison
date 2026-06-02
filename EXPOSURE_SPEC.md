# Exposure Spec — qrl-architecture-comparison

**Date:** 2026-04-17
**Surface class:** Controlled evidence surface under MyControl.
**Posture:** Demonstrates architecture; does not claim product.

This document defines what a user sees when they enter this surface,
what signals they should take away, and — explicitly — what must NOT
happen in the rendering or narrative layer. This is the exposure
contract for the demo, not a marketing description.

## A. Entry point

Users enter via the MyControl page that hosts the Architecture
Comparison Surface. The link goes to the deployed instance of this
repo (Railway or equivalent) at the root path `/`, which serves
`app/static/index.html` — the "MHT-FAISS Live Research Console" shell.

**Single entry point.** There is no authenticated route, no user
profile, no persisted session. The demo is live per page load, the
SSE stream starts on mount, and state resets on reload.

**Discovery contract (from MyControl):**

- The MyControl page that points here must label this as
  "Architecture Comparison — evidence demo" (or equivalent wording
  that names it as a demonstration, not a product).
- The link must not present this as a tool the user can use to make
  decisions.
- The link must not claim live PnL, live signals, or live predictions.

## B. What the user sees

The `/comparison/stream` SSE endpoint drives a **side-by-side**
layout. Each SSE frame is a `ComparisonFrame` carrying at most one
`SignalTick`, one `LLMPipelineState`, one `DecisionPipelineState`,
and optionally one `DivergenceEvent`. The frontend renders both
pipeline states simultaneously for the same tick.

### Left panel — LLM-only pipeline

| What renders | Source |
|---|---|
| The current parsed label | `LLMPipelineState.parsed_label` |
| Stated confidence (optional, self-reported) | `LLMPipelineState.confidence_stated` |
| Prompt token count + context window fullness | `LLMPipelineState.prompt_tokens`, `.context_window_pct` |
| Latency | `LLMPipelineState.latency_ms` |
| Failure-mode indicators | `uncertainty_collapsed = True`, `hypothesis_count = 1`, `traceable = False` |

### Right panel — decision architecture pipeline

| What renders | Source |
|---|---|
| Current hypothesis set (up to N live) | `DecisionPipelineState.hypothesis_set` |
| Per-hypothesis posterior weight | `Hypothesis.posterior` |
| Entropy (uncertainty metric) | `HypothesisSet.entropy` |
| Top similarity matches (FAISS / brute cosine) | `DecisionPipelineState.top_similarity_matches` |
| Ranked actions with confidence intervals | `DecisionPipelineState.ranked_actions` |
| Failure-mode indicators | `uncertainty_preserved = True`, `hypothesis_count > 1`, `traceable = True` |

### Shared

| What renders | Source |
|---|---|
| Current signal tick | `SignalTick` |
| Stage marker | `ComparisonFrame.stage` ∈ `{tick, llm_inference, hypothesis_update, divergence, end}` |
| Divergence chip/pulse when it fires | `DivergenceEvent` |
| Rolling history chart (entropy + divergence count) | `app/static/historyChart.js` |

## C. Key signals to convey

These are the three things a viewer should walk away having seen.
The animation is designed around them; the MyControl label should
hint at them; any accompanying copy should reinforce them.

### C.1 Divergence moments

When the LLM collapses to a label that disagrees with the tracker's
top hypothesis, `_compute_divergence()` fires a `DivergenceEvent`
with `magnitude` ∈ [0, 1] and a short `description`. The frontend
surfaces this as a visible pulse. **The divergence is the teaching
moment** — it is where the two architectures visibly do different
things with the same input.

### C.2 Hypothesis evolution

Over time, the tracker prunes low-weight hypotheses and rehydrates
when evidence accumulates. The viewer should see hypotheses *enter,
compete, and exit*, not a static bar chart. This is the
"state-across-time" thesis made visible.

### C.3 Confidence vs collapse

The left panel always shows exactly one label. The right panel always
shows a ranked set with per-item posterior weight. That asymmetry is
the whole thesis: *decision architecture preserves uncertainty
until evidence justifies collapsing it; LLM-only collapses on every
call.*

## D. What must NOT happen

The following are hard anti-patterns for this surface. They apply to
the rendered UI, to any narration/copy on the MyControl page that
links here, and to any screenshots or clips of this surface used
elsewhere.

### D.1 No overclaiming

- Do NOT present the decision architecture's output as a trading
  signal or as a forecast.
- Do NOT describe the LLM pipeline as "broken" or "wrong." It is a
  deliberate negative control — a single well-formed Claude call.
  The point is not that it is broken; the point is that it is
  *structurally constrained* in ways the decision pipeline is not.
- Do NOT claim that the tracker "always wins" or "is more accurate."
  The demo shows a structural property (uncertainty preservation,
  traceability), not a benchmark against which architecture produces
  better decisions on a held-out dataset.

### D.2 No "AI magic"

- Do NOT use language like "AI-powered," "intelligent decisions,"
  "understands the market," or similar vocabulary that hides the
  mechanism.
- The mechanism IS the point. `LatentEncoder → RegimeIndex →
  HypothesisUpdater → DecisionEvaluator → RankedAction`. Name the
  steps. Show them animating.
- Trust-safe language parity with Phoenix: no "validated," no
  "endorsed," no "the system recommends." Describe structural
  behavior, not authority.

### D.3 No hiding uncertainty

- Entropy must be visible whenever the decision pipeline is rendering.
- `uncertainty_collapsed` must render visibly as an indicator on the
  LLM side (it is always `True` there, and that is the teaching).
- When the tracker's top hypothesis has a posterior close to the
  second-ranked hypothesis, the closeness must be visible — do NOT
  suppress it to make the right panel look decisive.
- Do NOT compress the ranked actions down to a single "top pick"
  without the confidence interval. The interval is the shape.

### D.4 No scope creep

- Do NOT add Phoenix product surfaces to this demo.
- Do NOT use this surface to display a user's own portfolio, trades,
  or broker state.
- Do NOT connect this demo to any authenticated user data.
- Do NOT imply that running this demo produces a Phoenix result.

## E. Production-vs-evidence reminder

This surface is evidence. Phoenix is product. The MyControl page
linking here should say so — explicitly, in the label or subtitle —
so a visitor never confuses "watch the mechanism animate" with
"make a decision on real money."
