# System Placement — qrl-architecture-comparison

**Date:** 2026-04-17
**Primary system:** MyControl
**Secondary linkage:** Phoenix
**Posture:** Evidence and demonstration system — NOT production.

This document defines where this repo sits in the system architecture
and how it relates to the rest of the stack. It is the companion to
`INTEGRATION_CLASSIFICATION.md` — the latter says what each file IS,
this one says what the whole thing IS.

## A. MyControl role

### Named role

**Architecture Comparison Surface**
*LLM-only vs decision architecture, under identical input streams.*

### What this means inside MyControl

MyControl is the architectural home for the system-family decisions:
how reasoning is organized, how uncertainty is handled, how state
evolves across time, how multi-agent conclusions are reconciled. This
repo slots under MyControl as the surface that *demonstrates* those
decisions by running both approaches against the same signal stream
and rendering the divergence frame-by-frame.

### Explicit connections

| MyControl concern | How this repo supports it |
|---|---|
| Control plane | This repo is governed by the control plane (see `CONTROL_PLANE_LINK.md`). Not a renegade artifact. |
| Multi-agent reasoning | The LLM pipeline represents one agent's one answer. The decision pipeline represents a tracked hypothesis set. The comparison surface shows why the multi-agent framing is structurally different from asking one LLM N times. |
| Hypothesis tracking | `app/services/hypothesis_tracker.py` is the MHT reference implementation — `LatentEncoder`, `RegimeIndex`, `HypothesisUpdater`, `HypothesisManager`. Live code, not slides. |
| Reconciliation | `DivergenceEvent` in `app/models/comparison_models.py` is the canonical contract for "when do two reasoning surfaces disagree?" The frame pipeline emits one of these every time the LLM's single label diverges from the tracker's top hypothesis. Reconciliation in MyControl starts from that same question. |

### Boundary inside MyControl

This repo is **a surface under MyControl, not MyControl itself**. It
does not define the control-plane operating model, it does not govern
lane structure, and it does not hold strategic truth. Those remain in
S2 (`strategic-control-plane`). This repo supplies evidence that
underwrites MyControl architectural claims; it does not make those
claims.

## B. Phoenix linkage

Phoenix is the product-system that reached foundation COMPLETE on
2026-04-17 (see S2 `LANES/PHOENIX_FOUNDATION.md`). This repo relates
to Phoenix as an **evidence back-reference**, not a dependency.

### Four specific citation targets

1. **MHT-FAISS justification.**
   Phoenix uses multiple-hypothesis + similarity retrieval as its
   analog-matching primitive (IASG comparable surface, Condition
   Match pillar expansion, COT percentile positioning). This repo's
   `hypothesis_tracker.py` is the runnable reference that makes that
   choice legible: MHT is an algorithm with measurable entropy and
   prune/rehydrate dynamics, not a rhetorical flourish.

2. **Decision vs prompt inference.**
   Phoenix's `ConditionMatchResponse` returns `pillar_details` — five
   pillar objects with per-pillar `match` booleans and `favorable`
   sets — instead of a single bias label. That design decision mirrors
   this repo's `DecisionPipelineState`: `uncertainty_preserved = True`,
   `traceable = True`, `hypothesis_count > 1`. The comparison surface
   here is the "why" behind that API shape.

3. **Confidence ranking vs single-answer collapse.**
   Phoenix's IASG matcher (`qrl-phoenix-web/src/utils/iasgMatcher.ts`
   and the ported `phoenix-ios/Sources/PhoenixFoundation/IASGMatcher.swift`)
   returns a ranked list of `MatchCandidate` with `matchScore`,
   `matchLabel` (Direct Match / Similar / Analog / No Clear Match),
   and `rationale: [String]` — explicitly a ranking, not a pick. This
   repo's `RankedAction` + `SimilarityMatch` Pydantic models carry the
   same shape: score + confidence interval + supporting hypotheses,
   never a collapsed label.

4. **Regime-aware reasoning.**
   Phoenix's Condition Match engine evaluates five regime pillars
   (volatility, momentum, rate trend, dollar trend, term structure)
   before emitting `bias_score` and `regime_label`. This repo's
   `RegimeIndex` + `FinancialRegime` + `SensorRegime` constructs show
   the same commitment: regime is a discrete, labeled latent state,
   and the decision surface consumes it as a *substrate*, not as a
   model guess.

### What Phoenix does NOT do toward this repo

- Phoenix does not import this repo.
- Phoenix does not run against this repo's endpoints.
- Phoenix's production deployment does not depend on this repo being
  up, deployed, or even online.
- Changes here do not require a Phoenix re-deploy.
- The link is cite-based (evidence, doctrine) not runtime-based.

## C. Explicit boundary — production vs evidence

This repo is **NOT a production surface**. State this plainly in any
context where the distinction matters.

| Criterion | Phoenix | This repo |
|---|---|---|
| Product posture | investor-demo-MVP + foundation-complete | evidence / demonstration |
| User-facing stability contract | Yes — trust-safe language, doctrine-enforced | No — demo is an animation |
| Uptime obligation | Hardened (Cloudflare Tunnel, WAF, private repos) | Best-effort Railway demo |
| Data claims | PROVEN / IMPLEMENTED maturity ladder | Simulated or live signals, no claim graduation |
| Failure modes | Route error boundaries, fallback chains, doctrine PASS gates | Demo is allowed to fail loudly |

**This repo is an evidence and demonstration system.** It exists to
make the architectural thesis *legible in motion*: anyone can open the
comparison surface and watch the LLM pipeline collapse while the
decision pipeline preserves its hypothesis set. That's the entire
point. It is not a service, it is not a product, it is not on a
customer-facing surface, and it does not carry a shipping commitment.

Do not let downstream copy blur this line.
