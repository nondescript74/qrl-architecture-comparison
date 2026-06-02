# QRL Architecture Comparison Demo

Live side-by-side demonstration of **LLM-Only** vs **Decision Architecture** pipelines
processing the same financial or sensor signal stream in real time.

## Position in the system

This repo is the **Architecture Comparison Surface** under **MyControl**.

- **Role:** evidence and demonstration system (NOT a production surface).
- **Primary home:** MyControl — governed by the control plane.
- **Secondary linkage:** Phoenix — used as cite-based evidence for
  Phoenix's MHT / FAISS / decision-over-prompt design choices. Phoenix
  does **not** import or depend on this repo.
- **Authority:** classification and placement rules are in
  [`INTEGRATION_CLASSIFICATION.md`](./INTEGRATION_CLASSIFICATION.md),
  [`SYSTEM_PLACEMENT.md`](./SYSTEM_PLACEMENT.md), and
  [`EXPOSURE_SPEC.md`](./EXPOSURE_SPEC.md).
- **Control plane:** see [`CONTROL_PLANE_LINK.md`](./CONTROL_PLANE_LINK.md)
  for pointers to S2 (strategic-control-plane) and S3
  (chatgpt-comprehensive-scp).

Production claims, uptime contracts, trust-safe language enforcement,
and doctrine PASS gating all live with Phoenix — not here. This surface
is allowed to fail loudly; it's an animated demonstration of an
architectural thesis, not a service.

## What it shows

| LLM-Only | Decision Architecture |
|---|---|
| Single inference per tick | Competing hypotheses maintained |
| Uncertainty collapsed to one label | Bayesian MHT state evolution |
| Context limited to prompt window | FAISS historical regime retrieval |
| Decision path opaque | Ranked actions with confidence intervals |

## Architecture

```
app/
  main.py                    FastAPI entry point
  models/
    comparison_models.py     Pydantic schemas
  services/
    signal_service.py        Databento live feed + synthetic HMM fallback
    hypothesis_tracker.py    MHT engine — LatentEncoder + RegimeIndex + Bayesian updater
    llm_pipeline.py          LLM-only pipeline (Claude API)
    decision_pipeline.py     Full decision architecture pipeline
  routers/
    comparison.py            SSE streaming router
  static/
    index.html               Self-contained animated frontend
```

## Deploy to Railway

1. Fork/clone this repo
2. Create a new Railway project from this repo
3. Set environment variables:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   DATABENTO_API_KEY=db-...       # optional — falls back to synthetic
   ```
4. Railway auto-detects the `Procfile` and deploys

## Local development

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

## SSE stream

`GET /comparison/stream?max_frames=500`

Each frame is a `ComparisonFrame` JSON payload with `stage` values:
`tick` | `llm_inference` | `hypothesis_update` | `divergence` | `end`
