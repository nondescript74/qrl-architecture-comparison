"""
comparison.py
FastAPI router for the LLM-Only vs Decision Architecture comparison demo.

Endpoints:
  GET  /comparison/stream          → SSE: live frame-by-frame pipeline comparison
  GET  /comparison/config          → current DemoConfig
  POST /comparison/config          → update DemoConfig (hot-swap)
  POST /comparison/reset           → reset all pipeline state
  GET  /comparison/health          → liveness check

Mount in your main.py:
    from app.routers.comparison import router as comparison_router
    app.include_router(comparison_router, prefix="/comparison", tags=["comparison"])
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.comparison_models import (
    ComparisonFrame,
    DemoConfig,
    DivergenceEvent,
    SignalDomain,
)
from app.services.decision_pipeline import DecisionPipeline
from app.services.llm_pipeline import LLMPipeline
from app.services.signal_service import SignalService

router = APIRouter()


# ──────────────────────────────────────────────────────────────────
# Shared state (one session, reset-able)
# ──────────────────────────────────────────────────────────────────

class DemoSession:
    def __init__(self):
        self.config = DemoConfig()
        self._reset_pipelines()

    def _reset_pipelines(self):
        self.signal_svc    = SignalService(self.config)
        self.llm_pipeline  = LLMPipeline(self.config)
        self.dec_pipeline  = DecisionPipeline(self.config)
        self._started      = False

    async def ensure_started(self):
        if not self._started:
            await self.signal_svc.start()
            self._started = True

    def update_config(self, new_config: DemoConfig):
        self.config = new_config
        self._reset_pipelines()

    def reset(self):
        self._reset_pipelines()


_session = DemoSession()


# ──────────────────────────────────────────────────────────────────
# Divergence detector
# ──────────────────────────────────────────────────────────────────

def _compute_divergence(
    frame_id: int,
    llm_label: str,
    dec_label: str,
    entropy: float,
) -> DivergenceEvent | None:
    """
    Emit a divergence event when the two pipelines disagree.
    Magnitude is amplified by the decision architecture's uncertainty (entropy).
    """
    if llm_label == dec_label:
        return None

    descriptions = {
        frozenset({"bull_trend", "bear_trend"}): "Directional disagreement — LLM collapsed trend; MHT shows conflict",
        frozenset({"bull_trend", "mean_revert"}): "LLM sees momentum; MHT tracks reversion hypothesis",
        frozenset({"bear_trend", "high_volatility"}): "LLM attributes volatility to trend; MHT separates regime type",
        frozenset({"regime_shift", "mean_revert"}): "LLM missed regime shift signal preserved in hypothesis tree",
        frozenset({"nominal", "anomaly_drift"}): "LLM reports normal; MHT anomaly hypothesis is rising",
        frozenset({"fault_imminent", "recovery"}): "LLM sees recovery; MHT fault hypothesis still material",
    }
    key = frozenset({llm_label, dec_label})
    desc = descriptions.get(key, f"LLM: {llm_label} | MHT top: {dec_label}")

    magnitude = min(1.0, round(0.4 + entropy * 0.6, 3))

    return DivergenceEvent(
        tick_index=frame_id,
        llm_label=llm_label,
        decision_label=dec_label,
        magnitude=magnitude,
        description=desc,
    )


# ──────────────────────────────────────────────────────────────────
# SSE generator
# ──────────────────────────────────────────────────────────────────

async def _frame_generator(
    session: DemoSession,
    max_frames: int = 500,
) -> AsyncGenerator[str, None]:
    """
    Emits one SSE event per signal tick:
      data: <ComparisonFrame JSON>\n\n
    """
    await session.ensure_started()
    frame_id = 0

    # Emit config frame first so client knows the domain
    config_frame = ComparisonFrame(
        frame_id=0,
        signal=None,  # type: ignore
        stage="config",
        config=session.config,
    )
    yield f"data: {config_frame.model_dump_json()}\n\n"

    async for tick in session.signal_svc.stream():
        if frame_id >= max_frames:
            break

        frame_id += 1

        # ── Decision pipeline (synchronous, fast) ──
        dec_state = session.dec_pipeline.process(tick)

        # ── LLM pipeline (async, rate-limited) ──
        llm_state = await session.llm_pipeline.process(tick)

        # ── Determine stage label ──
        stage = "tick"
        if llm_state is not None:
            stage = "llm_inference"
        if frame_id % 5 == 0:
            stage = "hypothesis_update"

        # ── Divergence check (only when LLM fired) ──
        divergence = None
        if llm_state is not None:
            divergence = _compute_divergence(
                frame_id=frame_id,
                llm_label=llm_state.parsed_label,
                dec_label=dec_state.hypothesis_set.top_label.value,
                entropy=dec_state.hypothesis_set.entropy,
            )
            if divergence:
                stage = "divergence"

        frame = ComparisonFrame(
            frame_id=frame_id,
            signal=tick,
            llm_state=llm_state,
            decision_state=dec_state,
            divergence=divergence,
            stage=stage,
        )

        yield f"data: {frame.model_dump_json()}\n\n"

        # Heartbeat keepalive every 10 frames
        if frame_id % 10 == 0:
            yield ": keepalive\n\n"

    yield "data: {\"stage\": \"end\", \"frame_id\": " + str(frame_id) + "}\n\n"


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream_comparison(max_frames: int = 500):
    """
    SSE stream of comparison frames.
    Each frame contains the signal tick plus both pipeline states.
    """
    return StreamingResponse(
        _frame_generator(_session, max_frames=max_frames),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/config")
async def get_config() -> DemoConfig:
    return _session.config


@router.post("/config")
async def update_config(new_config: DemoConfig) -> DemoConfig:
    _session.update_config(new_config)
    return _session.config


@router.post("/reset")
async def reset_session() -> dict:
    _session.reset()
    return {"status": "reset", "ts": time.time()}


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "domain": _session.config.domain,
        "instrument": _session.config.instrument,
        "use_real_data": _session.config.use_real_data,
        "ts": time.time(),
    }
