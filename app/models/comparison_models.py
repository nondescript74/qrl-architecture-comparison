"""
comparison_models.py
Pydantic schemas for the LLM-Only vs Decision Architecture comparison demo.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
import time

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Config / Domain
# ─────────────────────────────────────────────

class SignalDomain(str, Enum):
    FINANCIAL = "financial"
    SENSOR    = "sensor"


class DemoConfig(BaseModel):
    domain: SignalDomain = SignalDomain.FINANCIAL
    use_real_data: bool = True           # False → synthetic fallback
    instrument: str = "ES.c.0"          # CME E-mini S&P front month
    sensor_id: str = "reactor-loop-1"   # used in sensor domain
    bar_interval_sec: int = 5           # how often a new signal tick arrives
    hypothesis_count: int = 5           # max live hypotheses
    llm_model: str = "claude-sonnet-4-20250514"
    prune_threshold: float = 0.03       # drop hypotheses below this weight


# ─────────────────────────────────────────────
# Signal
# ─────────────────────────────────────────────

class SignalTick(BaseModel):
    ts: float = Field(default_factory=time.time)
    domain: SignalDomain
    instrument: str
    price: Optional[float] = None
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    # sensor fields
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    flow_rate: Optional[float] = None
    # derived
    returns_1: Optional[float] = None   # 1-bar return
    returns_5: Optional[float] = None   # 5-bar return
    volatility: Optional[float] = None  # rolling std
    spread: Optional[float] = None


# ─────────────────────────────────────────────
# Hypothesis Tracker
# ─────────────────────────────────────────────

class RegimeLabel(str, Enum):
    # Financial
    BULL_TREND      = "bull_trend"
    BEAR_TREND      = "bear_trend"
    MEAN_REVERT     = "mean_revert"
    REGIME_SHIFT    = "regime_shift"
    HIGH_VOL        = "high_volatility"
    LOW_VOL         = "low_volatility"
    # Sensor
    NOMINAL         = "nominal"
    ANOMALY_DRIFT   = "anomaly_drift"
    FAULT_IMMINENT  = "fault_imminent"
    RECOVERY        = "recovery"


class Hypothesis(BaseModel):
    id: str
    label: RegimeLabel
    weight: float                        # posterior probability [0,1]
    likelihood: float                    # P(obs | hypothesis)
    evidence_trail: list[str] = []       # last N supporting observations
    similar_regime_ts: Optional[float] = None   # ts of retrieved historical match
    age_ticks: int = 0


class HypothesisSet(BaseModel):
    ts: float = Field(default_factory=time.time)
    hypotheses: list[Hypothesis]
    top_label: RegimeLabel
    top_weight: float
    entropy: float                       # uncertainty metric H = -Σ p log p
    tick_index: int


# ─────────────────────────────────────────────
# LLM-Only Pipeline
# ─────────────────────────────────────────────

class LLMPipelineState(BaseModel):
    ts: float = Field(default_factory=time.time)
    tick_index: int
    prompt_tokens: int
    context_window_pct: float            # how full the context is
    raw_answer: str
    parsed_label: str
    confidence_stated: Optional[float]   # self-reported by LLM
    latency_ms: float
    # Failure mode indicators
    uncertainty_collapsed: bool = True   # always True for LLM-only
    hypothesis_count: int = 1            # always 1 for LLM-only
    traceable: bool = False              # decision path not inspectable


# ─────────────────────────────────────────────
# Decision Architecture Pipeline
# ─────────────────────────────────────────────

class SimilarityMatch(BaseModel):
    regime_label: RegimeLabel
    similarity_score: float
    historical_ts: float
    context_snippet: str


class RankedAction(BaseModel):
    action: str
    score: float
    confidence_interval: tuple[float, float]
    supporting_hypotheses: list[str]


class DecisionPipelineState(BaseModel):
    ts: float = Field(default_factory=time.time)
    tick_index: int
    latent_vector_dim: int
    top_similarity_matches: list[SimilarityMatch]
    hypothesis_set: HypothesisSet
    ranked_actions: list[RankedAction]
    latency_ms: float
    # Resilience indicators
    uncertainty_preserved: bool = True
    hypothesis_count: int
    traceable: bool = True


# ─────────────────────────────────────────────
# Comparison Frame (SSE payload)
# ─────────────────────────────────────────────

class DivergenceEvent(BaseModel):
    tick_index: int
    llm_label: str
    decision_label: str
    magnitude: float                     # 0-1, how different they are
    description: str


class ComparisonFrame(BaseModel):
    """Single SSE frame emitted to the frontend animation."""
    frame_id: int
    ts: float = Field(default_factory=time.time)
    signal: SignalTick
    llm_state: Optional[LLMPipelineState] = None
    decision_state: Optional[DecisionPipelineState] = None
    divergence: Optional[DivergenceEvent] = None
    # Pipeline stage marker — lets the frontend know what just happened
    stage: str = "tick"                  # tick | llm_inference | hypothesis_update | divergence
    config: Optional[DemoConfig] = None
