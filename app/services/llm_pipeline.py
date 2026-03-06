"""
llm_pipeline.py
LLM-Only Pipeline for the comparison demo.

This intentionally represents what a single LLM inference does:
  - Assembles context from recent ticks into a text prompt
  - Calls Claude API once
  - Returns a single parsed answer

The point is NOT that this is wrong — it is that it collapses
uncertainty into one label too early, with no persistent state,
no competing hypotheses, and no traceable decision path.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from typing import Optional

import httpx

from app.models.comparison_models import (
    DemoConfig,
    LLMPipelineState,
    RegimeLabel,
    SignalDomain,
    SignalTick,
)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ──────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────

class PromptBuilder:
    """Assembles a text prompt from a window of ticks."""

    FINANCIAL_SYSTEM = (
        "You are a financial regime classifier. "
        "Given recent market observations, identify the current market regime. "
        "Respond ONLY with a JSON object: "
        '{"regime": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}. '
        "Valid regimes: bull_trend, bear_trend, mean_revert, high_volatility, "
        "regime_shift, low_volatility. "
        "Do not include any text outside the JSON."
    )

    SENSOR_SYSTEM = (
        "You are an industrial sensor anomaly classifier. "
        "Given recent sensor readings, identify the current operational state. "
        "Respond ONLY with a JSON object: "
        '{"regime": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}. '
        "Valid regimes: nominal, anomaly_drift, fault_imminent, recovery. "
        "Do not include any text outside the JSON."
    )

    def build(self, window: list[SignalTick], config: DemoConfig) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        system = (
            self.FINANCIAL_SYSTEM
            if config.domain == SignalDomain.FINANCIAL
            else self.SENSOR_SYSTEM
        )

        if config.domain == SignalDomain.FINANCIAL:
            rows = []
            for t in window[-10:]:     # last 10 ticks only — bounded context
                rows.append(
                    f"price={t.price}, vol={t.volume}, "
                    f"ret1={t.returns_1}, ret5={t.returns_5}, "
                    f"volatility={t.volatility}"
                )
            user = "Recent market observations:\n" + "\n".join(rows)
            user += f"\n\nInstrument: {config.instrument}. What is the current regime?"
        else:
            rows = []
            for t in window[-10:]:
                rows.append(
                    f"temp={t.temperature}, pressure={t.pressure}, "
                    f"flow={t.flow_rate}"
                )
            user = "Recent sensor readings:\n" + "\n".join(rows)
            user += f"\n\nSensor: {config.sensor_id}. What is the current state?"

        return system, user

    def token_estimate(self, prompt: str) -> int:
        """Rough token estimate: words * 1.3"""
        return int(len(prompt.split()) * 1.3)


# ──────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────

VALID_LABELS = {r.value for r in RegimeLabel}

class ResponseParser:
    def parse(self, raw: str) -> tuple[str, Optional[float], str]:
        """Returns (label, confidence, raw_answer)."""
        clean = raw.strip()

        # Strip markdown fences if present
        clean = re.sub(r"```json\s*", "", clean)
        clean = re.sub(r"```\s*", "", clean)

        try:
            obj = json.loads(clean)
            regime = obj.get("regime", "unknown").lower().replace(" ", "_")
            if regime not in VALID_LABELS:
                regime = "unknown"
            confidence = float(obj.get("confidence", 0.5))
            return regime, confidence, raw
        except Exception:
            # Fallback: scan for known label keywords
            lower = clean.lower()
            for label in VALID_LABELS:
                if label in lower:
                    return label, None, raw
            return "unknown", None, raw


# ──────────────────────────────────────────────────────────────────
# Async Claude caller
# ──────────────────────────────────────────────────────────────────

async def _call_claude(system: str, user: str, model: str) -> tuple[str, float]:
    """Returns (response_text, latency_ms)."""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 150,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        latency_ms = (time.monotonic() - t0) * 1000
        return text, latency_ms


# ──────────────────────────────────────────────────────────────────
# Public: LLMPipeline
# ──────────────────────────────────────────────────────────────────

class LLMPipeline:
    """
    Stateless (by design) LLM-only inference pipeline.
    Each call is independent — no persistent hypothesis state.
    Demonstrates: uncertainty collapsed into one answer per tick.
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._window: deque[SignalTick] = deque(maxlen=20)
        self._builder = PromptBuilder()
        self._parser  = ResponseParser()
        self._tick_idx = 0
        # Rate limiting: only call LLM every N ticks to control costs
        self._call_every_n = 3

    async def process(self, tick: SignalTick) -> Optional[LLMPipelineState]:
        """
        Returns LLMPipelineState if an inference was made, else None.
        Returns None on non-inference ticks (rate-limited).
        """
        self._window.append(tick)
        self._tick_idx += 1

        # Only call API every N ticks
        if self._tick_idx % self._call_every_n != 0:
            return None

        if not ANTHROPIC_API_KEY:
            return self._mock_inference(tick)

        try:
            system, user = self._builder.build(list(self._window), self.config)
            raw, latency = await _call_claude(system, user, self.config.llm_model)
            label, confidence, raw_answer = self._parser.parse(raw)
            tokens = self._builder.token_estimate(system + user)

            return LLMPipelineState(
                tick_index=self._tick_idx,
                prompt_tokens=tokens,
                context_window_pct=round(tokens / 200000, 4),  # Claude 200k window
                raw_answer=raw_answer[:300],
                parsed_label=label,
                confidence_stated=confidence,
                latency_ms=round(latency, 1),
            )
        except Exception as e:
            return self._mock_inference(tick, error=str(e))

    def _mock_inference(
        self,
        tick: SignalTick,
        error: Optional[str] = None,
    ) -> LLMPipelineState:
        """Synthetic LLM response for demo/test without API key."""
        import random
        labels = (
            ["bull_trend", "bear_trend", "mean_revert", "high_volatility", "regime_shift"]
            if self.config.domain == SignalDomain.FINANCIAL
            else ["nominal", "anomaly_drift", "fault_imminent", "recovery"]
        )
        label = random.choice(labels)
        confidence = round(random.uniform(0.55, 0.92), 2)
        reasoning = f"Based on recent price action, the market appears to be in a {label} regime."
        raw = json.dumps({"regime": label, "confidence": confidence, "reasoning": reasoning})

        tokens = 120 + len(list(self._window)) * 15
        return LLMPipelineState(
            tick_index=self._tick_idx,
            prompt_tokens=tokens,
            context_window_pct=round(tokens / 200000, 6),
            raw_answer=raw if not error else f"ERROR: {error}",
            parsed_label=label,
            confidence_stated=confidence,
            latency_ms=round(random.uniform(180, 900), 1),
        )
