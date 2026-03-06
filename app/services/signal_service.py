"""
signal_service.py
Unified signal generator supporting:
  - Databento live market data (financial domain)
  - Synthetic regime-aware simulation (financial + sensor domains)
Domain and data source are controlled by DemoConfig.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from collections import deque
from typing import AsyncGenerator, Optional

import numpy as np

from app.models.comparison_models import (
    DemoConfig,
    RegimeLabel,
    SignalDomain,
    SignalTick,
)


# ──────────────────────────────────────────────────────────────────
# Synthetic financial regime simulator
# ──────────────────────────────────────────────────────────────────

class FinancialRegime:
    """Hidden Markov regime that drives synthetic price paths."""

    REGIMES: dict[RegimeLabel, dict] = {
        RegimeLabel.BULL_TREND: {
            "drift":    0.0008,
            "vol":      0.0012,
            "label":    RegimeLabel.BULL_TREND,
            "duration": (20, 60),
        },
        RegimeLabel.BEAR_TREND: {
            "drift":   -0.0010,
            "vol":      0.0018,
            "label":    RegimeLabel.BEAR_TREND,
            "duration": (15, 45),
        },
        RegimeLabel.MEAN_REVERT: {
            "drift":    0.0000,
            "vol":      0.0006,
            "label":    RegimeLabel.MEAN_REVERT,
            "duration": (30, 80),
        },
        RegimeLabel.HIGH_VOL: {
            "drift":    0.0000,
            "vol":      0.0035,
            "label":    RegimeLabel.HIGH_VOL,
            "duration": (8, 25),
        },
        RegimeLabel.REGIME_SHIFT: {
            "drift":    0.0015,
            "vol":      0.0040,
            "label":    RegimeLabel.REGIME_SHIFT,
            "duration": (3, 12),   # short, transitional
        },
    }

    # Transition matrix rows: from → cols: to (probabilities)
    TRANSITIONS: dict[RegimeLabel, list[tuple[RegimeLabel, float]]] = {
        RegimeLabel.BULL_TREND:  [
            (RegimeLabel.BULL_TREND,   0.70),
            (RegimeLabel.MEAN_REVERT,  0.15),
            (RegimeLabel.REGIME_SHIFT, 0.10),
            (RegimeLabel.HIGH_VOL,     0.05),
        ],
        RegimeLabel.BEAR_TREND:  [
            (RegimeLabel.BEAR_TREND,   0.65),
            (RegimeLabel.MEAN_REVERT,  0.20),
            (RegimeLabel.REGIME_SHIFT, 0.10),
            (RegimeLabel.HIGH_VOL,     0.05),
        ],
        RegimeLabel.MEAN_REVERT: [
            (RegimeLabel.MEAN_REVERT,  0.60),
            (RegimeLabel.BULL_TREND,   0.20),
            (RegimeLabel.BEAR_TREND,   0.15),
            (RegimeLabel.REGIME_SHIFT, 0.05),
        ],
        RegimeLabel.HIGH_VOL:    [
            (RegimeLabel.REGIME_SHIFT, 0.40),
            (RegimeLabel.BEAR_TREND,   0.30),
            (RegimeLabel.MEAN_REVERT,  0.20),
            (RegimeLabel.HIGH_VOL,     0.10),
        ],
        RegimeLabel.REGIME_SHIFT:[
            (RegimeLabel.BULL_TREND,   0.35),
            (RegimeLabel.BEAR_TREND,   0.35),
            (RegimeLabel.HIGH_VOL,     0.20),
            (RegimeLabel.MEAN_REVERT,  0.10),
        ],
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.current_regime = RegimeLabel.BULL_TREND
        self.ticks_remaining = self._sample_duration()
        self.price = 5200.0          # ES-like starting price

    def _sample_duration(self) -> int:
        lo, hi = self.REGIMES[self.current_regime]["duration"]
        return int(self.rng.integers(lo, hi))

    def _transition(self) -> None:
        candidates, weights = zip(*self.TRANSITIONS[self.current_regime])
        idx = self.rng.choice(len(candidates), p=weights)
        self.current_regime = candidates[idx]
        self.ticks_remaining = self._sample_duration()

    def next_tick(self, instrument: str) -> SignalTick:
        r = self.REGIMES[self.current_regime]
        shock = self.rng.normal(r["drift"], r["vol"])
        self.price *= (1 + shock)
        spread = self.price * self.rng.uniform(0.00005, 0.00015)
        volume = int(self.rng.integers(200, 2000))

        self.ticks_remaining -= 1
        if self.ticks_remaining <= 0:
            self._transition()

        return SignalTick(
            domain=SignalDomain.FINANCIAL,
            instrument=instrument,
            price=round(self.price, 2),
            volume=float(volume),
            bid=round(self.price - spread / 2, 2),
            ask=round(self.price + spread / 2, 2),
            spread=round(spread, 4),
        )


# ──────────────────────────────────────────────────────────────────
# Synthetic sensor regime simulator
# ──────────────────────────────────────────────────────────────────

class SensorRegime:
    REGIMES = {
        RegimeLabel.NOMINAL:       {"drift": 0.00,  "vol": 0.02,  "duration": (40, 100)},
        RegimeLabel.ANOMALY_DRIFT: {"drift": 0.015, "vol": 0.05,  "duration": (10, 30)},
        RegimeLabel.FAULT_IMMINENT:{"drift": 0.04,  "vol": 0.12,  "duration": (5,  15)},
        RegimeLabel.RECOVERY:      {"drift":-0.02,  "vol": 0.03,  "duration": (20, 50)},
    }

    TRANSITIONS = {
        RegimeLabel.NOMINAL:       [(RegimeLabel.NOMINAL, 0.85), (RegimeLabel.ANOMALY_DRIFT, 0.15)],
        RegimeLabel.ANOMALY_DRIFT: [(RegimeLabel.ANOMALY_DRIFT, 0.55), (RegimeLabel.FAULT_IMMINENT, 0.25), (RegimeLabel.NOMINAL, 0.20)],
        RegimeLabel.FAULT_IMMINENT:[(RegimeLabel.FAULT_IMMINENT, 0.50), (RegimeLabel.RECOVERY, 0.35), (RegimeLabel.ANOMALY_DRIFT, 0.15)],
        RegimeLabel.RECOVERY:      [(RegimeLabel.RECOVERY, 0.60), (RegimeLabel.NOMINAL, 0.40)],
    }

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.current_regime = RegimeLabel.NOMINAL
        self.ticks_remaining = self._sample_duration()
        self.temperature = 72.0
        self.pressure    = 14.7
        self.flow_rate   = 100.0

    def _sample_duration(self) -> int:
        lo, hi = self.REGIMES[self.current_regime]["duration"]
        return int(self.rng.integers(lo, hi))

    def _transition(self):
        candidates, weights = zip(*self.TRANSITIONS[self.current_regime])
        idx = self.rng.choice(len(candidates), p=weights)
        self.current_regime = candidates[idx]
        self.ticks_remaining = self._sample_duration()

    def next_tick(self, sensor_id: str) -> SignalTick:
        r = self.REGIMES[self.current_regime]
        self.temperature += self.rng.normal(r["drift"] * 5,   r["vol"] * 3)
        self.pressure    += self.rng.normal(r["drift"] * 0.2, r["vol"] * 0.1)
        self.flow_rate   += self.rng.normal(-r["drift"] * 2,  r["vol"] * 1.5)

        self.ticks_remaining -= 1
        if self.ticks_remaining <= 0:
            self._transition()

        return SignalTick(
            domain=SignalDomain.SENSOR,
            instrument=sensor_id,
            temperature=round(float(self.temperature), 3),
            pressure=round(float(self.pressure), 4),
            flow_rate=round(float(self.flow_rate), 3),
        )


# ──────────────────────────────────────────────────────────────────
# Derived feature enrichment
# ──────────────────────────────────────────────────────────────────

class FeatureEnricher:
    """Computes rolling derived features from raw ticks."""

    def __init__(self, window: int = 20):
        self.window = window
        self._prices: deque[float] = deque(maxlen=window)
        self._returns: deque[float] = deque(maxlen=window)

    def enrich(self, tick: SignalTick) -> SignalTick:
        value = tick.price if tick.price is not None else tick.temperature
        if value is None:
            return tick

        if self._prices:
            ret = (value - self._prices[-1]) / self._prices[-1]
            self._returns.append(ret)

        self._prices.append(value)

        if len(self._returns) >= 2:
            tick.volatility = round(float(np.std(list(self._returns))), 6)
        if len(self._returns) >= 1:
            tick.returns_1 = round(self._returns[-1], 6)
        if len(self._returns) >= 5:
            tick.returns_5 = round(sum(list(self._returns)[-5:]), 6)

        return tick


# ──────────────────────────────────────────────────────────────────
# Databento live feed adapter
# ──────────────────────────────────────────────────────────────────

class DatabentoFeed:
    """
    Wraps the Databento Python client to produce SignalTick objects.
    Falls back to synthetic if key is missing or connection fails.
    """

    def __init__(self, instrument: str):
        self.instrument = instrument
        self.key = os.getenv("DATABENTO_API_KEY")
        self._client = None
        self._available = False
        self._buffer: asyncio.Queue[SignalTick] = asyncio.Queue(maxsize=100)

    async def connect(self) -> bool:
        if not self.key:
            return False
        try:
            import databento as db  # optional dep
            self._client = db.Live(key=self.key)
            self._available = True
            asyncio.create_task(self._stream_loop())
            return True
        except Exception:
            return False

    async def _stream_loop(self):
        """Background task: translate Databento records to SignalTick."""
        try:
            import databento as db
            self._client.subscribe(
                dataset="GLBX.MDP3",
                schema="mbp-1",
                symbols=[self.instrument],
            )
            async for record in self._client:
                if hasattr(record, "price") and hasattr(record, "size"):
                    tick = SignalTick(
                        domain=SignalDomain.FINANCIAL,
                        instrument=self.instrument,
                        price=record.price / 1e9,     # Databento fixed-point
                        volume=float(record.size),
                        bid=getattr(record, "bid_px", record.price / 1e9) / 1e9
                            if hasattr(record, "bid_px") else None,
                        ask=getattr(record, "ask_px", record.price / 1e9) / 1e9
                            if hasattr(record, "ask_px") else None,
                    )
                    if not self._buffer.full():
                        await self._buffer.put(tick)
        except Exception:
            self._available = False

    async def next_tick(self) -> Optional[SignalTick]:
        if not self._available:
            return None
        try:
            return self._buffer.get_nowait()
        except asyncio.QueueEmpty:
            return None


# ──────────────────────────────────────────────────────────────────
# Public interface: SignalService
# ──────────────────────────────────────────────────────────────────

class SignalService:
    """
    Unified signal entry point.
    Usage:
        svc = SignalService(config)
        await svc.start()
        async for tick in svc.stream():
            ...
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._fin_sim  = FinancialRegime()
        self._sensor_sim = SensorRegime()
        self._enricher = FeatureEnricher()
        self._databento: Optional[DatabentoFeed] = None
        self._use_databento = False

    async def start(self):
        if self.config.use_real_data and self.config.domain == SignalDomain.FINANCIAL:
            feed = DatabentoFeed(self.config.instrument)
            ok = await feed.connect()
            if ok:
                self._databento = feed
                self._use_databento = True

    async def stream(self) -> AsyncGenerator[SignalTick, None]:
        tick_index = 0
        while True:
            tick = await self._next_raw_tick()
            tick = self._enricher.enrich(tick)
            yield tick
            tick_index += 1
            await asyncio.sleep(self.config.bar_interval_sec)

    async def _next_raw_tick(self) -> SignalTick:
        # Try Databento first
        if self._use_databento and self._databento:
            tick = await self._databento.next_tick()
            if tick:
                return tick

        # Synthetic fallback
        if self.config.domain == SignalDomain.FINANCIAL:
            return self._fin_sim.next_tick(self.config.instrument)
        else:
            return self._sensor_sim.next_tick(self.config.sensor_id)

    def update_config(self, config: DemoConfig):
        """Hot-swap config without restarting the service."""
        self.config = config
