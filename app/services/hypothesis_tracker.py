"""
hypothesis_tracker.py
Multiple Hypothesis Tracker for regime estimation.

Architecture:
  1. Feature extraction → latent vector per tick
  2. FAISS ANN search   → retrieve nearest historical regime fingerprints
  3. Bayesian update    → update hypothesis posterior weights
  4. Entropy compute    → uncertainty metric for the animation
  5. Prune + rehydrate  → keep hypothesis set healthy

This is the core architectural differentiator vs. LLM-only systems.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from typing import Optional

import numpy as np

from app.models.comparison_models import (
    DemoConfig,
    Hypothesis,
    HypothesisSet,
    RegimeLabel,
    SignalDomain,
    SignalTick,
    SimilarityMatch,
)

# Try to import FAISS; fall back to brute-force cosine if unavailable
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# Latent Encoder
# ──────────────────────────────────────────────────────────────────

class LatentEncoder:
    """
    Encodes a window of SignalTicks into a fixed-dimensional feature vector.
    Dimensions: [returns_z, vol_z, spread_z, trend_z, momentum_z,
                 temp_z, pres_z, flow_z, regime_clock_sin, regime_clock_cos]
    → 10-D normalized vector
    """

    VECTOR_DIM = 10

    def __init__(self, window: int = 20):
        self._window = window
        self._history: deque[SignalTick] = deque(maxlen=window)
        # Running stats for z-score normalization
        self._means = np.zeros(8)
        self._stds  = np.ones(8)
        self._n     = 0
        self._tick_clock = 0

    def push(self, tick: SignalTick) -> np.ndarray:
        self._history.append(tick)
        self._tick_clock += 1
        raw = self._raw_features(tick)
        self._update_stats(raw)
        return self._encode(raw)

    def _raw_features(self, tick: SignalTick) -> np.ndarray:
        f = np.zeros(8)
        f[0] = tick.returns_1 or 0.0
        f[1] = tick.returns_5 or 0.0
        f[2] = tick.volatility or 0.0
        f[3] = tick.spread or 0.0
        f[4] = (tick.volume or 0.0) / 1000.0   # normalise volume
        f[5] = tick.temperature or 0.0
        f[6] = tick.pressure    or 0.0
        f[7] = tick.flow_rate   or 0.0
        return f

    def _update_stats(self, raw: np.ndarray):
        self._n += 1
        if self._n == 1:
            self._means = raw.copy()
        else:
            alpha = 0.05           # exponential moving average
            self._means = (1 - alpha) * self._means + alpha * raw
            self._stds  = (1 - alpha) * self._stds  + alpha * np.abs(raw - self._means)
        self._stds = np.where(self._stds < 1e-8, 1.0, self._stds)

    def _encode(self, raw: np.ndarray) -> np.ndarray:
        z = (raw - self._means) / self._stds
        # Clock features encode temporal position within assumed regime cycle
        phase = (self._tick_clock % 60) / 60.0 * 2 * math.pi
        clock = np.array([math.sin(phase), math.cos(phase)])
        vec = np.concatenate([z, clock]).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec


# ──────────────────────────────────────────────────────────────────
# Historical Regime Index (FAISS or brute-force)
# ──────────────────────────────────────────────────────────────────

class RegimeIndex:
    """
    Stores a growing archive of (vector, regime_label, ts) tuples.
    Supports ANN search to find similar historical regime fingerprints.
    """

    def __init__(self, dim: int = LatentEncoder.VECTOR_DIM, capacity: int = 2000):
        self.dim = dim
        self.capacity = capacity
        self._vectors: list[np.ndarray] = []
        self._labels:  list[RegimeLabel] = []
        self._timestamps: list[float] = []
        self._faiss_index = None

        if FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatIP(dim)   # inner product ≈ cosine on unit vecs

    def add(self, vec: np.ndarray, label: RegimeLabel, ts: float):
        if len(self._vectors) >= self.capacity:
            # Rolling window: drop oldest 10%
            keep = int(self.capacity * 0.9)
            self._vectors   = self._vectors[-keep:]
            self._labels    = self._labels[-keep:]
            self._timestamps = self._timestamps[-keep:]
            if FAISS_AVAILABLE and self._faiss_index:
                self._faiss_index.reset()
                batch = np.vstack(self._vectors)
                self._faiss_index.add(batch)

        self._vectors.append(vec)
        self._labels.append(label)
        self._timestamps.append(ts)

        if FAISS_AVAILABLE and self._faiss_index:
            self._faiss_index.add(vec.reshape(1, -1))

    def search(self, query: np.ndarray, k: int = 5) -> list[SimilarityMatch]:
        if len(self._vectors) < k:
            return []

        if FAISS_AVAILABLE and self._faiss_index and self._faiss_index.ntotal > 0:
            distances, indices = self._faiss_index.search(query.reshape(1, -1), k)
            hits = list(zip(distances[0], indices[0]))
        else:
            # Brute-force cosine
            mat = np.vstack(self._vectors)
            sims = mat @ query
            top_k = np.argsort(-sims)[:k]
            hits = [(float(sims[i]), i) for i in top_k]

        results = []
        for score, idx in hits:
            if idx < 0 or idx >= len(self._labels):
                continue
            results.append(SimilarityMatch(
                regime_label=self._labels[idx],
                similarity_score=round(float(score), 4),
                historical_ts=self._timestamps[idx],
                context_snippet=f"{self._labels[idx].value} @ sim={score:.3f}",
            ))
        return results


# ──────────────────────────────────────────────────────────────────
# Bayesian Hypothesis Updater
# ──────────────────────────────────────────────────────────────────

# Regime-conditional emission parameters (mean, std of 1-bar return)
REGIME_LIKELIHOODS: dict[RegimeLabel, dict] = {
    RegimeLabel.BULL_TREND:   {"mu":  0.0008, "sigma": 0.0012},
    RegimeLabel.BEAR_TREND:   {"mu": -0.0010, "sigma": 0.0018},
    RegimeLabel.MEAN_REVERT:  {"mu":  0.0000, "sigma": 0.0006},
    RegimeLabel.HIGH_VOL:     {"mu":  0.0000, "sigma": 0.0035},
    RegimeLabel.REGIME_SHIFT: {"mu":  0.0015, "sigma": 0.0040},
    RegimeLabel.NOMINAL:      {"mu":  0.000,  "sigma": 0.02},
    RegimeLabel.ANOMALY_DRIFT:{"mu":  0.015,  "sigma": 0.05},
    RegimeLabel.FAULT_IMMINENT:{"mu": 0.040,  "sigma": 0.12},
    RegimeLabel.RECOVERY:     {"mu": -0.020,  "sigma": 0.03},
    RegimeLabel.LOW_VOL:      {"mu":  0.000,  "sigma": 0.003},
}

def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


class HypothesisUpdater:
    """Applies Bayesian update to a list of Hypothesis objects."""

    def update(
        self,
        hypotheses: list[Hypothesis],
        tick: SignalTick,
        similarity_matches: list[SimilarityMatch],
    ) -> list[Hypothesis]:
        obs = tick.returns_1 if tick.returns_1 is not None else (
            (tick.temperature or 0) / 100.0
        )

        # Combine emission likelihood with similarity-based boost
        sim_boost: dict[RegimeLabel, float] = {}
        for m in similarity_matches:
            sim_boost[m.regime_label] = sim_boost.get(m.regime_label, 0.0) + m.similarity_score

        updated = []
        for h in hypotheses:
            params = REGIME_LIKELIHOODS.get(h.label, {"mu": 0.0, "sigma": 0.01})
            emission_like = _gaussian_pdf(obs, params["mu"], params["sigma"])
            sim_factor = 1.0 + 0.5 * sim_boost.get(h.label, 0.0)
            new_like = emission_like * sim_factor

            new_weight = h.weight * new_like
            evidence = f"obs={obs:.5f} | like={new_like:.4f}"
            trail = list(h.evidence_trail[-4:]) + [evidence]

            updated.append(Hypothesis(
                id=h.id,
                label=h.label,
                weight=max(new_weight, 1e-10),
                likelihood=new_like,
                evidence_trail=trail,
                similar_regime_ts=similarity_matches[0].historical_ts if similarity_matches else None,
                age_ticks=h.age_ticks + 1,
            ))

        # Normalise
        total = sum(h.weight for h in updated)
        if total > 0:
            for h in updated:
                h.weight = round(h.weight / total, 6)

        return updated


# ──────────────────────────────────────────────────────────────────
# Hypothesis Manager (lifecycle: prune + spawn)
# ──────────────────────────────────────────────────────────────────

FINANCIAL_REGIMES = [
    RegimeLabel.BULL_TREND, RegimeLabel.BEAR_TREND,
    RegimeLabel.MEAN_REVERT, RegimeLabel.HIGH_VOL, RegimeLabel.REGIME_SHIFT,
]
SENSOR_REGIMES = [
    RegimeLabel.NOMINAL, RegimeLabel.ANOMALY_DRIFT,
    RegimeLabel.FAULT_IMMINENT, RegimeLabel.RECOVERY,
]

class HypothesisManager:
    def __init__(self, config: DemoConfig):
        self.config = config
        domain_regimes = (
            FINANCIAL_REGIMES if config.domain == SignalDomain.FINANCIAL else SENSOR_REGIMES
        )
        n = min(config.hypothesis_count, len(domain_regimes))
        uniform = 1.0 / n
        self.hypotheses: list[Hypothesis] = [
            Hypothesis(
                id=str(uuid.uuid4())[:8],
                label=lbl,
                weight=uniform,
                likelihood=uniform,
                evidence_trail=[],
            )
            for lbl in domain_regimes[:n]
        ]
        self._domain_regimes = domain_regimes

    def prune_and_spawn(self) -> None:
        """Drop low-weight hypotheses; spawn new ones to maintain diversity."""
        threshold = self.config.prune_threshold
        surviving = [h for h in self.hypotheses if h.weight >= threshold]

        if not surviving:
            surviving = [max(self.hypotheses, key=lambda h: h.weight)]

        alive_labels = {h.label for h in surviving}
        missing = [lbl for lbl in self._domain_regimes if lbl not in alive_labels]

        n_to_spawn = max(0, self.config.hypothesis_count - len(surviving))
        for lbl in missing[:n_to_spawn]:
            min_weight = min(h.weight for h in surviving) * 0.5
            surviving.append(Hypothesis(
                id=str(uuid.uuid4())[:8],
                label=lbl,
                weight=min_weight,
                likelihood=min_weight,
                evidence_trail=["spawned"],
            ))

        # Re-normalise
        total = sum(h.weight for h in surviving)
        for h in surviving:
            h.weight = round(h.weight / total, 6)

        self.hypotheses = surviving


# ──────────────────────────────────────────────────────────────────
# Public: HypothesisTracker
# ──────────────────────────────────────────────────────────────────

class HypothesisTracker:
    """
    Full MHT pipeline: encode → retrieve → update → prune → emit.
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._encoder  = LatentEncoder()
        self._index    = RegimeIndex()
        self._updater  = HypothesisUpdater()
        self._manager  = HypothesisManager(config)
        self._tick_idx = 0

    def process(self, tick: SignalTick) -> tuple[HypothesisSet, list[SimilarityMatch], np.ndarray]:
        """
        Returns (HypothesisSet, similarity_matches, latent_vector)
        """
        vec = self._encoder.push(tick)

        # Retrieve similar regimes from history
        matches = self._index.search(vec, k=min(5, max(1, self._tick_idx)))

        # Top label (for indexing next time)
        top_h = max(self._manager.hypotheses, key=lambda h: h.weight)
        if self._tick_idx > 5:
            self._index.add(vec, top_h.label, tick.ts)

        # Bayesian update
        self._manager.hypotheses = self._updater.update(
            self._manager.hypotheses, tick, matches
        )

        # Prune every 5 ticks
        if self._tick_idx % 5 == 0:
            self._manager.prune_and_spawn()

        top_h = max(self._manager.hypotheses, key=lambda h: h.weight)
        entropy = self._entropy()
        hs = HypothesisSet(
            hypotheses=list(self._manager.hypotheses),
            top_label=top_h.label,
            top_weight=top_h.weight,
            entropy=round(entropy, 4),
            tick_index=self._tick_idx,
        )
        self._tick_idx += 1
        return hs, matches, vec

    def _entropy(self) -> float:
        weights = [h.weight for h in self._manager.hypotheses if h.weight > 0]
        return -sum(w * math.log(w) for w in weights) if weights else 0.0
