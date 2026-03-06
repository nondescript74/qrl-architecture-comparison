"""
decision_pipeline.py
Decision Architecture Pipeline — the resilient counterpart to LLM-only.

Layers:
  1. Latent State Encoder   (in HypothesisTracker._encoder)
  2. Similarity Retrieval   (FAISS / brute-force cosine)
  3. Hypothesis Tracking    (Bayesian MHT update)
  4. Decision Evaluation    (score actions against hypothesis set)
  5. Ranked Action Output   (confidence intervals included)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from app.models.comparison_models import (
    DecisionPipelineState,
    DemoConfig,
    Hypothesis,
    HypothesisSet,
    RankedAction,
    RegimeLabel,
    SignalDomain,
    SignalTick,
    SimilarityMatch,
)
from app.services.hypothesis_tracker import HypothesisTracker


# ──────────────────────────────────────────────────────────────────
# Action catalogue
# ──────────────────────────────────────────────────────────────────

FINANCIAL_ACTIONS = [
    "LONG  — trend following momentum entry",
    "SHORT — trend following momentum entry",
    "FLAT  — stand aside, await confirmation",
    "HEDGE — reduce gross exposure, add optionality",
    "SIZE_DOWN — reduce position size, elevated uncertainty",
]

SENSOR_ACTIONS = [
    "MONITOR — continue normal observation cadence",
    "ALERT   — escalate monitoring frequency",
    "INSPECT — dispatch maintenance team for inspection",
    "SHUTDOWN — initiate controlled shutdown sequence",
    "RECOVER  — execute recovery protocol",
]

# Regime → preferred action mapping
REGIME_ACTION_AFFINITY: dict[RegimeLabel, dict[str, float]] = {
    RegimeLabel.BULL_TREND:    {"LONG": 0.85,  "SHORT": 0.02, "FLAT": 0.08, "HEDGE": 0.03, "SIZE_DOWN": 0.02},
    RegimeLabel.BEAR_TREND:    {"LONG": 0.02,  "SHORT": 0.85, "FLAT": 0.06, "HEDGE": 0.05, "SIZE_DOWN": 0.02},
    RegimeLabel.MEAN_REVERT:   {"LONG": 0.25,  "SHORT": 0.25, "FLAT": 0.35, "HEDGE": 0.10, "SIZE_DOWN": 0.05},
    RegimeLabel.HIGH_VOL:      {"LONG": 0.05,  "SHORT": 0.05, "FLAT": 0.20, "HEDGE": 0.40, "SIZE_DOWN": 0.30},
    RegimeLabel.REGIME_SHIFT:  {"LONG": 0.10,  "SHORT": 0.10, "FLAT": 0.20, "HEDGE": 0.30, "SIZE_DOWN": 0.30},
    RegimeLabel.LOW_VOL:       {"LONG": 0.30,  "SHORT": 0.20, "FLAT": 0.40, "HEDGE": 0.05, "SIZE_DOWN": 0.05},
    RegimeLabel.NOMINAL:       {"MONITOR": 0.90, "ALERT": 0.05, "INSPECT": 0.03, "SHUTDOWN": 0.01, "RECOVER": 0.01},
    RegimeLabel.ANOMALY_DRIFT: {"MONITOR": 0.25, "ALERT": 0.60, "INSPECT": 0.12, "SHUTDOWN": 0.02, "RECOVER": 0.01},
    RegimeLabel.FAULT_IMMINENT:{"MONITOR": 0.02, "ALERT": 0.15, "INSPECT": 0.35, "SHUTDOWN": 0.45, "RECOVER": 0.03},
    RegimeLabel.RECOVERY:      {"MONITOR": 0.30, "ALERT": 0.20, "INSPECT": 0.10, "SHUTDOWN": 0.05, "RECOVER": 0.35},
}


# ──────────────────────────────────────────────────────────────────
# Decision Evaluator
# ──────────────────────────────────────────────────────────────────

class DecisionEvaluator:
    """
    Scores candidate actions against the full hypothesis set.
    This is the key step that LLM-only systems skip:
    actions are scored BEFORE commitment, across ALL hypotheses.
    """

    def evaluate(
        self,
        hypotheses: list[Hypothesis],
        domain: SignalDomain,
    ) -> list[RankedAction]:
        action_labels = (
            ["LONG", "SHORT", "FLAT", "HEDGE", "SIZE_DOWN"]
            if domain == SignalDomain.FINANCIAL
            else ["MONITOR", "ALERT", "INSPECT", "SHUTDOWN", "RECOVER"]
        )
        full_actions = (
            FINANCIAL_ACTIONS if domain == SignalDomain.FINANCIAL else SENSOR_ACTIONS
        )
        action_map = dict(zip(action_labels, full_actions))

        scores: dict[str, float] = {a: 0.0 for a in action_labels}
        variance: dict[str, float] = {a: 0.0 for a in action_labels}
        supporters: dict[str, list[str]] = {a: [] for a in action_labels}

        for h in hypotheses:
            affinity = REGIME_ACTION_AFFINITY.get(h.label, {})
            for action in action_labels:
                p_action_given_regime = affinity.get(action, 0.05)
                contribution = h.weight * p_action_given_regime
                scores[action] += contribution
                supporters[action].append(h.label.value) if h.weight > 0.1 else None

        # Normalise scores
        total = sum(scores.values())
        if total > 0:
            scores = {a: v / total for a, v in scores.items()}

        # Variance (uncertainty on the score itself)
        for action in action_labels:
            affinity_vals = [
                REGIME_ACTION_AFFINITY.get(h.label, {}).get(action, 0.05)
                for h in hypotheses
            ]
            weights = [h.weight for h in hypotheses]
            mean = scores[action]
            var = sum(w * (v - mean) ** 2 for w, v in zip(weights, affinity_vals))
            variance[action] = var

        # Build ranked list
        ranked = []
        for action in action_labels:
            s = scores[action]
            std = variance[action] ** 0.5
            z = 1.645  # 90% CI
            ranked.append(RankedAction(
                action=action_map[action],
                score=round(s, 4),
                confidence_interval=(
                    round(max(0.0, s - z * std), 4),
                    round(min(1.0, s + z * std), 4),
                ),
                supporting_hypotheses=supporters[action],
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked


# ──────────────────────────────────────────────────────────────────
# Public: DecisionPipeline
# ──────────────────────────────────────────────────────────────────

class DecisionPipeline:
    """
    Full layered decision architecture pipeline.
    Stateful: hypothesis beliefs persist and evolve across ticks.
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._tracker   = HypothesisTracker(config)
        self._evaluator = DecisionEvaluator()
        self._tick_idx  = 0

    def process(self, tick: SignalTick) -> DecisionPipelineState:
        t0 = time.monotonic()

        hypothesis_set, similarity_matches, latent_vec = self._tracker.process(tick)

        ranked_actions = self._evaluator.evaluate(
            hypothesis_set.hypotheses,
            self.config.domain,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        self._tick_idx += 1

        return DecisionPipelineState(
            tick_index=self._tick_idx,
            latent_vector_dim=len(latent_vec),
            top_similarity_matches=similarity_matches[:3],
            hypothesis_set=hypothesis_set,
            ranked_actions=ranked_actions[:3],
            latency_ms=round(latency_ms, 2),
            hypothesis_count=len(hypothesis_set.hypotheses),
        )
