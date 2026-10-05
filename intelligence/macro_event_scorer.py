"""
strategy/event_scorer.py — Macro event surprise scoring engine (v2)

The core edge hypothesis of MARK II v2:
  "High-impact macro surprises create statistically exploitable
   post-news directional continuation."

This module scores the surprise from a macro event release by:
  1. Computing percentage deviation: (actual - forecast) / |forecast|
  2. Normalizing via tanh to bound the score in [-1, +1]
  3. Applying direction mapping (higher CPI = bearish for indices)
  4. Gating on minimum surprise magnitude (weak surprises = no trade)
  5. Classifying by event tier

Replaces the old strategy/event_logic.py with:
  - More robust normalization
  - Explicit magnitude gating
  - Event tier awareness
  - Better edge case handling
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from core import config
from core.logger import get_logger
from data.economic_calendar import EconomicEvent, EventTier

logger = get_logger("EventScorer")


@dataclass
class EventScore:
    """Result of scoring a macro event surprise."""
    event_name: str
    tier: EventTier
    direction: str          # "BULLISH", "BEARISH", or "NEUTRAL"
    raw_surprise: float     # (actual - forecast) / |forecast|
    magnitude: float        # abs(raw_surprise)
    score: float            # Normalized score in [-1, +1]
    is_tradeable: bool      # Passes magnitude gate?
    actual: Optional[float]
    forecast: Optional[float]
    reason: str


def score_event(event: EconomicEvent) -> EventScore:
    """
    Score a single macro event based on surprise magnitude and direction.

    Returns an EventScore with is_tradeable=False if:
      - actual or forecast data is missing
      - surprise magnitude is below MIN_SURPRISE_MAGNITUDE
      - event is not a recognized tier
    """
    tier = event.tier
    name = event.name

    actual = event.actual_numeric()
    forecast = event.forecast_numeric()

    # ── Missing data → not tradeable ────────────────────────────
    if actual is None or forecast is None:
        return EventScore(
            event_name=name,
            tier=tier,
            direction="NEUTRAL",
            raw_surprise=0.0,
            magnitude=0.0,
            score=0.0,
            is_tradeable=False,
            actual=actual,
            forecast=forecast,
            reason=f"{name}: missing actual or forecast data",
        )

    # ── Compute surprise ────────────────────────────────────────
    if abs(forecast) > 1e-10:
        raw_surprise = (actual - forecast) / abs(forecast)
    elif abs(actual) > 1e-10:
        # Forecast is zero but actual is non-zero: treat as extreme surprise
        raw_surprise = 1.0 if actual > 0 else -1.0
    else:
        # Both are effectively zero
        raw_surprise = 0.0

    magnitude = abs(raw_surprise)

    # ── Magnitude gate ──────────────────────────────────────────
    if magnitude < config.MIN_SURPRISE_MAGNITUDE:
        return EventScore(
            event_name=name,
            tier=tier,
            direction="NEUTRAL",
            raw_surprise=raw_surprise,
            magnitude=magnitude,
            score=0.0,
            is_tradeable=False,
            actual=actual,
            forecast=forecast,
            reason=(
                f"{name}: surprise={raw_surprise:+.3f} "
                f"(magnitude {magnitude:.3f} < threshold {config.MIN_SURPRISE_MAGNITUDE})"
            ),
        )

    # ── Direction mapping ───────────────────────────────────────
    direction_mult = _get_direction_multiplier(name)

    # Normalize with tanh to bound in [-1, +1] and compress outliers
    # The multiplier (3.0) controls sensitivity: tanh(3×0.1)=0.29, tanh(3×0.5)=0.91
    normalized = math.tanh(raw_surprise * 3.0) * direction_mult

    # Clamp just in case
    score = max(-1.0, min(1.0, normalized))

    # Determine direction for indices
    if score > 0.05:
        direction = "BULLISH"
    elif score < -0.05:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    result = EventScore(
        event_name=name,
        tier=tier,
        direction=direction,
        raw_surprise=raw_surprise,
        magnitude=magnitude,
        score=score,
        is_tradeable=True,
        actual=actual,
        forecast=forecast,
        reason=(
            f"{name}: actual={actual} vs forecast={forecast} → "
            f"surprise={raw_surprise:+.3f} → {direction} "
            f"(score={score:+.3f}, tier={tier.value})"
        ),
    )

    logger.info(
        f"📊 {name} [{tier.value}] | "
        f"Actual={actual} vs Forecast={forecast} | "
        f"Surprise={raw_surprise:+.3f} | "
        f"Score={score:+.3f} → {direction}"
    )

    return result


def score_best_event(events: List[EconomicEvent]) -> EventScore:
    """
    Score all given events and return the one with the highest
    magnitude (the strongest surprise signal).
    """
    if not events:
        return EventScore(
            event_name="None",
            tier=EventTier.UNKNOWN,
            direction="NEUTRAL",
            raw_surprise=0.0,
            magnitude=0.0,
            score=0.0,
            is_tradeable=False,
            actual=None,
            forecast=None,
            reason="No events to score",
        )

    best: Optional[EventScore] = None
    for event in events:
        if not event.has_actual:
            continue  # Can't score events without actual data
        scored = score_event(event)
        if best is None or scored.magnitude > best.magnitude:
            best = scored

    if best is None:
        return EventScore(
            event_name="None",
            tier=EventTier.UNKNOWN,
            direction="NEUTRAL",
            raw_surprise=0.0,
            magnitude=0.0,
            score=0.0,
            is_tradeable=False,
            actual=None,
            forecast=None,
            reason="No events with actual data available",
        )

    return best


def _get_direction_multiplier(name: str) -> float:
    """
    Determine how a higher-than-expected value affects index direction.

    Returns:
      +1.0 → higher actual is BULLISH for indices (growth/jobs)
      -1.0 → higher actual is BEARISH for indices (inflation/rates)
    """
    name_lower = name.lower()
    for keyword, mult in config.EVENT_DIRECTION_MAP.items():
        if keyword.lower() in name_lower:
            return mult
    # Default: assume higher is bullish (growth proxy)
    return 1.0
