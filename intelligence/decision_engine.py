"""
strategy/decision_engine.py — Gated decision tree for MARK II v2

Replaces the old linear-weighted composite scoring system with an
explicit gated decision tree. Every gate must pass independently —
no single strong component can override a weak one.

Gate sequence:
  1. STATE GATE:  Is the state POST_EVENT? (only trading window)
  2. EVENT GATE:  Is the event surprise tradeable? (magnitude > threshold)
  3. TREND GATE:  Is the EMA 50/200 trend aligned with event direction?
  4. RSI VETO:    Is RSI at extreme levels against the trade?
  5. SPREAD VETO: Is the spread too wide (even for post-news)?
  6. NLP CHECK:   (Optional) Is sentiment aligned? (advisory only)
  7. PASS:        All gates passed → generate BUY or SELL

Design principles:
  - Each gate is binary (pass/fail), not weighted
  - A veto from ANY gate kills the trade
  - Confidence comes from event magnitude × tier, not from composite scoring
  - The decision is fully traceable: each gate logs its result
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core import config
from core.logger import get_logger
from strategy.event_scorer import EventScore
from strategy.technical_filter import TechnicalContext

logger = get_logger("DecisionEngine")


@dataclass
class Decision:
    """Final trading decision produced by the gated decision tree."""
    action: str              # "BUY", "SELL", "NO_TRADE"
    confidence: float        # 0.0 to 1.0 (from event magnitude)
    sl_price: float
    tp_price: float
    event_name: str
    event_direction: str
    gates_passed: list[str] = field(default_factory=list)
    gates_failed: list[str] = field(default_factory=list)
    reason: str = ""


class DecisionEngine:
    """
    Gated decision tree that replaces linear weighted scoring.

    In EVENT_ONLY mode (the only mode in v2), trades can only be
    generated when:
      - State is POST_EVENT
      - A macro event surprise exceeds the magnitude threshold
      - The technical trend confirms the event direction
      - No veto conditions are active (RSI extreme, spread too wide)
    """

    def decide(
        self,
        state: str,
        event_score: Optional[EventScore],
        tech_context: Optional[TechnicalContext],
        current_spread: float = 0.0,
        max_spread: float = 5.0,
        symbol_key: str = "",
        sentiment_score: float = 0.0,
    ) -> Decision:
        """
        Run the gated decision tree and return a Decision.

        Args:
            state: Current state machine state (e.g., "POST_EVENT")
            event_score: Score from the macro event surprise
            tech_context: Technical context for confirmation
            current_spread: Current bid-ask spread
            max_spread: Max acceptable spread for the symbol
            symbol_key: Symbol identifier for logging
            sentiment_score: NLP sentiment score (only used if ENABLE_NLP)
        """
        passed = []
        failed = []

        # ── Gate 1: STATE GATE ──────────────────────────────────────
        if state != "POST_EVENT":
            failed.append(f"STATE: {state} (requires POST_EVENT)")
            return self._no_trade(
                event_score, passed, failed,
                f"State={state}: trading only allowed in POST_EVENT"
            )
        passed.append("STATE: POST_EVENT ✓")

        # ── Gate 2: EVENT GATE ──────────────────────────────────────
        if event_score is None or not event_score.is_tradeable:
            reason = "no event data"
            if event_score:
                reason = f"magnitude {event_score.magnitude:.3f} < threshold"
            failed.append(f"EVENT: {reason}")
            return self._no_trade(
                event_score, passed, failed,
                f"Event surprise too weak or missing: {reason}"
            )
        passed.append(
            f"EVENT: {event_score.event_name} surprise={event_score.raw_surprise:+.3f} ✓"
        )

        # Determine proposed direction from event
        if event_score.direction == "BULLISH":
            proposed_action = "BUY"
        elif event_score.direction == "BEARISH":
            proposed_action = "SELL"
        else:
            failed.append("EVENT: direction is NEUTRAL")
            return self._no_trade(
                event_score, passed, failed,
                "Event direction is NEUTRAL — no trade"
            )

        # ── Gate 3: TREND GATE ──────────────────────────────────────
        if tech_context is None or not tech_context.is_valid:
            failed.append("TREND: no valid technical data")
            return self._no_trade(
                event_score, passed, failed,
                "No valid technical context for trend confirmation"
            )
        
        # News bypass: if the surprise is very strong, ignore trend misalignment
        STRONG_SURPRISE_THRESHOLD = 0.05
        is_strong_event = event_score.magnitude >= STRONG_SURPRISE_THRESHOLD

        if not tech_context.is_trend_aligned(proposed_action):
            if is_strong_event:
                passed.append(
                    f"TREND: {tech_context.trend_direction} misaligned BUT strong surprise "
                    f"({event_score.magnitude:.3f} >= {STRONG_SURPRISE_THRESHOLD}) ✓"
                )
            else:
                failed.append(
                    f"TREND: {tech_context.trend_direction} ≠ {proposed_action}"
                )
                return self._no_trade(
                    event_score, passed, failed,
                    f"Trend misaligned: EMA trend is {tech_context.trend_direction} "
                    f"but event suggests {proposed_action} (surprise below {STRONG_SURPRISE_THRESHOLD})"
                )
        else:
            passed.append(
                f"TREND: {tech_context.trend_direction} aligned with {proposed_action} ✓"
            )

        # ── Gate 4: RSI VETO ────────────────────────────────────────
        if tech_context.is_rsi_vetoed(proposed_action):
            rsi_val = tech_context.rsi
            failed.append(f"RSI: {rsi_val:.1f} vetoes {proposed_action}")
            return self._no_trade(
                event_score, passed, failed,
                f"RSI veto: RSI={rsi_val:.1f} at extreme for {proposed_action}"
            )
        passed.append(f"RSI: {tech_context.rsi:.1f} within range ✓")

        # ── Gate 5: SPREAD VETO ─────────────────────────────────────
        post_news_max = max_spread * config.MAX_SPREAD_MULTIPLIER_POST_NEWS
        if current_spread > post_news_max:
            failed.append(
                f"SPREAD: {current_spread:.1f} > {post_news_max:.1f} (post-news max)"
            )
            return self._no_trade(
                event_score, passed, failed,
                f"Spread too wide: {current_spread:.1f} > {post_news_max:.1f}"
            )
        passed.append(f"SPREAD: {current_spread:.1f} ≤ {post_news_max:.1f} ✓")

        # ── Gate 6: NLP CHECK (optional, advisory only) ─────────────
        if config.ENABLE_NLP and sentiment_score != 0.0:
            if proposed_action == "BUY" and sentiment_score < -0.3:
                passed.append(f"NLP: sentiment={sentiment_score:+.2f} ⚠️ bearish (advisory)")
            elif proposed_action == "SELL" and sentiment_score > 0.3:
                passed.append(f"NLP: sentiment={sentiment_score:+.2f} ⚠️ bullish (advisory)")
            else:
                passed.append(f"NLP: sentiment={sentiment_score:+.2f} ✓")

        # ── ALL GATES PASSED → TRADE ───────────────────────────────
        # Compute SL/TP from ATR
        current_price = tech_context.price
        sl_price, tp_price = tech_context.compute_sl_tp(
            proposed_action, current_price, symbol_key
        )

        # Confidence from event magnitude (0.0 to 1.0)
        confidence = min(1.0, event_score.magnitude * 2.0)

        decision = Decision(
            action=proposed_action,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            event_name=event_score.event_name,
            event_direction=event_score.direction,
            gates_passed=passed,
            gates_failed=failed,
            reason=f"ALL GATES PASSED → {proposed_action} (confidence={confidence:.2f})",
        )

        logger.info(
            f"🎯 [{symbol_key}] DECISION: {proposed_action} | "
            f"Event={event_score.event_name} ({event_score.direction}) | "
            f"Confidence={confidence:.2f} | "
            f"SL={sl_price:.2f} TP={tp_price:.2f} | "
            f"Gates: {len(passed)} passed, {len(failed)} failed"
        )

        # Log all gates for transparency
        for g in passed:
            logger.debug(f"  ✅ {g}")

        return decision

    @staticmethod
    def _no_trade(
        event_score: Optional[EventScore],
        passed: list[str],
        failed: list[str],
        reason: str,
    ) -> Decision:
        """Construct a NO_TRADE decision with full gate trace."""
        event_name = event_score.event_name if event_score else "None"
        event_dir = event_score.direction if event_score else "NEUTRAL"

        logger.debug(
            f"⛔ NO_TRADE: {reason} | "
            f"Gates passed: {len(passed)}, failed: {len(failed)}"
        )

        return Decision(
            action="NO_TRADE",
            confidence=0.0,
            sl_price=0.0,
            tp_price=0.0,
            event_name=event_name,
            event_direction=event_dir,
            gates_passed=passed,
            gates_failed=failed,
            reason=reason,
        )
