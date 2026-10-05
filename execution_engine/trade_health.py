"""
execution_engine/trade_health.py — Trade Health Monitor v2 for MARK III

Context-aware thesis validation system. Instead of blindly checking
"am I losing money?", this monitor asks:
"Is the reason I entered this trade still valid right now?"

Components (100 pts total, v2 — 5 components):
  20 pts → Time Decay (regime-aware grace periods)
  20 pts → P&L Trajectory (ATR-normalized, not binary)
  20 pts → Delta Alignment (does latest order flow agree with trade?)
  15 pts → Spread Health (has spread widened dangerously?)
  25 pts → Thesis Validation (regime persistence + structural integrity)

Actions:
  HOLD       → score >= 60 (thesis intact, keep it)
  TIGHTEN_SL → score 40-59 (thesis weakening, move SL to breakeven)
  EXIT       → score < 40 or timeout (thesis failed, close immediately)

v2 Design Principles:
  - Does NOT increase risk tolerance (SL, max loss, drawdown untouched)
  - Structural reference frozen at entry time (no moving goalposts)
  - FAST regime ≠ permanent immunity (structure break overrides)
  - Lightweight, deterministic, explainable — no recursive scoring
  - Does NOT re-run the SignalScorer (entry logic ≠ management logic)
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

from config import settings
from execution_engine.regime_profile import RegimeProfile, get_profile
from monitoring.logger import get_logger

logger = get_logger("TradeHealth")


# ─── Health States ──────────────────────────────────────────────
HEALTHY   = "HEALTHY"
DEGRADING = "DEGRADING"
CRITICAL  = "CRITICAL"

# ─── Recommended Actions ───────────────────────────────────────
HOLD       = "HOLD"
TIGHTEN_SL = "TIGHTEN_SL"
EXIT       = "EXIT"

# ─── Regime hierarchy (for persistence scoring) ────────────────
_REGIME_RANK = {
    "DEAD": 0,
    "SLOW_TREND": 1,
    "NORMAL": 2,
    "FAST": 3,
    "EXPLOSIVE": 4,
}

# ─── Regime-aware grace periods (seconds) ──────────────────────
_REGIME_GRACE = {
    "EXPLOSIVE": 180,   # 3 min — violent pullbacks are normal
    "FAST":      180,   # 3 min — same as explosive
    "NORMAL":    120,   # 2 min — standard
    "SLOW_TREND": 90,   # 1.5 min — slow trends shouldn't need long grace
    "DEAD":       60,   # 1 min — shouldn't be trading here anyway
}


@dataclass
class TradeHealth:
    """Result of a trade health evaluation."""
    score: int             # 0-100
    state: str             # HEALTHY | DEGRADING | CRITICAL
    action: str            # HOLD | TIGHTEN_SL | EXIT
    reason: str            # Human-readable explanation
    components: dict       # Score breakdown per component
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.state == HEALTHY

    @property
    def should_exit(self) -> bool:
        return self.action == EXIT

    @property
    def should_tighten(self) -> bool:
        return self.action == TIGHTEN_SL


@dataclass
class _EntryContext:
    """Frozen context from the moment the trade was entered."""
    regime: str = "NORMAL"
    atr: float = 0.0
    structural_ref: float = 0.0   # Swing low (BUY) or swing high (SELL)
    direction: str = "BUY"
    speed: int = 50               # Raw speed_score (0-100) at entry
    # v2.3: Full regime profile for regime-aware health thresholds
    profile: Optional[RegimeProfile] = None
    # P1 fix: setup type so health monitor applies correct adverse threshold
    setup_type: str = "UNKNOWN"


@dataclass
class _ExcursionState:
    """Per-ticket favorable excursion tracking."""
    peak_profit: float = 0.0
    max_favorable_distance: float = 0.0
    peak_favorable_price: float = 0.0


class TradeHealthMonitor:
    """
    Context-aware trade health monitor (v2).

    Evaluates whether an open trade's THESIS is still valid,
    not just whether the P&L is positive.

    Called from the main engine every HEALTH_CHECK_INTERVAL_SEC.
    """

    def __init__(self) -> None:
        # Cache of last states to avoid log spam
        self._last_states: Dict[int, str] = {}
        # Track peak profit per ticket
        self._peak_profits: Dict[int, float] = {}
        # Track MFE/peak favorable price per ticket
        self._excursions: Dict[int, _ExcursionState] = {}
        # Track entry time per ticket
        self._entry_times: Dict[int, float] = {}
        # v2: Frozen entry context per ticket
        self._entry_contexts: Dict[int, _EntryContext] = {}
        logger.info("TradeHealthMonitor initialized")

    def register_trade(
        self,
        ticket: int,
        entry_time: float = None,
        entry_regime: str = "NORMAL",
        entry_atr: float = 0.0,
        structural_ref: float = 0.0,
        direction: str = "BUY",
        entry_speed: int = 50,
        profile: Optional[RegimeProfile] = None,
        setup_type: str = "UNKNOWN",
    ) -> None:
        """
        Register a new trade for health tracking.

        v2.3: Also stores the frozen RegimeProfile for regime-aware
        health thresholds. The profile is NEVER recalculated.
        """
        # v2.3: If no profile provided, look it up from regime
        if profile is None:
            profile = get_profile(entry_regime)

        self._entry_times[ticket] = entry_time or time.time()
        self._peak_profits[ticket] = 0.0
        self._excursions[ticket] = _ExcursionState()
        self._last_states[ticket] = HEALTHY
        self._entry_contexts[ticket] = _EntryContext(
            regime=entry_regime,
            atr=entry_atr,
            structural_ref=structural_ref,
            direction=direction,
            speed=entry_speed,
            profile=profile,
            setup_type=setup_type,
        )
        logger.info(
            f"  📋 Trade #{ticket} registered for health monitoring │ "
            f"Regime={entry_regime} │ Speed={entry_speed} │ ATR={entry_atr:.2f} │ "
            f"StructRef={structural_ref:.2f} │ Dir={direction} │ "
            f"Profile: Degr={profile.health_degrading} Crit={profile.health_critical}"
        )

    def clear_trade(self, ticket: int) -> None:
        """Remove a closed trade from tracking."""
        self._entry_times.pop(ticket, None)
        self._peak_profits.pop(ticket, None)
        self._excursions.pop(ticket, None)
        self._last_states.pop(ticket, None)
        self._entry_contexts.pop(ticket, None)

    def evaluate(
        self,
        ticket: int,
        symbol: str,
        direction: str,       # "BUY" or "SELL"
        profit: float,        # Current unrealized P&L in USD
        entry_price: float,
        current_price: float,
        current_spread: float,
        max_spread: float,
        # v1 legacy (still used for delta alignment)
        recent_candles: Optional[str] = None,
        # v2: Current market context for thesis validation
        current_regime: str = "NORMAL",
        current_speed: int = 50,
        current_atr: float = 0.0,
        current_time: float = None,  # Fix for simulation parity
    ) -> TradeHealth:
        """
        Evaluates the health of an open trade using thesis validation.

        v2: Now receives current market context to compare against
        the frozen entry context. Does NOT re-run the scorer.

        Returns:
            TradeHealth with score, state, and recommended action.
        """
        eval_time = current_time or time.time()

        # ── Update peak profit tracking ──────────────────────────
        entry_time = self._entry_times.get(ticket, eval_time)
        entry_ctx = self._entry_contexts.get(ticket, _EntryContext())
        age_sec = eval_time - entry_time
        
        excursion = self._update_excursion(
            ticket=ticket,
            direction=direction,
            profit=profit,
            entry_price=entry_price,
            current_price=current_price,
            entry_atr=entry_ctx.atr,
        )
        peak = excursion["peak_profit"]

        # ══════════════════════════════════════════════════════════
        # 1. TIME DECAY (20 pts) — regime-aware grace
        # ══════════════════════════════════════════════════════════
        time_score, time_reason = self._score_time_decay(
            age_sec, entry_ctx.regime
        )

        # ══════════════════════════════════════════════════════════
        # 2. P&L TRAJECTORY (20 pts) — ATR-normalized
        # ══════════════════════════════════════════════════════════
        pnl_score, pnl_reason = self._score_pnl_trajectory(
            profit, peak, current_price, entry_price,
            direction, entry_ctx.atr,
        )

        # ══════════════════════════════════════════════════════════
        # 3. DELTA ALIGNMENT (20 pts)
        # ══════════════════════════════════════════════════════════
        delta_score, delta_reason = self._score_delta_alignment(
            direction, recent_candles
        )

        # ══════════════════════════════════════════════════════════
        # 4. SPREAD HEALTH (15 pts)
        # ══════════════════════════════════════════════════════════
        spread_score, spread_reason = self._score_spread_health(
            current_spread, max_spread
        )

        # ══════════════════════════════════════════════════════════
        # 5. THESIS VALIDATION (25 pts) — THE CORE UPGRADE
        # ══════════════════════════════════════════════════════════
        thesis_score, thesis_reason = self._score_thesis_validation(
            direction=direction,
            current_price=current_price,
            current_regime=current_regime,
            current_speed=current_speed,
            entry_ctx=entry_ctx,
            pressure_dir=recent_candles,
            current_atr=current_atr,
            age_sec=age_sec,
        )

        # ══════════════════════════════════════════════════════════
        # TOTAL SCORE
        # ══════════════════════════════════════════════════════════
        total_score = (
            time_score + pnl_score + delta_score
            + spread_score + thesis_score
        )
        total_score = max(0, min(100, total_score))

        components = {
            "time":   {"score": time_score,   "max": 20, "reason": time_reason},
            "pnl":    {"score": pnl_score,    "max": 20, "reason": pnl_reason},
            "delta":  {"score": delta_score,  "max": 20, "reason": delta_reason},
            "spread": {"score": spread_score, "max": 15, "reason": spread_reason},
            "thesis": {"score": thesis_score, "max": 25, "reason": thesis_reason},
        }
        giveback = self._analyze_profit_giveback(
            excursion=excursion,
            profit=profit,
            entry_ctx=entry_ctx,
            current_regime=current_regime,
            current_speed=current_speed,
            pressure_dir=recent_candles,
            thesis_score=thesis_score,
            delta_score=delta_score,
            spread_score=spread_score,
        )
        diagnostics = {
            "excursion": excursion,
            "giveback": giveback,
        }

        # ── Determine state and action ───────────────────────────
        max_age = settings.TRADE_MAX_AGE_SEC
        timed_out = age_sec >= max_age

        # v2.3: Use regime-aware thresholds from frozen profile
        profile = entry_ctx.profile
        degrading_threshold = (
            profile.health_degrading if profile
            else settings.HEALTH_DEGRADING_THRESHOLD
        )
        critical_threshold = (
            profile.health_critical if profile
            else settings.HEALTH_CRITICAL_THRESHOLD
        )

        # Grace period: regime-aware (v2.3: from frozen profile)
        grace = profile.grace_period_sec if profile else _REGIME_GRACE.get(entry_ctx.regime, 120)
        in_grace = age_sec < grace

        if in_grace:
            # During grace period, always hold unless deeply negative
            if profit < -settings.MAX_LOSS_PER_TRADE_USD:
                state = CRITICAL
                action = EXIT
                reason = f"Max loss exceeded during grace: ${profit:+.2f}"
            # ── v2.5: Immediate adverse movement exit ─────────────
            # If price moves too far against the trade within the
            # first 60 seconds the entry thesis already failed.
            # Don't wait for grace to expire — cut the loser fast.
            #
            # P1 fix: PULLBACK setups enter at a temporary discount
            # in the direction of trend — they need a wider band
            # before the reversal materialises.
            #   PULLBACK:     0.8×ATR  (was 0.5×ATR)
            #   CONTINUATION: 0.5×ATR  (unchanged)
            elif age_sec < 60 and entry_ctx.atr > 0:
                if direction == "BUY":
                    adverse = entry_price - current_price
                else:
                    adverse = current_price - entry_price
                setup_type = getattr(entry_ctx, "setup_type", "")
                adverse_mult = 0.8 if setup_type == "PULLBACK" else 0.5
                if adverse > entry_ctx.atr * adverse_mult:
                    state = CRITICAL
                    action = EXIT
                    reason = (
                        f"⚡ Immediate adverse: {adverse:.2f} > "
                        f"{adverse_mult}×ATR({entry_ctx.atr:.2f}) in {age_sec:.0f}s"
                        f" [{setup_type}]"
                    )
                else:
                    state = HEALTHY
                    action = HOLD
                    reason = (
                        f"🟢 Grace period ({age_sec:.0f}s/{grace}s) │ "
                        f"Score={total_score} │ Regime={entry_ctx.regime}"
                    )
            elif giveback["action"] == EXIT and age_sec >= 45:
                state = CRITICAL
                action = EXIT
                reason = (
                    f"Profit giveback during grace: {giveback['reason']} | "
                    f"Score={total_score}"
                )
            elif giveback["action"] == TIGHTEN_SL and age_sec >= 45:
                state = DEGRADING
                action = TIGHTEN_SL
                reason = (
                    f"Protective tighten during grace: {giveback['reason']} | "
                    f"Score={total_score}"
                )
            else:
                state = HEALTHY
                action = HOLD
                reason = (
                    f"🟢 Grace period ({age_sec:.0f}s/{grace}s) │ "
                    f"Score={total_score} │ Regime={entry_ctx.regime}"
                )
        elif timed_out:
            state = CRITICAL
            action = EXIT
            reason = (
                f"⏰ Timeout ({age_sec / 60:.0f}min ≥ {max_age / 60:.0f}min) │ "
                f"Score={total_score} │ P&L=${profit:+.2f}"
            )
        elif total_score < critical_threshold:
            state = CRITICAL
            action = EXIT
            reason = (
                f"🔴 Critical: {total_score}/100 (threshold={critical_threshold}) │ "
                + " + ".join(
                    f"{k}={v['score']}/{v['max']}"
                    for k, v in components.items()
                )
            )
        elif giveback["action"] == EXIT:
            state = CRITICAL
            action = EXIT
            reason = (
                f"Profit giveback exit: {giveback['reason']} | "
                f"Score={total_score}/100 | "
                + " + ".join(
                    f"{k}={v['score']}/{v['max']}"
                    for k, v in components.items()
                )
            )
        elif total_score < degrading_threshold:
            # ── Smart Profit Lock ─────────────────────────────────
            # If the trade was ever meaningfully in profit (>= 0.5 ATR) but is now
            # degrading, EXIT immediately to lock in gains instead of
            # trying to tighten SL (which often fails on chop).
            min_peak_usd = (entry_ctx.atr * 0.1) * 0.5  # Approx half ATR in USD for 0.1 lots
            if peak > min_peak_usd and giveback["giveback_pct"] >= 0.50:  
                state = CRITICAL
                action = EXIT
                reason = (
                    f"💰 Profit lock: peak=${peak:+.2f} → now=${profit:+.2f} │ "
                    f"Score={total_score}/100 │ "
                    + " + ".join(
                        f"{k}={v['score']}/{v['max']}"
                        for k, v in components.items()
                    )
                )
            else:
                state = DEGRADING
                action = TIGHTEN_SL
                reason = (
                    f"📉 Degrading: {total_score}/100 │ "
                    + " + ".join(
                        f"{k}={v['score']}/{v['max']}"
                        for k, v in components.items()
                    )
                )
        elif giveback["action"] == TIGHTEN_SL:
            state = DEGRADING
            action = TIGHTEN_SL
            reason = (
                f"Profit giveback tighten: {giveback['reason']} | "
                f"Score={total_score}/100 | Thesis={thesis_score}/25"
            )
        else:
            state = HEALTHY
            action = HOLD
            reason = (
                f"🟢 Healthy: {total_score}/100 │ P&L=${profit:+.2f} │ "
                f"Thesis={thesis_score}/25"
            )

        # ── Log state changes ────────────────────────────────────
        prev_state = self._last_states.get(ticket, HEALTHY)
        if state != prev_state:
            self._last_states[ticket] = state
            logger.info(
                f"[{symbol}] Health: {prev_state} → {state} │ "
                f"Score={total_score} │ Age={age_sec / 60:.1f}m │ "
                f"P&L=${profit:+.2f} │ {reason}"
            )

        result = TradeHealth(
            score=total_score,
            state=state,
            action=action,
            reason=reason,
            components=components,
            diagnostics=diagnostics,
        )

        # ── Save data for future ML training ─────────────────────
        self._log_to_csv(
            ticket, symbol, direction, age_sec, profit,
            current_spread, max_spread, recent_candles, result,
            current_regime, thesis_score, diagnostics,
        )

        return result

    def _log_to_csv(
        self, ticket: int, symbol: str, direction: str, age_sec: float,
        profit: float, current_spread: float, max_spread: float,
        recent_candles: Optional[str], result: TradeHealth,
        current_regime: str = "NORMAL", thesis_score: int = 0,
        diagnostics: Optional[dict] = None,
    ) -> None:
        """Saves current state and Health Score to a CSV for future AI training."""
        csv_file = os.path.join(settings.LOG_DIR, "ml_training_data.csv")
        file_exists = os.path.exists(csv_file)

        # v2: recent_candles is now a pressure_direction string ("BUY"/"SELL"/"NEUTRAL")
        pressure_dir = recent_candles if isinstance(recent_candles, str) else "UNKNOWN"
        diagnostics = diagnostics or {}
        excursion = diagnostics.get("excursion", {})
        giveback = diagnostics.get("giveback", {})

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "trade_age_sec": int(age_sec),
            "profit": f"{profit:.2f}",
            "current_spread": f"{current_spread:.2f}",
            "spread_ratio": f"{current_spread / max_spread:.2f}" if max_spread > 0 else "0.0",
            "pressure_direction": pressure_dir,
            "current_regime": current_regime,
            "thesis_score": thesis_score,
            "peak_profit": f"{excursion.get('peak_profit', 0.0):.2f}",
            "peak_favorable_price": f"{excursion.get('peak_favorable_price', 0.0):.2f}",
            "mfe_atr": f"{excursion.get('mfe_atr', 0.0):.2f}",
            "giveback_pct": f"{giveback.get('giveback_pct', 0.0):.2f}",
            "giveback_action": giveback.get("action", HOLD),
            "giveback_reason": giveback.get("reason", ""),
            "health_score": result.score,
            "health_state": result.state,
        }

        try:
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.debug(f"Error writing to ML CSV: {e}")

    # ══════════════════════════════════════════════════════════════
    # Scoring Components
    # ══════════════════════════════════════════════════════════════

    def get_trade_diagnostics(self, ticket: int) -> dict:
        """Returns current MFE/giveback diagnostics for another management module."""
        state = self._excursions.get(ticket, _ExcursionState())
        return {
            "peak_profit": state.peak_profit,
            "max_favorable_distance": state.max_favorable_distance,
            "peak_favorable_price": state.peak_favorable_price,
        }

    def _update_excursion(
        self,
        ticket: int,
        direction: str,
        profit: float,
        entry_price: float,
        current_price: float,
        entry_atr: float,
    ) -> dict:
        """Updates peak profit, peak favorable price, and ATR-normalized MFE."""
        state = self._excursions.setdefault(ticket, _ExcursionState())
        if state.peak_favorable_price <= 0:
            state.peak_favorable_price = entry_price

        if direction == "BUY":
            favorable_distance = current_price - entry_price
        else:
            favorable_distance = entry_price - current_price

        if profit > state.peak_profit:
            state.peak_profit = profit
            self._peak_profits[ticket] = profit

        if favorable_distance > state.max_favorable_distance:
            state.max_favorable_distance = favorable_distance
            state.peak_favorable_price = current_price

        giveback_usd = max(0.0, state.peak_profit - profit)
        giveback_pct = giveback_usd / state.peak_profit if state.peak_profit > 0 else 0.0
        mfe_atr = (
            state.max_favorable_distance / entry_atr
            if entry_atr > 0 and state.max_favorable_distance > 0
            else 0.0
        )

        return {
            "peak_profit": state.peak_profit,
            "peak_favorable_price": state.peak_favorable_price,
            "max_favorable_distance": state.max_favorable_distance,
            "current_favorable_distance": favorable_distance,
            "mfe_atr": mfe_atr,
            "giveback_usd": giveback_usd,
            "giveback_pct": giveback_pct,
        }

    def _analyze_profit_giveback(
        self,
        excursion: dict,
        profit: float,
        entry_ctx: _EntryContext,
        current_regime: str,
        current_speed: int,
        pressure_dir: Optional[str],
        thesis_score: int,
        delta_score: int,
        spread_score: int,
    ) -> dict:
        """Conservative overlay for trades that worked first, then stopped working."""
        regime = entry_ctx.regime
        mfe_atr = excursion.get("mfe_atr", 0.0)
        peak_profit = excursion.get("peak_profit", 0.0)
        giveback_pct = excursion.get("giveback_pct", 0.0)
        giveback_usd = excursion.get("giveback_usd", 0.0)

        min_mfe_atr = {
            "SLOW_TREND": 0.85,
            "NORMAL": 1.00,
            "FAST": 1.25,
            "EXPLOSIVE": 1.50,
        }.get(regime, 1.00)
        tighten_pct = {
            "SLOW_TREND": 0.45,
            "NORMAL": 0.50,
            "FAST": 0.60,
            "EXPLOSIVE": 0.70,
        }.get(regime, 0.50)
        exit_pct = {
            "SLOW_TREND": 0.75,
            "NORMAL": 0.80,
            "FAST": 0.85,
            "EXPLOSIVE": 0.90,
        }.get(regime, 0.80)

        pressure_opposing = (
            isinstance(pressure_dir, str)
            and pressure_dir not in (entry_ctx.direction, "NEUTRAL", "UNKNOWN")
        )
        speed_ratio = current_speed / entry_ctx.speed if entry_ctx.speed > 0 else 1.0
        regime_drop = _REGIME_RANK.get(current_regime, 2) < _REGIME_RANK.get(regime, 2)
        speed_decay = speed_ratio < 0.65
        speed_collapse = speed_ratio < 0.45
        thesis_weak = thesis_score <= 15
        liquidity_bad = spread_score <= 5
        context_flags = []

        if pressure_opposing:
            context_flags.append(f"pressure_opposing={pressure_dir}")
        if speed_decay:
            context_flags.append(f"speed_decay={entry_ctx.speed}->{current_speed}")
        if regime_drop:
            context_flags.append(f"regime_drop={regime}->{current_regime}")
        if thesis_weak:
            context_flags.append(f"thesis={thesis_score}/25")
        if liquidity_bad:
            context_flags.append(f"spread={spread_score}/15")

        meaningful_mfe = mfe_atr >= min_mfe_atr or peak_profit >= 0.25
        reason = (
            f"MFE={mfe_atr:.2f}xATR peak=${peak_profit:+.2f} "
            f"now=${profit:+.2f} giveback={giveback_pct:.0%}"
        )

        action = HOLD
        if meaningful_mfe and giveback_pct >= exit_pct:
            severe_context = (
                (pressure_opposing and (speed_collapse or thesis_weak))
                or (regime_drop and thesis_weak)
                or (delta_score == 0 and speed_collapse)
                or (liquidity_bad and pressure_opposing)
            )
            if severe_context:
                action = EXIT
        elif meaningful_mfe and giveback_pct >= tighten_pct:
            weak_context = pressure_opposing or speed_decay or regime_drop or thesis_weak
            if weak_context:
                action = TIGHTEN_SL

        if context_flags:
            reason = f"{reason} | " + ", ".join(context_flags)

        return {
            "action": action,
            "reason": reason,
            "giveback_pct": giveback_pct,
            "giveback_usd": giveback_usd,
            "meaningful_mfe": meaningful_mfe,
            "min_mfe_atr": min_mfe_atr,
            "tighten_pct": tighten_pct,
            "exit_pct": exit_pct,
            "speed_ratio": speed_ratio,
        }

    def _score_time_decay(
        self, age_sec: float, entry_regime: str
    ) -> tuple:
        """
        20 pts: Progressive time decay with regime-aware grace.

        Grace periods:
          FAST/EXPLOSIVE: 180s (3 min)
          NORMAL:         120s (2 min)
          SLOW_TREND:      90s (1.5 min)
          DEAD:            60s (1 min)

        After grace, linear decay to max_age.
        """
        grace = _REGIME_GRACE.get(entry_regime, 120)
        max_age = settings.TRADE_MAX_AGE_SEC

        if age_sec <= grace:
            return 20, f"{age_sec / 60:.1f}m (grace/{grace}s)"

        # Linear decay from grace to max_age
        decay_window = max_age - grace
        if decay_window <= 0:
            return 0, f"{age_sec / 60:.1f}m (timeout)"

        elapsed_after_grace = age_sec - grace
        decay_ratio = min(1.0, elapsed_after_grace / decay_window)
        score = int(20 * (1.0 - decay_ratio))

        return score, f"{age_sec / 60:.1f}m ({score}/20)"

    def _score_pnl_trajectory(
        self, current_pnl: float, peak_profit: float,
        current_price: float, entry_price: float,
        direction: str, entry_atr: float,
    ) -> tuple:
        """
        20 pts: ATR-normalized P&L assessment.

        Instead of binary "profitable vs losing", uses ATR distance
        to determine if the drawdown is normal or concerning.

        - In profit:                    20/20
        - Near breakeven (< 0.3×ATR):   15/20
        - Loss within 1×ATR:            12/20 (normal pullback territory)
        - Loss beyond 1×ATR:             5/20 (extended, concerning)
        - Loss beyond SL distance:        0/20 (should be stopped out)
        """
        # Calculate price distance from entry
        if direction == "BUY":
            price_distance = current_price - entry_price
        else:
            price_distance = entry_price - current_price

        # ATR-normalize the distance
        if entry_atr > 0:
            atr_distance = abs(price_distance) / entry_atr
        else:
            atr_distance = 0.0

        if current_pnl > 0.10:  # Clearly profitable
            score = 20
            reason = f"P&L=${current_pnl:+.2f}✓"
        elif current_pnl > -0.10:  # Near breakeven
            score = 15
            reason = f"P&L=${current_pnl:+.2f}~ ({atr_distance:.2f}×ATR)"
        elif price_distance < 0 and atr_distance < 1.0:
            # Losing but within 1×ATR — normal pullback territory
            score = 12
            reason = f"P&L=${current_pnl:+.2f} ({atr_distance:.2f}×ATR pullback)"
        elif price_distance < 0 and atr_distance < 1.5:
            # Extended pullback, concerning
            score = 5
            reason = f"P&L=${current_pnl:+.2f} ({atr_distance:.2f}×ATR extended)"
        else:
            score = 0
            reason = f"P&L=${current_pnl:+.2f}✗ ({atr_distance:.2f}×ATR deep)"

        # Penalty if retreated significantly from peak
        if peak_profit > 0.20 and current_pnl < peak_profit * 0.3:
            # Lost more than 70% of peak profit
            score = max(0, score - 5)
            reason += f" (retreat from ${peak_profit:+.2f})"

        return score, reason

    def _score_delta_alignment(
        self, direction: str, recent_candles: Optional[str]
    ) -> tuple:
        """
        20 pts: Does the latest pressure direction agree with the trade?

        v2: Accepts a pressure_direction string ("BUY"/"SELL"/"NEUTRAL")
        from PseudoDelta instead of mock Candle objects.
        """
        if not recent_candles or not isinstance(recent_candles, str):
            return 10, "No data (neutral)"

        pressure_dir = recent_candles

        if pressure_dir == direction:
            return 20, f"Pressure aligned ({pressure_dir})"
        elif pressure_dir == "NEUTRAL":
            return 10, "Pressure neutral"
        else:
            return 0, f"Pressure opposing ({pressure_dir})"

    def _score_spread_health(
        self, current_spread: float, max_spread: float
    ) -> tuple:
        """
        15 pts: Has the spread widened dangerously since entry?
        - Spread within normal range: 15/15
        - Spread elevated (> 80% of max): 10/15
        - Spread at/above max: 0/15 (broker conditions deteriorating)
        """
        if max_spread <= 0:
            return 15, "No spread data"

        spread_ratio = current_spread / max_spread

        if spread_ratio <= 0.6:
            return 15, f"Spread OK ({current_spread:.2f})"
        elif spread_ratio <= 0.8:
            return 10, f"Spread normal ({current_spread:.2f})"
        elif spread_ratio < 1.0:
            return 5, f"Spread elevated ({current_spread:.2f})"
        else:
            return 0, f"Spread critical ({current_spread:.2f})"

    def _score_thesis_validation(
        self,
        direction: str,
        current_price: float,
        current_regime: str,
        current_speed: int,
        entry_ctx: _EntryContext,
        pressure_dir: Optional[str] = None,
        current_atr: float = 0.0,
        age_sec: float = 0.0,
    ) -> tuple:
        """
        25 pts: Is the original trade thesis still valid?

        Three sub-components:
          A) Regime Persistence (10 pts):
             Is the market still active enough to support the trade?
          B) Structural Integrity (15 pts):
             Has the key invalidation level been broken?

        Two penalty modifiers (applied after base score):
          C) Momentum Decay Penalty (0 to -8 pts):
             Detects slow death of directional intent — the trade
             isn't "wrong" but it's no longer "working."
          D) Intent Collapse Penalty (0 to -7 pts):
             Detects aggressive reversal conditions even before
             structural failure.

        CRITICAL EDGE CASES:
          - FAST regime does NOT grant immunity.
          - Structure break overrides regime protection.
          - Delta flip in fast market accelerates degradation.
        """
        regime_score = 0
        struct_score = 0
        reasons = []

        # ── A) Regime Persistence (10 pts) ───────────────────────
        entry_rank = _REGIME_RANK.get(entry_ctx.regime, 2)
        current_rank = _REGIME_RANK.get(current_regime, 2)

        if current_rank >= entry_rank:
            # Regime maintained or improved → full points
            regime_score = 10
            reasons.append(f"regime={current_regime}✓")
        elif current_rank >= entry_rank - 1:
            # Dropped 1 level (e.g., EXPLOSIVE→FAST or FAST→NORMAL)
            # Still reasonable
            regime_score = 6
            reasons.append(f"regime={current_regime}~")
        elif current_rank >= 1:
            # Dropped significantly but not dead
            regime_score = 3
            reasons.append(f"regime={current_regime}↓")
        else:
            # Regime is DEAD — thesis is very weak
            regime_score = 0
            reasons.append("regime=DEAD✗")

        # ── B) Structural Integrity (15 pts) ─────────────────────
        # Check if the frozen reference level has been broken
        ref = entry_ctx.structural_ref
        if ref > 0:
            if direction == "BUY":
                # For BUY: swing low must NOT be broken
                if current_price > ref:
                    struct_score = 15
                    reasons.append(f"struct_intact(>{ref:.2f})")
                else:
                    struct_score = 0
                    reasons.append(f"struct_broken(<{ref:.2f})")
            else:
                # For SELL: swing high must NOT be broken
                if current_price < ref:
                    struct_score = 15
                    reasons.append(f"struct_intact(<{ref:.2f})")
                else:
                    struct_score = 0
                    reasons.append(f"struct_broken(>{ref:.2f})")
        else:
            # No structural reference provided — assume intact
            struct_score = 15
            reasons.append("struct_unknown")

        base_score = regime_score + struct_score

        # ── Penalty C: Momentum Decay ────────────────────────────
        # Trade is taking too long without moving
        speed_ratio = current_speed / entry_ctx.speed if entry_ctx.speed > 0 else 1.0
        penalty_c = 0
        if age_sec > 300 and speed_ratio < 0.5:
            penalty_c = -8
            reasons.append("momentum_decay")
        elif age_sec > 180 and speed_ratio < 0.7:
            penalty_c = -4
            reasons.append("momentum_slow")

        # ── Penalty D: Intent Collapse ───────────────────────────
        # Opposing pressure in a dropping regime = dangerous reversal
        penalty_d = 0
        opposing = (
            isinstance(pressure_dir, str)
            and pressure_dir not in (direction, "NEUTRAL", "UNKNOWN")
        )
        if opposing and current_rank < entry_rank:
            penalty_d = -7
            reasons.append(f"intent_collapse({pressure_dir})")
        elif opposing:
            penalty_d = -3
            reasons.append(f"opposing_pressure({pressure_dir})")

        final_score = max(0, base_score + penalty_c + penalty_d)
        reason_str = " | ".join(reasons)

        # Truncate reason if too long to keep logs clean
        if len(reason_str) > 80:
            reason_str = reason_str[:77] + "..."

        return final_score, reason_str
