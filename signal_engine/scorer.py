"""
signal_engine/scorer.py — Transparent Signal Scoring Engine for MARK III v2.4

Orchestrates all analysis modules and produces final scored trading signals.
Every point in the score is traceable to a specific module and condition.

Scoring System (100 pts max, 6 factors):
  ┌────────────────────────┬────────┬─────────────────────────────┐
  │ Factor                 │ Points │ Source Module                │
  ├────────────────────────┼────────┼─────────────────────────────┤
  │ 1. Trend Context       │  0-20  │ MarketStructure (M15 EMA)   │
  │ 2. Structure Break     │  0-20  │ MarketStructure (BOS/sweep)  │
  │ 3. Rejection / Sweep   │  0-15  │ MarketStructure + Absorption │
  │ 4. Volume Expansion    │  0-15  │ MomentumSpeed               │
  │ 5. Pressure Alignment  │  0-15  │ PseudoDelta                 │
  │ 6. Cross-Market Confirm│  0-15  │ CorrelationAnalyzer          │
  └────────────────────────┴────────┴─────────────────────────────┘

Rules:
  - Minimum score to trade: MIN_SCORE_TO_TRADE (default 60)
  - Minimum factors: 2 in strong trend regime, 3 otherwise
  - Dead market filter: MomentumSpeed.is_dead → block all signals
  - Direction conflicts: adaptive ratio with trend tiebreaker

v2.1 Changes:
  - Speed contribution uses candle direction (not dominance-gated)
  - Conflict ratio raised to 0.75 with trend-based tiebreaker
  - Minimum factors: 2 in strong trend (strength >= 15), 3 otherwise
  - MomentumSpeed receives trend context for adaptive dead-market logic
  - Enhanced rejection logging with filter breakdown

v2.4 Changes (conservative signal tuning):
  - Direction conflict ratio: 0.75 → 0.85 (only block truly ambiguous)
  - Regime profiles lowered: NORMAL 60→50, SLOW_TREND 55→45
  - MarketStructure now provides "developing trend" (5 pts) for
    transitional markets where EMA slope + alignment show bias
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from monitoring.logger import get_logger
from signal_engine.market_structure import MarketStructure, StructureContext
from signal_engine.pseudo_delta import PseudoDelta, PressureResult
from signal_engine.absorption import AbsorptionDetector, AbsorptionResult
from signal_engine.momentum_speed import MomentumSpeed, SpeedContext
from signal_engine.correlation import CorrelationAnalyzer, CorrelationResult
from execution_engine.regime_profile import get_profile, RegimeProfile

logger = get_logger("SignalEngine")


# ─── Signal Data Structure ──────────────────────────────────────
@dataclass
class Signal:
    """Trading signal with full transparency on what triggered it."""
    symbol: str           # Futures key: "ES" or "NQ"
    mt5_symbol: str       # MT5 CFD name: "US500Cash" or "US100Cash"
    action: str           # "BUY", "SELL", or "NEUTRAL"
    score: int            # Total score (0-100)
    atr: float            # Current M5 ATR value
    sl_distance: float    # Stop loss distance in price points
    tp_distance: float    # Take profit distance in price points
    reason: str           # Human-readable breakdown
    timestamp: float      # Unix timestamp

    # Score breakdown (for debugging and ML logging)
    trend_score: int = 0
    structure_score: int = 0
    rejection_score: int = 0
    volume_score: int = 0
    pressure_score: int = 0
    correlation_score: int = 0
    factors_active: int = 0

    # v2.2: Entry context for Trade Health Monitor thesis validation
    regime: str = "NORMAL"          # Market regime at entry time
    structural_ref: float = 0.0     # Frozen swing level (low for BUY, high for SELL)
    speed_score: int = 0            # Raw speed score (0-100) at entry time

    # v2.3: Frozen regime profile for downstream execution modules
    profile: Optional[RegimeProfile] = None

    # v2.6: Setup classification (CONTINUATION vs REVERSAL)
    setup_type: str = "UNKNOWN"

    @property
    def is_valid(self) -> bool:
        min_score = self.profile.min_score if self.profile else settings.MIN_SCORE_TO_TRADE
        return (
            self.action != "NEUTRAL"
            and self.score >= min_score
            and self.factors_active >= 2  # v2.1: min 2 (scorer enforces more when needed)
        )


class SignalScorer:
    """
    Orchestrates all analysis modules and produces scored signals.
    Transparent, rule-based, no randomness.

    v2.1: Adaptive regime-aware scoring with trend context.
    """

    def __init__(self) -> None:
        self._structure = MarketStructure()
        self._pseudo_delta = PseudoDelta()
        self._absorption = AbsorptionDetector()
        self._speed = MomentumSpeed()
        self._correlation = CorrelationAnalyzer()

        # Cooldown tracker: {symbol: last_signal_timestamp}
        self._last_signal_time: Dict[str, float] = {}

        logger.info(
            f"SignalScorer initialized │ "
            f"Regime-aware={settings.REGIME_AWARE_EXECUTION} │ "
            f"Default min score={settings.MIN_SCORE_TO_TRADE} │ "
            f"Min factors=2(trend)/profile(neutral)"
        )

    # ─── Main Entry Point ───────────────────────────────────────
    def evaluate(
        self, market_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> List[Signal]:
        """
        Evaluates all symbols and returns actionable signals.

        Args:
            market_data: {
                "ES": {"M5": DataFrame, "M15": DataFrame},
                "NQ": {"M5": DataFrame, "M15": DataFrame},
            }

        Returns:
            List of Signal objects (only those that pass all filters).
        """
        if not market_data:
            return []

        es_data = market_data.get("ES")
        nq_data = market_data.get("NQ")

        # ── Run analysis for each symbol ────────────────────────
        es_ctx = self._analyze_symbol("ES", es_data) if es_data else None
        nq_ctx = self._analyze_symbol("NQ", nq_data) if nq_data else None

        # ── Cross-market correlation ────────────────────────────
        correlation = CorrelationResult()
        if es_data and nq_data and es_ctx and nq_ctx:
            correlation = self._correlation.analyze(
                es_m5=es_data.get("M5"),
                nq_m5=nq_data.get("M5"),
                es_structure=es_ctx.get("structure"),
                nq_structure=nq_ctx.get("structure"),
            )

        # ── Score each symbol ───────────────────────────────────
        signals = []

        for sym_key, ctx in [("ES", es_ctx), ("NQ", nq_ctx)]:
            if ctx is None:
                continue

            signal = self._score_symbol(sym_key, ctx, correlation)

            # Log the evaluation regardless of validity
            self._log_evaluation(sym_key, signal, ctx)

            if signal.is_valid:
                if self._is_on_cooldown(sym_key):
                    continue
                self._mark_signal_fired(sym_key)
                signals.append(signal)

                logger.log(25,  # TRADE level
                    f"🎯 SIGNAL │ {sym_key}→{signal.mt5_symbol} │ "
                    f"{signal.action} │ Score={signal.score}/100 │ "
                    f"Factors={signal.factors_active}/6 │ "
                    f"SL={signal.sl_distance:.2f} │ TP={signal.tp_distance:.2f} │ "
                    f"{signal.reason}"
                )

        return signals

    # ─── Per-Symbol Analysis ────────────────────────────────────
    def _analyze_symbol(
        self, sym_key: str, sym_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Runs all analysis modules on one symbol.
        Returns dict with all analysis results.
        """
        m5 = sym_data.get("M5")
        m15 = sym_data.get("M15")

        if m5 is None or m15 is None:
            return None

        if len(m5) < 20 or len(m15) < 10:
            return None

        try:
            structure = self._structure.analyze(m5, m15)
            pressure = self._pseudo_delta.calculate(m5)
            absorption = self._absorption.detect(m5)

            # v2.1: Pass trend context to MomentumSpeed for adaptive
            # dead-market detection (slow trends != dead markets)
            speed = self._speed.measure(
                m5,
                trend=structure.trend,
                trend_strength=structure.trend_strength,
            )

            return {
                "m5": m5,
                "m15": m15,
                "structure": structure,
                "pressure": pressure,
                "absorption": absorption,
                "speed": speed,
            }
        except Exception as exc:
            logger.error(f"[{sym_key}] Analysis error: {exc}", exc_info=True)
            return None

    # ─── Scoring Logic ──────────────────────────────────────────
    def _score_symbol(
        self, sym_key: str, ctx: Dict, correlation: CorrelationResult
    ) -> Signal:
        """
        Scores a symbol based on all analysis results.
        Returns a Signal with full breakdown.
        """
        sym_cfg = settings.SYMBOL_MAP[sym_key]
        m5 = ctx["m5"]
        structure: StructureContext = ctx["structure"]
        pressure: PressureResult = ctx["pressure"]
        absorption: AbsorptionResult = ctx["absorption"]
        speed: SpeedContext = ctx["speed"]

        # v2.3: Regime-aware execution profile (frozen at signal time)
        profile = get_profile(speed.regime)

        # Get ATR from last M5 candle
        last_m5 = m5.iloc[-1]
        atr = float(last_m5.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            return self._neutral_signal(sym_key, sym_cfg, atr, "No ATR data")

        # ── Dead market filter (v2.1: adaptive) ─────────────────
        if speed.is_dead:
            return self._neutral_signal(
                sym_key, sym_cfg, atr,
                f"Market too quiet (regime={speed.regime}, speed={speed.speed_score})"
            )

        # ══════════════════════════════════════════════════════════
        # COLLECT DIRECTIONAL VOTES FROM EACH MODULE
        # ══════════════════════════════════════════════════════════
        buy_score = 0
        sell_score = 0
        buy_factors = 0
        sell_factors = 0
        buy_reasons = []
        sell_reasons = []

        # ── 1. Trend Context (0-20 pts) ─────────────────────────
        if structure.trend == "BULLISH":
            buy_score += structure.trend_strength
            buy_factors += 1
            buy_reasons.append(f"Trend={structure.trend_strength}")
        elif structure.trend == "BEARISH":
            sell_score += structure.trend_strength
            sell_factors += 1
            sell_reasons.append(f"Trend={structure.trend_strength}")

        # ── 2. Structure Break (0-20 pts) ───────────────────────
        if structure.bos_detected:
            if structure.bos_direction == "BULLISH":
                buy_score += structure.bos_strength
                buy_factors += 1
                buy_reasons.append(f"BOS={structure.bos_strength}")
            elif structure.bos_direction == "BEARISH":
                sell_score += structure.bos_strength
                sell_factors += 1
                sell_reasons.append(f"BOS={structure.bos_strength}")

        # ── 3. Rejection / Sweep / Absorption (0-15 pts) ────────
        # Take the best of rejection, sweep, or absorption
        best_reversal_score = 0
        best_reversal_dir = "NEUTRAL"
        best_reversal_label = ""

        if structure.sweep_detected:
            best_reversal_score = structure.sweep_strength
            best_reversal_dir = structure.sweep_direction
            best_reversal_label = "Sweep"

        if structure.rejection_detected and structure.rejection_strength > best_reversal_score:
            best_reversal_score = structure.rejection_strength
            best_reversal_dir = structure.rejection_direction
            best_reversal_label = "Reject"

        if absorption.detected and absorption.strength > best_reversal_score:
            best_reversal_score = absorption.strength
            best_reversal_dir = absorption.reversal_direction
            best_reversal_label = "Absorb"

        if best_reversal_score > 0:
            if best_reversal_dir == "BUY":
                buy_score += best_reversal_score
                buy_factors += 1
                buy_reasons.append(f"{best_reversal_label}={best_reversal_score}")
            elif best_reversal_dir == "SELL":
                sell_score += best_reversal_score
                sell_factors += 1
                sell_reasons.append(f"{best_reversal_label}={best_reversal_score}")

        # ── 4. Volume Expansion (0-15 pts) ──────────────────────
        # v2.1: Use current candle direction instead of requiring
        # one side to already be dominant. This prevents speed pts
        # from being thrown away when scores are tied.
        if speed.contribution > 0:
            last_body = float(last_m5.get("body", 0))
            if last_body > 0:
                buy_score += speed.contribution
                buy_factors += 1
                buy_reasons.append(f"Speed={speed.contribution}")
            elif last_body < 0:
                sell_score += speed.contribution
                sell_factors += 1
                sell_reasons.append(f"Speed={speed.contribution}")
            else:
                # Doji: add to dominant side if one exists
                if buy_score > sell_score:
                    buy_score += speed.contribution
                    buy_factors += 1
                    buy_reasons.append(f"Speed={speed.contribution}")
                elif sell_score > buy_score:
                    sell_score += speed.contribution
                    sell_factors += 1
                    sell_reasons.append(f"Speed={speed.contribution}")

        # ── 5. Pressure Alignment (0-15 pts) ────────────────────
        if pressure.alignment_score > 0:
            if pressure.pressure_direction == "BUY":
                buy_score += pressure.alignment_score
                buy_factors += 1
                buy_reasons.append(f"Pressure={pressure.alignment_score}")
            elif pressure.pressure_direction == "SELL":
                sell_score += pressure.alignment_score
                sell_factors += 1
                sell_reasons.append(f"Pressure={pressure.alignment_score}")

        # Pressure divergence can add to reversal side
        if pressure.divergence_detected:
            if pressure.divergence_direction == "BUY":
                buy_score += pressure.divergence_strength
                buy_factors += 1
                buy_reasons.append(f"PressDiv={pressure.divergence_strength}")
            elif pressure.divergence_direction == "SELL":
                sell_score += pressure.divergence_strength
                sell_factors += 1
                sell_reasons.append(f"PressDiv={pressure.divergence_strength}")

        # ── 6. Cross-Market Confirmation (0-15 pts) ─────────────
        if correlation.divergence_detected:
            if correlation.divergence_direction == "BUY":
                buy_score += correlation.strength
                buy_factors += 1
                buy_reasons.append(f"SMT={correlation.strength}")
            elif correlation.divergence_direction == "SELL":
                sell_score += correlation.strength
                sell_factors += 1
                sell_reasons.append(f"SMT={correlation.strength}")
        elif correlation.confirmed:
            if correlation.confirmation_direction == "BULLISH":
                buy_score += correlation.strength
                buy_factors += 1
                buy_reasons.append(f"Confirm={correlation.strength}")
            elif correlation.confirmation_direction == "BEARISH":
                sell_score += correlation.strength
                sell_factors += 1
                sell_reasons.append(f"Confirm={correlation.strength}")

        # ══════════════════════════════════════════════════════════
        # FINAL DECISION (v2.1: adaptive conflict + factor logic)
        # ══════════════════════════════════════════════════════════

        # v2.1: Trend-aware conflict resolution
        # In a confirmed trend, reduce the opposing side's score by 50%
        # before calculating the conflict ratio. This prevents normal
        # pullback patterns (rejection wicks, minor absorption) from
        # triggering false "direction conflict" blocks.
        adj_buy = buy_score
        adj_sell = sell_score

        trend_confirmed = (
            structure.trend in ("BULLISH", "BEARISH")
            and structure.trend_strength >= 15
        )

        if trend_confirmed and buy_score > 0 and sell_score > 0:
            if structure.trend == "BULLISH":
                adj_sell = int(sell_score * 0.5)
            else:
                adj_buy = int(buy_score * 0.5)

        # Check for direction conflict (v2.4: ratio raised to 0.85)
        # v2.1→v2.4: 0.75 was too aggressive — blocked legitimate setups
        # where mild opposing signals existed (e.g., B:19 S:15 = 0.79).
        # 0.85 still blocks truly ambiguous signals while allowing
        # setups with a clear directional lean.
        if adj_buy > 0 and adj_sell > 0:
            ratio = min(adj_buy, adj_sell) / max(adj_buy, adj_sell)
            if ratio > 0.85:
                return self._neutral_signal(
                    sym_key, sym_cfg, atr,
                    f"Direction conflict (B:{buy_score} S:{sell_score} "
                    f"adj_B:{adj_buy} adj_S:{adj_sell} ratio:{ratio:.2f})"
                )

        # Determine winner (using original scores for final score)
        # v2.3: Regime-aware entry threshold
        if buy_score >= sell_score and buy_score >= profile.min_score:
            action = "BUY"
            score = min(100, buy_score)
            factors = buy_factors
            reason = " + ".join(buy_reasons)
        elif sell_score > buy_score and sell_score >= profile.min_score:
            action = "SELL"
            score = min(100, sell_score)
            factors = sell_factors
            reason = " + ".join(sell_reasons)
        else:
            extra = ""
            if profile.regime == "EXPLOSIVE":
                extra = (
                    f" │ EXPLOSIVE_BLOCK "
                    f"buy_factors={buy_factors} sell_factors={sell_factors} "
                    f"buy_parts={'+'.join(buy_reasons) or 'none'} "
                    f"sell_parts={'+'.join(sell_reasons) or 'none'} "
                    f"speed={speed.speed_score} range={speed.range_expansion:.2f}x "
                    f"vol={speed.volume_burst:.2f}x atr_state={speed.atr_trend}"
                )
            return self._neutral_signal(
                sym_key, sym_cfg, atr,
                f"Score below min (B:{buy_score} S:{sell_score} "
                f"min:{profile.min_score} [{profile.regime}]){extra}"
            )

        # v2.3: Adaptive minimum factors (regime-aware base)
        # - Strong confirmed trend (M15 strength >= 15): 2 factors OK
        # - Otherwise: use profile's min_factors
        min_factors = 2 if trend_confirmed else profile.min_factors

        if factors < min_factors:
            return self._neutral_signal(
                sym_key, sym_cfg, atr,
                f"Only {factors}/{min_factors} factors "
                f"(regime={profile.regime}, Score={score})"
            )

        # ── Setup Classification (v2.7) ─────────────────────────────
        # REVERSAL is deprecated. We only trade CONTINUATION and PULLBACK.
        # Pullbacks are sweeps/rejections that align with the M15 Trend.
        is_pullback = (action == best_reversal_dir and best_reversal_score >= 10)
        setup_type = "PULLBACK" if is_pullback else "CONTINUATION"

        # ── v2.6/v2.7: Context-Aware M5 EMA21 alignment gate ────────
        # Continuation trades require strict EMA alignment (breakouts).
        # Pullback trades do NOT require M5 EMA alignment, because by definition 
        # a pullback dips below the EMA temporarily at a discount. Handled safely 
        # by the new M15 Trend enforcement in _detect_sweep.
        last_m5 = m5.iloc[-1]
        m5_ema21 = last_m5.get("ema_21", None)
        
        requires_alignment = (setup_type == "CONTINUATION")

        if m5_ema21 is not None and pd.notna(m5_ema21):
            if requires_alignment:
                fail_msg = f"M5 misaligned ({setup_type}): close={float(last_m5['close']):.2f} vs EMA21={float(m5_ema21):.2f} for {action}"
                
                if action == "BUY" and float(last_m5["close"]) < float(m5_ema21):
                    return self._neutral_signal(sym_key, sym_cfg, atr, fail_msg, setup_type=setup_type)
                if action == "SELL" and float(last_m5["close"]) > float(m5_ema21):
                    return self._neutral_signal(sym_key, sym_cfg, atr, fail_msg, setup_type=setup_type)

        # ── Build signal ────────────────────────────────────────
        # v2.2: Freeze structural reference at entry time
        # BUY → store latest M5 swing low (invalidation level)
        # SELL → store latest M5 swing high (invalidation level)
        if action == "BUY":
            struct_ref = structure.last_swing_low
        else:
            struct_ref = structure.last_swing_high

        # v2.5: PULLBACK trades enter against short-term momentum —
        # they need wider SL to survive the initial counter-move.
        # Audit: 55 SL hits at 12.7% WR (-$34.43). Extra 0.3×ATR
        # breathing room for PULLBACK only.
        sl_mult = profile.sl_atr_mult
        if setup_type == "PULLBACK":
            sl_mult += 0.3  # e.g. NORMAL 1.5 → 1.8 for pullbacks

        return Signal(
            symbol=sym_key,
            mt5_symbol=sym_cfg["mt5_name"],
            action=action,
            score=score,
            atr=atr,
            sl_distance=atr * sl_mult,                   # v2.5: setup-aware
            tp_distance=atr * profile.tp_atr_mult,       # v2.3: regime-aware
            reason=reason,
            timestamp=time.time(),
            trend_score=structure.trend_strength if structure.trend != "NEUTRAL" else 0,
            structure_score=structure.bos_strength if structure.bos_detected else 0,
            rejection_score=best_reversal_score,
            volume_score=speed.contribution,
            pressure_score=pressure.alignment_score,
            correlation_score=correlation.strength,
            factors_active=factors,
            regime=speed.regime,
            structural_ref=struct_ref,
            speed_score=speed.speed_score,
            profile=profile,                            # v2.3: frozen profile
            setup_type=setup_type,
        )

    # ─── Helpers ────────────────────────────────────────────────
    @staticmethod
    def _neutral_signal(sym_key: str, sym_cfg: dict, atr: float, reason: str, setup_type: str = "UNKNOWN") -> Signal:
        return Signal(
            symbol=sym_key,
            mt5_symbol=sym_cfg["mt5_name"],
            action="NEUTRAL",
            score=0,
            atr=atr if not pd.isna(atr) else 0,
            sl_distance=0,
            tp_distance=0,
            reason=reason,
            timestamp=time.time(),
            setup_type=setup_type,
        )

    def _is_on_cooldown(self, symbol: str) -> bool:
        last = self._last_signal_time.get(symbol, 0)
        elapsed = time.time() - last
        if elapsed < settings.SIGNAL_COOLDOWN_SEC:
            remaining = settings.SIGNAL_COOLDOWN_SEC - elapsed
            logger.debug(f"[{symbol}] On cooldown, {remaining:.0f}s remaining")
            return True
        return False

    def _mark_signal_fired(self, symbol: str) -> None:
        self._last_signal_time[symbol] = time.time()

    def _log_evaluation(self, sym_key: str, signal: Signal, ctx: Dict) -> None:
        """
        v2.1: Enhanced logging with regime and filter breakdown.
        Logs every evaluation (even NEUTRAL) with full transparency
        on why a trade was taken or rejected.
        """
        structure: StructureContext = ctx["structure"]
        speed: SpeedContext = ctx["speed"]
        
        last_m5 = ctx["m5"].iloc[-1]
        m5_ema21 = last_m5.get("ema_21", None)
        close = last_m5.get("close", None)
        ema_str = f"C={close:.2f}|E21={m5_ema21:.2f}" if m5_ema21 is not None and pd.notna(m5_ema21) else "EMA=N/A"

        if signal.action == "NEUTRAL":
            trend_diag = structure.trend_diagnostics or {}
            trend_details = (
                f"TrendDiag close={trend_diag.get('close')} "
                f"ema21={trend_diag.get('ema21')} "
                f"ema50={trend_diag.get('ema50')} "
                f"ema21_slope={trend_diag.get('ema21_slope')} "
                f"ema50_slope={trend_diag.get('ema50_slope')} "
                f"ema50_valid={trend_diag.get('ema50_valid')} "
                f"atr={trend_diag.get('atr')} "
                f"proxATR={trend_diag.get('proximity_atr')} "
                f"developing={trend_diag.get('developing')} "
                f"reason={structure.trend_reason}"
            )
            logger.info(
                f"[{sym_key}] Score=  0 (NEUTRAL) │ "
                f"{structure.summary()} │ {speed.summary()} │ "
                f"Regime={speed.regime} │ {ema_str} │ "
                f"Type={signal.setup_type} │ "
                f"{signal.reason} │ {trend_details}"
            )
        else:
            profile_tag = f"[{signal.profile.regime}]" if signal.profile else ""
            trend_diag = structure.trend_diagnostics or {}
            trend_details = (
                f"TrendDiag close={trend_diag.get('close')} "
                f"ema21={trend_diag.get('ema21')} "
                f"ema50={trend_diag.get('ema50')} "
                f"ema21_slope={trend_diag.get('ema21_slope')} "
                f"ema50_slope={trend_diag.get('ema50_slope')} "
                f"ema50_valid={trend_diag.get('ema50_valid')} "
                f"atr={trend_diag.get('atr')} "
                f"proxATR={trend_diag.get('proximity_atr')} "
                f"developing={trend_diag.get('developing')} "
                f"reason={structure.trend_reason}"
            )
            logger.info(
                f"[{sym_key}] Score={signal.score:3d} ({signal.action:7s}) │ "
                f"Type={signal.setup_type} │ Factors={signal.factors_active}/6 │ "
                f"T={signal.trend_score} S={signal.structure_score} "
                f"R={signal.rejection_score} V={signal.volume_score} "
                f"P={signal.pressure_score} C={signal.correlation_score} │ "
                f"Regime={speed.regime} {profile_tag} │ {ema_str} │ "
                f"SL={signal.sl_distance:.2f} TP={signal.tp_distance:.2f} │ "
                f"ATR={signal.atr:.2f} │ {trend_details}"
            )
