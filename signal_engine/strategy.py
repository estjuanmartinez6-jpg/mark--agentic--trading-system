"""
signal_engine/strategy.py — Signal Generation Engine for MARK III

Implements a multi-factor scoring system:
  1. Delta Strength  (35 pts) — Current delta vs rolling average
  2. Divergence      (35 pts) — Price direction vs delta direction mismatch
  3. Confirmation    (30 pts) — ES and NQ agreeing on direction

Trade fires when total score >= MIN_SCORE_TO_TRADE (default: 65).
This means at least 2 of 3 factors must align — no single-factor trades.

Includes:
  - ATR-based volatility filter (blocks trades in flat markets)
  - Cooldown tracking per symbol (prevents signal spam)
  - Detailed score breakdown logging
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import settings
from data_handler.order_flow import Candle
from monitoring.logger import get_logger

logger = get_logger("SignalEngine")


# ─── Signal Data Structure ──────────────────────────────────────
@dataclass
class Signal:
    """Represents a trading signal with full context."""
    symbol: str           # Futures symbol (ES, NQ)
    mt5_symbol: str       # MT5 CFD symbol (US500Cash, US100Cash)
    action: str           # BUY, SELL, or NEUTRAL
    score: int            # Total combined score (0-100)
    atr: float            # Current ATR value
    sl_distance: float    # Stop loss distance in price points
    tp_distance: float    # Take profit distance in price points
    # Score breakdown
    delta_score: int
    delta_direction: str
    divergence_score: int
    divergence_direction: str
    confirmation_score: int
    reason: str           # Human-readable explanation
    timestamp: float

    @property
    def is_valid(self) -> bool:
        return self.action != "NEUTRAL" and self.score >= settings.MIN_SCORE_TO_TRADE


class SignalEngine:
    """
    Evaluates order flow data and generates scored trading signals.
    """

    def __init__(self):
        # Cooldown tracker: {symbol: last_signal_timestamp}
        self._last_signal_time: Dict[str, float] = {}
        logger.info(
            f"SignalEngine initialized │ "
            f"Delta×{settings.DELTA_THRESHOLD_MULTIPLIER} │ "
            f"Div lookback={settings.DIVERGENCE_LOOKBACK} │ "
            f"Min score={settings.MIN_SCORE_TO_TRADE} │ "
            f"SL={settings.ATR_SL_MULTIPLIER}×ATR │ "
            f"TP={settings.ATR_TP_MULTIPLIER}×ATR"
        )

    # ─── ATR Calculation ────────────────────────────────────────
    def calculate_atr(self, candles: List[Candle], period: int) -> float:
        """Calculates Average True Range over the given period."""
        if len(candles) < period + 1:
            return 0.0

        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)

        return statistics.mean(trs[-period:])

    # ─── Factor 1: Delta Strength ───────────────────────────────
    def _analyze_delta(self, candles: List[Candle]) -> Tuple[int, str]:
        """
        Compares the latest candle's absolute delta against the rolling average.
        If current |delta| > DELTA_THRESHOLD_MULTIPLIER × avg|delta| → strong signal.

        Returns: (score, direction)
        """
        if len(candles) < 6:
            return 0, "NEUTRAL"

        # Use all candles except the latest for the average baseline
        historical_deltas = [abs(c.delta) for c in candles[:-1]]
        avg_delta = statistics.mean(historical_deltas) if historical_deltas else 1.0

        # Avoid division by zero in very quiet markets
        if avg_delta < 10:
            avg_delta = 10.0

        current = candles[-1]
        ratio = abs(current.delta) / avg_delta

        if ratio >= settings.DELTA_THRESHOLD_MULTIPLIER:
            direction = "BUY" if current.delta > 0 else "SELL"
            logger.debug(
                f"[{current.symbol}] Strong delta detected │ "
                f"Delta={current.delta:+d} │ Avg={avg_delta:.0f} │ "
                f"Ratio={ratio:.2f}x → {direction}"
            )
            return settings.SCORE_DELTA_STRONG, direction

        return 0, "NEUTRAL"

    # ─── Factor 2: Price vs Delta Divergence ────────────────────
    def _check_divergence(self, candles: List[Candle], lookback: int) -> Tuple[int, str]:
        """
        Compares price trend vs cumulative delta trend over the lookback window.

        Bearish Divergence: Price ↑ but Cumulative Delta ↓ → SELL
        Bullish Divergence: Price ↓ but Cumulative Delta ↑ → BUY

        Uses percentage-based thresholds to avoid false signals on tiny moves.

        Returns: (score, direction)
        """
        if len(candles) < lookback:
            return 0, "NEUTRAL"

        window = candles[-lookback:]

        # Price trend
        price_change = window[-1].close - window[0].open
        price_pct = abs(price_change) / window[0].open if window[0].open > 0 else 0

        # Ignore very small price moves (noise)
        if price_pct < 0.0002:  # Less than 0.02% move
            return 0, "NEUTRAL"

        price_up = price_change > 0

        # Cumulative delta over the window
        window_delta = sum(c.delta for c in window)

        # Also check if delta is trending (comparing first half vs second half)
        mid = lookback // 2
        first_half_delta = sum(c.delta for c in window[:mid])
        second_half_delta = sum(c.delta for c in window[mid:])
        delta_weakening = abs(second_half_delta) < abs(first_half_delta) * 0.6

        delta_up = window_delta > 0

        # Divergence detected
        if price_up and (not delta_up or delta_weakening):
            logger.debug(
                f"[{window[-1].symbol}] Bearish divergence │ "
                f"Price Δ={price_change:+.2f} │ CumDelta={window_delta:+d} │ "
                f"Weakening={delta_weakening}"
            )
            return settings.SCORE_DIVERGENCE, "SELL"

        elif not price_up and (delta_up or delta_weakening):
            logger.debug(
                f"[{window[-1].symbol}] Bullish divergence │ "
                f"Price Δ={price_change:+.2f} │ CumDelta={window_delta:+d} │ "
                f"Weakening={delta_weakening}"
            )
            return settings.SCORE_DIVERGENCE, "BUY"

        return 0, "NEUTRAL"

    # ─── Cooldown Check ─────────────────────────────────────────
    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if this symbol is still within its signal cooldown period."""
        last = self._last_signal_time.get(symbol, 0)
        elapsed = time.time() - last
        if elapsed < settings.SIGNAL_COOLDOWN_SEC:
            remaining = settings.SIGNAL_COOLDOWN_SEC - elapsed
            logger.debug(f"[{symbol}] On cooldown, {remaining:.0f}s remaining")
            return True
        return False

    def _mark_signal_fired(self, symbol: str) -> None:
        self._last_signal_time[symbol] = time.time()

    # ─── Main Evaluation ────────────────────────────────────────
    def evaluate(self, market_data: Dict[str, List[Candle]]) -> List[Signal]:
        """
        Evaluates all market data and returns a list of actionable signals.
        This is the main entry point called by the engine each cycle.
        """
        es_candles = market_data.get("ES", [])
        nq_candles = market_data.get("NQ", [])

        if len(es_candles) < settings.MIN_HISTORY_CANDLES or \
           len(nq_candles) < settings.MIN_HISTORY_CANDLES:
            logger.debug(
                f"Insufficient history: ES={len(es_candles)}, NQ={len(nq_candles)} "
                f"(need {settings.MIN_HISTORY_CANDLES})"
            )
            return []

        # ── Calculate ATR for both symbols ───────────────────────
        es_atr = self.calculate_atr(es_candles, settings.ATR_PERIOD)
        nq_atr = self.calculate_atr(nq_candles, settings.ATR_PERIOD)

        # ── Volatility filter ────────────────────────────────────
        if es_atr < settings.MIN_VOLATILITY_ATR_ES:
            logger.info(f"[ES] Low volatility filter │ ATR={es_atr:.2f} < {settings.MIN_VOLATILITY_ATR_ES}")
            return []
        if nq_atr < settings.MIN_VOLATILITY_ATR_NQ:
            logger.info(f"[NQ] Low volatility filter │ ATR={nq_atr:.2f} < {settings.MIN_VOLATILITY_ATR_NQ}")
            return []

        # ── Evaluate each symbol's individual factors ────────────
        es_delta_score, es_delta_dir = self._analyze_delta(es_candles)
        es_div_score, es_div_dir = self._check_divergence(es_candles, settings.DIVERGENCE_LOOKBACK)

        nq_delta_score, nq_delta_dir = self._analyze_delta(nq_candles)
        nq_div_score, nq_div_dir = self._check_divergence(nq_candles, settings.DIVERGENCE_LOOKBACK)

        # ── Determine base direction for each symbol ─────────────
        # Priority: Delta > Divergence (delta is real-time, divergence is lagging)
        es_base_dir = self._resolve_direction(es_delta_dir, es_div_dir)
        nq_base_dir = self._resolve_direction(nq_delta_dir, nq_div_dir)

        # ── Factor 3: Cross-instrument confirmation ──────────────
        es_conf_score = 0
        nq_conf_score = 0
        if es_base_dir == nq_base_dir and es_base_dir != "NEUTRAL":
            es_conf_score = settings.SCORE_CONFIRMATION
            nq_conf_score = settings.SCORE_CONFIRMATION
            logger.debug(
                f"ES/NQ Confirmation │ Both signaling {es_base_dir} │ +{settings.SCORE_CONFIRMATION}pts"
            )

        # ── Build total scores ───────────────────────────────────
        es_total = es_delta_score + es_div_score + es_conf_score
        nq_total = nq_delta_score + nq_div_score + nq_conf_score

        # ── Log the score breakdown ──────────────────────────────
        self._log_score_breakdown("ES", es_total, es_base_dir,
                                   es_delta_score, es_delta_dir,
                                   es_div_score, es_div_dir,
                                   es_conf_score, es_atr)
        self._log_score_breakdown("NQ", nq_total, nq_base_dir,
                                   nq_delta_score, nq_delta_dir,
                                   nq_div_score, nq_div_dir,
                                   nq_conf_score, nq_atr)

        # ── Generate signals for symbols that pass the threshold ─
        signals = []
        now = time.time()

        for sym, total, base_dir, d_score, d_dir, div_score, div_dir, conf_score, atr in [
            ("ES", es_total, es_base_dir, es_delta_score, es_delta_dir,
             es_div_score, es_div_dir, es_conf_score, es_atr),
            ("NQ", nq_total, nq_base_dir, nq_delta_score, nq_delta_dir,
             nq_div_score, nq_div_dir, nq_conf_score, nq_atr),
        ]:
            if total < settings.MIN_SCORE_TO_TRADE or base_dir == "NEUTRAL":
                continue

            if self._is_on_cooldown(sym):
                continue

            # Build reason string
            parts = []
            if d_score > 0:
                parts.append(f"Delta:{d_dir}")
            if div_score > 0:
                parts.append(f"Div:{div_dir}")
            if conf_score > 0:
                parts.append("ES=NQ")

            sym_cfg = settings.SYMBOL_MAP[sym]
            signal = Signal(
                symbol=sym,
                mt5_symbol=sym_cfg["mt5_name"],
                action=base_dir,
                score=total,
                atr=atr,
                sl_distance=atr * settings.ATR_SL_MULTIPLIER,
                tp_distance=atr * settings.ATR_TP_MULTIPLIER,
                delta_score=d_score,
                delta_direction=d_dir,
                divergence_score=div_score,
                divergence_direction=div_dir,
                confirmation_score=conf_score,
                reason=" + ".join(parts),
                timestamp=now,
            )
            signals.append(signal)
            self._mark_signal_fired(sym)

            logger.log(25,  # TRADE level
                f"🎯 SIGNAL │ {sym}→{sym_cfg['mt5_name']} │ {base_dir} │ "
                f"Score={total}/100 │ {signal.reason} │ "
                f"SL={signal.sl_distance:.2f} │ TP={signal.tp_distance:.2f}"
            )

        return signals

    # ─── Helpers ─────────────────────────────────────────────────
    @staticmethod
    def _resolve_direction(delta_dir: str, div_dir: str) -> str:
        """
        Resolves the overall direction from delta and divergence signals.
        If both are active and conflict, delta takes priority (more immediate).
        """
        if delta_dir != "NEUTRAL":
            return delta_dir
        if div_dir != "NEUTRAL":
            return div_dir
        return "NEUTRAL"

    @staticmethod
    def _log_score_breakdown(
        sym: str, total: int, direction: str,
        d_score: int, d_dir: str,
        div_score: int, div_dir: str,
        conf_score: int, atr: float,
    ) -> None:
        d_tag = f"✅{d_dir}" if d_score > 0 else "—"
        div_tag = f"✅{div_dir}" if div_score > 0 else "—"
        conf_tag = "✅" if conf_score > 0 else "—"

        logger.info(
            f"[{sym}] Score={total:3d} ({direction:7s}) │ "
            f"Delta[{d_score:2d}]={d_tag:10s} │ "
            f"Div[{div_score:2d}]={div_tag:10s} │ "
            f"Conf[{conf_score:2d}]={conf_tag:3s} │ "
            f"ATR={atr:.2f}"
        )
