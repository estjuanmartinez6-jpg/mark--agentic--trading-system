"""
signal_engine/momentum_speed.py — Volatility Expansion Proxy for MARK III v2.1

Approximates "tape speed" and market urgency using candle-level metrics.

IMPORTANT: This is NOT real Time & Sales tape speed. Real tape speed
requires tick-level institutional data with millisecond timestamps.

This module uses practical proxies:
  - Range expansion vs rolling average (3-candle smoothed)
  - Volume burst detection
  - ATR acceleration (expanding or contracting volatility)

v2.1 Changes:
  - range_expansion now uses 3-candle average (not single candle)
  - is_dead threshold lowered from 25 to 15
  - range < 0.5 no longer a hard-block (converted to score penalty)
  - Trend-aware soft override: confirmed M15 trend + speed > 10 → not dead
  - Graduated NORMAL contribution (6-12 pts based on speed)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monitoring.logger import get_logger

logger = get_logger("Speed")


@dataclass
class SpeedContext:
    """Result of momentum/speed analysis."""
    speed_score: int = 0     # 0-100 overall speed rating
    is_fast: bool = False    # Market is active and directional
    is_dead: bool = False    # Market is too quiet to trade

    # Components
    range_expansion: float = 0.0    # Current range / avg range ratio
    volume_burst: float = 0.0       # Current volume / avg volume ratio
    atr_trend: str = "FLAT"         # "EXPANDING", "CONTRACTING", "FLAT"

    # Score contribution
    contribution: int = 0   # 0-15 contribution to signal score

    # v2.1: Regime classification for logging
    regime: str = "NORMAL"  # "DEAD", "SLOW_TREND", "NORMAL", "FAST", "EXPLOSIVE"

    def summary(self) -> str:
        state = "🟢FAST" if self.is_fast else ("🔴DEAD" if self.is_dead else "🟡NORMAL")
        return (
            f"Speed={self.speed_score} {state} │ "
            f"Range={self.range_expansion:.1f}× │ "
            f"Vol={self.volume_burst:.1f}× │ "
            f"ATR={self.atr_trend}"
        )


class MomentumSpeed:
    """
    Volatility and speed proxy using candle-level metrics.
    NOT real Time & Sales tape speed.

    v2.1: Adaptive regime detection with trend-awareness.
    """

    def __init__(self) -> None:
        self._lookback = 20  # Rolling window for averages
        logger.info("MomentumSpeed initialized (volatility proxy, v2.1 adaptive)")

    def measure(
        self,
        df: pd.DataFrame,
        trend: str = "NEUTRAL",
        trend_strength: int = 0,
    ) -> SpeedContext:
        """
        Measures market speed/activity from OHLCV data.

        Args:
            df: M5 DataFrame with OHLCV + indicators from MT5DataProvider.
            trend: M15 trend direction ("BULLISH", "BEARISH", "NEUTRAL")
            trend_strength: M15 trend strength (0-20)

        Returns:
            SpeedContext with speed metrics and trading suitability.
        """
        ctx = SpeedContext()

        if df is None or len(df) < self._lookback + 2:
            ctx.is_dead = True
            ctx.regime = "DEAD"
            return ctx

        # ── 1. Range Expansion (3-candle smoothed) ───────────────
        # v2.1: Use average of last 3 candles instead of single candle
        # to prevent a single doji from triggering DEAD classification
        ranges = df["range"].values
        avg_range = float(np.mean(ranges[-self._lookback:-3]))  # Exclude last 3 from baseline
        current_range = float(np.mean(ranges[-3:]))  # Smooth with 3-candle avg

        if avg_range > 0:
            ctx.range_expansion = current_range / avg_range
        else:
            ctx.range_expansion = 1.0

        # ── 2. Volume Burst ─────────────────────────────────────
        # Also use 3-candle average for consistency
        if "vol_ratio" in df.columns:
            ctx.volume_burst = float(np.mean(
                df["vol_ratio"].values[-3:]
            ))
        else:
            ctx.volume_burst = 1.0

        # ── 3. ATR Trend ────────────────────────────────────────
        ctx.atr_trend = self._atr_direction(df)

        # ── 4. Combined Speed Score ─────────────────────────────
        # Range expansion: 0-40 pts
        range_pts = min(40, int(ctx.range_expansion * 20))

        # Volume burst: 0-30 pts
        vol_pts = min(30, int(ctx.volume_burst * 15))

        # ATR trend: 0-30 pts
        atr_pts = {"EXPANDING": 30, "FLAT": 15, "CONTRACTING": 5}.get(ctx.atr_trend, 15)

        ctx.speed_score = min(100, range_pts + vol_pts + atr_pts)

        # ── 5. Classify market state (v2.1 adaptive) ────────────
        ctx.is_fast = ctx.speed_score >= 60 and ctx.range_expansion >= 1.3

        # v2.1: Dead market detection — adaptive, not rigid
        #   - Base threshold: speed_score < 15 (lowered from 25)
        #   - No more range_expansion < 0.5 hard-block
        #   - Trend-aware soft override: if M15 trend is confirmed
        #     AND speed is at least 10, the market is NOT dead
        #     (it's a slow continuation, not a dead market)
        raw_dead = ctx.speed_score < 15

        # Trend-aware override (soft, not full bypass)
        trend_confirmed = trend in ("BULLISH", "BEARISH") and trend_strength >= 15
        if raw_dead and trend_confirmed and ctx.speed_score > 10:
            # Market is slow but trending — classify as SLOW_TREND, not DEAD
            ctx.is_dead = False
            ctx.regime = "SLOW_TREND"
        else:
            ctx.is_dead = raw_dead
            if ctx.is_dead:
                ctx.regime = "DEAD"

        # ── 6. Regime classification ────────────────────────────
        if not ctx.is_dead and ctx.regime != "SLOW_TREND":
            if ctx.is_fast and ctx.range_expansion >= 2.0:
                ctx.regime = "EXPLOSIVE"
            elif ctx.is_fast:
                ctx.regime = "FAST"
            else:
                ctx.regime = "NORMAL"

        # ── 7. Score contribution (v2.1 graduated) ──────────────
        if ctx.is_fast:
            ctx.contribution = 15
        elif ctx.regime == "SLOW_TREND":
            # Slow but confirmed trend: award moderate points
            ctx.contribution = 8
        elif not ctx.is_dead:
            # NORMAL: graduate from 6-12 based on speed score
            ctx.contribution = min(12, max(6, ctx.speed_score // 5))
        else:
            ctx.contribution = 0

        return ctx

    # ═══════════════════════════════════════════════════════════════
    # ATR Direction
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _atr_direction(df: pd.DataFrame) -> str:
        """
        Determines if ATR is expanding, contracting, or flat
        by comparing the last 3 ATR values.
        """
        if "atr" not in df.columns or len(df) < 5:
            return "FLAT"

        atr_vals = df["atr"].dropna().values
        if len(atr_vals) < 5:
            return "FLAT"

        recent_3 = atr_vals[-3:]

        # Check if consistently rising
        if recent_3[-1] > recent_3[-2] > recent_3[-3]:
            return "EXPANDING"

        # Check if consistently falling
        if recent_3[-1] < recent_3[-2] < recent_3[-3]:
            return "CONTRACTING"

        # Mixed or flat
        change_pct = abs(recent_3[-1] - recent_3[0]) / recent_3[0] if recent_3[0] > 0 else 0
        if change_pct > 0.10:
            return "EXPANDING" if recent_3[-1] > recent_3[0] else "CONTRACTING"

        return "FLAT"
