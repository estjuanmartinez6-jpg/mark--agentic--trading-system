"""
signal_engine/market_structure.py — Price Structure Analysis for MARK III v2

Detects market structure using REAL price data from MT5:
  - Swing highs and swing lows (fractal-based)
  - Break of Structure (BOS) — trend continuation signal
  - Liquidity sweeps — price pierces a level then reverses (trap)
  - Rejection candles — long wicks showing institutional defense
  - Trend context from M15 EMA alignment

This module operates on pandas DataFrames from MT5DataProvider.
No random data, no simulations, no fake signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import settings
from monitoring.logger import get_logger

logger = get_logger("Structure")


@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    index: int
    price: float
    type: str  # "HIGH" or "LOW"
    broken: bool = False


@dataclass
class StructureContext:
    """Result of market structure analysis for one symbol."""
    # Trend
    trend: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    trend_strength: int = 0  # 0-20 (contribution to signal score)

    # Structure Break
    bos_detected: bool = False
    bos_direction: str = "NEUTRAL"  # "BULLISH" or "BEARISH"
    bos_strength: int = 0  # 0-20

    # Liquidity Sweep
    sweep_detected: bool = False
    sweep_direction: str = "NEUTRAL"  # Direction of expected reversal
    sweep_strength: int = 0  # 0-15

    # Rejection Candle
    rejection_detected: bool = False
    rejection_direction: str = "NEUTRAL"
    rejection_strength: int = 0  # 0-15

    # Raw swing data
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)

    # Current price level context
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0

    # Trend diagnostics for forensic logging
    trend_reason: str = "not evaluated"
    trend_diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"Trend={self.trend}({self.trend_strength})"]
        if self.bos_detected:
            parts.append(f"BOS={self.bos_direction}({self.bos_strength})")
        if self.sweep_detected:
            parts.append(f"Sweep={self.sweep_direction}({self.sweep_strength})")
        if self.rejection_detected:
            parts.append(f"Reject={self.rejection_direction}({self.rejection_strength})")
        return " │ ".join(parts)


class MarketStructure:
    """
    Analyzes price structure on M5 and M15 timeframes.
    All analysis is based on real OHLCV data.
    """

    def __init__(self) -> None:
        self._lookback = getattr(settings, "SWING_LOOKBACK", 5)
        logger.info(
            f"MarketStructure initialized │ "
            f"Swing lookback={self._lookback}"
        )

    def analyze(
        self, df_m5: pd.DataFrame, df_m15: pd.DataFrame
    ) -> StructureContext:
        """
        Full structure analysis combining M15 trend with M5 structure.

        Args:
            df_m5:  M5 DataFrame with OHLCV + ema_21, ema_50, atr
            df_m15: M15 DataFrame with OHLCV + ema_21, ema_50, atr

        Returns:
            StructureContext with all detected features.
        """
        ctx = StructureContext()

        if df_m5 is None or len(df_m5) < 20:
            return ctx
        if df_m15 is None or len(df_m15) < 10:
            return ctx

        # ── 1. M15 Trend Context ────────────────────────────────
        (
            ctx.trend,
            ctx.trend_strength,
            ctx.trend_reason,
            ctx.trend_diagnostics,
        ) = self._analyze_trend(df_m15)

        # ── 2. M5 Swing Detection ──────────────────────────────
        ctx.swing_highs = self._find_swing_highs(df_m5)
        ctx.swing_lows = self._find_swing_lows(df_m5)

        if ctx.swing_highs:
            ctx.last_swing_high = ctx.swing_highs[-1].price
        if ctx.swing_lows:
            ctx.last_swing_low = ctx.swing_lows[-1].price

        # ── 3. Break of Structure (BOS) ────────────────────────
        ctx.bos_detected, ctx.bos_direction, ctx.bos_strength = (
            self._detect_bos(df_m5, ctx.swing_highs, ctx.swing_lows, ctx.trend)
        )

        # ── 4. Liquidity Sweep ─────────────────────────────────
        ctx.sweep_detected, ctx.sweep_direction, ctx.sweep_strength = (
            self._detect_sweep(df_m5, ctx.swing_highs, ctx.swing_lows, ctx.trend)
        )

        # ── 5. Rejection Candle ────────────────────────────────
        ctx.rejection_detected, ctx.rejection_direction, ctx.rejection_strength = (
            self._detect_rejection(df_m5, ctx.trend)
        )

        return ctx

    # ═══════════════════════════════════════════════════════════════
    # 1. TREND ANALYSIS (M15)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _analyze_trend(df: pd.DataFrame) -> tuple:
        """
        Determines trend from M15 EMA alignment.
        Returns (trend_direction, score 0-20, reason, diagnostics).

        v2.5: Developing trend uses ATR-normalized proximity to EMA21
        instead of requiring price to sit exactly on EMA21.
        """
        diagnostics = {
            "close": None,
            "ema21": None,
            "ema50": None,
            "ema21_slope": None,
            "ema50_slope": None,
            "ema50_valid": False,
            "atr": None,
            "proximity_atr": None,
            "developing": False,
            "reason": "not evaluated",
        }

        if "ema_21" not in df.columns or "ema_50" not in df.columns:
            diagnostics["reason"] = "missing EMA columns"
            return "NEUTRAL", 0, diagnostics["reason"], diagnostics

        last = df.iloc[-1]
        ema21 = last["ema_21"]
        ema50 = last["ema_50"]
        close = last["close"]
        atr = last.get("atr", np.nan)

        diagnostics.update({
            "close": round(float(close), 2) if pd.notna(close) else None,
            "ema21": round(float(ema21), 2) if pd.notna(ema21) else None,
            "ema50": round(float(ema50), 2) if pd.notna(ema50) else None,
            "ema50_valid": bool(pd.notna(ema50)),
            "atr": round(float(atr), 2) if pd.notna(atr) else None,
        })

        if pd.isna(ema21) or pd.isna(ema50):
            diagnostics["reason"] = "EMA not initialized"
            return "NEUTRAL", 0, diagnostics["reason"], diagnostics

        ema21_slope = 0.0
        ema50_slope = 0.0
        if len(df) >= 4:
            ema21_prev = df.iloc[-3]["ema_21"]
            ema50_prev = df.iloc[-3]["ema_50"]
            if pd.notna(ema21_prev):
                ema21_slope = float(ema21 - ema21_prev)
            if pd.notna(ema50_prev):
                ema50_slope = float(ema50 - ema50_prev)

        proximity_atr = None
        if pd.notna(atr) and atr > 0:
            proximity_atr = abs(float(close - ema21)) / float(atr)

        diagnostics.update({
            "ema21_slope": round(ema21_slope, 4),
            "ema50_slope": round(ema50_slope, 4),
            "proximity_atr": round(proximity_atr, 3) if proximity_atr is not None else None,
        })

        # Strong bullish: price > EMA21 > EMA50
        if close > ema21 and ema21 > ema50:
            # Bonus: check EMA separation (wider = stronger trend)
            separation = (ema21 - ema50) / ema50 if ema50 > 0 else 0
            strength = 20 if separation > 0.001 else 15
            diagnostics["reason"] = f"strong bullish EMA alignment sep={separation:.4f}"
            return "BULLISH", strength, diagnostics["reason"], diagnostics

        # Strong bearish: price < EMA21 < EMA50
        if close < ema21 and ema21 < ema50:
            separation = (ema50 - ema21) / ema50 if ema50 > 0 else 0
            strength = 20 if separation > 0.001 else 15
            diagnostics["reason"] = f"strong bearish EMA alignment sep={separation:.4f}"
            return "BEARISH", strength, diagnostics["reason"], diagnostics

        # Weak bullish: price above EMA21 with directional slope support.
        # v2.5: Reduced from 10 → 5 pts to prevent permanent directional
        # bias that made SELL signals structurally unreachable.
        if close > ema21 and ema21_slope >= 0:
            diagnostics["reason"] = "weak bullish: price above EMA21 with non-negative EMA21 slope"
            return "BULLISH", 5, diagnostics["reason"], diagnostics

        # Weak bearish: price below EMA21 with directional slope support.
        # v2.5: Reduced from 10 → 5 pts (symmetric with weak bullish).
        if close < ema21 and ema21_slope <= 0:
            diagnostics["reason"] = "weak bearish: price below EMA21 with non-positive EMA21 slope"
            return "BEARISH", 5, diagnostics["reason"], diagnostics

        # Developing trend: conservative, ATR-normalized EMA21 retest.
        # This grants only 5 points and requires slope + EMA structure.
        near_ema21 = proximity_atr is not None and proximity_atr <= 0.35
        ema_spread_tolerance = float(atr) * 0.10 if pd.notna(atr) and atr > 0 else 0.0

        if near_ema21:
            bullish_structure = ema21 >= (ema50 - ema_spread_tolerance)
            bearish_structure = ema21 <= (ema50 + ema_spread_tolerance)

            if ema21_slope > 0 and bullish_structure and close >= ema21 - ema_spread_tolerance:
                diagnostics["developing"] = True
                diagnostics["reason"] = "developing bullish: near EMA21, rising EMA21, EMA21 near/above EMA50"
                return "BULLISH", 5, diagnostics["reason"], diagnostics

            if ema21_slope < 0 and bearish_structure and close <= ema21 + ema_spread_tolerance:
                diagnostics["developing"] = True
                diagnostics["reason"] = "developing bearish: near EMA21, falling EMA21, EMA21 near/below EMA50"
                return "BEARISH", 5, diagnostics["reason"], diagnostics

        diagnostics["reason"] = "neutral: no EMA alignment, weak trend, or developing bias"
        return "NEUTRAL", 0, diagnostics["reason"], diagnostics

    # ═══════════════════════════════════════════════════════════════
    # 2. SWING DETECTION
    # ═══════════════════════════════════════════════════════════════

    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Finds swing highs using fractal logic.
        A swing high is a candle whose HIGH is higher than the
        N candles on each side.
        """
        highs = df["high"].values
        n = self._lookback
        swings = []

        for i in range(n, len(highs) - n):
            is_swing = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append(SwingPoint(index=i, price=highs[i], type="HIGH"))

        return swings

    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Finds swing lows using fractal logic.
        A swing low is a candle whose LOW is lower than the
        N candles on each side.
        """
        lows = df["low"].values
        n = self._lookback
        swings = []

        for i in range(n, len(lows) - n):
            is_swing = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append(SwingPoint(index=i, price=lows[i], type="LOW"))

        return swings

    # ═══════════════════════════════════════════════════════════════
    # 3. BREAK OF STRUCTURE (BOS)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_bos(
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        trend: str,
    ) -> tuple:
        """
        Detects Break of Structure: price closing beyond the last swing level.

        Bullish BOS: Current close > last swing high
        Bearish BOS: Current close < last swing low

        v2.1: Added persistence window — if break happened within last
        3 candles, still counts with reduced strength (10 pts).

        Returns (detected, direction, score 0-20).
        """
        if not swing_highs and not swing_lows:
            return False, "NEUTRAL", 0

        last_close = df.iloc[-1]["close"]
        prev_close = df.iloc[-2]["close"]

        # ── Bullish BOS ──────────────────────────────────────────
        if swing_highs:
            last_sh = swing_highs[-1].price

            # Fresh break (this candle): full strength
            if last_close > last_sh and prev_close <= last_sh:
                score = 20 if trend == "BULLISH" else 15
                return True, "BULLISH", score

            # Recent break (within last 3 candles): reduced strength
            # Price is still above the level, and a recent candle was the break
            if last_close > last_sh:
                lookback = min(5, len(df) - 1)
                for i in range(2, lookback):
                    if df.iloc[-i]["close"] <= last_sh:
                        score = 10  # Reduced but still active
                        return True, "BULLISH", score

        # ── Bearish BOS ──────────────────────────────────────────
        if swing_lows:
            last_sl = swing_lows[-1].price

            # Fresh break (this candle): full strength
            if last_close < last_sl and prev_close >= last_sl:
                score = 20 if trend == "BEARISH" else 15
                return True, "BEARISH", score

            # Recent break (within last 3 candles): reduced strength
            if last_close < last_sl:
                lookback = min(5, len(df) - 1)
                for i in range(2, lookback):
                    if df.iloc[-i]["close"] >= last_sl:
                        score = 10
                        return True, "BEARISH", score

        return False, "NEUTRAL", 0

    # ═══════════════════════════════════════════════════════════════
    # 4. LIQUIDITY SWEEP
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_sweep(
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        trend: str,
    ) -> tuple:
        """
        Detects liquidity sweeps: price pierces a swing level with a wick
        but closes back inside.

        Bearish sweep: High exceeds last swing high but close is below it.
        Bullish sweep: Low exceeds last swing low but close is above it.

        v2.7: Trend-Compliant Pullbacks. Full score ONLY if sweep aligns with trend.
        Counter-trend sweeps get 0-5 pts max, killing anti-trend entries.

        Returns (detected, reversal_direction, score 0-15).
        """
        if len(df) < 3:
            return False, "NEUTRAL", 0

        last = df.iloc[-1]

        # Bearish sweep of highs (trap buyers, expect reversal down)
        if swing_highs:
            sh = swing_highs[-1].price
            if last["high"] > sh and last["close"] < sh:
                # Wick pierced the level but close came back — trap
                score = 15 if trend == "BEARISH" else 5
                return True, "SELL", score

        # Bullish sweep of lows (trap sellers, expect reversal up)
        if swing_lows:
            sl = swing_lows[-1].price
            if last["low"] < sl and last["close"] > sl:
                score = 15 if trend == "BULLISH" else 5
                return True, "BUY", score

        return False, "NEUTRAL", 0

    # ═══════════════════════════════════════════════════════════════
    # 5. REJECTION CANDLE
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_rejection(df: pd.DataFrame, trend: str) -> tuple:
        """
        Detects rejection candles: candles with a long wick (>60% of range)
        showing strong rejection at a price level.

        Bullish rejection: Long lower wick (hammer-like)
        Bearish rejection: Long upper wick (shooting star-like)

        v2.7: Trend-compliant Pullbacks. Downgrade counter-trend wicks.
        Returns (detected, direction, score 0-15).
        """
        if len(df) < 2:
            return False, "NEUTRAL", 0

        last = df.iloc[-1]
        candle_range = last["range"]

        if candle_range <= 0:
            return False, "NEUTRAL", 0

        upper_wick = last["upper_wick"]
        lower_wick = last["lower_wick"]
        body = last["body_abs"]

        # Minimum range threshold (avoid noise on tiny candles)
        atr = last.get("atr", 0)
        if pd.notna(atr) and atr > 0 and candle_range < atr * 0.3:
            return False, "NEUTRAL", 0

        # Bullish rejection: lower wick > 60% of range, small body
        if lower_wick > candle_range * 0.60 and body < candle_range * 0.30:
            score = 15 if trend == "BULLISH" else 5
            return True, "BUY", score

        # Bearish rejection: upper wick > 60% of range, small body
        if upper_wick > candle_range * 0.60 and body < candle_range * 0.30:
            score = 15 if trend == "BEARISH" else 5
            return True, "SELL", score

        # Moderate rejection (wick > 50%)
        if lower_wick > candle_range * 0.50 and body < candle_range * 0.35:
            score = 10 if trend == "BULLISH" else 5
            return True, "BUY", score

        if upper_wick > candle_range * 0.50 and body < candle_range * 0.35:
            score = 10 if trend == "BEARISH" else 5
            return True, "SELL", score

        return False, "NEUTRAL", 0
