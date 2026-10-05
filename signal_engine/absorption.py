"""
signal_engine/absorption.py — Price-Action Absorption Detection for MARK III v2

Detects absorption-like behavior WITHOUT DOM or Level 2 data.
Uses only OHLCV + volume relationships to identify when aggressive
moves are being absorbed by passive orders.

Detection methods:
  1. Volume expansion with weak price progress (big volume, small body)
  2. Repeated rejection at the same level (level defense)
  3. Failed continuation (strong candle immediately reversed)
  4. Post-sweep reversal (from MarketStructure)

IMPORTANT: This is a price-action interpretation, NOT real order book
analysis. Real absorption detection requires DOM data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monitoring.logger import get_logger

logger = get_logger("Absorption")


@dataclass
class AbsorptionResult:
    """Result of absorption detection."""
    detected: bool = False
    absorption_type: str = "NEUTRAL"  # "BUY_ABSORPTION" or "SELL_ABSORPTION"
    reversal_direction: str = "NEUTRAL"  # Expected reversal direction
    strength: int = 0  # 0-15 contribution to signal score
    reason: str = ""

    def summary(self) -> str:
        if not self.detected:
            return "None"
        return f"{self.absorption_type} → {self.reversal_direction}({self.strength}) │ {self.reason}"


class AbsorptionDetector:
    """
    Detects absorption-like behavior through price action and volume.
    NOT DOM-based. Uses candle morphology as a practical proxy.
    """

    def __init__(self) -> None:
        logger.info("AbsorptionDetector initialized (price-action proxy)")

    def detect(self, df: pd.DataFrame) -> AbsorptionResult:
        """
        Runs all absorption detection methods and returns the strongest signal.

        Args:
            df: M5 DataFrame with OHLCV + indicators from MT5DataProvider.

        Returns:
            AbsorptionResult with detection details.
        """
        if df is None or len(df) < 10:
            return AbsorptionResult()

        # Run each detection method
        results = [
            self._check_volume_absorption(df),
            self._check_level_defense(df),
            self._check_failed_continuation(df),
        ]

        # Return the strongest detection
        best = max(results, key=lambda r: r.strength)
        return best

    # ═══════════════════════════════════════════════════════════════
    # 1. Volume Expansion + Weak Price Progress
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _check_volume_absorption(df: pd.DataFrame) -> AbsorptionResult:
        """
        High volume but small body = passive orders absorbing aggression.

        Conditions:
        - tick_volume > 1.5× average
        - body_abs < 0.3× ATR (weak progress despite volume)
        - The candle direction indicates who was absorbed

        If volume spikes during a bullish candle but price barely moves up,
        sellers are absorbing buyers → expect reversal DOWN.
        """
        last = df.iloc[-1]
        vol_ratio = last.get("vol_ratio", 1.0)
        body_abs = last.get("body_abs", 0)
        atr = last.get("atr", 1.0)
        body = last.get("body", 0)

        if pd.isna(atr) or atr <= 0:
            return AbsorptionResult()

        # Volume must be significantly above average
        if vol_ratio < 1.5:
            return AbsorptionResult()

        # Body must be small relative to ATR (weak progress)
        if body_abs > atr * 0.3:
            return AbsorptionResult()

        # Direction of the weak move tells us who got absorbed
        if body > 0:
            # Bullish candle with high vol but small body → buy absorption
            return AbsorptionResult(
                detected=True,
                absorption_type="BUY_ABSORPTION",
                reversal_direction="SELL",
                strength=15 if vol_ratio > 2.0 else 10,
                reason=f"High vol ({vol_ratio:.1f}×) + small body ({body_abs:.2f}/{atr:.2f} ATR)"
            )
        elif body < 0:
            # Bearish candle with high vol but small body → sell absorption
            return AbsorptionResult(
                detected=True,
                absorption_type="SELL_ABSORPTION",
                reversal_direction="BUY",
                strength=15 if vol_ratio > 2.0 else 10,
                reason=f"High vol ({vol_ratio:.1f}×) + small body ({body_abs:.2f}/{atr:.2f} ATR)"
            )

        return AbsorptionResult()

    # ═══════════════════════════════════════════════════════════════
    # 2. Repeated Rejection at Same Level (Level Defense)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _check_level_defense(df: pd.DataFrame) -> AbsorptionResult:
        """
        Multiple candles with wicks touching the same zone = level being defended.

        Checks last 5 candles for 3+ wicks touching the same price zone
        (within 0.5× ATR).
        """
        if len(df) < 5:
            return AbsorptionResult()

        recent = df.iloc[-5:]
        atr = recent.iloc[-1].get("atr", 1.0)
        if pd.isna(atr) or atr <= 0:
            return AbsorptionResult()

        zone_tolerance = atr * 0.5

        # Check for lower wick clustering (support defense)
        lows = recent["low"].values
        for i in range(len(lows)):
            level = lows[i]
            touches = sum(1 for low in lows if abs(low - level) <= zone_tolerance)
            if touches >= 3:
                return AbsorptionResult(
                    detected=True,
                    absorption_type="SELL_ABSORPTION",
                    reversal_direction="BUY",
                    strength=10,
                    reason=f"Support defense: {touches} wicks near {level:.2f}"
                )

        # Check for upper wick clustering (resistance defense)
        highs = recent["high"].values
        for i in range(len(highs)):
            level = highs[i]
            touches = sum(1 for high in highs if abs(high - level) <= zone_tolerance)
            if touches >= 3:
                return AbsorptionResult(
                    detected=True,
                    absorption_type="BUY_ABSORPTION",
                    reversal_direction="SELL",
                    strength=10,
                    reason=f"Resistance defense: {touches} wicks near {level:.2f}"
                )

        return AbsorptionResult()

    # ═══════════════════════════════════════════════════════════════
    # 3. Failed Continuation
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _check_failed_continuation(df: pd.DataFrame) -> AbsorptionResult:
        """
        A strong candle immediately followed by an equal or stronger
        opposite candle = failed continuation (engulfing pattern).

        This suggests the initial move was absorbed and reversed.
        """
        if len(df) < 3:
            return AbsorptionResult()

        prev = df.iloc[-2]
        last = df.iloc[-1]

        prev_body = prev.get("body", 0)
        last_body = last.get("body", 0)
        prev_body_abs = prev.get("body_abs", 0)
        last_body_abs = last.get("body_abs", 0)
        atr = last.get("atr", 1.0)

        if pd.isna(atr) or atr <= 0:
            return AbsorptionResult()

        # Both candles must be significant (body > 0.3× ATR)
        if prev_body_abs < atr * 0.3 or last_body_abs < atr * 0.3:
            return AbsorptionResult()

        # Opposite directions and current engulfs previous
        if prev_body > 0 and last_body < 0 and last_body_abs >= prev_body_abs * 0.8:
            # Previous was bullish, current is bearish engulfing
            return AbsorptionResult(
                detected=True,
                absorption_type="BUY_ABSORPTION",
                reversal_direction="SELL",
                strength=15,
                reason="Bearish engulfing (failed bullish continuation)"
            )

        if prev_body < 0 and last_body > 0 and last_body_abs >= prev_body_abs * 0.8:
            # Previous was bearish, current is bullish engulfing
            return AbsorptionResult(
                detected=True,
                absorption_type="SELL_ABSORPTION",
                reversal_direction="BUY",
                strength=15,
                reason="Bullish engulfing (failed bearish continuation)"
            )

        return AbsorptionResult()
