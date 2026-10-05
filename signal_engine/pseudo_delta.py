"""
signal_engine/pseudo_delta.py — Candle-Based Pressure Proxy for MARK III v2

IMPORTANT: This is NOT real bid/ask delta. Real delta requires institutional
data feeds (Rithmic, CQG) with per-tick bid/ask classification.

This module approximates directional pressure using only OHLCV candle data:
  - Body direction and size relative to range
  - Close position within the candle (near high = buying pressure)
  - Volume weighting (high-volume candles carry more weight)
  - Rolling cumulative pressure (similar concept to CVD, but candle-based)
  - Pressure divergence (pressure rising but price falling = absorption)

These are honest approximations, NOT institutional-grade signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monitoring.logger import get_logger

logger = get_logger("PseudoDelta")


@dataclass
class PressureResult:
    """Result of candle-based pressure analysis."""
    # Current state
    pressure: float = 0.0           # Last candle pressure (-1 to +1)
    cum_pressure: float = 0.0       # Rolling cumulative pressure
    pressure_direction: str = "NEUTRAL"  # "BUY", "SELL", "NEUTRAL"

    # Divergence
    divergence_detected: bool = False
    divergence_direction: str = "NEUTRAL"  # Direction of expected reversal
    divergence_strength: int = 0     # 0-15 contribution to score

    # Alignment score
    alignment_score: int = 0         # 0-15 contribution to signal score

    def summary(self) -> str:
        parts = [f"Pressure={self.pressure:+.2f}"]
        if self.divergence_detected:
            parts.append(f"Div={self.divergence_direction}({self.divergence_strength})")
        parts.append(f"Align={self.alignment_score}")
        return " │ ".join(parts)


class PseudoDelta:
    """
    Candle-based directional pressure approximation.

    NOT real delta. Uses candle morphology + volume to infer
    buying/selling pressure as a practical proxy.
    """

    def __init__(self) -> None:
        self._lookback = 10  # Rolling pressure window
        logger.info("PseudoDelta initialized (candle-based pressure proxy)")

    def calculate(self, df: pd.DataFrame) -> PressureResult:
        """
        Calculates pressure metrics from OHLCV data.

        Args:
            df: DataFrame with columns from MT5DataProvider
                (open, high, low, close, tick_volume, body, range,
                 close_position, vol_ratio)

        Returns:
            PressureResult with pressure values and divergence detection.
        """
        result = PressureResult()

        if df is None or len(df) < self._lookback + 5:
            return result

        # ── Calculate per-candle pressure ────────────────────────
        pressures = self._candle_pressure(df)

        if len(pressures) == 0:
            return result

        # ── Current pressure ────────────────────────────────────
        result.pressure = pressures[-1]

        if result.pressure > 0.15:
            result.pressure_direction = "BUY"
        elif result.pressure < -0.15:
            result.pressure_direction = "SELL"
        else:
            result.pressure_direction = "NEUTRAL"

        # ── Rolling cumulative pressure ─────────────────────────
        window = pressures[-self._lookback:]
        result.cum_pressure = float(np.sum(window))

        # ── Check for pressure divergence ───────────────────────
        result.divergence_detected, result.divergence_direction, result.divergence_strength = (
            self._detect_divergence(df, pressures)
        )

        # ── Alignment score ─────────────────────────────────────
        result.alignment_score = self._score_alignment(pressures)

        return result

    # ═══════════════════════════════════════════════════════════════
    # Per-Candle Pressure Calculation
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _candle_pressure(df: pd.DataFrame) -> np.ndarray:
        """
        Calculates directional pressure for each candle.

        Pressure = close_position_score × body_ratio × volume_weight

        Where:
        - close_position_score: -1 (close at low) to +1 (close at high)
        - body_ratio: 0 (doji) to 1 (full body) — conviction
        - volume_weight: capped at 2× average (amplifies high-volume candles)

        Result: values from -2.0 to +2.0 (volume can amplify beyond ±1)
        """
        close_pos = df["close_position"].values
        body_ratio = df["body_ratio"].values
        vol_ratio = df["vol_ratio"].values

        # Close position score: map 0-1 → -1 to +1
        direction_score = (close_pos * 2.0) - 1.0

        # Volume weight: cap at 2× to prevent outlier dominance
        vol_weight = np.clip(vol_ratio, 0.5, 2.0)

        # Combined pressure
        pressure = direction_score * body_ratio * vol_weight

        return pressure

    # ═══════════════════════════════════════════════════════════════
    # Pressure Divergence Detection
    # ═══════════════════════════════════════════════════════════════

    def _detect_divergence(
        self, df: pd.DataFrame, pressures: np.ndarray
    ) -> tuple:
        """
        Detects pressure divergence: pressure and price moving in
        opposite directions over the lookback window.

        Bearish divergence: Price rising but cumulative pressure falling
        Bullish divergence: Price falling but cumulative pressure rising

        Returns (detected, reversal_direction, score 0-15).
        """
        if len(df) < self._lookback + 2:
            return False, "NEUTRAL", 0

        window = self._lookback
        recent_close = df["close"].values[-window:]
        recent_pressure = pressures[-window:]

        # Price trend over window
        price_change = recent_close[-1] - recent_close[0]
        price_pct = abs(price_change) / recent_close[0] if recent_close[0] > 0 else 0

        # Ignore tiny price moves (noise)
        if price_pct < 0.0003:
            return False, "NEUTRAL", 0

        price_rising = price_change > 0

        # Pressure trend: compare first half vs second half
        mid = window // 2
        first_half = float(np.sum(recent_pressure[:mid]))
        second_half = float(np.sum(recent_pressure[mid:]))

        # Cumulative pressure direction
        cum_pressure = float(np.sum(recent_pressure))
        pressure_rising = cum_pressure > 0

        # Pressure weakening (second half weaker than first)
        pressure_weakening = abs(second_half) < abs(first_half) * 0.5

        # Bearish divergence: price up but pressure down or weakening
        if price_rising and (not pressure_rising or pressure_weakening):
            return True, "SELL", 15

        # Bullish divergence: price down but pressure up or weakening
        if not price_rising and (pressure_rising or pressure_weakening):
            return True, "BUY", 15

        return False, "NEUTRAL", 0

    # ═══════════════════════════════════════════════════════════════
    # Alignment Scoring
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _score_alignment(pressures: np.ndarray) -> int:
        """
        Scores how consistently the last 5 candles agree in direction.

        5/5 same direction: 15 pts
        4/5 same direction: 10 pts
        3/5 same direction: 5 pts
        Otherwise: 0 pts
        """
        if len(pressures) < 5:
            return 0

        last5 = pressures[-5:]
        positive = int(np.sum(last5 > 0.1))
        negative = int(np.sum(last5 < -0.1))

        dominant = max(positive, negative)

        if dominant >= 5:
            return 15
        elif dominant >= 4:
            return 10
        elif dominant >= 3:
            return 5
        return 0
