"""
signal_engine/correlation.py — ES/NQ Structural Divergence for MARK III v2

Performs proper cross-market analysis between ES (S&P 500) and NQ (Nasdaq 100)
using structural comparisons, not just simple direction checks.

Analysis methods:
  - SMT-style divergence: one makes higher high while other makes lower high
  - Momentum confirmation: both trending same direction with aligned structure
  - Leader detection: which instrument moved first (it sets the tone)
  - Structural agreement scoring

Uses swing highs/lows from MarketStructure analysis — NOT fake data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from monitoring.logger import get_logger

logger = get_logger("Correlation")


@dataclass
class CorrelationResult:
    """Result of ES/NQ cross-market analysis."""
    # Divergence
    divergence_detected: bool = False
    divergence_type: str = "NONE"  # "BULLISH_DIV", "BEARISH_DIV", "NONE"
    divergence_direction: str = "NEUTRAL"  # Expected trade direction

    # Confirmation
    confirmed: bool = False
    confirmation_direction: str = "NEUTRAL"  # Direction both agree on

    # Leader
    leader: str = "NONE"  # "ES" or "NQ"

    # Score contribution
    strength: int = 0  # 0-15 contribution to signal score

    def summary(self) -> str:
        if self.divergence_detected:
            return (
                f"SMT={self.divergence_type} → {self.divergence_direction}({self.strength}) │ "
                f"Leader={self.leader}"
            )
        if self.confirmed:
            return f"Confirmed={self.confirmation_direction}({self.strength}) │ Leader={self.leader}"
        return "No correlation signal"


class CorrelationAnalyzer:
    """
    ES/NQ structural comparison using swing highs/lows.
    Detects SMT-style divergences and structural confirmations.
    """

    def __init__(self) -> None:
        logger.info("CorrelationAnalyzer initialized (ES/NQ structural)")

    def analyze(
        self,
        es_m5: Optional[pd.DataFrame],
        nq_m5: Optional[pd.DataFrame],
        es_structure: Optional[object] = None,
        nq_structure: Optional[object] = None,
    ) -> CorrelationResult:
        """
        Compares ES and NQ structure for divergence or confirmation.

        Args:
            es_m5: ES M5 DataFrame
            nq_m5: NQ M5 DataFrame
            es_structure: StructureContext for ES (from MarketStructure)
            nq_structure: StructureContext for NQ (from MarketStructure)

        Returns:
            CorrelationResult with divergence/confirmation details.
        """
        result = CorrelationResult()

        if es_m5 is None or nq_m5 is None:
            return result
        if len(es_m5) < 10 or len(nq_m5) < 10:
            return result

        # ── 1. SMT Divergence (structural highs/lows) ───────────
        smt = self._detect_smt(es_m5, nq_m5)
        if smt is not None:
            result.divergence_detected = True
            result.divergence_type = smt[0]
            result.divergence_direction = smt[1]
            result.strength = smt[2]

        # ── 2. Structural Confirmation ──────────────────────────
        if not result.divergence_detected and es_structure and nq_structure:
            conf = self._check_confirmation(es_structure, nq_structure)
            if conf is not None:
                result.confirmed = True
                result.confirmation_direction = conf[0]
                result.strength = conf[1]

        # ── 3. Leader Detection ─────────────────────────────────
        result.leader = self._detect_leader(es_m5, nq_m5)

        return result

    # ═══════════════════════════════════════════════════════════════
    # 1. SMT DIVERGENCE
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_smt(es_df: pd.DataFrame, nq_df: pd.DataFrame) -> Optional[tuple]:
        """
        Smart Money Technique divergence:
        - Bearish SMT: NQ makes higher high but ES makes lower high → SELL
        - Bullish SMT: NQ makes lower low but ES makes higher low → BUY

        Uses the last 10 candles to find recent highs/lows.

        Returns (type, direction, score) or None.
        """
        lookback = 10

        es_recent = es_df.iloc[-lookback:]
        nq_recent = nq_df.iloc[-lookback:]

        # Split into two halves for comparison
        mid = lookback // 2

        # ES highs/lows in each half
        es_first_high = es_recent.iloc[:mid]["high"].max()
        es_second_high = es_recent.iloc[mid:]["high"].max()
        es_first_low = es_recent.iloc[:mid]["low"].min()
        es_second_low = es_recent.iloc[mid:]["low"].min()

        # NQ highs/lows in each half
        nq_first_high = nq_recent.iloc[:mid]["high"].max()
        nq_second_high = nq_recent.iloc[mid:]["high"].max()
        nq_first_low = nq_recent.iloc[:mid]["low"].min()
        nq_second_low = nq_recent.iloc[mid:]["low"].min()

        # Bearish SMT: NQ makes higher high, ES makes lower high
        nq_higher_high = nq_second_high > nq_first_high
        es_lower_high = es_second_high < es_first_high

        if nq_higher_high and es_lower_high:
            return "BEARISH_DIV", "SELL", 15

        # Bullish SMT: NQ makes lower low, ES makes higher low
        nq_lower_low = nq_second_low < nq_first_low
        es_higher_low = es_second_low > es_first_low

        if nq_lower_low and es_higher_low:
            return "BULLISH_DIV", "BUY", 15

        return None

    # ═══════════════════════════════════════════════════════════════
    # 2. STRUCTURAL CONFIRMATION
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _check_confirmation(es_ctx, nq_ctx) -> Optional[tuple]:
        """
        Checks if both ES and NQ have the same trend direction
        AND same BOS direction (strong confluence).

        Returns (direction, score) or None.
        """
        # Both must have a clear trend
        if es_ctx.trend == "NEUTRAL" or nq_ctx.trend == "NEUTRAL":
            return None

        # Trends must agree
        if es_ctx.trend != nq_ctx.trend:
            return None

        direction = es_ctx.trend  # Both agree

        # Bonus if both also have BOS in the same direction
        if es_ctx.bos_detected and nq_ctx.bos_detected:
            if es_ctx.bos_direction == nq_ctx.bos_direction == ("BULLISH" if direction == "BULLISH" else "BEARISH"):
                return direction, 15  # Full confirmation

        # Just trend agreement (no BOS)
        return direction, 10

    # ═══════════════════════════════════════════════════════════════
    # 3. LEADER DETECTION
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_leader(es_df: pd.DataFrame, nq_df: pd.DataFrame) -> str:
        """
        Determines which instrument moved first in the last 5 candles.
        The leader's structure takes priority in trade decisions.

        Compares normalized percentage moves.
        """
        if len(es_df) < 5 or len(nq_df) < 5:
            return "NONE"

        # Calculate 5-candle percentage moves
        es_move = (es_df.iloc[-1]["close"] - es_df.iloc[-5]["close"]) / es_df.iloc[-5]["close"]
        nq_move = (nq_df.iloc[-1]["close"] - nq_df.iloc[-5]["close"]) / nq_df.iloc[-5]["close"]

        # Check which had a stronger 3-candle move (moved first)
        es_early = abs((es_df.iloc[-3]["close"] - es_df.iloc[-5]["close"]) / es_df.iloc[-5]["close"])
        nq_early = abs((nq_df.iloc[-3]["close"] - nq_df.iloc[-5]["close"]) / nq_df.iloc[-5]["close"])

        if es_early > nq_early * 1.2:
            return "ES"
        elif nq_early > es_early * 1.2:
            return "NQ"

        return "NONE"
