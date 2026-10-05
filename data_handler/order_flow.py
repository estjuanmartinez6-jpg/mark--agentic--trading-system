"""
data_handler/order_flow.py — Order Flow Data Provider (Pluggable Architecture)

This module defines the abstract interface for order flow data and provides
a mock implementation for development/testing. The mock generates realistic
correlated ES/NQ data with proper delta simulation.

To connect to a real data source (Rithmic, CQG, dxFeed), simply create a new
class that inherits from OrderFlowProviderBase and implements fetch_latest_candles().
"""
from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from monitoring.logger import get_logger

logger = get_logger("OrderFlow")


# ─── Data Model ─────────────────────────────────────────────────
@dataclass
class Candle:
    """Represents a single M5 candle with order flow data."""
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_volume: int
    sell_volume: int
    delta: int              # buy_volume - sell_volume
    cumulative_delta: float # Running sum of delta


# ─── Abstract Base Class ────────────────────────────────────────
class OrderFlowProviderBase(ABC):
    """
    Interface for order flow data providers.
    Swap this out with a real provider (Rithmic, CQG) later.
    """

    @abstractmethod
    def fetch_latest_candles(self, lookback: int = 20) -> Dict[str, List[Candle]]:
        """Returns {symbol: [Candle, ...]} for all tracked symbols."""
        ...

    def get_cached_candles(self) -> Dict[str, List[Candle]]:
        """Returns the last fetched candles without generating new ones."""
        return {}

    @abstractmethod
    def reset(self) -> None:
        """Clears all cached history."""
        ...


# ─── Mock Implementation ────────────────────────────────────────
class MockOrderFlowProvider(OrderFlowProviderBase):
    """
    Simulates a realistic data feed for ES and NQ futures with:
    - Correlated price movements (ES and NQ tend to move together)
    - Realistic volume profiles
    - Delta that generally follows price but with divergence scenarios
    - Cumulative delta tracking
    """

    # Realistic base params per symbol
    PROFILES = {
        "ES": {"base_price": 5900.0, "volatility": 4.0, "avg_volume": 3000},
        "NQ": {"base_price": 21000.0, "volatility": 12.0, "avg_volume": 2500},
    }

    # ES→NQ correlation coefficient (realistic: ~0.85-0.95)
    CORRELATION: float = 0.90

    def __init__(self):
        self.current_prices: Dict[str, float] = {
            sym: p["base_price"] for sym, p in self.PROFILES.items()
        }
        self.history: Dict[str, List[Candle]] = {"ES": [], "NQ": []}
        self.cumulative_deltas: Dict[str, float] = {"ES": 0.0, "NQ": 0.0}
        self._candle_count: int = 0
        logger.info(
            f"MockOrderFlowProvider initialized │ "
            f"ES={self.current_prices['ES']:.0f} │ NQ={self.current_prices['NQ']:.0f} │ "
            f"Correlation={self.CORRELATION}"
        )

    def reset(self) -> None:
        self.history = {"ES": [], "NQ": []}
        self.cumulative_deltas = {"ES": 0.0, "NQ": 0.0}
        self._candle_count = 0

    def get_cached_candles(self) -> Dict[str, List[Candle]]:
        """Returns previously generated candles without creating new ones."""
        return self.history

    def _generate_correlated_pair(self) -> Dict[str, Candle]:
        """
        Generates one ES and one NQ candle with correlated price movements.
        The NQ candle is influenced by the ES candle's direction.
        """
        self._candle_count += 1
        now = time.time()

        # ── Step 1: Generate the ES candle (leader) ──────────────
        es_profile = self.PROFILES["ES"]
        es_base = self.current_prices["ES"]

        # Random walk for ES
        es_move = random.gauss(0, es_profile["volatility"])
        es_open = es_base
        es_close = es_base + es_move
        es_high = max(es_open, es_close) + abs(random.gauss(0, es_profile["volatility"] * 0.3))
        es_low = min(es_open, es_close) - abs(random.gauss(0, es_profile["volatility"] * 0.3))
        self.current_prices["ES"] = es_close

        # ── Step 2: Generate NQ candle (correlated follower) ─────
        nq_profile = self.PROFILES["NQ"]
        nq_base = self.current_prices["NQ"]

        # Correlated component: NQ follows ES direction, scaled by its own volatility
        es_direction = 1.0 if es_move > 0 else -1.0
        correlated_move = es_direction * abs(random.gauss(0, nq_profile["volatility"])) * self.CORRELATION
        independent_move = random.gauss(0, nq_profile["volatility"]) * (1 - self.CORRELATION)
        nq_move = correlated_move + independent_move

        nq_open = nq_base
        nq_close = nq_base + nq_move
        nq_high = max(nq_open, nq_close) + abs(random.gauss(0, nq_profile["volatility"] * 0.3))
        nq_low = min(nq_open, nq_close) - abs(random.gauss(0, nq_profile["volatility"] * 0.3))
        self.current_prices["NQ"] = nq_close

        # ── Step 3: Generate volume & delta for both ─────────────
        result = {}
        for sym, o, h, l, c, profile in [
            ("ES", es_open, es_high, es_low, es_close, es_profile),
            ("NQ", nq_open, nq_high, nq_low, nq_close, nq_profile),
        ]:
            volume = max(500, int(random.gauss(profile["avg_volume"], profile["avg_volume"] * 0.3)))
            is_up = c > o

            # Delta correlates with price movement ~80% of the time
            # 20% chance of generating a divergence scenario
            is_diverging = random.random() < 0.18

            if is_up:
                buy_pct = random.uniform(0.55, 0.75) if not is_diverging else random.uniform(0.30, 0.45)
            else:
                buy_pct = random.uniform(0.25, 0.45) if not is_diverging else random.uniform(0.55, 0.70)

            buy_volume = int(volume * buy_pct)
            sell_volume = volume - buy_volume
            delta = buy_volume - sell_volume

            self.cumulative_deltas[sym] += delta

            result[sym] = Candle(
                symbol=sym,
                timestamp=now,
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=volume,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                delta=delta,
                cumulative_delta=self.cumulative_deltas[sym],
            )

        return result

    def fetch_latest_candles(self, lookback: int = 20) -> Dict[str, List[Candle]]:
        """
        Fetches the latest candles for both ES and NQ.
        On first call, generates `lookback` historical candles.
        On subsequent calls, appends one new candle pair.
        """
        if not self.history["ES"]:
            # Bootstrap history with correlated pairs
            logger.info(f"Bootstrapping {lookback} historical candles...")
            for _ in range(lookback):
                pair = self._generate_correlated_pair()
                for sym, candle in pair.items():
                    self.history[sym].append(candle)
        else:
            # Add new candle pair
            pair = self._generate_correlated_pair()
            for sym, candle in pair.items():
                self.history[sym].append(candle)
                # Keep rolling window
                if len(self.history[sym]) > lookback:
                    self.history[sym] = self.history[sym][-lookback:]

        return self.history
