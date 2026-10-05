"""
simulator/simulation_clock.py — Isolated Simulation Time for MARK III Replay

CRITICAL DESIGN PRINCIPLE:
    Do NOT globally monkeypatch time.time() application-wide.
    Global patching causes race conditions in logging, threading,
    and any other module that depends on wall-clock time.

Instead, this module provides:
  1. SimulationClock — tracks the current simulated timestamp
  2. patch_scorer_context() — a context manager that temporarily
     patches time.time ONLY within the signal_engine.scorer module,
     scoped to a single with-block, thread-safe via unittest.mock.patch.

Usage in StrategyAdapter:
    clock = SimulationClock()
    clock.advance(candle_timestamp)

    with clock.patch_scorer_context():
        signals = scorer.evaluate(market_data)
    # time.time() is fully restored here - no global impact
"""
from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import patch

logger = logging.getLogger("Sim.Clock")


class SimulationClock:
    """
    Maintains the current simulated time for the replay session.

    The clock advances one M5 candle at a time, always using the
    candle's timestamp as the simulated 'now'.

    Thread-safety note: patch_scorer_context() uses unittest.mock.patch
    which is safe within a single thread. Do not call from multiple
    threads simultaneously.
    """

    def __init__(self, start_timestamp: float = 0.0) -> None:
        """
        Args:
            start_timestamp: Initial simulated Unix timestamp.
                             0.0 means uninitialized (advance() must be
                             called before patch_scorer_context()).
        """
        self._simulated_ts: float = start_timestamp
        self._wall_start: float = time.time()
        self._advance_count: int = 0

    # ── Clock control ────────────────────────────────────────────────

    def advance(self, timestamp: float) -> None:
        """
        Set the simulated clock to the given Unix timestamp.
        Called once per M5 candle, using candle['time'].timestamp().

        Args:
            timestamp: Unix timestamp (float seconds since epoch UTC)
        """
        if timestamp <= 0:
            raise ValueError(f"Invalid simulated timestamp: {timestamp}")
        if timestamp < self._simulated_ts:
            logger.warning(
                f"[Clock] Time moved backwards: "
                f"{self._simulated_ts:.0f} → {timestamp:.0f}. "
                f"This should not happen in a forward replay."
            )
        self._simulated_ts = timestamp
        self._advance_count += 1

    def now(self) -> float:
        """Returns the current simulated Unix timestamp."""
        if self._simulated_ts == 0.0:
            raise RuntimeError(
                "SimulationClock not initialized. "
                "Call advance() with a candle timestamp before using now()."
            )
        return self._simulated_ts

    def now_dt(self) -> datetime:
        """Returns the current simulated time as a UTC datetime."""
        return datetime.fromtimestamp(self._simulated_ts, tz=timezone.utc)

    def now_str(self) -> str:
        """Returns simulated time as ISO 8601 string (UTC)."""
        return self.now_dt().isoformat()

    # ── Scorer clock patching ────────────────────────────────────────

    @contextmanager
    def patch_scorer_context(self) -> Generator:
        """
        Temporarily replaces time.time() within the signal_engine.scorer
        module with a function that returns the simulated timestamp.

        Scope: ONLY signal_engine.scorer — no other modules are affected.
        Duration: Only for the duration of the with-block.
        Thread-safety: Safe for single-threaded replay (standard use case).

        Why scorer only?
            SignalScorer uses time.time() in two places:
            1. _is_on_cooldown() — compares last signal time to now
            2. _mark_signal_fired() — records signal time
            Both must use simulated time so that cooldowns are relative
            to replay time, not wall-clock time.

        All other modules (logger, health monitor, risk manager) are
        called with explicit simulated timestamps via adapter methods,
        so they do not need time.time() patching.

        Usage:
            with clock.patch_scorer_context():
                signals = scorer.evaluate(market_data)
            # time.time() fully restored here
        """
        if self._simulated_ts == 0.0:
            raise RuntimeError("Clock not advanced before patch_scorer_context()")

        sim_ts = self._simulated_ts

        def _simulated_time() -> float:
            return sim_ts

        # Patch only the scorer's reference to time.time
        # This leaves global time.time() and all other module references intact
        with patch("signal_engine.scorer.time.time", side_effect=_simulated_time):
            yield

    # ── Properties ──────────────────────────────────────────────────

    @property
    def simulated_timestamp(self) -> float:
        return self._simulated_ts

    @property
    def advance_count(self) -> int:
        """Number of times advance() has been called (= candles processed)."""
        return self._advance_count

    @property
    def elapsed_wall_seconds(self) -> float:
        """Wall-clock seconds since this clock was created."""
        return time.time() - self._wall_start

    def summary(self) -> dict:
        return {
            "simulated_ts": self._simulated_ts,
            "simulated_dt": self.now_str() if self._simulated_ts > 0 else "uninitialized",
            "advance_count": self._advance_count,
            "wall_elapsed_sec": round(self.elapsed_wall_seconds, 2),
        }
