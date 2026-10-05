"""
simulator/data_feed.py — Progressive Event-Emitting Candle Feed for MARK III Replay

The core anti-leakage mechanism. Ensures the strategy can only ever
see candles at or before the current replay index — exactly as in live trading.

Anti-leakage guarantee:
    At step i, m5_snapshot = source_df.iloc[0 : warmup_bars + i + 1]
    No future candle (index >= warmup_bars + i + 1) is ever accessible.

Memory safety (Req 8):
    source_df is an immutable reference; DataFeed never copies it in full.
    iloc slices are pandas VIEWS (not copies) for non-fancy indexing.
    The IndicatorEngine receives the view, makes a copy internally,
    and returns a fresh annotated DataFrame.
    M15 candles accumulate row-by-row in M15Builder's list (no repeated concat).

Immutability (Req 7):
    DataFeed does not modify source_df at any point.
    advance() operates on iloc views exclusively.

Event-driven design (Req 14):
    DataFeed does not call the strategy directly.
    Instead, advance() emits "new_candle" via EventBus.
    The StrategyAdapter subscribes and reacts — decoupled.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from simulator.event_bus import EventBus
from simulator.indicators import IndicatorEngine
from simulator.m15_builder import M15Builder
from simulator.simulation_clock import SimulationClock

logger = logging.getLogger("Sim.DataFeed")

# Minimum M5 candles needed before strategy sees any data
# Matches live CANDLE_HISTORY_M5 setting (default 50)
DEFAULT_WARMUP_BARS = 50


class DataFeed:
    """
    Streams M5 candles progressively from an immutable historical DataFrame.

    On each advance() call:
      1. Increments the replay index by 1
      2. Provides an iloc slice of M5 data [0..current_index]
      3. Runs IndicatorEngine on that slice
      4. Passes the M5 candle to M15Builder (checks for window completion)
      5. Emits "new_candle" via EventBus with both M5 and M15 snapshots
      6. Advances the SimulationClock to this candle's timestamp

    The warmup period pre-loads `warmup_bars` candles so that EMAs/ATR
    are initialized before the strategy starts evaluating. These warm-up
    candles are NOT emitted as events — they prime the M15Builder and
    ensure indicator values are stable at replay start.
    """

    def __init__(
        self,
        source_m5: pd.DataFrame,
        indicator_engine: IndicatorEngine,
        m15_builder: M15Builder,
        sim_clock: SimulationClock,
        event_bus: EventBus,
        warmup_bars: int = DEFAULT_WARMUP_BARS,
        replay_start_idx: int = 0,  # Index in source_m5 where ACTIVE replay begins
    ) -> None:
        """
        Args:
            source_m5:       Full immutable historical M5 DataFrame.
                             Must include 'warmup_bars' candles BEFORE
                             the actual replay session start.
            indicator_engine: Computes EMA/ATR on each slice.
            m15_builder:      Accumulates M15 candles from M5 stream.
            sim_clock:        Advances to each candle's timestamp.
            event_bus:        Emits "new_candle" events.
            warmup_bars:      Number of M5 candles to pre-process before
                              emitting events (primes indicators/M15).
            replay_start_idx: Index in source_m5 where warmup begins.
                              (Session controller sets this after adding
                               warmup candles to the front of the DataFrame.)
        """
        if source_m5 is None or len(source_m5) == 0:
            raise ValueError("source_m5 DataFrame is empty or None")
        if len(source_m5) <= warmup_bars:
            raise ValueError(
                f"source_m5 has {len(source_m5)} rows but needs > {warmup_bars} "
                f"(warmup_bars). Add more history before the session start."
            )

        # Store reference (NOT copy) to immutable source
        self._src = source_m5
        self._indicator_engine = indicator_engine
        self._m15_builder = m15_builder
        self._clock = sim_clock
        self._bus = event_bus
        self._warmup_bars = warmup_bars

        # Replay state
        # After warmup, _replay_idx starts at warmup_bars and runs to len(source_m5)-1
        self._current_idx: int = warmup_bars - 1   # Will be incremented on first advance()
        self._replay_candle_count: int = 0          # Candles emitted AFTER warmup
        self._exhausted: bool = False

        # Pre-process warmup candles (prime M15Builder but don't emit events)
        self._run_warmup()

    # ── Warmup ──────────────────────────────────────────────────────

    def _run_warmup(self) -> None:
        """
        Pre-processes the first `warmup_bars` candles.
        Primes M15Builder so completed M15 candles are available
        at replay start. Does NOT emit any EventBus events.
        """
        logger.info(
            f"[DataFeed] Running warmup: {self._warmup_bars} M5 candles "
            f"(no events emitted during warmup)"
        )
        for i in range(self._warmup_bars):
            row = self._src.iloc[i]
            self._m15_builder.add_m5_candle(row)

        m15_ready = self._m15_builder.completed_count
        logger.info(
            f"[DataFeed] Warmup complete. M15s built during warmup: {m15_ready}. "
            f"Replay begins at index {self._warmup_bars} of source DataFrame."
        )

    # ── Main interface ───────────────────────────────────────────────

    def advance(self) -> bool:
        """
        Advance replay by one M5 candle.

        Returns:
            True  — candle was processed and event was emitted
            False — no more candles (replay exhausted)

        The "new_candle" event is emitted with:
            timestamp     : pd.Timestamp (UTC) of this candle's open time
            candle_index  : int replay candle number (0-based, post-warmup)
            m5_snapshot   : pd.DataFrame — M5 candles [0..current_idx] with indicators
            m15_snapshot  : pd.DataFrame — all completed M15 candles so far
            raw_candle    : pd.Series — the current M5 candle row (raw, no indicators)
        """
        if self._exhausted:
            return False

        next_idx = self._current_idx + 1

        if next_idx >= len(self._src):
            self._exhausted = True
            logger.info(
                f"[DataFeed] All {self._replay_candle_count} replay candles processed."
            )
            return False

        self._current_idx = next_idx
        row = self._src.iloc[self._current_idx]
        ts = row["time"]

        # Advance simulation clock to this candle's timestamp
        self._clock.advance(ts.timestamp())

        # Build M15 snapshot (check if this M5 completes a window)
        self._m15_builder.add_m5_candle(row)
        m15_snapshot = self._m15_builder.get_m15_dataframe()

        # Compute indicators on visible M5 slice [0 .. current_idx]
        # This is a VIEW (not a copy) — IndicatorEngine makes its own copy internally
        m5_visible_slice = self._src.iloc[0 : self._current_idx + 1]
        m5_with_indicators = self._indicator_engine.compute(m5_visible_slice)

        # Apply indicators to completed M15 candles as well
        m15_with_indicators = pd.DataFrame()
        if not m15_snapshot.empty:
            m15_with_indicators = self._indicator_engine.compute(m15_snapshot)

        self._replay_candle_count += 1

        # Emit event — all subscribers react independently
        self._bus.emit(
            EventBus.NEW_CANDLE,
            timestamp=ts,
            candle_index=self._replay_candle_count - 1,
            m5_snapshot=m5_with_indicators,
            m15_snapshot=m15_with_indicators,
            raw_candle=row,
        )

        return True

    # ── Properties and state ────────────────────────────────────────

    @property
    def current_timestamp(self) -> Optional[pd.Timestamp]:
        """Simulated current time (the last processed candle's open time)."""
        if self._current_idx < self._warmup_bars:
            return None
        return self._src.iloc[self._current_idx]["time"]

    @property
    def candle_index(self) -> int:
        """Zero-based index of the current replay candle (post-warmup)."""
        return self._replay_candle_count

    @property
    def source_total_bars(self) -> int:
        """Total candles in source DataFrame (warmup + replay)."""
        return len(self._src)

    @property
    def replay_total_bars(self) -> int:
        """Total replay-phase candles (excluding warmup)."""
        return len(self._src) - self._warmup_bars

    @property
    def progress_pct(self) -> float:
        """How far through the replay we are (0.0 to 1.0)."""
        total = self.replay_total_bars
        if total <= 0:
            return 1.0
        return min(1.0, self._replay_candle_count / total)

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    def summary(self) -> dict:
        return {
            "warmup_bars": self._warmup_bars,
            "current_idx": self._current_idx,
            "replay_candle_count": self._replay_candle_count,
            "replay_total_bars": self.replay_total_bars,
            "progress_pct": round(self.progress_pct * 100, 1),
            "m15_built": self._m15_builder.completed_count,
            "exhausted": self._exhausted,
            "current_ts": str(self.current_timestamp),
        }
