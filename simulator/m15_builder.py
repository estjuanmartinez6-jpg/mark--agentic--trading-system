"""
simulator/m15_builder.py — NY-Clock-Aligned M15 Aggregation for MARK III Replay

CRITICAL REQUIREMENT (Req 2):
    M15 candles MUST align to :00, :15, :30, :45 minute boundaries
    in New York timezone — NOT simply "every 3 M5 candles".

    Example (correct):
        08:00, 08:05, 08:10 NY → M15 window [08:00-08:15) closes at 08:10+5min
        The M5 candle timestamped 08:10 NY represents 08:10–08:15.
        When it closes, the M15 for [08:00-08:15) is complete.

    Example (WRONG — naive 3-candle aggregation):
        If replay starts at 08:05, grouping (08:05, 08:10, 08:15) creates
        a fake M15 window [08:05-08:20) which the live system never saw.
        This introduces a subtle lookahead bias via M15 trend context.

Boundary logic:
    A M5 candle CLOSES an M15 window when its minute (in NY time) is:
        9, 24, 39, 54  (the END of the :00, :15, :30, :45 windows)
    
    MT5 candle timestamp = candle OPEN time.
    M5 at 08:10 NY opens at 08:10 and closes at 08:15.
    So minute 10 (open) = end of the 08:00-08:15 window.
    
    Equivalently:
        window_start_minute = (open_minute // 15) * 15
        The candle at open_minute (10) belongs to window starting at 0 (floor(10/15)*15=0 → 00).
        The LAST M5 in this window has open_minute = window_start + 10.
        i.e., minutes 0, 15, 30, 45 → last M5 open minutes 10, 25, 40, 55 (= window_start + 10).

OHLCV aggregation rules:
    open   = first M5 open in window
    high   = max(all M5 highs)
    low    = min(all M5 lows)
    close  = last M5 close
    tick_volume = sum(all M5 tick_volumes)
    spread = max(all M5 spreads)   ← conservative: worst spread seen
    time   = window START timestamp (e.g., 08:00 for the 08:00–08:15 window)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
import pytz

logger = logging.getLogger("Sim.M15Builder")

# New York timezone — all M15 boundary checks use this
NY_TZ = pytz.timezone("America/New_York")

# The open-minute of the LAST M5 candle in each 15-minute window
# Window :00 → M5 opens at :10 (and closes at :15, completing the window)
# Window :15 → M5 opens at :25, Window :30 → :40, Window :45 → :55
_CLOSING_MINUTES = frozenset({10, 25, 40, 55})


class M15Builder:
    """
    Builds M15 candles from M5 candles, strictly aligned to NY-clock boundaries.

    State machine:
        - accumulate() receives M5 candles one at a time
        - When an M5 candle closes its M15 window, a completed M15 is returned
        - All completed M15s are stored in an internal list
        - get_m15_dataframe() returns a view of all completed M15s

    Thread-safety: Not thread-safe. Single-threaded replay use only.
    """

    def __init__(self) -> None:
        self._completed: List[dict] = []   # Completed M15 candle rows
        self._window_candles: List[pd.Series] = []  # Current window accumulation
        self._current_window_start: Optional[pd.Timestamp] = None

    # ── Main interface ───────────────────────────────────────────────

    def add_m5_candle(self, row: pd.Series) -> Optional[dict]:
        """
        Accept one M5 candle row. Returns a completed M15 dict if the
        current window just closed, or None if still accumulating.

        Args:
            row: pd.Series with columns: time (UTC Timestamp), open, high,
                 low, close, tick_volume, spread (plus computed columns)

        Returns:
            Completed M15 candle as dict, or None.
        """
        ts_utc = row["time"]

        # Convert to NY time for boundary checks
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        ts_ny = ts_utc.astimezone(NY_TZ)

        # Determine which M15 window this M5 belongs to
        window_start_min = (ts_ny.minute // 15) * 15
        window_start_ny = ts_ny.replace(
            minute=window_start_min, second=0, microsecond=0
        )
        window_start_utc = window_start_ny.astimezone(pytz.utc)
        window_start_utc = pd.Timestamp(window_start_utc).tz_convert("UTC")

        # If this is a new window, start fresh accumulation
        if (
            self._current_window_start is None
            or window_start_utc != self._current_window_start
        ):
            # If we had a partial window accumulation and now jumped to a new
            # window, those partial candles are orphaned (replay started mid-window).
            # Discard them — do NOT create a partial/misaligned M15.
            if self._window_candles and self._current_window_start is not None:
                logger.debug(
                    f"[M15Builder] Discarding {len(self._window_candles)} orphaned M5 "
                    f"candles from partial window starting {self._current_window_start}"
                )
            self._window_candles = []
            self._current_window_start = window_start_utc

        self._window_candles.append(row)

        # Check if this M5 closes the current M15 window
        if ts_ny.minute in _CLOSING_MINUTES:
            m15 = self._aggregate_window(self._current_window_start)
            self._completed.append(m15)
            # Reset for next window
            self._window_candles = []
            self._current_window_start = None
            return m15

        return None

    # ── Output ──────────────────────────────────────────────────────

    def get_m15_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all completed M15 candles.

        This is a snapshot — called by DataFeed to provide
        M15 context to the SignalScorer.

        Returns empty DataFrame with correct columns if no M15s yet.
        """
        if not self._completed:
            return pd.DataFrame(columns=[
                "time", "open", "high", "low", "close",
                "tick_volume", "spread"
            ])
        return pd.DataFrame(self._completed)

    @property
    def completed_count(self) -> int:
        """Number of completed M15 candles."""
        return len(self._completed)

    def reset(self) -> None:
        """Clear all state. Called between replay sessions."""
        self._completed.clear()
        self._window_candles.clear()
        self._current_window_start = None

    # ── Private ─────────────────────────────────────────────────────

    def _aggregate_window(self, window_start: pd.Timestamp) -> dict:
        """
        Aggregates the accumulated M5 candles into one M15 candle.
        The window always contains exactly 3 M5 candles, but may contain
        fewer if the replay started mid-window (those are discarded above).
        """
        candles = self._window_candles

        if not candles:
            raise RuntimeError("Attempted to aggregate empty window")

        if len(candles) < 3:
            logger.debug(
                f"[M15Builder] Window at {window_start} has only "
                f"{len(candles)}/3 M5 candles (replay start boundary)"
            )

        # OHLCV aggregation
        m15 = {
            "time":        window_start,
            "open":        float(candles[0]["open"]),
            "high":        float(max(c["high"] for c in candles)),
            "low":         float(min(c["low"]  for c in candles)),
            "close":       float(candles[-1]["close"]),
            "tick_volume": int(sum(c.get("tick_volume", 0) for c in candles)),
            "spread":      float(max(c.get("spread", 0) for c in candles)),
        }

        logger.debug(
            f"[M15Builder] M15 built: {window_start.strftime('%H:%M')} UTC "
            f"O={m15['open']:.2f} H={m15['high']:.2f} "
            f"L={m15['low']:.2f} C={m15['close']:.2f} "
            f"({len(candles)} M5 candles)"
        )
        return m15
