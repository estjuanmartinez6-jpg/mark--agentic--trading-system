"""
simulator/session_controller.py — Trading Session & Event Window Management

Handles all date/time logic for replay sessions:
  - Date selection (single day or range)
  - Session window filtering (NY or Colombia local times)
  - Predefined macro-event windows (CPI, NFP, FOMC, etc.)
  - Timezone conversion (always UTC internally, local zone for display)
  - Warmup data window calculation

By default, custom session boundaries use NY time unless Colombia local time arguments are explicitly provided.
Macro-event windows still use NY timezone definitions for US economic events.

All returned datetimes are UTC-aware.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import pytz

logger = logging.getLogger("Sim.SessionController")

# ── Timezone constants ───────────────────────────────────────────────
UTC = timezone.utc
NY_TZ = pytz.timezone("America/New_York")
COL_TZ = pytz.timezone("America/Bogota")

# ── Predefined macro-event windows (in NY time HH:MM) ────────────────
# Format: (start_NY, end_NY)
PREDEFINED_EVENTS: dict[str, Tuple[str, str]] = {
    "CPI":       ("08:20", "10:00"),   # Consumer Price Index release
    "NFP":       ("08:20", "09:30"),   # Non-Farm Payrolls (first Friday)
    "FOMC":      ("13:50", "15:30"),   # Fed meeting / statement
    "FOMC_MINS": ("13:50", "15:30"),   # FOMC Minutes
    "OPEN":      ("09:25", "10:30"),   # Market open volatility window
    "CLOSE":     ("15:30", "16:15"),   # Market close window
    "FULL_DAY":  ("08:00", "16:00"),   # Full regular trading session
    "FULL":      ("08:00", "16:00"),   # Alias
    "PPI":       ("08:20", "10:00"),   # Producer Price Index
    "RETAIL":    ("08:20", "10:00"),   # Retail Sales
    "ISM":       ("09:55", "11:00"),   # ISM Manufacturing/Services
    "JOBS":      ("08:20", "10:00"),   # Weekly Jobless Claims
    "PCE":       ("08:20", "10:00"),   # PCE Price Index
    "GDP":       ("08:20", "10:00"),   # GDP release
}

# Warmup duration: fetch this many minutes of M5 data BEFORE session start
# 750 minutes = 150 M5 candles = 50 M15 candles (required for M15 EMA-50)
WARMUP_MINUTES = 750


class SessionController:
    """
    Controls which historical data window to replay.

    Usage examples:
        # Full day (default uses Colombia local time start)
        sc = SessionController(date="2026-05-22")

        # CPI event window (NY times, predefined)
        sc = SessionController(date="2026-05-22", event="CPI")

        # Custom window in NY time
        sc = SessionController(date="2026-05-22", start_ny="09:30", end_ny="14:00")

        # Custom window in Colombia local time
        sc = SessionController(date="2026-05-22", start_colombia="09:00", end_colombia="16:00")

        # Access the computed UTC windows
        warmup_start, session_end = sc.get_data_window()
    """

    def __init__(
        self,
        date: str,
        start_ny: str = None,
        end_ny: str = None,
        start_colombia: str = None,
        end_colombia: str = None,
        event: str = None,
    ) -> None:
        """
        Args:
            date:           Date string "YYYY-MM-DD"
            start_ny:       Session start in NY time "HH:MM" (overridden by event)
            end_ny:         Session end in NY time "HH:MM"   (overridden by event)
            start_colombia: Session start in Colombia local time "HH:MM"
            end_colombia:   Session end in Colombia local time "HH:MM"
            event:          Predefined event name (see PREDEFINED_EVENTS).
                            If provided, start/end times are derived from it.
        """
        self._date_str = date
        self._date_ny = self._parse_date_ny(date)

        # Resolve session window from macro-event if requested
        if event:
            event_key = event.upper()
            if event_key not in PREDEFINED_EVENTS:
                raise ValueError(
                    f"Unknown event '{event}'. "
                    f"Available: {sorted(PREDEFINED_EVENTS.keys())}"
                )
            start_ny, end_ny = PREDEFINED_EVENTS[event_key]
            logger.info(f"[SessionCtrl] Event '{event_key}': {start_ny}–{end_ny} NY")
            self._session_tz = NY_TZ
            self._session_start_str = start_ny
            self._session_end_str = end_ny
        else:
            if start_colombia is not None or end_colombia is not None:
                if start_ny is not None or end_ny is not None:
                    raise ValueError(
                        "Cannot mix NY and Colombia session window arguments"
                    )
                self._session_tz = COL_TZ
                self._session_start_str = start_colombia or "09:00"
                self._session_end_str = end_colombia or "16:00"
            elif start_ny is not None or end_ny is not None:
                self._session_tz = NY_TZ
                self._session_start_str = start_ny or "09:30"
                self._session_end_str = end_ny or "16:00"
            else:
                # Default: Use Colombia local time 04:00-17:00
                # This maps to ~09:00-22:00 UTC, matching the full window
                # the live bot trades in (settings.MARKET_HOURS_UTC ES: 0-22 UTC).
                # Previous default of NY 09:30-16:00 missed all pre-market
                # trades that the live bot regularly captures.
                self._session_tz = COL_TZ
                self._session_start_str = "04:00"
                self._session_end_str = "17:00"

        self._event = event

        # Compute UTC timestamps
        self._session_start_utc = self._tz_to_utc(date, self._session_start_str, self._session_tz)
        self._session_end_utc   = self._tz_to_utc(date, self._session_end_str, self._session_tz)
        
        # We need at least 150 M5 candles (50 M15s) to initialize EMA50.
        # Since CFDs have overnight gaps where no candles print, subtracting 750 
        # clock minutes is not enough. We instead fetch 3 days back to guarantee
        # there is enough actual market activity in the dataframe before the session.
        self._warmup_start_utc  = self._session_start_utc - timedelta(days=3)

        logger.info(
            f"[SessionCtrl] {date} | "
            f"Session: {self._session_start_str}–{self._session_end_str} "
            f"({self._session_tz.zone}) | "
            f"UTC: {self._session_start_utc.strftime('%H:%M')}–"
            f"{self._session_end_utc.strftime('%H:%M')} | "
            f"Warmup from: {self._warmup_start_utc.strftime('%Y-%m-%d %H:%M')} UTC"
        )

    # ── Public API ─────────────────────────────────────────────────

    def get_data_window(self) -> Tuple[datetime, datetime]:
        """
        Returns (warmup_start_utc, session_end_utc).
        Use this to query the DataLoader for the full data range needed.

        The warmup window precedes the session start by WARMUP_MINUTES,
        ensuring enough M5 history for EMA/ATR initialization.
        """
        return self._warmup_start_utc, self._session_end_utc

    def get_session_window(self) -> Tuple[datetime, datetime]:
        """
        Returns (session_start_utc, session_end_utc).
        The active replay window (events emitted only within this window).
        """
        return self._session_start_utc, self._session_end_utc

    def get_warmup_bar_count(self, source_df: pd.DataFrame) -> int:
        """
        Counts how many bars in source_df fall BEFORE session_start.
        These become the warmup bars passed to DataFeed.

        Warns if fewer than 50 warmup bars are available (indicator quality).

        Args:
            source_df: Normalized M5 DataFrame returned by a DataLoader.

        Returns:
            Number of warmup bars (may be less than 50 if history is limited).
        """
        if source_df.empty or "time" not in source_df.columns:
            return 0

        session_start_ts = pd.Timestamp(self._session_start_utc).tz_convert("UTC")
        warmup_mask = source_df["time"] < session_start_ts
        warmup_count = warmup_mask.sum()

        if warmup_count < 50:
            logger.warning(
                f"[SessionCtrl] Only {warmup_count} warmup bars available "
                f"(recommended: 50+). Indicator quality may be reduced for the "
                f"first few candles of the session."
            )
        else:
            logger.info(f"[SessionCtrl] {warmup_count} warmup bars available")

        return int(warmup_count)

    def is_in_session(self, ts: pd.Timestamp) -> bool:
        """
        Returns True if the given UTC timestamp falls within the active
        replay session window [session_start, session_end].
        """
        session_start_ts = pd.Timestamp(self._session_start_utc).tz_convert("UTC")
        session_end_ts   = pd.Timestamp(self._session_end_utc).tz_convert("UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return session_start_ts <= ts <= session_end_ts

    def as_ny_str(self, ts: pd.Timestamp) -> str:
        """Converts a UTC Timestamp to a NY time string for display."""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_ny = ts.astimezone(NY_TZ)
        return ts_ny.strftime("%Y-%m-%d %H:%M:%S %Z")

    # ── Properties ──────────────────────────────────────────────────

    @property
    def date_str(self) -> str:
        return self._date_str

    @property
    def event_name(self) -> Optional[str]:
        return self._event

    @property
    def session_start_utc(self) -> datetime:
        return self._session_start_utc

    @property
    def session_end_utc(self) -> datetime:
        return self._session_end_utc

    @property
    def warmup_start_utc(self) -> datetime:
        return self._warmup_start_utc

    def summary(self) -> dict:
        return {
            "date": self._date_str,
            "event": self._event,
            "session_start": self._session_start_str,
            "session_end": self._session_end_str,
            "session_time_zone": self._session_tz.zone,
            "session_start_utc": self._session_start_utc.isoformat(),
            "session_end_utc": self._session_end_utc.isoformat(),
            "warmup_start_utc": self._warmup_start_utc.isoformat(),
            "warmup_minutes": WARMUP_MINUTES,
        }

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_date_ny(date_str: str) -> datetime:
        """Parse 'YYYY-MM-DD' into a naive date."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format '{date_str}'. Expected 'YYYY-MM-DD'."
            )

    @staticmethod
    def _tz_to_utc(date_str: str, time_str: str, tz: pytz.BaseTzInfo) -> datetime:
        """
        Convert a local time string "HH:MM" on date_str in the provided timezone
        to a UTC-aware datetime. Handles DST automatically via pytz localize.
        """
        dt_naive = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_local = tz.localize(dt_naive)  # Handles DST correctly
        dt_utc = dt_local.astimezone(pytz.utc)
        return dt_utc.replace(tzinfo=timezone.utc)
