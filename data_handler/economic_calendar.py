"""
data/economic_calendar.py — High-impact economic event awareness (v2)

Fetches the weekly economic calendar from ForexFactory's public JSON endpoint,
classifies events by tier (Tier 1 / Tier 2 / Tier 3), and provides methods
to query upcoming and recently-released events.

v2 changes:
  - EventTier enum for classification
  - Tier-based filtering (only Tier 1 by default)
  - Improved numeric parser (handles negatives, edge cases)
  - get_tradeable_events() method
  - Better separation of filtering logic

Data source: https://nfs.faireconomy.media/ff_calendar_thisweek.json
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import requests

from core import config
from core.logger import get_logger

logger = get_logger("EconCalendar")

# ForexFactory public JSON (unofficial but widely used and reliable)
_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class EventTier(str, Enum):
    """Classification of economic events by market impact."""
    TIER_1 = "TIER_1"  # NFP, CPI, FOMC, Interest Rate, Core PCE
    TIER_2 = "TIER_2"  # GDP, Retail Sales, PPI, Unemployment
    TIER_3 = "TIER_3"  # ISM, ADP, Jobless Claims, Trade Balance
    UNKNOWN = "UNKNOWN"


def classify_event_tier(name: str) -> EventTier:
    """Classify an event by its name into a tier."""
    name_lower = name.lower()
    for kw in config.TIER_1_KEYWORDS:
        if kw.lower() in name_lower:
            return EventTier.TIER_1
    for kw in config.TIER_2_KEYWORDS:
        if kw.lower() in name_lower:
            return EventTier.TIER_2
    for kw in config.TIER_3_KEYWORDS:
        if kw.lower() in name_lower:
            return EventTier.TIER_3
    return EventTier.UNKNOWN


@dataclass
class EconomicEvent:
    """Represents a single economic calendar event."""
    name: str
    time: datetime           # UTC datetime of the event
    impact: str              # "High", "Medium", "Low"
    currency: str            # "USD", "EUR", etc.
    forecast: Optional[str] = None
    actual: Optional[str] = None
    previous: Optional[str] = None

    @property
    def tier(self) -> EventTier:
        return classify_event_tier(self.name)

    @property
    def has_actual(self) -> bool:
        return self.actual is not None and self.actual.strip() != ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def is_tradeable(self) -> bool:
        """Check if this event qualifies for trading (Tier 1, or Tier 2 if enabled)."""
        if self.currency != "USD":
            return False
        tier = self.tier
        if tier == EventTier.TIER_1:
            return True
        if tier == EventTier.TIER_2 and config.ENABLE_TIER_2_EVENTS:
            return True
        return False

    def forecast_numeric(self) -> Optional[float]:
        """Try to parse forecast as a number (strip %, K, M, B suffixes)."""
        return self._parse_numeric(self.forecast)

    def actual_numeric(self) -> Optional[float]:
        """Try to parse actual as a number."""
        return self._parse_numeric(self.actual)

    def previous_numeric(self) -> Optional[float]:
        """Try to parse previous as a number."""
        return self._parse_numeric(self.previous)

    @staticmethod
    def _parse_numeric(val: Optional[str]) -> Optional[float]:
        """
        Robust numeric parser for economic data values.
        Handles: percentages, K/M/B suffixes, negative numbers, commas.
        """
        if not val:
            return None
        cleaned = val.strip().replace(",", "")

        # Handle empty or non-numeric
        if not cleaned:
            return None

        multiplier = 1.0
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        elif cleaned.upper().endswith("K"):
            cleaned = cleaned[:-1]
            multiplier = 1_000
        elif cleaned.upper().endswith("M"):
            cleaned = cleaned[:-1]
            multiplier = 1_000_000
        elif cleaned.upper().endswith("B"):
            cleaned = cleaned[:-1]
            multiplier = 1_000_000_000

        # Handle negative numbers and edge cases
        cleaned = cleaned.strip()
        if not cleaned or cleaned in ("-", "+", "."):
            return None

        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None


class EconomicCalendar:
    """
    Fetches and caches the weekly economic calendar.
    Provides methods to query upcoming and recently-released events.

    v2: Events are classified by tier. Only tradeable events
    (Tier 1, optionally Tier 2) are returned by trading queries.
    """

    def __init__(self) -> None:
        self._events: List[EconomicEvent] = []
        self._last_fetch: float = 0.0
        self._fetch_interval: float = config.CALENDAR_REFRESH_SEC
        self._consecutive_errors: int = 0  # suppress log spam

    def refresh(self, force: bool = False) -> None:
        """Fetch calendar data if cache has expired (or force=True)."""
        now = time.time()
        if not force and (now - self._last_fetch) < self._fetch_interval:
            return

        events, loaded_from_cache = self._fetch_from_ff()
        if events:
            self._events = events
            self._last_fetch = now  # reset timer whether live or cache
            self._consecutive_errors = 0

            if not loaded_from_cache:
                # Log tier breakdown only on live fetch
                t1 = [e for e in events if e.tier == EventTier.TIER_1]
                t2 = [e for e in events if e.tier == EventTier.TIER_2]
                t3 = [e for e in events if e.tier == EventTier.TIER_3]
                logger.info(
                    f"📅 Calendar refreshed: {len(events)} events | "
                    f"Tier 1: {len(t1)} | Tier 2: {len(t2)} | Tier 3: {len(t3)}"
                )
            else:
                logger.info("📅 Calendar loaded from disk cache (API unavailable)")
        else:
            self._consecutive_errors += 1
            # Only warn on first failure and every 10th after, to avoid log spam
            if self._consecutive_errors == 1 or self._consecutive_errors % 10 == 0:
                logger.warning(
                    f"Calendar unavailable (attempt {self._consecutive_errors}), "
                    "keeping previous data. Will retry next cycle."
                )
            # Still update _last_fetch so we wait a full interval before retrying
            self._last_fetch = now

    def get_upcoming_events(
        self,
        minutes_ahead: int = 25,
        tradeable_only: bool = True,
    ) -> List[EconomicEvent]:
        """Return tradeable events happening within the next `minutes_ahead` minutes."""
        self.refresh()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=minutes_ahead)

        results = []
        for ev in self._events:
            if tradeable_only and not ev.is_tradeable:
                continue
            if now <= ev.time <= cutoff:
                results.append(ev)

        return results

    def get_recent_events(
        self,
        minutes_back: int = 30,
        tradeable_only: bool = True,
    ) -> List[EconomicEvent]:
        """Return tradeable events released within the last `minutes_back` minutes."""
        self.refresh()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=minutes_back)

        results = []
        for ev in self._events:
            if tradeable_only and not ev.is_tradeable:
                continue
            if cutoff <= ev.time <= now:
                results.append(ev)

        return results

    def get_next_event(
        self, tradeable_only: bool = True,
    ) -> Optional[EconomicEvent]:
        """Return the next upcoming tradeable event (if any)."""
        self.refresh()
        now = datetime.now(timezone.utc)

        best = None
        for ev in self._events:
            if tradeable_only and not ev.is_tradeable:
                continue
            if ev.time > now:
                if best is None or ev.time < best.time:
                    best = ev

        return best

    def minutes_to_next_event(
        self, tradeable_only: bool = True,
    ) -> Optional[float]:
        """Return minutes until the next tradeable event, or None if none upcoming."""
        ev = self.get_next_event(tradeable_only)
        if ev is None:
            return None
        delta = (ev.time - datetime.now(timezone.utc)).total_seconds() / 60.0
        return max(0.0, delta)

    # ─── Data Fetching ──────────────────────────────────────────────
    def _fetch_from_ff(self):
        """
        Fetch from ForexFactory JSON endpoint with local file fallback.
        Returns (list[EconomicEvent], loaded_from_cache: bool).
        """
        import json
        cache_file = config.ROOT_DIR / "data" / "ff_cache.json"
        data = None
        loaded_from_cache = False

        try:
            resp = requests.get(_FF_CALENDAR_URL, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; MARK-II/2.0)",
            })
            resp.raise_for_status()
            data = resp.json()

            # Persist to cache for future fallbacks
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Could not write calendar cache file: {e}")

        except requests.RequestException as exc:
            logger.error(f"ForexFactory calendar fetch failed: {exc}")
            # Try to load from disk cache
            if cache_file.exists():
                logger.info("Loading calendar from local disk cache…")
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_from_cache = True
                except Exception as e:
                    logger.error(f"Failed to read calendar cache: {e}")

            if not data:
                return [], False

        try:
            events = []
            for item in data:
                country = item.get("country", "")
                if country != "USD":
                    continue

                title = item.get("title", "")
                impact = item.get("impact", "Low")

                # Parse datetime
                dt_str = item.get("date", "")
                ev_time = self._parse_ff_datetime(dt_str)
                if ev_time is None:
                    continue

                event = EconomicEvent(
                    name=title,
                    time=ev_time,
                    impact=impact if impact else "Medium",
                    currency=country,
                    forecast=item.get("forecast"),
                    actual=item.get("actual"),
                    previous=item.get("previous"),
                )

                # Only keep events with a recognised tier
                if event.tier != EventTier.UNKNOWN:
                    events.append(event)

            return events, loaded_from_cache

        except Exception as exc:
            logger.error(f"Error parsing calendar data: {exc}")
            return [], loaded_from_cache

    @staticmethod
    def _parse_ff_datetime(dt_str: str) -> Optional[datetime]:
        """Parse ForexFactory datetime string to UTC datetime."""
        if not dt_str:
            return None
        try:
            # FF uses format like "2024-01-05T13:30:00-05:00"
            dt = datetime.fromisoformat(dt_str)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass

        # Fallback: try common formats
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(dt_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue

        return None
