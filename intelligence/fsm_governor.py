"""
core/state_machine.py — 6-state finite state machine for MARK II v2

States:
  IDLE            — No nearby events. Bot watches calendar. NO trading.
  PRE_EVENT       — High-impact event within 12 min: flatten positions, block entries.
  LOCKDOWN        — Event release ±3 min: total silence, no actions.
  POST_EVENT      — 3-20 min after event: THE ONLY TRADING WINDOW.
  COOLDOWN        — 5 min rest after POST_EVENT expires or trade taken.
  SESSION_CLOSED  — Outside US trading hours. Everything off.

Transition Logic (evaluated every loop cycle):
  SESSION_CLOSED takes highest priority (outside hours = nothing runs).
  Then event proximity is evaluated against upcoming/recent events.

v2 changes over v1:
  - NORMAL state eliminated (replaced by IDLE — no trading in idle)
  - LOCKDOWN replaces NEWS with explicit naming
  - POST_EVENT is the ONLY state where trades can be taken
  - COOLDOWN prevents re-entry after a window expires
  - SESSION_CLOSED enforces hard trading hours
  - Only Tier 1 events (optionally Tier 2) trigger transitions
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

from core import config
from core.logger import get_logger

logger = get_logger("StateMachine")


class State(str, Enum):
    IDLE = "IDLE"
    PRE_EVENT = "PRE_EVENT"
    LOCKDOWN = "LOCKDOWN"
    POST_EVENT = "POST_EVENT"
    COOLDOWN = "COOLDOWN"
    MARKET_OPEN_BLOCK = "MARKET_OPEN_BLOCK"


# Behavior descriptors for each state
_STATE_BEHAVIORS = {
    State.IDLE: {
        "allow_new_trades": False,
        "close_on_enter": False,
        "description": "No nearby events — watching calendar only",
    },
    State.PRE_EVENT: {
        "allow_new_trades": False,
        "close_on_enter": True,
        "description": "High-impact event approaching — flatten exposure",
    },
    State.LOCKDOWN: {
        "allow_new_trades": False,
        "close_on_enter": False,
        "description": "Event release window — total silence",
    },
    State.POST_EVENT: {
        "allow_new_trades": True,
        "close_on_enter": False,
        "description": "Post-event continuation — TRADING WINDOW",
    },
    State.COOLDOWN: {
        "allow_new_trades": False,
        "close_on_enter": False,
        "description": "Post-window rest — no re-entry",
    },
    State.MARKET_OPEN_BLOCK: {
        "allow_new_trades": False,
        "close_on_enter": False,
        "description": "Volatility safety block at market open",
    },
}


class StateMachine:
    """
    Evaluates the current state based on proximity to economic events
    and trading session hours. Logs every state transition.

    v2: Only Tier 1 events trigger transitions. IDLE is the default
    state (no trading). POST_EVENT is the only trading window.
    """

    def __init__(self) -> None:
        self._state: State = State.IDLE
        self._active_event = None  # EconomicEvent or None
        self._state_entered_at: float = time.time()
        self._cooldown_until: float = 0.0
        self._last_transition: Optional[str] = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.value

    @property
    def active_event(self):
        return self._active_event

    @property
    def minutes_in_current_state(self) -> float:
        return (time.time() - self._state_entered_at) / 60.0

    def get_behavior(self) -> dict:
        """Return the behavior descriptor for the current state."""
        return _STATE_BEHAVIORS[self._state]

    def trigger_cooldown(self) -> None:
        """
        Called externally when a trade is taken or POST_EVENT window
        should end early. Forces transition to COOLDOWN.
        """
        self._cooldown_until = time.time() + (config.COOLDOWN_MINUTES * 60)
        self._transition_to(State.COOLDOWN, reason="trade_taken_or_window_end")

    def update(
        self,
        upcoming_events: list,
        recent_events: list,
    ) -> State:
        """
        Evaluate events and session hours, transition to appropriate state.

        Args:
            upcoming_events: Events happening within the next ~20 minutes
            recent_events: Events released within the last ~25 minutes
        """
        now = datetime.now(timezone.utc)
        new_state = self._evaluate(now, upcoming_events, recent_events)

        if new_state != self._state:
            self._transition_to(new_state)

        return self._state

    def _transition_to(self, new_state: State, reason: str = "") -> None:
        """Execute a state transition with logging."""
        old_state = self._state
        self._state = new_state
        self._state_entered_at = time.time()

        event_name = "—"
        if self._active_event:
            event_name = getattr(self._active_event, "name", "—")

        extra = f" | Reason: {reason}" if reason else ""
        logger.info(
            f"🔄 STATE: {old_state.value} → {new_state.value} | "
            f"Event: {event_name}{extra}"
        )
        self._last_transition = datetime.now(timezone.utc).isoformat()

    def _evaluate(
        self,
        now: datetime,
        upcoming: list,
        recent: list,
    ) -> State:
        """Determine the correct state based on session hours and event proximity."""

        # ── Priority 1: Market Open Volatility Block ─────────────────
        if self._is_market_opening(now):
            self._active_event = None
            return State.MARKET_OPEN_BLOCK

        # ── Priority 2: Active cooldown ──────────────────────────────
        if self._state == State.COOLDOWN:
            if time.time() < self._cooldown_until:
                return State.COOLDOWN
            # Cooldown expired → fall through to normal evaluation

        # ── Priority 3: Check upcoming events for PRE_EVENT or LOCKDOWN
        for event in upcoming:
            if not self._is_tradeable_event(event):
                continue

            minutes_to = (event.time - now).total_seconds() / 60.0

            # LOCKDOWN: within ± LOCKDOWN_MINUTES of release
            if abs(minutes_to) <= config.LOCKDOWN_MINUTES:
                self._active_event = event
                return State.LOCKDOWN

            # PRE_EVENT: within PRE_EVENT_MINUTES before release
            if 0 < minutes_to <= config.PRE_EVENT_MINUTES:
                self._active_event = event
                return State.PRE_EVENT

        # ── Priority 4: Check recent events for LOCKDOWN or POST_EVENT
        for event in recent:
            if not self._is_tradeable_event(event):
                continue

            minutes_since = (now - event.time).total_seconds() / 60.0

            # LOCKDOWN: still within LOCKDOWN_MINUTES after release
            if minutes_since <= config.LOCKDOWN_MINUTES:
                self._active_event = event
                return State.LOCKDOWN

            # POST_EVENT: between lockdown end and window expiry
            lockdown_end = config.LOCKDOWN_MINUTES
            window_end = config.POST_EVENT_WINDOW_MINUTES
            if lockdown_end < minutes_since <= window_end:
                self._active_event = event
                return State.POST_EVENT

            # Just past the POST_EVENT window → trigger cooldown
            if window_end < minutes_since <= (window_end + config.COOLDOWN_MINUTES):
                if self._state == State.POST_EVENT:
                    # Auto-transition to cooldown when window expires
                    self._cooldown_until = time.time() + (config.COOLDOWN_MINUTES * 60)
                    self._active_event = event
                    return State.COOLDOWN

        # ── Default: IDLE (no nearby events) ─────────────────────────
        self._active_event = None
        return State.IDLE

    # ─── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_market_opening(now: datetime) -> bool:
        """Check if current UTC time falls within the 5-minute chaotic market open."""
        # Weekend check
        if now.weekday() >= 5:
            return False

        current_minutes = now.hour * 60 + now.minute
        
        # Check US Open Block
        us_open_minutes = config.US_MARKET_OPEN_HOUR_UTC * 60 + config.US_MARKET_OPEN_MIN_UTC
        if us_open_minutes <= current_minutes < (us_open_minutes + config.MARKET_OPEN_BLOCK_MINUTES):
            return True

        # Check Global Open Block
        global_open_minutes = config.GLOBAL_MARKET_OPEN_HOUR_UTC * 60 + config.GLOBAL_MARKET_OPEN_MIN_UTC
        if global_open_minutes <= current_minutes < (global_open_minutes + config.MARKET_OPEN_BLOCK_MINUTES):
            return True
            
        return False

    @staticmethod
    def _is_tradeable_event(event) -> bool:
        """Check if an event qualifies for state transitions (Tier 1, optionally Tier 2)."""
        # Must be USD
        if hasattr(event, "currency") and event.currency != "USD":
            return False

        name = getattr(event, "name", "")
        name_lower = name.lower()

        # Check Tier 1
        for kw in config.TIER_1_KEYWORDS:
            if kw.lower() in name_lower:
                return True

        # Check Tier 2 (only if enabled)
        if config.ENABLE_TIER_2_EVENTS:
            for kw in config.TIER_2_KEYWORDS:
                if kw.lower() in name_lower:
                    return True

        return False

    def force_state(self, state: State) -> None:
        """Manually override the state (for testing/emergency)."""
        self._transition_to(state, reason="FORCED")

    def status_summary(self) -> dict:
        """Return a summary dict for dashboard/heartbeat."""
        event_name = None
        event_time = None
        if self._active_event:
            event_name = getattr(self._active_event, "name", None)
            event_time = getattr(self._active_event, "time", None)
            if event_time:
                event_time = event_time.isoformat()

        return {
            "state": self._state.value,
            "minutes_in_state": round(self.minutes_in_current_state, 1),
            "active_event": event_name,
            "event_time": event_time,
            "market_open_block": self._is_market_opening(datetime.now(timezone.utc)),
        }
