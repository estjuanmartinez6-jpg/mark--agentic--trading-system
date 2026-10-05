"""
simulator/event_bus.py — Central event dispatcher for MARK III Replay

Makes the simulation pipeline fully event-driven (not loop-driven).
Handlers fire automatically when events are emitted — no caller
needs to know which subscribers exist.

Design:
  Phase 1: Synchronous dispatch (handlers execute in registration order)
  Phase 2 upgrade path: Replace emit() body with asyncio.create_task()
                        without changing any subscriber code.

Supported events and their standard payloads:

  replay_start    date, session_start_ny, session_end_ny, config
  new_candle      timestamp, candle_index, m5_snapshot, m15_snapshot
  signal_evaluated  signal, context (score breakdown dict)
  trade_open      ticket, signal, fill_price, spread, slippage, latency_ms
  trade_close     ticket, exit_price, pnl_usd, reason, duration_sec
  stop_hit        ticket, hit_type ("SL"|"TP"), price, candle_ts
  health_event    ticket, health_score, action, reason
  risk_event      reason, daily_pnl, action ("HALT"|"TRAILING"|"TARGET")
  replay_end      total_trades, final_equity, metrics_summary
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Any

logger = logging.getLogger("Sim.EventBus")


class EventBus:
    """
    Central synchronous event dispatcher.

    Usage:
        bus = EventBus()
        bus.subscribe("new_candle", my_handler)
        bus.emit("new_candle", timestamp=ts, m5_snapshot=df, ...)

    All handlers for an event receive identical keyword arguments.
    Exceptions in one handler do NOT prevent other handlers from running.
    """

    # ── Canonical event names ────────────────────────────────────────
    REPLAY_START     = "replay_start"
    NEW_CANDLE       = "new_candle"
    SIGNAL_EVALUATED = "signal_evaluated"
    TRADE_OPEN       = "trade_open"
    TRADE_CLOSE      = "trade_close"
    STOP_HIT         = "stop_hit"
    HEALTH_EVENT     = "health_event"
    RISK_EVENT       = "risk_event"
    REPLAY_END       = "replay_end"

    def __init__(self, debug: bool = False) -> None:
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._debug = debug
        self._emission_count: Dict[str, int] = defaultdict(int)

    # ── Subscription ────────────────────────────────────────────────

    def subscribe(self, event: str, handler: Callable) -> None:
        """
        Register a callback for the given event name.

        Args:
            event:   Event name string (use class constants e.g. EventBus.NEW_CANDLE)
            handler: Callable(**payload) — receives event payload as kwargs
        """
        if not callable(handler):
            raise TypeError(f"Handler must be callable, got {type(handler)}")
        self._handlers[event].append(handler)
        if self._debug:
            logger.debug(f"[EventBus] Subscribed {handler.__qualname__} → '{event}'")

    def unsubscribe(self, event: str, handler: Callable) -> bool:
        """
        Remove a previously registered handler.
        Returns True if handler was found and removed.
        """
        handlers = self._handlers.get(event, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def clear(self, event: str = None) -> None:
        """Clear all handlers, or handlers for a specific event."""
        if event:
            self._handlers.pop(event, None)
        else:
            self._handlers.clear()
            self._emission_count.clear()

    # ── Emission ────────────────────────────────────────────────────

    def emit(self, event: str, **payload: Any) -> int:
        """
        Emit an event with keyword payload to all registered handlers.

        Handlers execute synchronously in registration order.
        Exceptions in individual handlers are caught and logged —
        they do NOT propagate or prevent other handlers from running.

        Args:
            event:   Event name
            **payload: Arbitrary keyword arguments passed to each handler

        Returns:
            Number of handlers successfully called
        """
        self._emission_count[event] += 1
        handlers = self._handlers.get(event, [])

        if not handlers:
            if self._debug:
                logger.debug(f"[EventBus] emit('{event}') — no subscribers")
            return 0

        success_count = 0
        for handler in handlers:
            try:
                handler(**payload)
                success_count += 1
            except Exception as exc:
                logger.error(
                    f"[EventBus] Handler {handler.__qualname__} raised on "
                    f"event '{event}': {exc}",
                    exc_info=True,
                )

        if self._debug:
            logger.debug(
                f"[EventBus] emit('{event}') → {success_count}/{len(handlers)} handlers OK"
            )

        return success_count

    # ── Introspection ────────────────────────────────────────────────

    def subscriber_count(self, event: str) -> int:
        """Returns number of registered handlers for an event."""
        return len(self._handlers.get(event, []))

    def emission_count(self, event: str) -> int:
        """Returns how many times an event has been emitted."""
        return self._emission_count.get(event, 0)

    def stats(self) -> Dict[str, Any]:
        """Returns a summary dict for logging/debugging."""
        return {
            "events_registered": list(self._handlers.keys()),
            "emission_counts": dict(self._emission_count),
            "subscriber_counts": {
                evt: len(hdls) for evt, hdls in self._handlers.items()
            },
        }
