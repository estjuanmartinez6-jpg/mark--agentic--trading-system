"""
simulator/trade_logger.py — Comprehensive Trade Recording for MARK III Replay

Subscribes to EventBus trade events and records exhaustive detail about
every trade: not just the outcome, but WHY the trade happened (Req 13).

Two outputs:
  1. CSV — same schema as existing trades/trades_YYYY-MM-DD.csv for compatibility
  2. JSON sidecar — richer format with full decision context and health trajectory

Event subscriptions:
  trade_open  → capture fill details + full signal context
  trade_close → record outcome, compute final P&L, write to CSV/JSON
  stop_hit    → record the specific SL/TP trigger price
  health_event → track health score trajectory during trade life
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from simulator.event_bus import EventBus

logger = logging.getLogger("Sim.TradeLogger")


@dataclass
class OpenTradeRecord:
    """Accumulates data about an open trade until it closes."""
    # Fill details
    ticket: int
    symbol: str
    direction: str
    lot_size: float
    fill_price: float
    sl_price: float
    tp_price: float
    spread_at_entry: float
    slippage_applied: float
    latency_ms: float
    open_time: pd.Timestamp

    # Signal context (WHY the trade happened — Req 13)
    signal_score: int = 0
    trend_score: int = 0
    structure_score: int = 0
    rejection_score: int = 0
    volume_score: int = 0
    pressure_score: int = 0
    correlation_score: int = 0
    factors_active: int = 0
    regime: str = ""
    setup_type: str = ""
    entry_atr: float = 0.0
    sl_distance: float = 0.0
    tp_distance: float = 0.0
    structural_ref: float = 0.0
    entry_speed: int = 0
    profile_min_score: Optional[int] = None
    profile_sl_mult: Optional[float] = None
    profile_tp_mult: Optional[float] = None
    simulated_session: str = ""

    # Health trajectory (accumulated across health events during trade life)
    health_scores: List[int] = field(default_factory=list)
    health_actions: List[str] = field(default_factory=list)
    peak_profit: float = 0.0


class SimTradeLogger:
    """
    Records all trade events during a replay session.

    Usage:
        logger = SimTradeLogger(output_dir="simulator/results/run_001")
        logger.subscribe(event_bus)
        # ... run replay ...
        logger.finalize()
    """

    # CSV columns matching existing MARK III trade log schema + simulation extras
    CSV_COLUMNS = [
        # Core (matches monitoring/logger.py CSV_HEADERS)
        "timestamp_utc", "symbol", "direction", "lot_size",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "sl_price", "tp_price", "pnl_usd", "pnl_pct",
        "duration_min", "close_reason", "signal_score", "setup_type",
        "ticket", "trajectory", "market_regime",
        # Simulation extras
        "spread_at_entry", "slippage_applied", "latency_ms", "commission",
        "fill_mode", "ohlc_scenario",
        # Signal decision context (Req 13)
        "trend_score", "structure_score", "rejection_score",
        "volume_score", "pressure_score", "correlation_score",
        "factors_active", "entry_atr", "sl_distance", "tp_distance",
        "structural_ref", "entry_speed",
        "profile_min_score", "profile_sl_mult", "profile_tp_mult",
        "simulated_session",
        # Health trajectory
        "health_scores", "health_actions", "peak_profit",
    ]

    def __init__(
        self,
        output_dir: str,
        fill_mode: str = "pessimistic",
        commission: float = 0.0,
        session_config: dict = None,
    ) -> None:
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._fill_mode = fill_mode
        self._commission = commission
        self._session_config = session_config or {}

        # Trade state
        self._open_trades: Dict[int, OpenTradeRecord] = {}
        self._closed_trades: List[dict] = []

        # Output file paths
        self._csv_path  = self._out_dir / "trades.csv"
        self._json_path = self._out_dir / "trades.json"

        # Initialize CSV
        self._init_csv()

        logger.info(
            f"[TradeLogger] Output dir: {self._out_dir} │ "
            f"Fill mode: {fill_mode} │ Commission: ${commission}"
        )

    def subscribe(self, bus: EventBus) -> None:
        """Subscribe to all relevant EventBus events."""
        bus.subscribe(EventBus.TRADE_OPEN,   self._on_trade_open)
        bus.subscribe(EventBus.TRADE_CLOSE,  self._on_trade_close)
        bus.subscribe(EventBus.HEALTH_EVENT, self._on_health_event)
        bus.subscribe(EventBus.STOP_HIT,     self._on_stop_hit)
        logger.info("[TradeLogger] Subscribed to trade events")

    # ── Event handlers ───────────────────────────────────────────────

    def _on_trade_open(
        self,
        ticket: int,
        signal,
        fill_price: float,
        spread: float,
        slippage: float,
        latency_ms: float,
        timestamp: pd.Timestamp,
        context: dict = None,
        **kwargs,
    ) -> None:
        ctx = context or {}

        rec = OpenTradeRecord(
            ticket=ticket,
            symbol=signal.symbol,
            direction=signal.action,
            # P2 fix: lot_size is passed explicitly in the event payload,
            # not via signal (Signal has no lot_size attribute).
            lot_size=kwargs.get("lot_size", 0.0),
            fill_price=fill_price,
            sl_price=getattr(signal, "sl", 0.0),
            tp_price=getattr(signal, "tp", 0.0),
            spread_at_entry=spread,
            slippage_applied=slippage,
            latency_ms=latency_ms,
            open_time=timestamp,
            # Signal context
            signal_score=signal.score,
            trend_score=signal.trend_score,
            structure_score=signal.structure_score,
            rejection_score=signal.rejection_score,
            volume_score=signal.volume_score,
            pressure_score=signal.pressure_score,
            correlation_score=signal.correlation_score,
            factors_active=signal.factors_active,
            regime=signal.regime,
            setup_type=getattr(signal, "setup_type", "UNKNOWN"),
            entry_atr=signal.atr,
            sl_distance=signal.sl_distance,
            tp_distance=signal.tp_distance,
            structural_ref=signal.structural_ref,
            entry_speed=signal.speed_score,
            profile_min_score=ctx.get("profile_min_score"),
            profile_sl_mult=ctx.get("profile_sl_mult"),
            profile_tp_mult=ctx.get("profile_tp_mult"),
            simulated_session=ctx.get("simulated_session", ""),
        )
        self._open_trades[ticket] = rec

    def _on_trade_close(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        reason: str,
        duration_sec: Optional[float] = None,
        **kwargs,
    ) -> None:
        rec = self._open_trades.pop(ticket, None)
        if rec is None:
            logger.warning(f"[TradeLogger] Close for unknown ticket {ticket}")
            return

        now_utc = datetime.now(timezone.utc)
        exit_time = kwargs.get("timestamp", now_utc)

        # Compute duration
        if duration_sec is None and hasattr(rec.open_time, "timestamp"):
            if hasattr(exit_time, "timestamp"):
                try:
                    duration_sec = (
                        exit_time.timestamp() - rec.open_time.timestamp()
                    )
                except Exception:
                    duration_sec = 0.0

        pnl_pct = (
            (pnl_usd / (rec.fill_price * rec.lot_size)) * 100
            if rec.fill_price > 0 and rec.lot_size > 0
            else 0.0
        )

        row = {
            "timestamp_utc":    now_utc.isoformat(),
            "symbol":           rec.symbol,
            "direction":        rec.direction,
            "lot_size":         rec.lot_size,
            "entry_time":       str(rec.open_time),
            "exit_time":        str(exit_time),
            "entry_price":      rec.fill_price,
            "exit_price":       exit_price,
            "sl_price":         rec.sl_price,
            "tp_price":         rec.tp_price,
            "pnl_usd":          round(pnl_usd, 4),
            "pnl_pct":          round(pnl_pct, 4),
            "duration_min":     round(duration_sec / 60, 2) if duration_sec else 0.0,
            "close_reason":     reason,
            "signal_score":     rec.signal_score,
            "setup_type":       rec.setup_type,
            "ticket":           ticket,
            "trajectory":       str(rec.health_scores),  # Health score trajectory
            "market_regime":    rec.regime,
            # Simulation extras
            "spread_at_entry":  rec.spread_at_entry,
            "slippage_applied": rec.slippage_applied,
            "latency_ms":       rec.latency_ms,
            "commission":       self._commission,
            "fill_mode":        self._fill_mode,
            "ohlc_scenario":    kwargs.get("ohlc_scenario", "unknown"),
            # Signal context
            "trend_score":      rec.trend_score,
            "structure_score":  rec.structure_score,
            "rejection_score":  rec.rejection_score,
            "volume_score":     rec.volume_score,
            "pressure_score":   rec.pressure_score,
            "correlation_score":rec.correlation_score,
            "factors_active":   rec.factors_active,
            "entry_atr":        rec.entry_atr,
            "sl_distance":      rec.sl_distance,
            "tp_distance":      rec.tp_distance,
            "structural_ref":   rec.structural_ref,
            "entry_speed":      rec.entry_speed,
            "profile_min_score":rec.profile_min_score,
            "profile_sl_mult":  rec.profile_sl_mult,
            "profile_tp_mult":  rec.profile_tp_mult,
            "simulated_session":rec.simulated_session,
            # Health
            "health_scores":    str(rec.health_scores),
            "health_actions":   str(rec.health_actions),
            "peak_profit":      rec.peak_profit,
        }

        self._closed_trades.append(row)
        self._write_csv_row(row)

    def _on_health_event(
        self, ticket: int, health_score: int, action: str, **kwargs
    ) -> None:
        rec = self._open_trades.get(ticket)
        if rec:
            rec.health_scores.append(health_score)
            rec.health_actions.append(action)

    def _on_stop_hit(self, ticket: int, hit_type: str, price: float, **kwargs) -> None:
        # Stop fills are handled via trade_close — nothing extra needed here
        logger.debug(f"[TradeLogger] Stop hit: #{ticket} {hit_type} @ {price:.2f}")

    # ── Output ──────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        """Write CSV header if file doesn't exist."""
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def _write_csv_row(self, row: dict) -> None:
        """Append one trade row to the CSV."""
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
                writer.writerow(row)
        except Exception as exc:
            logger.error(f"[TradeLogger] CSV write error: {exc}")

    def finalize(self) -> None:
        """Write the full JSON export after replay ends."""
        try:
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session": self._session_config,
                        "trades":  self._closed_trades,
                        "total":   len(self._closed_trades),
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(
                f"[TradeLogger] Finalized │ "
                f"{len(self._closed_trades)} trades → {self._json_path}"
            )
        except Exception as exc:
            logger.error(f"[TradeLogger] JSON export error: {exc}")

    # ── Properties ──────────────────────────────────────────────────

    @property
    def trade_count(self) -> int:
        return len(self._closed_trades)

    @property
    def closed_trades(self) -> List[dict]:
        return list(self._closed_trades)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def json_path(self) -> Path:
        return self._json_path
