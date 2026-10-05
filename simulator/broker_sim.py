"""
simulator/broker_sim.py — Simulated Broker Execution for MARK III Replay

Implements SimulatedExecutionBackend — a full simulation of broker
execution without connecting to MT5.

Key features:
  - Deterministic randomness via seeded np.random.RandomState (Req 3)
  - Intra-candle OHLC ambiguity handled via configurable FillMode (Req 4)
  - Three spread models: historical, dynamic, fixed
  - Slippage drawn from seeded RNG (always adverse)
  - Session latency model (Req 10)
  - Market microstructure placeholder interface (Req 11)
  - Virtual positions with real-time P&L calculation
  - Tick-value based P&L (US500/ES: $1 per point per lot)

Determinism contract (Req 3, 15):
    All random draws go through self._rng (np.random.RandomState).
    The RNG state is initialized with `seed` and advances predictably.
    Identical seed → identical fills → identical trade outcomes.
    RNG state is serializable for checkpoint/restore (Req 12).
"""
from __future__ import annotations

import logging
import time as wall_time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from simulator.execution_backend import (
    ExecutionBackend, OrderRequest, OrderResult, SimPosition
)

logger = logging.getLogger("Sim.Broker")


# ═══════════════════════════════════════════════════════════════════════
# FILL MODE
# ═══════════════════════════════════════════════════════════════════════

class FillMode(Enum):
    """
    Controls how intra-candle OHLC ambiguity is resolved (Req 4).

    OHLC candles reveal price extremes but not their ORDER within the bar.
    This matters when both SL and TP fall within the candle's range.

    PESSIMISTIC (default): The adverse extreme is assumed to occur first.
        BUY trade:  LOW checked before HIGH → SL hit before TP
        SELL trade: HIGH checked before LOW → SL hit before TP
        Use for conservative backtesting (worst-case fills).

    OPTIMISTIC: The favourable extreme is assumed first.
        BUY trade:  HIGH checked before LOW → TP hit before SL
        SELL trade: LOW checked before HIGH → TP hit before SL
        Useful for comparing best/worst case scenarios.

    NEAREST: Whichever of SL/TP is closest to candle OPEN hits first.
        More realistic — assumes price moved from open toward nearest level.

    RANDOM (seeded): RNG decides which extreme occurred first.
        Ensures determinism within a seed while still sampling uncertainty.
    """
    PESSIMISTIC = "pessimistic"
    OPTIMISTIC  = "optimistic"
    NEAREST     = "nearest"
    RANDOM      = "random"


# ═══════════════════════════════════════════════════════════════════════
# SPREAD MODEL
# ═══════════════════════════════════════════════════════════════════════

class SpreadModel(Enum):
    HISTORICAL = "historical"   # Use candle's recorded spread column
    DYNAMIC    = "dynamic"      # Historical × vol_ratio multiplier
    FIXED      = "fixed"        # Constant max_spread_pts from settings


# ═══════════════════════════════════════════════════════════════════════
# LATENCY MODEL (Req 10)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LatencyModel:
    """
    Simulates execution latency without actually sleeping.

    Modeled as price impact: if total latency > 1 candle duration,
    the fill is deferred to the next candle's open.

    For sub-candle latency, we assume the fill occurs at the candle
    close (conservative — price may move during latency period).

    All values in milliseconds.
    """
    order_transmission_ms: float = 0.0    # Signal → broker
    execution_ms: float = 0.0             # Broker → fill
    candle_processing_ms: float = 0.0     # Strategy computation lag

    @property
    def total_ms(self) -> float:
        return self.order_transmission_ms + self.execution_ms + self.candle_processing_ms

    @property
    def deferred_to_next_candle(self) -> bool:
        """If latency > M5 duration (300s), fill on next candle."""
        return self.total_ms >= 300_000


# ═══════════════════════════════════════════════════════════════════════
# MARKET MICROSTRUCTURE PLACEHOLDER (Req 11)
# ═══════════════════════════════════════════════════════════════════════

class MarketMicrostructure:
    """
    PLACEHOLDER — not implemented in Phase 1.

    Defines the interface for future tick/DOM integration.
    SimulatedBroker accepts an optional microstructure parameter.
    In Phase 1, all methods raise NotImplementedError.

    Future implementations:
        TickReplayMicrostructure — feeds actual tick data
        SyntheticTickMicrostructure — generates ticks from OHLCV
        DOMSnapshotMicrostructure — loads LOB snapshots
    """

    def get_bid_ask(self, timestamp: pd.Timestamp) -> tuple:
        raise NotImplementedError("Microstructure not available in Phase 1 replay")

    def get_tick_stream(self, timestamp: pd.Timestamp):
        raise NotImplementedError("Tick replay not available in Phase 1")

    def get_dom_snapshot(self, timestamp: pd.Timestamp):
        raise NotImplementedError("DOM data not available in Phase 1")


# ═══════════════════════════════════════════════════════════════════════
# SIMULATED BROKER
# ═══════════════════════════════════════════════════════════════════════

class SimulatedExecutionBackend(ExecutionBackend):
    """
    Full broker simulation for MARK III replay sessions.

    Virtual account:
        - Tracks balance, equity, unrealized P&L
        - Computes P&L using US500 tick value (1 point = $1 × lot_size)
        - Commission deducted at trade close

    Ticket numbering:
        - Starts at 9_000_000 to avoid confusion with real MT5 tickets
        - Auto-increments for each new position
    """

    # US500/ES: $1 per point per standard lot
    TICK_VALUE_PER_POINT: float = 1.0
    TICKET_BASE: int = 9_000_000

    def __init__(
        self,
        initial_balance: float = 30.0,
        spread_model: str = "historical",
        spread_fixed_pts: float = 0.5,
        slippage_max_pts: float = 0.5,
        fill_mode: str = "pessimistic",
        commission_per_trade: float = 0.0,
        latency: Optional[LatencyModel] = None,
        microstructure: Optional[MarketMicrostructure] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            initial_balance:       Starting account balance (USD)
            spread_model:          "historical" | "dynamic" | "fixed"
            spread_fixed_pts:      Fixed spread (used if spread_model="fixed")
            slippage_max_pts:      Maximum slippage in points (adverse only)
            fill_mode:             "pessimistic"|"optimistic"|"nearest"|"random"
            commission_per_trade:  USD per trade (round-turn at close)
            latency:               LatencyModel instance (None = zero latency)
            microstructure:        MarketMicrostructure (None = Phase 1 placeholder)
            seed:                  Random seed for reproducibility (Req 3)
        """
        # ── Deterministic RNG (Req 3, 15) ────────────────────────────
        # All random draws use this isolated RNG — no global random state
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        # ── Execution config ─────────────────────────────────────────
        self._spread_model = SpreadModel(spread_model)
        self._spread_fixed_pts = spread_fixed_pts
        self._slippage_max_pts = slippage_max_pts
        self._fill_mode = FillMode(fill_mode)
        self._commission = commission_per_trade
        self._latency = latency or LatencyModel()
        self._microstructure = microstructure  # Placeholder — not used in Phase 1

        # ── Account state ─────────────────────────────────────────────
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._equity = initial_balance

        # ── Position tracking ─────────────────────────────────────────
        self._positions: Dict[int, SimPosition] = {}
        self._ticket_counter: int = self.TICKET_BASE
        self._current_candle: Optional[pd.Series] = None  # Latest M5 candle
        self._current_spread: float = 0.0

        # ── Closed trade history ──────────────────────────────────────
        self._closed_trades: List[dict] = []
        self._pending_fills: List[dict] = []  # Deferred (latency > 1 candle)

        logger.info(
            f"[Broker] SimulatedBroker initialized │ "
            f"Balance=${initial_balance:.2f} │ "
            f"Spread={spread_model} │ "
            f"Slippage=0-{slippage_max_pts}pts │ "
            f"Fill={fill_mode} │ "
            f"Seed={seed}"
        )

    # ── ExecutionBackend interface ───────────────────────────────────

    def get_account(self) -> dict:
        self._update_equity()
        return {
            "balance":      self._balance,
            "equity":       self._equity,
            "margin":       0.0,   # CFD — no margin requirement in sim
            "free_margin":  self._equity,
        }

    def get_spread(self, symbol: str) -> float:
        return self._current_spread

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Gap 3 fix: Returns symbol spec reading lot constraints from
        settings.SYMBOL_MAP so lot-sizing parity with the live system is exact.
        """
        from config import settings
        sym_cfg = settings.SYMBOL_MAP.get(symbol, {})
        return {
            "point":         1.0,
            "tick_value":    self.TICK_VALUE_PER_POINT,
            "contract_size": 1.0,
            "min_lot":       sym_cfg.get("min_lot",  0.1),
            "max_lot":       sym_cfg.get("max_lot",  0.3),
            "lot_step":      sym_cfg.get("lot_step", 0.1),
        }

    def get_open_positions(self, symbol: str = None) -> List[SimPosition]:
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Execute a simulated market order.

        Fill price = candle close ± spread/2 ± slippage (adverse).
        If latency defers to next candle, records a pending fill.
        """
        if self._current_candle is None:
            return OrderResult(
                success=False, ticket=0, fill_price=0.0,
                spread_applied=0.0, slippage_applied=0.0,
                latency_ms=0.0, error="No current candle (broker not updated)"
            )

        # Compute spread at this moment
        spread = self._compute_spread(self._current_candle)
        self._current_spread = spread

        # Compute fill price (with spread and slippage)
        base_price = float(self._current_candle["close"])
        fill_price, slippage = self._apply_fill_cost(
            base_price, order.direction, spread
        )

        # Handle latency deferral
        latency_ms = self._latency.total_ms
        if self._latency.deferred_to_next_candle:
            self._pending_fills.append({
                "order": order,
                "fill_price": fill_price,
                "spread": spread,
                "slippage": slippage,
            })
            logger.info(
                f"[Broker] Order deferred to next candle "
                f"(latency={latency_ms:.0f}ms > 300,000ms)"
            )
            return OrderResult(
                success=False, ticket=0, fill_price=0.0,
                spread_applied=spread, slippage_applied=slippage,
                latency_ms=latency_ms, error="Deferred to next candle"
            )

        return self._create_position(order, fill_price, spread, slippage, latency_ms)

    def close_position(self, ticket: int, reason: str = "Manual") -> bool:
        """Close an open position at current market price."""
        pos = self._positions.get(ticket)
        if not pos:
            logger.warning(f"[Broker] close_position: ticket {ticket} not found")
            return False

        exit_price = float(self._current_candle["close"])
        self._settle_position(pos, exit_price, reason)
        return True

    def modify_sl_tp(
        self,
        ticket: int,
        sl: float = None,
        tp: float = None,
    ) -> bool:
        pos = self._positions.get(ticket)
        if not pos:
            return False
        if sl is not None:
            pos.sl_price = sl
        if tp is not None:
            pos.tp_price = tp
        return True

    def on_new_candle(self, candle: pd.Series) -> List[dict]:
        """
        Called at the start of each new M5 candle.
        1. Processes any pending deferred fills (from latency model)
        2. Checks all open positions for SL/TP hits within candle range
        3. Updates all position P&L with current candle close
        4. Returns list of fill events for EventBus propagation

        Args:
            candle: M5 candle row with OHLCV + indicators

        Returns:
            List of fill events. Each event is a dict:
            {"ticket": int, "hit_type": "SL"|"TP", "price": float,
             "pnl_usd": float, "reason": str}
        """
        self._current_candle = candle
        fill_events = []

        # Process any deferred fills from previous candle's latency
        if self._pending_fills:
            for pending in self._pending_fills:
                result = self._create_position(
                    pending["order"],
                    pending["fill_price"],
                    pending["spread"],
                    pending["slippage"],
                    self._latency.total_ms,
                )
                logger.info(f"[Broker] Deferred fill executed: ticket={result.ticket}")
            self._pending_fills.clear()

        # Check SL/TP for each open position
        for ticket, pos in list(self._positions.items()):
            event = self._check_stops(pos, candle)
            if event:
                fill_events.append(event)

        # Update equity
        self._update_equity()

        return fill_events

    def reset(self) -> None:
        """Reset to initial state for a new replay session."""
        self._balance = self._initial_balance
        self._equity = self._initial_balance
        self._positions.clear()
        self._closed_trades.clear()
        self._pending_fills.clear()
        self._ticket_counter = self.TICKET_BASE
        self._current_candle = None
        # Reset RNG to original seed for determinism (Req 3)
        self._rng = np.random.RandomState(self._seed)
        logger.info(f"[Broker] Reset complete. Balance=${self._initial_balance:.2f}")

    # ── Stop-loss / Take-profit checking ────────────────────────────

    def _check_stops(self, pos: SimPosition, candle: pd.Series) -> Optional[dict]:
        """
        Determines if an open position hit SL or TP within this candle.

        Uses FillMode to resolve intra-candle OHLC ambiguity (Req 4).
        """
        high = float(candle["high"])
        low  = float(candle["low"])

        sl_hit = False
        tp_hit = False

        if pos.direction == "BUY":
            sl_hit = low  <= pos.sl_price
            tp_hit = high >= pos.tp_price
        else:  # SELL
            sl_hit = high >= pos.sl_price
            tp_hit = low  <= pos.tp_price

        if not sl_hit and not tp_hit:
            return None

        # ── Resolve ambiguity when BOTH hit in same candle ──────────
        hit_type, fill_price = self._resolve_fill(
            pos, candle, sl_hit, tp_hit
        )

        pnl = self._compute_pnl(pos, fill_price)
        self._settle_position(pos, fill_price, f"MT5_{hit_type}")

        logger.info(
            f"[Broker] {hit_type} hit on {pos.symbol} #{pos.ticket} │ "
            f"Dir={pos.direction} │ Fill={fill_price:.2f} │ P&L=${pnl:+.2f}"
        )

        return {
            "ticket":   pos.ticket,
            "hit_type": hit_type,
            "price":    fill_price,
            "pnl_usd":  pnl,
            "reason":   f"MT5_{hit_type}",
        }

    def _resolve_fill(
        self,
        pos: SimPosition,
        candle: pd.Series,
        sl_hit: bool,
        tp_hit: bool,
    ) -> tuple:
        """
        Determines which level hit first and returns (hit_type, fill_price).
        """
        if sl_hit and not tp_hit:
            return "SL", pos.sl_price
        if tp_hit and not sl_hit:
            return "TP", pos.tp_price

        # Both hit — use FillMode to determine order (Req 4)
        if self._fill_mode == FillMode.PESSIMISTIC:
            # Adverse extreme first → SL hit before TP
            return "SL", pos.sl_price

        elif self._fill_mode == FillMode.OPTIMISTIC:
            # Favourable extreme first → TP hit before SL
            return "TP", pos.tp_price

        elif self._fill_mode == FillMode.NEAREST:
            # Whichever is closer to candle open
            candle_open = float(candle["open"])
            if pos.direction == "BUY":
                dist_sl = abs(candle_open - pos.sl_price)
                dist_tp = abs(candle_open - pos.tp_price)
            else:
                dist_sl = abs(candle_open - pos.sl_price)
                dist_tp = abs(candle_open - pos.tp_price)
            if dist_sl <= dist_tp:
                return "SL", pos.sl_price
            return "TP", pos.tp_price

        elif self._fill_mode == FillMode.RANDOM:
            # Seeded RNG — deterministic within seed (Req 3)
            choice = self._rng.choice(["SL", "TP"])
            price = pos.sl_price if choice == "SL" else pos.tp_price
            return choice, price

        return "SL", pos.sl_price  # Fallback: pessimistic

    # ── Spread / Slippage / Fill costs ──────────────────────────────

    def _compute_spread(self, candle: pd.Series) -> float:
        """
        Compute the spread to apply at this candle, in price points.

        P0 fix: MT5 CSV exports spread as INTEGER TICKS, not price points.
        For US500/ES, 1 tick = 0.01 points (broker minimum price move).
        Example: MT5 spread=70 → 70 × 0.01 = 0.70 price points.

        Without this conversion, every fill was displaced by ~35 pts
        (spread/2 = 70/2 = 35 pts) instead of the real 0.35 pts.
        """
        if self._spread_model == SpreadModel.FIXED:
            return self._spread_fixed_pts

        raw_ticks = float(candle.get("spread", 0))

        # Convert ticks → price points.
        # US500 point size = 0.01 (i.e. minimum price increment).
        # If the candle has no spread column, fall back to the fixed default.
        if raw_ticks <= 0:
            historical_spread_pts = self._spread_fixed_pts
        else:
            historical_spread_pts = raw_ticks * 0.01  # ticks → points

        if self._spread_model == SpreadModel.HISTORICAL:
            return historical_spread_pts

        # DYNAMIC: scale by vol_ratio (capped at 3×)
        vol_ratio = float(candle.get("vol_ratio", 1.0))
        scale = min(3.0, max(1.0, vol_ratio))
        return historical_spread_pts * scale

    def _apply_fill_cost(
        self,
        base_price: float,
        direction: str,
        spread: float,
    ) -> tuple:
        """
        Compute fill price after spread and slippage.
        Returns (fill_price, slippage_applied).

        BUY  fills at ask (base + spread/2) + slippage (higher)
        SELL fills at bid (base - spread/2) - slippage (lower)
        """
        half_spread = spread / 2.0

        # Slippage: always adverse, scaled by vol_ratio
        slippage = self._rng.uniform(0, self._slippage_max_pts)

        if direction == "BUY":
            fill = base_price + half_spread + slippage
        else:
            fill = base_price - half_spread - slippage

        return fill, slippage

    # ── Position lifecycle ────────────────────────────────────────────

    def _create_position(
        self,
        order: OrderRequest,
        fill_price: float,
        spread: float,
        slippage: float,
        latency_ms: float,
    ) -> OrderResult:
        """Create and register a new virtual position."""
        self._ticket_counter += 1
        ticket = self._ticket_counter

        pos = SimPosition(
            ticket=ticket,
            symbol=order.symbol,
            direction=order.direction,
            lot_size=order.lot_size,
            entry_price=fill_price,
            sl_price=order.sl_price,
            tp_price=order.tp_price,
            open_time=self._current_candle["time"].timestamp()
                if hasattr(self._current_candle["time"], "timestamp")
                else float(wall_time.time()),
            entry_atr=order.entry_atr,
            structural_ref=order.structural_ref,
            entry_regime=order.market_regime,
            entry_speed=order.entry_speed,
            signal_score=order.signal_score,
            setup_type=order.setup_type,
            context=order.context,
        )
        self._positions[ticket] = pos

        logger.info(
            f"[Broker] ✅ {order.direction} {order.symbol} │ "
            f"Ticket={ticket} │ Fill={fill_price:.2f} │ "
            f"SL={order.sl_price:.2f} │ TP={order.tp_price:.2f} │ "
            f"Spread={spread:.2f}pts │ Slip={slippage:.2f}pts"
        )

        return OrderResult(
            success=True,
            ticket=ticket,
            fill_price=fill_price,
            spread_applied=spread,
            slippage_applied=slippage,
            latency_ms=latency_ms,
        )

    def _settle_position(
        self, pos: SimPosition, exit_price: float, reason: str
    ) -> dict:
        """Close a position, compute P&L, update balance, record trade."""
        pnl = self._compute_pnl(pos, exit_price) - self._commission

        self._balance += pnl
        self._equity = self._balance  # Will be updated with open positions

        exit_time = (
            self._current_candle["time"].timestamp()
            if self._current_candle is not None and
            hasattr(self._current_candle["time"], "timestamp")
            else wall_time.time()
        )

        trade_record = {
            "ticket":        pos.ticket,
            "symbol":        pos.symbol,
            "direction":     pos.direction,
            "lot_size":      pos.lot_size,
            "entry_price":   pos.entry_price,
            "exit_price":    exit_price,
            "sl_price":      pos.sl_price,
            "tp_price":      pos.tp_price,
            "open_time":     pos.open_time,
            "close_time":    exit_time,
            "pnl_usd":       round(pnl, 4),
            "close_reason":  reason,
            "signal_score":  pos.signal_score,
            "setup_type":    pos.setup_type,
            "entry_regime":  pos.entry_regime,
            "entry_atr":     pos.entry_atr,
            "context":       pos.context,
            "commission":    self._commission,
        }

        self._closed_trades.append(trade_record)
        del self._positions[pos.ticket]

        logger.info(
            f"[Broker] ❌ Closed #{pos.ticket} {pos.symbol} {pos.direction} │ "
            f"Entry={pos.entry_price:.2f} → Exit={exit_price:.2f} │ "
            f"P&L=${pnl:+.2f} │ Reason={reason}"
        )
        return trade_record

    def _compute_pnl(self, pos: SimPosition, exit_price: float) -> float:
        """Compute P&L in USD for a position."""
        if pos.direction == "BUY":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price
        return round(price_diff * pos.lot_size * self.TICK_VALUE_PER_POINT, 4)

    def _update_equity(self) -> None:
        """Recalculate equity = balance + sum of open position P&L."""
        unrealized = sum(
            self._compute_pnl(pos, float(self._current_candle["close"]))
            for pos in self._positions.values()
        ) if self._current_candle is not None else 0.0
        self._equity = self._balance + unrealized
        for pos in self._positions.values():
            pos.unrealized_pnl = self._compute_pnl(
                pos, float(self._current_candle["close"])
            ) if self._current_candle is not None else 0.0

    # ── Properties / State export ─────────────────────────────────

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def closed_trades(self) -> List[dict]:
        return list(self._closed_trades)

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    def get_rng_state(self) -> tuple:
        """Export RNG state for checkpoint (Req 12)."""
        return self._rng.get_state()

    def set_rng_state(self, state: tuple) -> None:
        """Restore RNG state from checkpoint (Req 12)."""
        self._rng.set_state(state)

    def summary(self) -> dict:
        return {
            "balance":        round(self._balance, 4),
            "equity":         round(self._equity, 4),
            "pnl":            round(self._equity - self._initial_balance, 4),
            "open_positions": self.open_position_count,
            "total_trades":   len(self._closed_trades),
            "seed":           self._seed,
            "fill_mode":      self._fill_mode.value,
            "spread_model":   self._spread_model.value,
        }
