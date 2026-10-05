"""
simulator/execution_backend.py — Unified Execution Interface for MARK III (Req 9)

Defines the abstract ExecutionBackend that decouples the strategy
from the execution mechanism. The strategy adapter never knows whether
it's trading live, paper, or replaying history.

Implementations:
    SimulatedExecutionBackend — for REPLAY mode (implemented in broker_sim.py)
    MT5ExecutionBackend       — for LIVE/PAPER mode (placeholder for future)

Benefits:
  1. Strategy code is identical across all modes
  2. Paper trading = swap SimulatedExecutionBackend for a live feed
  3. Exhaustive testing of execution logic in isolation
  4. Future: cloud execution backends (Interactive Brokers, Alpaca, etc.)

Data contracts:
    OrderRequest — encapsulates everything needed to submit an order
    OrderResult  — encapsulates the fill confirmation
    SimPosition  — represents an open position in any backend
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════
# DATA CONTRACTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OrderRequest:
    """
    Everything needed to submit a trade order.
    Created by StrategyAdapter from a Signal object.
    """
    symbol: str
    direction: str           # "BUY" or "SELL"
    lot_size: float
    sl_price: float          # Stop-loss price
    tp_price: float          # Take-profit price
    signal_score: int
    setup_type: str          # "CONTINUATION" or "REVERSAL"
    market_regime: str
    entry_atr: float
    structural_ref: float    # Swing level the trade is based on
    entry_speed: int
    context: dict = field(default_factory=dict)  # Signal score breakdown

    @classmethod
    def from_signal(cls, signal, context: dict = None, lot_size: float = None) -> "OrderRequest":
        """
        Build an OrderRequest from a MARK III Signal object.
        Bridges the Signal dataclass to the execution backend.
        """
        return cls(
            symbol=signal.symbol,
            direction=signal.action,
            lot_size=lot_size if lot_size is not None else getattr(signal, "lot_size", 1.0),
            sl_price=getattr(signal, "sl", getattr(signal, "sl_distance", 0.0)),
            tp_price=getattr(signal, "tp", getattr(signal, "tp_distance", 0.0)),
            signal_score=signal.score,
            setup_type=getattr(signal, "setup_type", "UNKNOWN"),
            market_regime=getattr(signal, "market_regime", "NORMAL"),
            entry_atr=getattr(signal, "atr", 0.0),
            structural_ref=getattr(signal, "structural_ref", 0.0),
            entry_speed=getattr(signal, "speed_score", 50),
            context=context or {},
        )


@dataclass
class OrderResult:
    """Fill confirmation returned after order submission."""
    success: bool
    ticket: int              # Virtual ticket number (unique per session)
    fill_price: float        # Actual fill price (after spread + slippage)
    spread_applied: float    # Spread points applied at fill
    slippage_applied: float  # Slippage points applied
    latency_ms: float        # Simulated execution latency
    error: str = ""


@dataclass
class SimPosition:
    """
    An open position in any execution backend.
    Normalized representation for StrategyAdapter management.
    """
    ticket: int
    symbol: str
    direction: str           # "BUY" or "SELL"
    lot_size: float
    entry_price: float
    sl_price: float
    tp_price: float
    open_time: float         # Unix timestamp (simulated)
    unrealized_pnl: float = 0.0
    current_price: float = 0.0

    # Context frozen at entry (for health monitor)
    entry_atr: float = 0.0
    structural_ref: float = 0.0
    entry_regime: str = "NORMAL"
    entry_speed: int = 50
    signal_score: int = 0
    setup_type: str = "UNKNOWN"
    context: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════

class ExecutionBackend(ABC):
    """
    Abstract execution interface.

    StrategyAdapter calls these methods — it never accesses MT5 or
    the SimulatedBroker directly. Swap the backend to change mode.
    """

    # ── Account ──────────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> dict:
        """
        Returns current account state.
        Keys: balance (float), equity (float), margin (float), free_margin (float)
        """

    # ── Market data ──────────────────────────────────────────────────

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """Returns current spread in points for the symbol."""

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> dict:
        """
        Returns symbol specification.
        Keys: point (float), tick_value (float), contract_size (float),
              min_lot (float), max_lot (float), lot_step (float)
        """

    # ── Positions ────────────────────────────────────────────────────

    @abstractmethod
    def get_open_positions(self, symbol: str = None) -> List[SimPosition]:
        """Returns all open positions, optionally filtered by symbol."""

    # ── Order management ─────────────────────────────────────────────

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit a market order.
        Fill is immediate (market order semantics).
        """

    @abstractmethod
    def close_position(self, ticket: int, reason: str = "Manual") -> bool:
        """Close an open position at current market price."""

    @abstractmethod
    def modify_sl_tp(
        self,
        ticket: int,
        sl: float = None,
        tp: float = None,
    ) -> bool:
        """
        Modify stop-loss and/or take-profit of an open position.
        Pass None for sl or tp to leave unchanged.
        """

    # ── Session management ────────────────────────────────────────────

    @abstractmethod
    def on_new_candle(self, candle: "pd.Series") -> List[dict]:
        """
        Called on every new M5 candle. Backend checks if any open
        positions hit their SL or TP within the candle's price range.

        Returns:
            List of fill events: [{"ticket": int, "type": "SL"|"TP", "price": float}, ...]
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset backend state for a new replay session."""


# ═══════════════════════════════════════════════════════════════════════
# MT5 EXECUTION BACKEND (PLACEHOLDER)
# ═══════════════════════════════════════════════════════════════════════

class MT5ExecutionBackend(ExecutionBackend):
    """
    FUTURE IMPLEMENTATION — Live/Paper trading via MT5.

    This class will wrap the existing MT5Connector + OrderManager
    to satisfy the ExecutionBackend interface for live deployment.

    When implemented:
        backend = MT5ExecutionBackend(connector=MT5Connector())
        engine  = StrategyAdapter(backend=backend)
        # All strategy logic identical to replay — only backend changes
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "MT5ExecutionBackend is a future implementation placeholder. "
            "Use SimulatedExecutionBackend for replay mode."
        )

    def get_account(self) -> dict: ...
    def get_spread(self, symbol: str) -> float: ...
    def get_symbol_info(self, symbol: str) -> dict: ...
    def get_open_positions(self, symbol: str = None) -> List[SimPosition]: ...
    def submit_order(self, order: OrderRequest) -> OrderResult: ...
    def close_position(self, ticket: int, reason: str = "Manual") -> bool: ...
    def modify_sl_tp(self, ticket: int, sl: float = None, tp: float = None) -> bool: ...
    def on_new_candle(self, candle) -> List[dict]: ...
    def reset(self) -> None: ...
