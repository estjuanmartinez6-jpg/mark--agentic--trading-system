"""
simulator/__init__.py — MARK III Replay Simulator Package

Provides historical simulation and backtesting for the MARK III
trading system without modifying any live trading code.

Quick start:
    from simulator.replay_engine import ReplayEngine

    engine = ReplayEngine(
        date="2026-05-22",
        event="CPI",
        seed=42,
        speed="instant",
    )
    result = engine.run()
    print(f"P&L: ${result.metrics['summary']['net_pnl_usd']:+.2f}")
"""

__version__ = "1.0.0"
__author__  = "MARK III Team"

# Public API
from simulator.replay_engine import ReplayEngine, ReplayResult
from simulator.event_bus import EventBus
from simulator.broker_sim import FillMode, SpreadModel
from simulator.session_controller import SessionController, PREDEFINED_EVENTS
from simulator.metrics_engine import MetricsEngine
from simulator.checkpoint import Checkpointer, ReplaySnapshot

__all__ = [
    "ReplayEngine",
    "ReplayResult",
    "EventBus",
    "FillMode",
    "SpreadModel",
    "SessionController",
    "PREDEFINED_EVENTS",
    "MetricsEngine",
    "Checkpointer",
    "ReplaySnapshot",
]
