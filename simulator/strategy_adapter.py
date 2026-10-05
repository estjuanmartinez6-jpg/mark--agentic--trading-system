"""
simulator/strategy_adapter.py — Bridge Between DataFeed and Unmodified SignalScorer

This module is the heart of the bridge pattern. It subscribes to EventBus
"new_candle" events and calls the real, unmodified MARK III SignalScorer
exactly as the live TradingEngine._run_analysis_cycle() does.

Key guarantees:
  1. SignalScorer instance is real — same class, same initialization
  2. market_data format is identical to what MT5DataProvider.get_all_symbols() returns
  3. SimulationClock patches ONLY scorer's time.time() during evaluation
  4. RiskManager and TradeHealthMonitor receive simulated timestamps explicitly
  5. Health monitor logic mirrors _manage_open_positions() exactly
  6. Trailing stop logic mirrors main.py _manage_open_positions() exactly

US500-only mode:
    market_data = {"ES": {"M5": df, "M15": df}}  (no NQ key)
    SignalScorer.evaluate() handles missing NQ gracefully (returns no NQ signal).

Context capture for trade logging (Req 13):
    The _analyze_symbol() result dict contains all score components.
    This is captured and passed to EventBus "trade_open" payload so
    TradeLogger can record WHY the trade happened.
"""
from __future__ import annotations

import logging
import time as wall_time
from typing import Dict, Optional

import pandas as pd

from config import settings
from execution_engine.regime_profile import get_profile
from execution_engine.trade_health import TradeHealthMonitor
from execution_engine.risk_manager import RiskManager
from signal_engine.scorer import Signal
from signal_engine.ichimoku_strategy import IchimokuStrategy
from signal_engine.momentum_speed import MomentumSpeed
from signal_engine.pseudo_delta import PseudoDelta
from simulator.event_bus import EventBus
from simulator.execution_backend import ExecutionBackend, OrderRequest, SimPosition
from simulator.simulation_clock import SimulationClock

logger = logging.getLogger("Sim.Adapter")

# Minimum M5 and M15 bars needed before strategy runs (matches live behavior)
MIN_M5_BARS = 20
MIN_M15_BARS = 10

# Health check interval in candles (every 6 M5 candles ≈ HEALTH_CHECK_SEC=30)
# In live: HEALTH_CHECK_SEC=30, signal check every 5s → every 30s = every 6 candles
# In replay: check every candle to maintain thesis integrity
HEALTH_CHECK_EVERY_N_CANDLES = 1


class StrategyAdapter:
    """
    Bridges the EventBus "new_candle" events to the real SignalScorer.evaluate().

    Instantiated once per replay session. Subscribes to NEW_CANDLE events.
    On each new candle, mirrors the full lifecycle from TradingEngine:
      1. Update simulated account (RiskManager)
      2. Manage open positions (health, trailing)
      3. Evaluate signals (with isolated clock patch)
      4. Execute valid signals through ExecutionBackend
      5. Register trades with health monitor
      6. Emit trade_open / trade_close events via EventBus
    """

    def __init__(
        self,
        execution_backend: ExecutionBackend,
        sim_clock: SimulationClock,
        event_bus: EventBus,
    ) -> None:
        """
        Args:
            execution_backend: SimulatedExecutionBackend (or MT5 in future)
            sim_clock:         Isolated simulation clock (no global patching)
            event_bus:         Central event bus for publishing trade events
        """
        # ── Core MARK III components (REAL, UNMODIFIED) ─────────────
        self._scorer       = IchimokuStrategy()  # v4.0: Ichimoku Cloud strategy
        self._health_mon   = TradeHealthMonitor() # Real health monitor
        self._risk_mgr     = RiskManager()        # Real risk manager
        self._pseudo_delta = PseudoDelta()        # For health context
        self._speed_check  = MomentumSpeed()      # For current regime

        # ── Simulation infrastructure ────────────────────────────────
        self._backend = execution_backend
        self._clock   = sim_clock
        self._bus     = event_bus

        # ── Internal state ───────────────────────────────────────────
        self._candle_count   = 0
        self._signals_fired  = 0
        self._last_market_data: Optional[Dict] = None  # For health check

        # Subscribe to data events
        event_bus.subscribe(EventBus.NEW_CANDLE, self._on_new_candle)

        logger.info(
            "[Adapter] StrategyAdapter initialized — "
            "real SignalScorer, RiskManager, TradeHealthMonitor"
        )

    # ── Event handler ────────────────────────────────────────────────

    def _get_simulated_session(self, dt_utc) -> str:
        import pytz
        cot_tz = pytz.timezone("America/Bogota")
        dt_cot = dt_utc.astimezone(cot_tz)
        h = dt_cot.hour + dt_cot.minute / 60.0
        
        # v2.5: Optimized session windows discovered via timezone audit.
        # European/London open (05:30-08:30 COT) and Late US Morning
        # (11:00-14:00 COT) produce clean directional trends ideal
        # for PULLBACK entries. Avoids the chaotic US Open (08:30-10:00).
        if 5.5 <= h <= 8.5:
            return "MORNING"
        elif 11.0 <= h <= 14.0:
            return "AFTERNOON"
        elif 20.5 <= h <= 23.5:
            return "NIGHT"
        else:
            return "OUT_OF_SESSION"

    def _on_new_candle(
        self,
        timestamp: pd.Timestamp,
        candle_index: int,
        m5_snapshot: pd.DataFrame,
        m15_snapshot: pd.DataFrame,
        raw_candle: pd.Series,
        **kwargs,
    ) -> None:
        """
        Called on every new_candle event from DataFeed.
        Mirrors TradingEngine._main_loop() signal check cycle.
        """
        self._candle_count += 1
        sim_ts = self._clock.now()

        # ── 1. Notify broker of new candle (SL/TP checks) ───────────
        fill_events = self._backend.on_new_candle(raw_candle)
        for fe in fill_events:
            self._handle_fill_event(fe, timestamp)

        # ── 2. Update RiskManager with simulated account state ───────
        account = self._backend.get_account()
        
        dt_utc = self._clock.now_dt()
        session_name = self._get_simulated_session(dt_utc)
        sim_date_str = dt_utc.strftime("%Y-%m-%d")
        
        # Suffix the date string to naturally trigger standard daily
        # risk manager resets per discrete trading session block.
        if session_name != "OUT_OF_SESSION":
            override_str = f"{sim_date_str}_{session_name}"
        else:
            override_str = sim_date_str

        self._risk_mgr.update_balance(
            balance=account["balance"],
            equity=account["equity"],
            today_override=override_str,
        )

        # ── 3. Check risk limits ─────────────────────────────────────
        can_trade, risk_reason = self._risk_mgr.can_trade()
        
        # Enforce discrete 3-hour operation windows natively
        if session_name == "OUT_OF_SESSION":
            can_trade = False
            risk_reason = "Out of scheduled 3-hour trading session"

        if not can_trade:
            logger.debug(f"[Adapter] Risk block: {risk_reason}")
            # Still manage open positions even when blocked from new trades
            self._manage_open_positions(m5_snapshot, m15_snapshot, timestamp)
            return

        # ── 4. Manage open positions (health + trailing) ─────────────
        self._manage_open_positions(m5_snapshot, m15_snapshot, timestamp)

        # ── 5. Build market_data dict (identical format to live) ─────
        if len(m5_snapshot) < MIN_M5_BARS:
            return  # Not enough data for strategy evaluation
        if len(m15_snapshot) < MIN_M15_BARS:
            return  # M15 not yet initialized

        market_data = {
            "ES": {
                "M5":  m5_snapshot,
                "M15": m15_snapshot,
            }
            # NQ intentionally omitted — US500-only mode
            # SignalScorer handles missing NQ gracefully (returns no NQ signal)
        }
        self._last_market_data = market_data

        # ── 6. Evaluate signals with isolated clock patch ────────────
        # SimulationClock patches time.time() ONLY within scorer module,
        # only for the duration of this with-block (no global side effects)
        with self._clock.patch_scorer_context():
            signals = self._scorer.evaluate(market_data)

        # ── 7. Execute valid signals ─────────────────────────────────
        for signal in signals:
            self._signals_fired += 1
            self._execute_signal(signal, market_data, timestamp, m5_snapshot=m5_snapshot)

    # ── Signal execution ─────────────────────────────────────────────

    def _validate_signal_sim(
        self,
        signal: Signal,
        m5_snapshot: pd.DataFrame,
    ) -> bool:
        """
        Gap 4 fix: Pre-trade validations that mirror OrderManager._validate_signal().
        Checks spread, duplicate positions, and max positions.
        Returns True if the signal passes all checks.
        """
        sym_cfg = settings.SYMBOL_MAP.get(signal.symbol, {})

        # 1. Check current spread against max allowed
        current_spread = self._backend.get_spread(signal.symbol)
        max_spread = sym_cfg.get("max_spread_pts", 5.0)
        if current_spread > max_spread:
            logger.debug(
                f"[Adapter] ⛔ Spread too wide: {current_spread:.2f} > "
                f"{max_spread:.2f} │ Skipping {signal.symbol}"
            )
            return False

        # 2. Avoid duplicate positions on the same symbol
        open_positions = self._backend.get_open_positions()
        on_same_symbol = [
            p for p in open_positions if p.symbol == signal.symbol
        ]
        if on_same_symbol:
            logger.debug(
                f"[Adapter] Already have {len(on_same_symbol)} open position(s) "
                f"on {signal.symbol} │ Skipping"
            )
            return False

        # 3. Check max positions limit
        if len(open_positions) >= settings.MAX_POSITIONS:
            logger.debug(
                f"[Adapter] Max positions reached "
                f"({len(open_positions)}/{settings.MAX_POSITIONS}) │ Skipping"
            )
            return False

        return True

    def _execute_signal(
        self,
        signal: Signal,
        market_data: Dict,
        timestamp: pd.Timestamp,
        m5_snapshot: pd.DataFrame = None,
    ) -> None:
        """
        Mirrors TradingEngine._run_analysis_cycle() trade execution block.
        Executes via ExecutionBackend, registers with health monitor.
        """
        if signal.action == "NEUTRAL":
            return

        # Gap 4 fix: run pre-trade validations (spread, duplicates, max positions)
        if m5_snapshot is not None and not self._validate_signal_sim(signal, m5_snapshot):
            return

        # Capture full context for trade logging (Req 13)
        context = self._capture_signal_context(signal, market_data)

        # ── Calculate realistic lot size using simulated equity ──
        account_info = self._backend.get_account()
        equity = account_info.get("equity", 0.0)
        sym_cfg = settings.SYMBOL_MAP.get(signal.symbol, {})

        lot = sym_cfg.get("min_lot", 0.1)
        if equity > 0 and signal.sl_distance > 0:
            risk_usd = equity * (settings.RISK_PER_TRADE_PCT / 100.0)
            point_value = self._backend.get_symbol_info(signal.symbol).get("tick_value", 1.0)
            raw_lot = risk_usd / (signal.sl_distance * point_value)

            lot_step = sym_cfg.get("lot_step", 0.1)
            lot = round(raw_lot / lot_step) * lot_step
            lot = max(sym_cfg.get("min_lot", 0.1), min(lot, sym_cfg.get("max_lot", 0.3)))
            lot = max(settings.MIN_LOT_ABSOLUTE, min(lot, settings.MAX_LOT_ABSOLUTE))

            # Safety fuse: Max loss per trade
            estimated_loss = signal.sl_distance * point_value * lot
            if estimated_loss > settings.MAX_LOSS_PER_TRADE_USD:
                raw_safe_lot = settings.MAX_LOSS_PER_TRADE_USD / (signal.sl_distance * point_value)
                safe_lot = round(raw_safe_lot / lot_step) * lot_step
                lot = max(sym_cfg.get("min_lot", 0.1), safe_lot)

        # Build order request
        order = OrderRequest.from_signal(signal, context=context, lot_size=lot)
        
        # Calculate absolute SL/TP prices (Req 9 simulation parity)
        m5_df = market_data[signal.symbol].get("M5")
        if m5_df is not None and not m5_df.empty:
            current_price = m5_df.iloc[-1]["close"]
            if signal.action == "BUY":
                order.sl_price = round(current_price - signal.sl_distance, 2)
                order.tp_price = round(current_price + signal.tp_distance, 2)
            else:
                order.sl_price = round(current_price + signal.sl_distance, 2)
                order.tp_price = round(current_price - signal.tp_distance, 2)

        result = self._backend.submit_order(order)

        if not result.success:
            if "Deferred" not in result.error:
                logger.warning(f"[Adapter] Order rejected: {result.error}")
            return

        # Register with risk manager (trade count)
        self._risk_mgr.register_trade()

        # Register with health monitor (thesis validation from entry context)
        self._health_mon.register_trade(
            ticket=result.ticket,
            entry_time=self._clock.now(),  # Simulated timestamp, not wall time
            entry_regime=signal.regime,
            entry_atr=signal.atr,
            structural_ref=signal.structural_ref,
            direction=signal.action,
            entry_speed=signal.speed_score,
            profile=signal.profile,
            setup_type=getattr(signal, "setup_type", "UNKNOWN"),  # P1 fix
        )

        logger.info(
            f"[Adapter] 🎯 {signal.action} {signal.symbol} │ "
            f"Ticket={result.ticket} │ Score={signal.score} │ "
            f"Fill={result.fill_price:.2f} │ "
            f"SL={order.sl_price:.2f} │ TP={order.tp_price:.2f}"
        )

        # Emit trade_open event for logger/visualizer/dashboard subscribers
        self._bus.emit(
            EventBus.TRADE_OPEN,
            ticket=result.ticket,
            signal=signal,
            fill_price=result.fill_price,
            spread=result.spread_applied,
            slippage=result.slippage_applied,
            latency_ms=result.latency_ms,
            timestamp=timestamp,
            context=context,
            lot_size=lot,   # P2 fix: pass explicit lot so logger records it
        )

    # ── Position management (mirrors _manage_open_positions) ─────────

    def _manage_open_positions(
        self,
        m5_snapshot: pd.DataFrame,
        m15_snapshot: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> None:
        """
        Mirrors TradingEngine._manage_open_positions() exactly.
        Called every candle (health check interval = 1 candle in replay).
        """
        positions: list[SimPosition] = self._backend.get_open_positions()
        if not positions:
            return

        # Compute current market context for thesis validation
        pressure_dir, current_regime, current_speed, current_atr = (
            self._compute_market_context(m5_snapshot)
        )

        sym_cfg = settings.SYMBOL_MAP.get("ES", {})
        max_spread = sym_cfg.get("max_spread_pts", 5.0)

        for pos in positions:
            current_price = float(m5_snapshot.iloc[-1]["close"])
            current_spread = self._backend.get_spread(pos.symbol)

            # Track P&L trajectory (mirrors order_manager.append_trajectory)
            # (In simulation, P&L is computed from prices, not MT5 pos.profit)
            if pos.direction == "BUY":
                unrealized_pnl = (current_price - pos.entry_price) * pos.lot_size
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.lot_size

            # Evaluate health
            health = self._health_mon.evaluate(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction=pos.direction,
                profit=unrealized_pnl,
                entry_price=pos.entry_price,
                current_price=current_price,
                current_spread=current_spread,
                max_spread=max_spread,
                recent_candles=pressure_dir,
                current_regime=current_regime,
                current_speed=current_speed,
                current_atr=current_atr,
                current_time=self._clock.now(),
            )

            # ── Trailing stop (mirrors main.py exactly) ──────────────
            entry_ctx = self._health_mon._entry_contexts.get(pos.ticket)
            if entry_ctx and entry_ctx.profile and health.is_healthy:
                new_sl = self._compute_trail_stop(
                    pos=pos,
                    entry_ctx=entry_ctx,
                    current_price=current_price,
                    current_regime=current_regime,
                    current_speed=current_speed,
                    pressure_dir=pressure_dir,
                    current_atr=current_atr,
                    health_score=health.score,
                    giveback_pct=health.diagnostics.get("giveback", {}).get("giveback_pct", 0.0),
                )
                if new_sl is not None:
                    self._backend.modify_sl_tp(pos.ticket, sl=new_sl)

            # ── Act on health recommendation ──────────────────────────
            if health.should_exit:
                logger.info(
                    f"[Adapter] 🏥 Health EXIT #{pos.ticket} │ {health.reason}"
                )
                self._backend.close_position(pos.ticket, "MK3_health")
                self._health_mon.clear_trade(pos.ticket)
                self._bus.emit(
                    EventBus.HEALTH_EVENT,
                    ticket=pos.ticket,
                    health_score=health.score,
                    action="EXIT",
                    reason=health.reason,
                )
                # Also emit trade_close for logger
                closed_pnl = unrealized_pnl
                self._bus.emit(
                    EventBus.TRADE_CLOSE,
                    ticket=pos.ticket,
                    exit_price=current_price,
                    pnl_usd=closed_pnl,
                    reason="MK3_health",
                    duration_sec=self._clock.now() - pos.open_time,
                )

            elif health.should_tighten:
                logger.info(
                    f"[Adapter] 🏥 Health TIGHTEN #{pos.ticket} │ {health.reason}"
                )
                # Move SL to breakeven (entry price)
                self._backend.modify_sl_tp(pos.ticket, sl=pos.entry_price)
                self._bus.emit(
                    EventBus.HEALTH_EVENT,
                    ticket=pos.ticket,
                    health_score=health.score,
                    action="TIGHTEN",
                    reason=health.reason,
                )

    def _handle_fill_event(self, fill_event: dict, timestamp: pd.Timestamp) -> None:
        """
        Called when broker reports an SL or TP hit.
        Cleans up health monitor and emits trade_close event.
        """
        ticket = fill_event["ticket"]
        self._health_mon.clear_trade(ticket)

        self._bus.emit(
            EventBus.STOP_HIT,
            ticket=ticket,
            hit_type=fill_event["hit_type"],
            price=fill_event["price"],
            candle_ts=timestamp,
        )
        self._bus.emit(
            EventBus.TRADE_CLOSE,
            ticket=ticket,
            exit_price=fill_event["price"],
            pnl_usd=fill_event["pnl_usd"],
            reason=fill_event["reason"],
            duration_sec=None,   # TradeLogger computes from open_time
        )

    # ── Market context helpers ───────────────────────────────────────

    def _compute_market_context(
        self, m5_snapshot: pd.DataFrame
    ) -> tuple:
        """
        Compute current regime, speed, pressure, and ATR for health monitor.
        Mirrors the context computation in _manage_open_positions().
        Returns (pressure_dir, regime, speed_score, atr).
        """
        pressure_dir   = "NEUTRAL"
        current_regime = "NORMAL"
        current_speed  = 50
        current_atr    = 0.0

        if m5_snapshot is None or len(m5_snapshot) < 15:
            return pressure_dir, current_regime, current_speed, current_atr

        try:
            pressure = self._pseudo_delta.calculate(m5_snapshot)
            pressure_dir = pressure.pressure_direction
        except Exception:
            pass

        try:
            speed_ctx = self._speed_check.measure(m5_snapshot)
            current_regime = speed_ctx.regime
            current_speed  = speed_ctx.speed_score
        except Exception:
            pass

        try:
            current_atr = float(m5_snapshot.iloc[-1].get("atr", 0))
        except Exception:
            pass

        return pressure_dir, current_regime, current_speed, current_atr

    def _compute_trail_stop(
        self,
        pos: SimPosition,
        entry_ctx,
        current_price: float,
        current_regime: str,
        current_speed: int,
        pressure_dir: str,
        current_atr: float,
        health_score: int,
        giveback_pct: float,
    ) -> Optional[float]:
        """
        Gap 2 fix: Full adaptive trailing stop — mirrors OrderManager.trail_stop() exactly.

        Applies the same 6 contextual adjustments that the live system uses:
          1. Health score < 60     → tighten activation and distance
          2. Giveback >= 45%       → tighten activation and distance further
          3. Speed decay < 65%     → reduce activation and distance
          4. ATR contraction < 65% → reduce trail distance
          5. Pressure opposing     → tighten trail distance
          6. Regime drop           → tighten activation and distance

        STRICT RULE: may only tighten risk (move SL closer to price).
        Returns new SL price, or None if no adjustment is needed/valid.
        """
        if entry_ctx.atr <= 0 or current_atr <= 0:
            return None

        profile = entry_ctx.profile
        trail_activation_atr = profile.trail_activation_atr
        trail_distance_atr   = profile.trail_distance_atr

        # ── Regime rank table (mirrors OrderManager) ─────────────
        regime_rank = {
            "DEAD": 0, "SLOW_TREND": 1, "NORMAL": 2,
            "FAST": 3, "EXPLOSIVE": 4,
        }

        speed_ratio    = current_speed / entry_ctx.speed if entry_ctx.speed > 0 else 1.0
        atr_ratio      = current_atr / entry_ctx.atr if entry_ctx.atr > 0 else 1.0
        pressure_opposing = (
            isinstance(pressure_dir, str)
            and pressure_dir not in (pos.direction, "NEUTRAL", "UNKNOWN")
        )
        regime_dropped = (
            regime_rank.get(current_regime, 2)
            < regime_rank.get(entry_ctx.regime, 2)
        )

        effective_activation = trail_activation_atr
        effective_distance   = trail_distance_atr
        trail_notes = []

        # 1. Health score degraded
        if health_score < 60:
            effective_activation *= 0.80
            effective_distance   *= 0.75
            trail_notes.append(f"health={health_score}")

        # 2. Significant profit giveback
        if giveback_pct >= 0.45:
            effective_activation *= 0.75
            effective_distance   *= 0.65
            trail_notes.append(f"giveback={giveback_pct:.0%}")

        # 3. Speed decay
        if speed_ratio < 0.65:
            effective_activation *= 0.85
            effective_distance   *= 0.75
            trail_notes.append(
                f"speed={entry_ctx.speed}->{current_speed}"
            )

        # 4. ATR contraction
        if atr_ratio < 0.65:
            effective_distance *= 0.80
            trail_notes.append(f"atr={atr_ratio:.2f}x")

        # 5. Pressure opposing trade direction
        if pressure_opposing:
            effective_distance *= 0.70
            trail_notes.append(f"pressure={pressure_dir}")

        # 6. Regime drop (momentum collapsed)
        if regime_dropped:
            effective_activation *= 0.85
            effective_distance   *= 0.80
            trail_notes.append(
                f"regime={entry_ctx.regime}->{current_regime}"
            )

        # Clamp: never go below 0.35 × base, never exceed base
        effective_activation = max(
            0.35, min(trail_activation_atr, effective_activation)
        )
        effective_distance = max(
            0.35, min(trail_distance_atr, effective_distance)
        )

        # ── Calculate profit distance in ATR units ────────────────
        if pos.direction == "BUY":
            profit_distance = current_price - pos.entry_price
        else:
            profit_distance = pos.entry_price - current_price

        profit_in_atr = profit_distance / entry_ctx.atr if entry_ctx.atr > 0 else 0.0

        # ── Check activation threshold ────────────────────────────
        if profit_in_atr < effective_activation:
            return None  # Not enough profit yet

        # ── Calculate new SL ──────────────────────────────────────
        trail_distance_price = effective_distance * entry_ctx.atr

        if pos.direction == "BUY":
            new_sl = current_price - trail_distance_price
            # TIGHTEN-ONLY: new SL must be strictly above current SL
            if pos.sl_price > 0 and new_sl <= pos.sl_price:
                return None
            logger.debug(
                f"[Adapter] Trail BUY #{pos.ticket}: "
                f"SL {pos.sl_price:.2f} → {new_sl:.2f} "
                f"({profit_in_atr:.2f}xATR profit"
                + (f" | {', '.join(trail_notes)}" if trail_notes else "") + ")"
            )
            return new_sl
        else:  # SELL
            new_sl = current_price + trail_distance_price
            # TIGHTEN-ONLY: new SL must be strictly below current SL
            if pos.sl_price > 0 and new_sl >= pos.sl_price:
                return None
            logger.debug(
                f"[Adapter] Trail SELL #{pos.ticket}: "
                f"SL {pos.sl_price:.2f} → {new_sl:.2f} "
                f"({profit_in_atr:.2f}xATR profit"
                + (f" | {', '.join(trail_notes)}" if trail_notes else "") + ")"
            )
            return new_sl

    # ── Context capture for trade logging (Req 13) ───────────────────

    def _capture_signal_context(
        self, signal: Signal, market_data: Dict
    ) -> dict:
        """
        Captures the full signal decision context for the trade logger.
        This is WHY the trade happened — all score components + market state.
        """
        return {
            # Session Tracking
            "simulated_session":  self._get_simulated_session(self._clock.now_dt()),
            # Score components
            "signal_score":       signal.score,
            "trend_score":        signal.trend_score,
            "structure_score":    signal.structure_score,
            "rejection_score":    signal.rejection_score,
            "volume_score":       signal.volume_score,
            "pressure_score":     signal.pressure_score,
            "correlation_score":  signal.correlation_score,
            "factors_active":     signal.factors_active,
            # Entry context
            "regime":             signal.regime,
            "setup_type":         signal.setup_type,
            "atr":                signal.atr,
            "sl_distance":        signal.sl_distance,
            "tp_distance":        signal.tp_distance,
            "structural_ref":     signal.structural_ref,
            "speed_score":        signal.speed_score,
            # Profile
            "profile_min_score":  signal.profile.min_score if signal.profile else None,
            "profile_sl_mult":    signal.profile.sl_atr_mult if signal.profile else None,
            "profile_tp_mult":    signal.profile.tp_atr_mult if signal.profile else None,
        }

    # ── Properties ──────────────────────────────────────────────────

    @property
    def signals_fired(self) -> int:
        return self._signals_fired

    @property
    def candles_processed(self) -> int:
        return self._candle_count

    def summary(self) -> dict:
        account = self._backend.get_account()
        return {
            "candles_processed": self._candle_count,
            "signals_fired":     self._signals_fired,
            "balance":           account["balance"],
            "equity":            account["equity"],
            "risk_halted":       self._risk_mgr.trading_halted,
            "halt_reason":       self._risk_mgr.halt_reason,
            "trades_today":      self._risk_mgr.trades_today,
        }
