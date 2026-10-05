"""
main.py — MARK III Trading Engine Orchestrator

Main loop lifecycle:
  1. Initialize all modules (data, signal, execution, risk, health)
  2. Connect to MT5 and enable symbols
  3. Bootstrap order flow history (warmup)
  4. Loop:
     a. Check market hours
     b. Update risk manager (daily P&L, drawdown, trailing profit)
     c. Manage open positions (trade health monitor)
     d. Fetch new order flow data
     e. Evaluate signals
     f. Execute actionable signals on MT5
     g. Heartbeat logging
  5. Graceful shutdown on Ctrl+C

Improvements over v1 (from MARK I lessons):
  - Trade Health Monitor: prevents stale trades
  - Daily Risk Manager: drawdown halt, profit target, trailing profit
  - Market Hours Filter: avoids 10018 errors
  - Proper daily counter reset at midnight UTC
"""
from __future__ import annotations

import signal
import time
from datetime import datetime, timezone

from colorama import Fore, Style, init as colorama_init

from config import settings
from data_handler.mt5_data import MT5DataProvider
from execution_engine.mt5_connector import MT5Connector
from execution_engine.order_manager import OrderManager
from execution_engine.risk_manager import RiskManager
from execution_engine.trade_health import TradeHealthMonitor
from monitoring.api_server import APIServer
from monitoring.logger import get_logger

logger = get_logger("Engine")


class TradingEngine:
    """
    Orchestrates the MARK III order flow trading system.
    Connects futures order flow analysis with CFD execution on MT5.
    """

    def __init__(self) -> None:
        self._running = False
        self._cycle_count = 0

        # ── Initialize modules ──────────────────────────────────
        self._connector = MT5Connector()
        self._data_provider = None  # Initialized after MT5 connects

        # Import here to avoid circular imports
        # v4.0: Ichimoku Cloud strategy replaces 6-module SignalScorer
        from signal_engine.ichimoku_strategy import IchimokuStrategy
        self._signal_engine = IchimokuStrategy()
        self._pseudo_delta = None  # Initialized after imports

        self._order_manager = OrderManager(self._connector)
        self._risk_manager = RiskManager()
        self._health_monitor = TradeHealthMonitor()

        # ── State tracking ──────────────────────────────────────
        self._signals_generated: int = 0
        self._start_time: float = 0.0
        self.recent_signals = []  # For dashboard
        self._trailing_acknowledged: bool = False  # One-shot trailing profit handler

        # ── Dashboard ───────────────────────────────────────────
        self._api_server = None
        if settings.DASHBOARD_ENABLED:
            self._api_server = APIServer(self)

    # ─── Properties ─────────────────────────────────────────────
    def uptime_sec(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.time() - self._start_time

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    # ─── Start ──────────────────────────────────────────────────
    def start(self) -> None:
        """Starts the engine. Blocks until stop() is called."""
        self._print_banner()

        # Connect to MT5
        logger.info("🔌 Connecting to MetaTrader 5...")
        if not self._connector.connect():
            logger.critical("❌ Cannot connect to MT5. Exiting.")
            return

        # Enable symbols in Market Watch
        for sym_key, sym_cfg in settings.SYMBOL_MAP.items():
            mt5_name = sym_cfg["mt5_name"]
            if self._connector.enable_symbol(mt5_name):
                logger.info(f"   ✅ Symbol enabled: {mt5_name} ({sym_cfg['display_name']})")
            else:
                logger.warning(f"   ⚠️  Could not enable: {mt5_name}")

        self._running = True
        self._start_time = time.time()

        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            f"🚀 MARK III v4.0 Ichimoku started │ Mode: {settings.TRADING_MODE} │ "
            f"Symbols: {', '.join(settings.ACTIVE_SYMBOLS)} │ "
            f"Strategy: Ichimoku+VWAP+RSI │ "
            f"SL=1.5×ATR TP=3.0×ATR (1:2 R:R) │ "
            f"Risk: {settings.RISK_PER_TRADE_PCT}%"
        )

        if self._api_server:
            self._api_server.run_in_background()
            logger.info(f"🖥️  Dashboard running at http://{settings.DASHBOARD_HOST}:{settings.DASHBOARD_PORT}")

        # Initialize data provider now that MT5 is connected
        self._data_provider = MT5DataProvider(self._connector)

        # Initialize PseudoDelta for health monitor
        from signal_engine.pseudo_delta import PseudoDelta
        self._pseudo_delta = PseudoDelta()

        logger.info("✅ Data provider ready. Entering main loop.")

        self._main_loop()

    def stop(self) -> None:
        """Signals the engine to stop gracefully."""
        logger.info("🛑 Stopping MARK III...")
        self._running = False

    # ─── Main Loop ──────────────────────────────────────────────
    def _main_loop(self) -> None:
        last_signal_check = 0.0
        last_heartbeat = 0.0
        last_health_check = 0.0

        while self._running:
            now = time.time()

            # ── Update Risk Manager (every cycle) ────────────────
            try:
                account = self._connector.get_account_info()
                if account:
                    self._risk_manager.update_balance(
                        account.get("balance", 0),
                        account.get("equity", 0),
                    )
            except Exception:
                pass  # Don't crash the loop on account info failure

            # ── Heartbeat ────────────────────────────────────────
            if now - last_heartbeat >= settings.HEARTBEAT_INTERVAL_SEC:
                self._heartbeat()
                last_heartbeat = now

            # ── Trade Health Monitor & Logger ────────────────────
            if now - last_health_check >= settings.HEALTH_CHECK_INTERVAL_SEC:
                self._manage_open_positions()
                self._order_manager.check_closed_trades()
                last_health_check = now

            # ── Risk Manager: check trailing profit trigger ──────
            if self._risk_manager.trailing_profit_triggered:
                if not self._trailing_acknowledged:
                    # ONE-SHOT: close positions, log, then go quiet
                    logger.warning(
                        f"🛑 Trailing profit triggered — closing all positions │ "
                        f"Peak=${self._risk_manager.peak_daily_pnl:+.2f} → "
                        f"Current=${self._risk_manager.daily_pnl:+.2f}"
                    )
                    self._order_manager.close_all_positions()
                    self._trailing_acknowledged = True
                    logger.info("🏁 Day finished — engine entering quiet mode (heartbeat only)")
                # Halted state: only heartbeat runs, skip all trading logic
                time.sleep(1)
                continue

            # ── Signal Check Cycle ───────────────────────────────
            if now - last_signal_check >= settings.SIGNAL_CHECK_INTERVAL_SEC:
                # Ensure MT5 is alive
                if not self._connector.ensure_connected():
                    logger.error("MT5 unavailable. Waiting 10s...")
                    time.sleep(10)
                    last_signal_check = now
                    continue

                # Check general market hours (weekend/daily resets)
                if not OrderManager.is_market_open():
                    if self._cycle_count % 60 == 0:  # Log once every ~15 min
                        logger.info("🌙 Market closed — waiting for next general session")
                    self._cycle_count += 1
                    last_signal_check = now
                    time.sleep(1)
                    continue

                # NEW in v2.5: Check optimized strategy sessions
                from execution_engine.session_manager import SessionManager
                in_session, session_name = SessionManager.is_in_session()
                if not in_session:
                    if self._cycle_count % 60 == 0:
                        logger.info("⏳ Strategy Rest Mode — waiting for optimal hours (05:30-08:30 or 11:00-14:00 COT)")
                    self._cycle_count += 1
                    last_signal_check = now
                    time.sleep(1)
                    continue

                # Check risk manager
                can_trade, reason = self._risk_manager.can_trade()
                if not can_trade:
                    if self._cycle_count % 20 == 0:  # Log every ~5 min
                        logger.info(f"📊 Trading paused: {reason}")
                else:
                    self._run_analysis_cycle()

                self._cycle_count += 1
                last_signal_check = now

            time.sleep(1)  # Main loop granularity

        self._shutdown()

    # ─── Analysis Cycle ─────────────────────────────────────────
    def _run_analysis_cycle(self) -> None:
        """Fetches real market data, evaluates signals, executes trades."""
        try:
            # Fetch real OHLCV data from MT5
            market_data = self._data_provider.get_all_symbols()

            if not market_data:
                logger.debug("No market data available")
                return

            # Evaluate signals using the new market structure engine
            signals = self._signal_engine.evaluate(market_data)

            # Execute actionable signals
            for sig in signals:
                self._signals_generated += 1
                self.recent_signals.insert(0, sig)  # Add to top of list
                if len(self.recent_signals) > 50:   # Keep last 50
                    self.recent_signals.pop()

                result = self._order_manager.execute_signal(sig)
                if result:
                    # Register with risk manager (proper daily tracking)
                    self._risk_manager.register_trade()
                    # Register with health monitor (v2.3: with frozen regime profile)
                    self._health_monitor.register_trade(
                        ticket=result["ticket"],
                        entry_time=time.time(),
                        entry_regime=sig.regime,
                        entry_atr=sig.atr,
                        structural_ref=sig.structural_ref,
                        direction=sig.action,
                        entry_speed=sig.speed_score,
                        profile=sig.profile,
                    )

        except Exception as exc:
            logger.error(f"Error in analysis cycle: {exc}", exc_info=True)

    # ─── Trade Health Management ────────────────────────────────
    def _manage_open_positions(self) -> None:
        """
        Checks the health of all open MARK III positions.
        Takes action based on health score (tighten SL or close).

        v2: Passes current market context (regime, speed, ATR) to the
        health monitor for thesis validation. Does NOT re-run the scorer.
        """
        if settings.TRADING_MODE != "AUTO":
            return

        try:
            positions = self._connector.get_open_positions(
                magic=settings.MAGIC_NUMBER
            )
            if not positions:
                return

            # Get current market data for pressure direction + regime
            market_data = self._data_provider.get_cached()

            for pos in positions:
                # Determine which symbol key this position belongs to
                sym_key = None
                for key, cfg in settings.SYMBOL_MAP.items():
                    if cfg["mt5_name"] == pos.symbol:
                        sym_key = key
                        break

                if not sym_key:
                    continue

                sym_cfg = settings.SYMBOL_MAP[sym_key]
                direction = "BUY" if pos.type == 0 else "SELL"

                # Get current spread
                spread = self._connector.get_spread(pos.symbol)
                max_spread = sym_cfg.get("max_spread_pts", 5.0)

                # ── v2: Compute current market context ────────────
                pressure_dir = "NEUTRAL"
                current_regime = "NORMAL"
                current_speed = 50
                current_atr = 0.0

                if market_data and sym_key in market_data:
                    sym_data = market_data[sym_key]
                    m5 = sym_data.get("M5")
                    m15 = sym_data.get("M15")

                    if m5 is not None and len(m5) > 15:
                        # Pressure direction
                        if self._pseudo_delta:
                            try:
                                pressure = self._pseudo_delta.calculate(m5)
                                pressure_dir = pressure.pressure_direction
                            except Exception:
                                pass

                        # Current regime from MomentumSpeed
                        try:
                            from signal_engine.momentum_speed import MomentumSpeed
                            speed_checker = MomentumSpeed()
                            speed_ctx = speed_checker.measure(m5)
                            current_regime = speed_ctx.regime
                            current_speed = speed_ctx.speed_score
                        except Exception:
                            pass

                        # Current ATR
                        try:
                            last_m5 = m5.iloc[-1]
                            atr_val = float(last_m5.get("atr", 0))
                            if atr_val > 0:
                                current_atr = atr_val
                        except Exception:
                            pass

                # Prepend trajectory tracking
                self._order_manager.append_trajectory(pos.ticket, pos.profit)

                # Evaluate health (v2: with full market context)
                health = self._health_monitor.evaluate(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    profit=pos.profit,
                    entry_price=pos.price_open,
                    current_price=pos.price_current,
                    current_spread=spread,
                    max_spread=max_spread,
                    recent_candles=pressure_dir,
                    current_regime=current_regime,
                    current_speed=current_speed,
                    current_atr=current_atr,
                )

                # ── v2.3: Regime-aware trailing stop ─────────────
                # Attempt trail BEFORE acting on health, so the SL
                # is already tightened when health evaluates.
                entry_ctx = self._health_monitor._entry_contexts.get(pos.ticket)
                if entry_ctx and entry_ctx.profile and health.is_healthy:
                    self._order_manager.trail_stop(
                        ticket=pos.ticket,
                        trail_activation_atr=entry_ctx.profile.trail_activation_atr,
                        trail_distance_atr=entry_ctx.profile.trail_distance_atr,
                        entry_atr=entry_ctx.atr,
                        entry_price=pos.price_open,
                        direction=direction,
                        entry_speed=entry_ctx.speed,
                        current_speed=current_speed,
                        entry_regime=entry_ctx.regime,
                        current_regime=current_regime,
                        pressure_dir=pressure_dir,
                        current_atr=current_atr,
                        health_score=health.score,
                        giveback_pct=health.diagnostics.get("giveback", {}).get("giveback_pct", 0.0),
                    )

                # Act on health result
                if health.should_exit:
                    logger.warning(
                        f"[{sym_key}] 🏥 Health EXIT │ {health.reason}"
                    )
                    closed = self._order_manager.close_position(
                        pos.ticket, "MK3_health"
                    )
                    if closed:
                        self._health_monitor.clear_trade(pos.ticket)

                elif health.should_tighten:
                    logger.info(
                        f"[{sym_key}] 🏥 Health TIGHTEN │ {health.reason}"
                    )
                    self._order_manager.modify_sl_to_breakeven(pos.ticket)

        except Exception as exc:
            logger.error(f"Error in health check: {exc}", exc_info=True)

    # ─── Heartbeat ──────────────────────────────────────────────
    def _heartbeat(self) -> None:
        """Periodic status log with account info and system state."""
        try:
            account = self._connector.get_account_info()
            equity = account.get("equity", 0)
            balance = account.get("balance", 0)
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

            # Count open MARK III positions
            positions = self._connector.get_open_positions(magic=settings.MAGIC_NUMBER)
            open_count = len(positions)

            uptime_min = (time.time() - self._start_time) / 60

            # Risk manager info
            rm = self._risk_manager
            day_pnl = rm.daily_pnl
            market_status = "🟢 OPEN" if OrderManager.is_market_open() else "🌙 CLOSED"

            logger.info(
                f"💓 Heartbeat │ {ts} │ "
                f"Balance=${balance:.2f} │ Equity=${equity:.2f} │ "
                f"Day P&L=${day_pnl:+.2f} │ "
                f"Positions={open_count} │ "
                f"Signals={self._signals_generated} │ "
                f"Trades={rm.trades_today}/{settings.MAX_TRADES_PER_DAY} │ "
                f"Market={market_status} │ "
                f"Regime-Aware={settings.REGIME_AWARE_EXECUTION} │ "
                f"Uptime={uptime_min:.0f}m"
            )
        except Exception as exc:
            logger.error(f"Heartbeat error: {exc}")

    # ─── Shutdown ───────────────────────────────────────────────
    def _handle_shutdown(self, signum, frame) -> None:
        logger.info(f"Shutdown signal received ({signum})")
        self.stop()

    def _shutdown(self) -> None:
        logger.info("Closing modules...")
        try:
            # Try to close positions on shutdown (only if market is open)
            if settings.TRADING_MODE == "AUTO" and OrderManager.is_market_open():
                positions = self._connector.get_open_positions(
                    magic=settings.MAGIC_NUMBER
                )
                if positions:
                    logger.info(
                        f"⚠️ {len(positions)} open positions detected at shutdown. "
                        f"Attempting to close..."
                    )
                    self._order_manager.close_all_positions()
            self._connector.disconnect()
        except Exception:
            pass
        logger.info("👋 MARK III stopped successfully.")

    # ─── Banner ─────────────────────────────────────────────────
    @staticmethod
    def _print_banner() -> None:
        colorama_init()
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}
  ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗    ██╗██╗██╗
  ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝    ██║██║██║
  ██╔████╔██║███████║██████╔╝█████╔╝     ██║██║██║
  ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗     ██║██║██║
  ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗    ██║██║██║
  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝╚═╝╚═╝
{Style.RESET_ALL}
   {Fore.YELLOW}Ichimoku Cloud Strategy (v4.0){Style.RESET_ALL}
   {Fore.WHITE}ES (S&P 500) → US500Cash │ NQ (Nasdaq) → US100Cash{Style.RESET_ALL}
   {Fore.GREEN}Mode: {settings.TRADING_MODE} │ Health Monitor: ON │ Risk Manager: ON{Style.RESET_ALL}
   {Fore.CYAN}Ichimoku + VWAP + RSI │ SL=1.5×ATR TP=3.0×ATR{Style.RESET_ALL}
        """
        print(banner)


# ─── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    engine = TradingEngine()
    engine.start()
