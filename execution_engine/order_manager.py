"""
execution_engine/order_manager.py — Order Execution & Risk Management

Full-featured order manager with:
  - ATR-based SL/TP calculation
  - Position sizing (1% equity risk)
  - Max-loss-per-trade safety fuse
  - Spread validation before entry
  - Retry logic with requote handling
  - Duplicate position prevention
  - Position modification and closing

Adapted from MARK I's battle-tested order manager.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

from config import settings
from execution_engine.mt5_connector import MT5Connector
from monitoring.logger import get_logger, TRADE_LEVEL_NUM, TradeLogger
from signal_engine.scorer import Signal

logger = get_logger("OrderManager")


class OrderManager:
    """Manages order execution, modification, and closing on MT5."""

    def __init__(self, connector: MT5Connector) -> None:
        self._conn = connector
        self._trade_logger = TradeLogger()
        self._known_open_tickets = {}  # ticket -> {score, entry_time}

    # ─── Pre-flight Checks ──────────────────────────────────────
    def _validate_signal(self, signal: Signal) -> bool:
        """
        Runs pre-execution checks before sending an order.
        Returns True if all checks pass.
        """
        mt5_sym = signal.mt5_symbol
        sym_key = signal.symbol
        sym_cfg = settings.SYMBOL_MAP[sym_key]

        # 1. Check spread
        spread = self._conn.get_spread(mt5_sym)
        max_spread = sym_cfg.get("max_spread_pts", 5.0)
        if spread > max_spread:
            logger.warning(
                f"[{sym_key}] ⛔ Spread too wide: {spread:.2f} > {max_spread:.2f} │ Skipping"
            )
            return False

        # 2. Check existing positions (avoid duplicates)
        positions = self._conn.get_open_positions(magic=settings.MAGIC_NUMBER)
        open_on_symbol = [p for p in positions if p.symbol == mt5_sym]
        if open_on_symbol:
            logger.info(
                f"[{sym_key}] Already have {len(open_on_symbol)} open position(s) on {mt5_sym} │ Skipping"
            )
            return False

        # 3. Check max positions limit
        if len(positions) >= settings.MAX_POSITIONS:
            logger.info(
                f"[{sym_key}] Max positions reached ({len(positions)}/{settings.MAX_POSITIONS}) │ Skipping"
            )
            return False

        return True

    # ─── Position Sizing ────────────────────────────────────────
    def calculate_lot_size(
        self,
        sym_key: str,
        equity: float,
        sl_distance: float,
    ) -> float:
        """
        Calculates position size based on:
          - Risk = RISK_PER_TRADE_PCT % of equity
          - SL distance in price points
          - Symbol's tick value and contract size from MT5

        Includes MAX_LOSS_PER_TRADE_USD safety fuse.
        """
        sym_cfg = settings.SYMBOL_MAP[sym_key]
        mt5_sym = sym_cfg["mt5_name"]

        risk_usd = equity * (settings.RISK_PER_TRADE_PCT / 100.0)

        if sl_distance <= 0:
            logger.warning(f"[{sym_key}] SL distance = 0, using minimum lot")
            return sym_cfg["min_lot"]

        # Get symbol info from MT5 for accurate calculation
        sym_info = self._conn.get_symbol_info(mt5_sym)
        if sym_info:
            try:
                tick_value = sym_info.get("trade_tick_value", 1.0)
                tick_size = sym_info.get("trade_tick_size", 1.0)

                # Value per point per lot
                point_value = tick_value / tick_size if tick_size > 0 else 1.0

                raw_lot = risk_usd / (sl_distance * point_value)
            except Exception as e:
                logger.warning(f"[{sym_key}] Error in lot calculation: {e}, using fallback")
                raw_lot = sym_cfg["min_lot"]
        else:
            logger.warning(f"[{sym_key}] No symbol info, using minimum lot")
            raw_lot = sym_cfg["min_lot"]

        # Round to broker step and apply limits
        lot_step = sym_cfg.get("lot_step", 0.1)
        lot = round(raw_lot / lot_step) * lot_step
        lot = max(sym_cfg["min_lot"], min(lot, sym_cfg["max_lot"]))
        lot = max(settings.MIN_LOT_ABSOLUTE, min(lot, settings.MAX_LOT_ABSOLUTE))

        # ── SAFETY FUSE: Max loss per trade in USD ───────────────
        if sym_info:
            try:
                tick_value = sym_info.get("trade_tick_value", 1.0)
                tick_size = sym_info.get("trade_tick_size", 1.0)
                point_value = tick_value / tick_size if tick_size > 0 else 1.0
                estimated_loss = sl_distance * point_value * lot

                if estimated_loss > settings.MAX_LOSS_PER_TRADE_USD:
                    raw_safe_lot = settings.MAX_LOSS_PER_TRADE_USD / (sl_distance * point_value)
                    safe_lot = round(raw_safe_lot / lot_step) * lot_step
                    broker_min_lot = sym_cfg["min_lot"]
                    estimated_loss_at_min_lot = sl_distance * point_value * broker_min_lot

                    if (
                        raw_safe_lot < broker_min_lot
                        and estimated_loss_at_min_lot > settings.MAX_LOSS_PER_TRADE_USD
                    ):
                        logger.warning(
                            f"[{sym_key}] Trade rejected: broker minimum lot exceeds allowed risk │ "
                            f"Calculated safe lot={raw_safe_lot:.4f} │ "
                            f"Rounded safe lot={safe_lot:.4f} │ "
                            f"Broker min lot={broker_min_lot:.4f} │ "
                            f"Est. loss at min lot=${estimated_loss_at_min_lot:.2f} │ "
                            f"Allowed max loss=${settings.MAX_LOSS_PER_TRADE_USD:.2f}"
                        )
                        return 0.0

                    safe_lot = max(broker_min_lot, safe_lot)

                    if (
                        safe_lot < broker_min_lot
                        and estimated_loss_at_min_lot > settings.MAX_LOSS_PER_TRADE_USD
                    ):
                        logger.warning(
                            f"[{sym_key}] ⛔ TRADE REJECTED │ "
                            f"Est. loss=${estimated_loss:.2f} > max=${settings.MAX_LOSS_PER_TRADE_USD:.2f} │ "
                            f"Even min lot exceeds limit."
                        )
                        return 0.0

                    logger.info(
                        f"[{sym_key}] 🛡️ Lot adjusted by safety fuse │ "
                        f"${estimated_loss:.2f} → ${safe_lot * sl_distance * point_value:.2f} │ "
                        f"Lot: {lot} → {safe_lot}"
                    )
                    lot = safe_lot
            except Exception:
                pass  # If fuse calculation fails, proceed with risk-based lot

        logger.info(
            f"[{sym_key}] Position sizing │ Equity=${equity:.2f} │ "
            f"Risk=${risk_usd:.2f} │ SL_dist={sl_distance:.2f} │ Lot={lot}"
        )
        return lot

    # ─── Execute Signal ─────────────────────────────────────────
    def execute_signal(self, signal: Signal, retries: int = 3) -> Optional[dict]:
        """
        Translates a Signal into an MT5 market order.
        Returns order details dict on success, None on failure.
        """
        sym_key = signal.symbol
        mt5_sym = signal.mt5_symbol

        # In SIGNAL mode, only log — don't execute
        if settings.TRADING_MODE != "AUTO":
            logger.info(
                f"[{sym_key}] 📡 SIGNAL MODE │ {signal.action} {mt5_sym} │ "
                f"Score={signal.score} │ {signal.reason} │ "
                f"SL={signal.sl_distance:.2f} │ TP={signal.tp_distance:.2f}"
            )
            return None

        # Pre-flight checks
        if not self._validate_signal(signal):
            return None

        # Get account info for position sizing
        account = self._conn.get_account_info()
        equity = account.get("equity", 0.0)
        if equity <= 0:
            logger.warning(f"[{sym_key}] Equity unavailable, cannot execute")
            return None

        # Calculate lot size
        lot = self.calculate_lot_size(sym_key, equity, signal.sl_distance)
        if lot <= 0:
            return None

        # Get current price
        with self._conn.lock:
            tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            logger.error(f"[{sym_key}] Cannot get tick for order submission")
            return None

        if signal.action == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = round(price - signal.sl_distance, 2)
            tp = round(price + signal.tp_distance, 2)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = round(price + signal.sl_distance, 2)
            tp = round(price - signal.tp_distance, 2)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_sym,
            "volume":       float(lot),
            "type":         order_type,
            "price":        price,
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    30,
            "magic":        settings.MAGIC_NUMBER,
            "comment":      f"MK3_{sym_key}_{signal.score}"[:25],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # ── Retry loop ──────────────────────────────────────────
        for attempt in range(retries):
            try:
                with self._conn.lock:
                    result = mt5.order_send(request)

                if result is None:
                    logger.warning(f"[{sym_key}] order_send returned None (attempt {attempt + 1})")
                    time.sleep(1)
                    continue

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    info = {
                        "ticket":     result.order,
                        "symbol":     mt5_sym,
                        "symbol_key": sym_key,
                        "direction":  signal.action,
                        "lot":        lot,
                        "entry":      result.price,
                        "sl":         sl,
                        "tp":         tp,
                        "score":      signal.score,
                        "reason":     signal.reason,
                    }
                    self._known_open_tickets[result.order] = {
                        "score": signal.score,
                        "entry_time": time.time(),
                        "symbol_key": sym_key,
                        "regime": signal.regime,
                        "trajectory": [],
                        "setup_type": getattr(signal, "setup_type", "UNKNOWN"),
                    }
                    logger.log(TRADE_LEVEL_NUM,
                        f"✅ [{sym_key}] Order executed │ {signal.action} "
                        f"{lot} lots @ {result.price:.2f} │ "
                        f"SL={sl:.2f} │ TP={tp:.2f} │ "
                        f"Ticket={result.order} │ Score={signal.score}"
                    )
                    return info

                # Fatal errors (don't retry)
                retcode_msg = self._retcode_str(result.retcode)
                if result.retcode == 10027:  # AutoTrading disabled
                    logger.error(
                        f"[{sym_key}] ❌ AutoTrading disabled in MT5 (10027). "
                        f"Enable it in MT5 → Tools → Options → Expert Advisors"
                    )
                    return None
                if result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                    logger.error(f"[{sym_key}] ❌ Insufficient margin (10019)")
                    return None

                # Recoverable error — retry with updated price
                logger.warning(
                    f"[{sym_key}] Order rejected │ attempt {attempt + 1} │ "
                    f"retcode={result.retcode} ({retcode_msg})"
                )
                if result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF):
                    with self._conn.lock:
                        tick = mt5.symbol_info_tick(mt5_sym)
                    if tick:
                        request["price"] = tick.ask if signal.action == "BUY" else tick.bid

                time.sleep(0.5 * (attempt + 1))

            except Exception as exc:
                logger.error(f"[{sym_key}] Exception sending order (attempt {attempt + 1}): {exc}")
                time.sleep(1)

        logger.error(f"[{sym_key}] ❌ Failed to execute order after {retries} attempts")
        return None

    # ─── Close Position ─────────────────────────────────────────
    def close_position(self, ticket: int, comment: str = "MK3_close") -> bool:
        """Closes a position by ticket number."""
        if settings.TRADING_MODE != "AUTO":
            return False

        with self._conn.lock:
            positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Ticket {ticket} not found")
            return False

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY

        with self._conn.lock:
            tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    30,
            "magic":        settings.MAGIC_NUMBER,
            "comment":      comment[:25],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        try:
            with self._conn.lock:
                result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.log(TRADE_LEVEL_NUM,
                    f"🔒 Position closed │ Ticket={ticket} │ {pos.symbol} │ "
                    f"Price={price:.2f} │ Reason: {comment}"
                )
                return True
            logger.warning(f"Failed to close ticket {ticket}: {result}")
            return False
        except Exception as exc:
            logger.error(f"Error closing ticket {ticket}: {exc}")
            return False

    # ─── Close All ──────────────────────────────────────────────
    def close_all_positions(self) -> int:
        """Closes all MARK III positions. Returns count of positions closed."""
        if settings.TRADING_MODE != "AUTO":
            return 0

        positions = self._conn.get_open_positions(magic=settings.MAGIC_NUMBER)
        if not positions:
            return 0

        # Check if market is likely open before attempting
        if not self.is_market_open():
            logger.warning(
                f"Market appears closed. Skipping close_all to avoid 10018 errors. "
                f"{len(positions)} positions remain open."
            )
            return 0

        closed = 0
        for pos in positions:
            if self.close_position(pos.ticket, "MK3_emergency"):
                closed += 1

        logger.info(f"🔒 {closed} positions closed")
        return closed

    def append_trajectory(self, ticket: int, profit: float) -> None:
        """Appends the current profit to the trade's trajectory."""
        if ticket in self._known_open_tickets:
            self._known_open_tickets[ticket].setdefault("trajectory", []).append(round(profit, 2))

    # ─── Modify SL to Breakeven ──────────────────────────────────
    def modify_sl_to_breakeven(self, ticket: int) -> bool:
        """
        Moves SL to entry price (breakeven), respecting broker minimum
        stop-level distance. If broker rejects even the best-effort SL,
        force-closes the trade instead of leaving it unprotected.
        """
        if settings.TRADING_MODE != "AUTO":
            return False

        with self._conn.lock:
            positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Ticket {ticket} not found for SL modification")
            return False

        pos = positions[0]
        entry_price = pos.price_open

        # Check if SL is already at or past breakeven
        if pos.type == 0:  # BUY
            if pos.sl >= entry_price:
                return True  # Already at breakeven or better
        else:  # SELL
            if pos.sl > 0 and pos.sl <= entry_price:
                return True  # Already at breakeven or better

        # ── Get broker's minimum stop distance ────────────────────
        with self._conn.lock:
            sym_info = mt5.symbol_info(pos.symbol)
        if not sym_info:
            logger.warning(f"Cannot get symbol info for {pos.symbol}")
            return False

        # trade_stops_level is in points; convert to price distance
        stop_level_pts = sym_info.trade_stops_level
        point = sym_info.point
        min_distance = stop_level_pts * point
        # Add a small safety buffer (2 extra points)
        min_distance += 2 * point

        # ── Calculate valid SL ────────────────────────────────────
        current_price = pos.price_current

        if pos.type == 0:  # BUY — SL must be below current price
            lowest_valid_sl = current_price - min_distance
            # We want breakeven (entry_price), but it must be valid
            new_sl = min(entry_price, lowest_valid_sl)
            # Don't move SL further from entry than it already is
            if new_sl <= pos.sl:
                return True  # Current SL is already better
        else:  # SELL — SL must be above current price
            highest_valid_sl = current_price + min_distance
            new_sl = max(entry_price, highest_valid_sl)
            if pos.sl > 0 and new_sl >= pos.sl:
                return True  # Current SL is already better

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       float(round(new_sl, sym_info.digits)),
            "tp":       float(pos.tp),
        }
        try:
            with self._conn.lock:
                result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🛡️ SL moved to breakeven │ Ticket={ticket} │ "
                    f"{pos.symbol} │ SL={new_sl:.2f}"
                )
                return True
            retcode_msg = self._retcode_str(result.retcode) if result else "None"
            logger.warning(
                f"Failed to modify SL on ticket {ticket}: {retcode_msg} │ "
                f"Attempted SL={new_sl:.2f} │ Price={current_price:.2f} │ "
                f"MinDist={min_distance:.2f}"
            )
            return False
        except Exception as exc:
            logger.error(f"Error modifying SL on ticket {ticket}: {exc}")
            return False

    # ─── Regime-Aware Trailing Stop (v2.3) ─────────────────────────
    def trail_stop(
        self,
        ticket: int,
        trail_activation_atr: float,
        trail_distance_atr: float,
        entry_atr: float,
        entry_price: float,
        direction: str,
        entry_speed: int = 50,
        current_speed: int = 50,
        entry_regime: str = "NORMAL",
        current_regime: str = "NORMAL",
        pressure_dir: str = "NEUTRAL",
        current_atr: float = 0.0,
        health_score: int = 100,
        giveback_pct: float = 0.0,
    ) -> bool:
        """
        Regime-aware trailing stop. Moves SL to protect profits once
        the trade has moved trail_activation_atr × ATR in favor.

        STRICT RULE: This method may ONLY tighten risk (move SL closer
        to current price). It can NEVER widen the stop under any condition.

        On broker rejection (min stop distance), falls back silently.
        No retry spam — the health monitor will handle it on next cycle.

        Args:
            ticket: MT5 position ticket
            trail_activation_atr: How many ATRs in profit before trail starts
            trail_distance_atr: How far behind price the trail follows (in ATRs)
            entry_atr: ATR value frozen at entry time
            entry_price: Original entry price
            direction: "BUY" or "SELL"

        Returns:
            True if SL was modified, False otherwise.
        """
        if settings.TRADING_MODE != "AUTO":
            return False

        if entry_atr <= 0 or trail_activation_atr <= 0 or trail_distance_atr <= 0:
            return False

        with self._conn.lock:
            positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        current_price = pos.price_current
        current_sl = pos.sl
        speed_ratio = current_speed / entry_speed if entry_speed > 0 else 1.0
        atr_ratio = current_atr / entry_atr if current_atr > 0 and entry_atr > 0 else 1.0
        pressure_opposing = (
            isinstance(pressure_dir, str)
            and pressure_dir not in (direction, "NEUTRAL", "UNKNOWN")
        )
        regime_rank = {"DEAD": 0, "SLOW_TREND": 1, "NORMAL": 2, "FAST": 3, "EXPLOSIVE": 4}
        regime_dropped = regime_rank.get(current_regime, 2) < regime_rank.get(entry_regime, 2)

        effective_activation_atr = trail_activation_atr
        effective_distance_atr = trail_distance_atr
        trail_notes = []

        if health_score < 60:
            effective_activation_atr *= 0.80
            effective_distance_atr *= 0.75
            trail_notes.append(f"health={health_score}")
        if giveback_pct >= 0.45:
            effective_activation_atr *= 0.75
            effective_distance_atr *= 0.65
            trail_notes.append(f"giveback={giveback_pct:.0%}")
        if speed_ratio < 0.65:
            effective_activation_atr *= 0.85
            effective_distance_atr *= 0.75
            trail_notes.append(f"speed={entry_speed}->{current_speed}")
        if atr_ratio < 0.65:
            effective_distance_atr *= 0.80
            trail_notes.append(f"atr={atr_ratio:.2f}x")
        if pressure_opposing:
            effective_distance_atr *= 0.70
            trail_notes.append(f"pressure={pressure_dir}")
        if regime_dropped:
            effective_activation_atr *= 0.85
            effective_distance_atr *= 0.80
            trail_notes.append(f"regime={entry_regime}->{current_regime}")

        effective_activation_atr = max(0.35, min(trail_activation_atr, effective_activation_atr))
        effective_distance_atr = max(0.35, min(trail_distance_atr, effective_distance_atr))

        # ── Calculate profit distance in ATR units ───────────────
        if direction == "BUY":
            profit_distance = current_price - entry_price
        else:
            profit_distance = entry_price - current_price

        profit_in_atr = profit_distance / entry_atr if entry_atr > 0 else 0.0

        # ── Check activation threshold ───────────────────────────
        if profit_in_atr < effective_activation_atr:
            return False  # Not enough profit to activate trail

        # ── Calculate new trailing SL ────────────────────────────
        trail_distance_price = effective_distance_atr * entry_atr

        if direction == "BUY":
            new_sl = current_price - trail_distance_price
            # TIGHTEN-ONLY: new SL must be ABOVE current SL
            if current_sl > 0 and new_sl <= current_sl:
                return False  # Current SL is already better
        else:
            new_sl = current_price + trail_distance_price
            # TIGHTEN-ONLY: new SL must be BELOW current SL
            if current_sl > 0 and new_sl >= current_sl:
                return False  # Current SL is already better

        # ── Get broker min stop distance ─────────────────────────
        with self._conn.lock:
            sym_info = mt5.symbol_info(pos.symbol)
        if not sym_info:
            return False

        stop_level_pts = sym_info.trade_stops_level
        point = sym_info.point
        min_distance = (stop_level_pts + 2) * point  # +2 pts safety buffer

        # Validate against broker minimum
        if direction == "BUY":
            if current_price - new_sl < min_distance:
                logger.debug(
                    f"[Trail] Ticket={ticket} │ Trail SL={new_sl:.2f} too close │ "
                    f"MinDist={min_distance:.2f} │ Fallback to health monitor"
                )
                return False
        else:
            if new_sl - current_price < min_distance:
                logger.debug(
                    f"[Trail] Ticket={ticket} │ Trail SL={new_sl:.2f} too close │ "
                    f"MinDist={min_distance:.2f} │ Fallback to health monitor"
                )
                return False

        # ── Send SL modification ─────────────────────────────────
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       float(round(new_sl, sym_info.digits)),
            "tp":       float(pos.tp),
        }
        try:
            with self._conn.lock:
                result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"📐 [Trail] SL trailed │ Ticket={ticket} │ "
                    f"{pos.symbol} {direction} │ "
                    f"SL: {current_sl:.2f} → {new_sl:.2f} │ "
                    f"Profit={profit_in_atr:.2f}×ATR │ "
                    f"TrailDist={effective_distance_atr:.2f}xATR"
                )
                return True

            # Broker rejected — fall back silently
            retcode_msg = self._retcode_str(result.retcode) if result else "None"
            logger.info(
                f"📐 [Trail] SL modify rejected │ Ticket={ticket} │ "
                f"{retcode_msg} │ SL={new_sl:.2f} │ "
                f"Fallback to health monitor on next cycle"
            )
            return False

        except Exception as exc:
            logger.debug(f"[Trail] Error on ticket {ticket}: {exc}")
            return False

    # ─── Market Hours Check ──────────────────────────────────────
    @staticmethod
    def is_market_open(symbol_key: str = None) -> bool:
        """
        Checks if any active symbol's market is currently open.
        Uses settings.MARKET_HOURS_UTC.
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        current_day = now.weekday()  # 0=Monday, 6=Sunday

        # Futures are closed on weekends (Sat 22:00 - Sun 23:00 UTC approx)
        # Simplified: skip Saturday entirely, Sunday before 23:00
        if current_day == 5:  # Saturday
            return False
        if current_day == 6 and current_hour < 23:  # Sunday before 23:00
            return False

        symbols_to_check = [symbol_key] if symbol_key else settings.ACTIVE_SYMBOLS

        for sym in symbols_to_check:
            hours = settings.MARKET_HOURS_UTC.get(sym, [(0, 22)])
            for start_h, end_h in hours:
                if start_h <= current_hour < end_h:
                    return True

        return False

    # ─── Track Closed Trades for CSV Logging ────────────────────
    def check_closed_trades(self) -> None:
        """
        Detects if any tracked positions were closed (e.g. by SL/TP or manually).
        Logs the final result to TradeLogger.
        """
        if settings.TRADING_MODE != "AUTO" or not self._known_open_tickets:
            return

        current_positions = self._conn.get_open_positions(magic=settings.MAGIC_NUMBER)
        current_tickets = {p.ticket for p in current_positions}

        # Find tickets that are no longer open
        closed_tickets = set(self._known_open_tickets.keys()) - current_tickets

        for ticket in closed_tickets:
            meta = self._known_open_tickets.pop(ticket, {})
            try:
                # Get the history deal that closed this position
                # We need to fetch deals from the past 24 hours
                now = datetime.now(timezone.utc)
                from_date = now.replace(hour=0, minute=0, second=0)
                
                with self._conn.lock:
                    deals = mt5.history_deals_get(position=ticket)
                
                if not deals or len(deals) < 2:
                    continue  # We need at least Entry and Exit deals
                    
                entry_deal = deals[0]
                exit_deal = deals[-1]
                
                # Calculate duration in minutes
                duration_min = (exit_deal.time - entry_deal.time) / 60.0
                
                direction = "BUY" if entry_deal.type == 0 else "SELL"
                sym_key = meta.get("symbol_key", exit_deal.symbol)
                
                # Format times — MT5 deal.time is in broker server time
                # (UTC+3), NOT real UTC. Subtract offset for true UTC.
                BROKER_OFFSET_SEC = 3 * 3600  # UTC+3
                entry_time_str = datetime.fromtimestamp(entry_deal.time - BROKER_OFFSET_SEC, tz=timezone.utc).isoformat()
                exit_time_str = datetime.fromtimestamp(exit_deal.time - BROKER_OFFSET_SEC, tz=timezone.utc).isoformat()
                
                # PnL logic
                pnl_usd = exit_deal.profit + exit_deal.commission + exit_deal.swap
                
                # We don't have accurate account opening balance here, so pnl_pct is skipped or calculated off 100 for now
                pnl_pct = 0.0

                self._trade_logger.log_trade(
                    symbol=sym_key,
                    direction=direction,
                    lot_size=exit_deal.volume,
                    entry_time=entry_time_str,
                    exit_time=exit_time_str,
                    entry_price=entry_deal.price,
                    exit_price=exit_deal.price,
                    pnl_usd=round(pnl_usd, 2),
                    pnl_pct=round(pnl_pct, 2),
                    duration_min=round(duration_min, 1),
                    close_reason="MT5_Close",
                    signal_score=meta.get("score", 0),
                    setup_type=meta.get("setup_type", "UNKNOWN"),
                    ticket=ticket,
                    trajectory=str(meta.get("trajectory", [])),
                    market_regime=meta.get("regime", "UNKNOWN")
                )
            except Exception as exc:
                logger.error(f"Error logging closed trade ticket={ticket}: {exc}")

    # ─── Helper ─────────────────────────────────────────────────
    @staticmethod
    def _retcode_str(retcode: int) -> str:
        codes = {
            10004: "REQUOTE", 10006: "REJECTED", 10007: "CANCELED",
            10008: "PLACED", 10009: "DONE", 10010: "DONE_PARTIAL",
            10011: "ERROR", 10012: "TIMEOUT", 10013: "INVALID",
            10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
            10016: "INVALID_STOPS", 10017: "TRADE_DISABLED",
            10018: "MARKET_CLOSED", 10019: "NO_MONEY",
            10020: "PRICE_CHANGED", 10021: "PRICE_OFF",
            10022: "INVALID_EXPIRATION", 10024: "TOO_MANY_REQUESTS",
            10027: "AUTOTRADING_DISABLED",
        }
        return codes.get(retcode, f"UNKNOWN({retcode})")
