"""
execution_engine/risk_manager.py — Daily Risk Management for MARK III

Ported from MARK I's battle-tested RiskManager.
Manages account-level risk controls:

  - Daily P&L tracking (equity vs opening balance)
  - Daily drawdown halt (default: -3% → stop trading)
  - Daily profit target (default: +$5 → protect gains)
  - Trailing daily profit (activate at +$3, pullback $1.50 → close all)
  - Trade counter with proper daily reset at midnight UTC

Philosophy: These are SAFETY NETS, not barriers. They only activate
in extreme scenarios to protect the account from catastrophe.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from config import settings
from monitoring.logger import get_logger

logger = get_logger("RiskManager")


class RiskManager:
    """
    Account-level risk management.
    Called from the main engine every heartbeat cycle.
    """

    def __init__(self) -> None:
        self._starting_balance: float = 0.0
        self._day_open_balance: float = 0.0
        self._day_pnl: float = 0.0
        self._day_date: Optional[str] = None
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        self._profit_target_hit: bool = False
        self._day_trades: int = 0

        # ── Trailing Profit (protect intermediate gains) ─────────
        self._peak_daily_pnl: float = 0.0
        self._trailing_profit_active: bool = False
        self._trailing_profit_triggered: bool = False

        logger.info("RiskManager initialized")

    # ─── Daily initialization ────────────────────────────────────
    def update_balance(
        self,
        balance: float,
        equity: float,
        today_override: Optional[str] = None,
    ) -> None:
        """
        Call every engine cycle with current balance.
        Detects day change and resets daily tracking.

        Args:
            balance:        Current account balance.
            equity:         Current account equity.
            today_override: Optional date string (YYYY-MM-DD) to use instead
                            of datetime.now(). Used by the simulator to inject
                            the simualted replay date so that daily limits reset
                            correctly across multi-day replays.
        """
        today = today_override or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._day_date != today:
            self._day_date = today
            self._day_open_balance = balance
            self._day_pnl = 0.0
            self._trading_halted = False
            self._halt_reason = ""
            self._profit_target_hit = False
            self._day_trades = 0
            self._peak_daily_pnl = 0.0
            self._trailing_profit_active = False
            self._trailing_profit_triggered = False
            logger.info(f"📅 New trading day │ Opening balance: ${balance:.2f}")

        if self._starting_balance == 0.0:
            self._starting_balance = balance

        # P&L = difference between current equity and day open balance
        self._day_pnl = equity - self._day_open_balance

        # Update trailing profit
        self._update_trailing_profit()

    # ─── Properties ──────────────────────────────────────────────
    @property
    def daily_pnl(self) -> float:
        return self._day_pnl

    @property
    def daily_pnl_pct(self) -> float:
        """Daily P&L as percentage of opening balance."""
        if self._day_open_balance <= 0:
            return 0.0
        return (self._day_pnl / self._day_open_balance) * 100.0

    @property
    def trading_halted(self) -> bool:
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def trades_today(self) -> int:
        return self._day_trades

    @property
    def max_trades_reached(self) -> bool:
        return self._day_trades >= settings.MAX_TRADES_PER_DAY

    @property
    def trailing_profit_triggered(self) -> bool:
        return self._trailing_profit_triggered

    @property
    def peak_daily_pnl(self) -> float:
        return self._peak_daily_pnl

    # ─── Trade Counter ───────────────────────────────────────────
    def register_trade(self) -> None:
        """Register a trade executed today."""
        self._day_trades += 1
        logger.info(
            f"📊 Trade #{self._day_trades}/{settings.MAX_TRADES_PER_DAY} today"
        )

    # ─── Risk Checks ─────────────────────────────────────────────
    def can_trade(self) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        Checks all daily risk conditions.
        """
        # Check if trading was halted
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # Check daily drawdown
        max_dd = settings.MAX_DAILY_DRAWDOWN_PCT
        if self.daily_pnl_pct <= -abs(max_dd):
            self._trading_halted = True
            self._halt_reason = (
                f"Daily drawdown exceeded: {self.daily_pnl_pct:.2f}% "
                f"(limit: -{max_dd}%)"
            )
            logger.warning(f"⛔ {self._halt_reason}")
            return False, self._halt_reason

        # Check profit target
        if self._profit_target_hit:
            return False, "Daily profit target reached"

        target_usd = settings.DAILY_PROFIT_TARGET_USD
        if target_usd > 0 and self._day_pnl >= target_usd:
            self._profit_target_hit = True
            self._halt_reason = (
                f"Daily profit target reached: ${self._day_pnl:+.2f} "
                f"(target: ${target_usd:.2f})"
            )
            logger.info(f"🎯 {self._halt_reason}")
            return False, self._halt_reason

        # Check trailing profit
        if self._trailing_profit_triggered:
            return False, "Trailing profit protection activated"

        # Check max trades
        if self.max_trades_reached:
            return False, f"Max trades reached ({self._day_trades}/{settings.MAX_TRADES_PER_DAY})"

        return True, "OK"

    # ─── Trailing Profit ─────────────────────────────────────────
    def _update_trailing_profit(self) -> None:
        """
        Trailing Stop at ACCOUNT level:
        - Activates when daily P&L exceeds activation threshold
        - Tracks peak P&L
        - If P&L drops more than pullback from peak → signal to close all
        """
        if self._trailing_profit_triggered:
            return

        activation = settings.TRAILING_PROFIT_ACTIVATION_USD
        pullback = settings.TRAILING_PROFIT_PULLBACK_USD

        # Update peak
        if self._day_pnl > self._peak_daily_pnl:
            self._peak_daily_pnl = self._day_pnl

        # Activate protection when P&L crosses threshold
        if not self._trailing_profit_active and self._peak_daily_pnl >= activation:
            self._trailing_profit_active = True
            logger.info(
                f"🛡️ Trailing profit protection ACTIVATED │ "
                f"Peak P&L: ${self._peak_daily_pnl:+.2f} │ "
                f"Close all if drops to ${self._peak_daily_pnl - pullback:+.2f}"
            )

        # Check if pullback triggered
        if self._trailing_profit_active:
            drawdown_from_peak = self._peak_daily_pnl - self._day_pnl
            if drawdown_from_peak >= pullback:
                self._trailing_profit_triggered = True
                self._halt_reason = (
                    f"Trailing profit: Peak ${self._peak_daily_pnl:+.2f} → "
                    f"Current ${self._day_pnl:+.2f} "
                    f"(drop: ${drawdown_from_peak:.2f})"
                )
                logger.warning(
                    f"🛑 TRAILING PROFIT TRIGGERED │ {self._halt_reason}"
                )

    # ─── Summary for logging ─────────────────────────────────────
    def summary(self) -> dict:
        return {
            "day_pnl":         round(self._day_pnl, 2),
            "day_pnl_pct":     round(self.daily_pnl_pct, 2),
            "halted":          self._trading_halted,
            "halt_reason":     self._halt_reason,
            "profit_target":   self._profit_target_hit,
            "trades_today":    self._day_trades,
            "max_trades":      settings.MAX_TRADES_PER_DAY,
            "peak_pnl":        round(self._peak_daily_pnl, 2),
            "trailing_active": self._trailing_profit_active,
        }
