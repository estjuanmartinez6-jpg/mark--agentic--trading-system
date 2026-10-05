"""
execution_engine/mt5_connector.py — MetaTrader 5 Connection Manager

Thread-safe, production-grade MT5 connector with:
  - Exponential backoff reconnection
  - Health check (ping-before-trade)
  - Symbol enablement
  - Account info caching

Adapted from MARK I's battle-tested connector.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import MetaTrader5 as mt5

from config import settings
from monitoring.logger import get_logger

logger = get_logger("MT5Connector")

# Global lock — MT5 Python API is NOT thread-safe
_mt5_lock = threading.Lock()


class MT5Connector:
    """Manages the connection lifecycle to MetaTrader 5."""

    def __init__(self) -> None:
        self._connected = False
        self._account_info: dict = {}

    # ─── Properties ─────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def lock(self) -> threading.Lock:
        """Exposes the lock so other modules can synchronize MT5 calls."""
        return _mt5_lock

    # ─── Connection ─────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Initializes MT5 with exponential backoff retries.
        If MT5_LOGIN is configured, performs explicit login.
        """
        for attempt in range(settings.MT5_RECONNECT_RETRIES):
            try:
                with _mt5_lock:
                    if not mt5.initialize():
                        err = mt5.last_error()
                        logger.warning(f"mt5.initialize() failed: {err}")
                        continue

                    # Explicit login if credentials are provided
                    if settings.MT5_LOGIN:
                        ok = mt5.login(
                            login=settings.MT5_LOGIN,
                            password=settings.MT5_PASSWORD,
                            server=settings.MT5_SERVER or None,
                        )
                        if not ok:
                            logger.warning(f"mt5.login() failed: {mt5.last_error()}")
                            mt5.shutdown()
                            continue

                self._connected = True
                info = self._fetch_account_info()
                logger.info(
                    f"✅ MT5 connected │ Account: {info.get('login')} │ "
                    f"Server: {info.get('server')} │ "
                    f"Balance: ${info.get('balance', 0):.2f} │ "
                    f"Equity: ${info.get('equity', 0):.2f}"
                )
                return True

            except Exception as exc:
                logger.warning(f"Connection attempt {attempt + 1}: {exc}")

            delay = settings.MT5_RECONNECT_BASE_DELAY * (2 ** attempt)
            logger.info(f"Retrying in {delay:.1f}s...")
            time.sleep(delay)

        logger.critical("❌ Failed to connect to MT5 after all attempts.")
        return False

    def disconnect(self) -> None:
        """Cleanly shuts down the MT5 connection."""
        try:
            with _mt5_lock:
                mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")
        except Exception as exc:
            logger.error(f"Error disconnecting MT5: {exc}")

    # ─── Health Check ───────────────────────────────────────────
    def ensure_connected(self) -> bool:
        """
        Checks if MT5 is alive. If not, attempts to reconnect.
        Call this before any critical operation.
        """
        if not self._is_alive():
            logger.warning("MT5 connection lost. Reconnecting...")
            self._connected = False
            return self.connect()
        return True

    def _is_alive(self) -> bool:
        """Lightweight ping: checks terminal status."""
        try:
            with _mt5_lock:
                info = mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    # ─── Account Info ───────────────────────────────────────────
    def get_account_info(self) -> dict:
        """Returns account info dict (balance, equity, margin, etc.)."""
        return self._fetch_account_info()

    def _fetch_account_info(self) -> dict:
        try:
            with _mt5_lock:
                ai = mt5.account_info()
            if ai is None:
                return {}
            self._account_info = ai._asdict()
            return self._account_info
        except Exception as exc:
            logger.error(f"Error fetching account info: {exc}")
            return {}

    # ─── Symbol Management ──────────────────────────────────────
    def enable_symbol(self, mt5_name: str) -> bool:
        """Ensures the symbol is visible in Market Watch."""
        try:
            with _mt5_lock:
                result = mt5.symbol_select(mt5_name, True)
            if not result:
                logger.warning(f"Failed to enable symbol: {mt5_name}")
            return result
        except Exception as exc:
            logger.error(f"Error enabling symbol {mt5_name}: {exc}")
            return False

    def get_symbol_info(self, mt5_name: str) -> Optional[dict]:
        """Returns symbol info dict (tick_value, tick_size, contract_size, etc.)."""
        try:
            with _mt5_lock:
                info = mt5.symbol_info(mt5_name)
            if info is None:
                return None
            return info._asdict()
        except Exception as exc:
            logger.error(f"Error getting symbol info for {mt5_name}: {exc}")
            return None

    def get_spread(self, mt5_name: str) -> float:
        """Returns the current spread for the symbol in points."""
        try:
            with _mt5_lock:
                tick = mt5.symbol_info_tick(mt5_name)
            if tick is None:
                return 999.0  # High value to block trading on error
            return tick.ask - tick.bid
        except Exception:
            return 999.0

    def get_open_positions(self, magic: int = None) -> list:
        """Returns all open positions, optionally filtered by magic number."""
        try:
            with _mt5_lock:
                positions = mt5.positions_get()
            if positions is None:
                return []
            if magic is not None:
                return [p for p in positions if p.magic == magic]
            return list(positions)
        except Exception as exc:
            logger.error(f"Error getting positions: {exc}")
            return []

    # ─── Last Error ─────────────────────────────────────────────
    @staticmethod
    def last_error() -> tuple:
        return mt5.last_error()
