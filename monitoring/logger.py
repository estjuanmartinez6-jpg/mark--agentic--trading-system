"""
monitoring/logger.py — Logging configuration for MARK III

Provides structured, colorized console logging and file logging.
Adapted from MARK I's battle-tested logger.
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOG_DIR, LOG_LEVEL, TRADES_DIR

# ─── Custom Trade level (between INFO and WARNING) ──────────────
TRADE_LEVEL_NUM = 25
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def _trade(self, message, *args, **kwargs):
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        self._log(TRADE_LEVEL_NUM, message, args, **kwargs)


logging.Logger.trade = _trade  # type: ignore[attr-defined]


# ─── Formatter with colors ──────────────────────────────────────
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[90m",     # Gray
        "INFO":     "\033[97m",     # White
        "TRADE":    "\033[92m",     # Green
        "WARNING":  "\033[93m",     # Yellow
        "ERROR":    "\033[91m",     # Red
        "CRITICAL": "\033[91;1m",   # Bright Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Creates a named logger with console + file handlers."""
    logger = logging.getLogger(f"MK3.{name}")

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # Console handler (colorized)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter(
        "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # File handler (plain text, append)
    log_file = LOG_DIR / "mark3.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ─── TradeLogger ────────────────────────────────────────────────
CSV_HEADERS = [
    "timestamp_utc", "bot_version", "symbol", "direction", "lot_size",
    "entry_time", "exit_time",
    "entry_price", "sl_price", "tp_price", "exit_price",
    "pnl_usd", "pnl_pct", "duration_min", "close_reason",
    "signal_score", "setup_type", "ticket", "trajectory", "market_regime",
]


class TradeLogger:
    """Registra cada operación cerrada en un CSV de historial."""

    def __init__(self) -> None:
        self._dir = TRADES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log = get_logger("TradeLogger")

    def _get_file(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"trades_{date_str}.csv"

    def log_trade(self, **kwargs) -> None:
        """Escribe una fila al CSV de trades del día actual."""
        file = self._get_file()
        file_exists = file.exists()

        row = {h: kwargs.get(h, "") for h in CSV_HEADERS}
        row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        if not row["bot_version"]:
            row["bot_version"] = kwargs.get("bot_version", "v4.0_Ichimoku")

        try:
            with open(file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            self._log.log(TRADE_LEVEL_NUM, f"Trade guardado en CSV → Ticket {row.get('ticket')}")
        except Exception as exc:
            self._log.error(f"Error guardando trade en CSV: {exc}")
