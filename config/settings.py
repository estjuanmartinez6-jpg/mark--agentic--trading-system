"""
config/settings.py — Configuration for MARK III Order Flow Trading System

Loads environment variables from .env file and provides sensible defaults.
Follows the same patterns established in MARK I for consistency.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Rutas del proyecto ─────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
LOG_DIR = ROOT_DIR / "logs"
TRADES_DIR = ROOT_DIR / "trades"

LOG_DIR.mkdir(exist_ok=True)
TRADES_DIR.mkdir(exist_ok=True)

# Cargar .env desde la raíz del proyecto
load_dotenv(ROOT_DIR / ".env")


# ─── Modo de operación ──────────────────────────────────────────
# SIGNAL: solo genera señales, NO ejecuta órdenes
# AUTO:   ejecuta órdenes automáticamente en MT5
TRADING_MODE: str = os.getenv("TRADING_MODE", "SIGNAL").upper()
assert TRADING_MODE in ("AUTO", "SIGNAL"), \
    f"TRADING_MODE debe ser AUTO o SIGNAL, recibido: {TRADING_MODE}"


# ─── MetaTrader 5 ───────────────────────────────────────────────
_login_raw = os.getenv("MT5_LOGIN", "")
MT5_LOGIN: int | None = int(_login_raw) if _login_raw.strip().isdigit() else None
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_RECONNECT_RETRIES: int = 5
MT5_RECONNECT_BASE_DELAY: float = 2.0

# Magic number identifica operaciones de MARK III en el historial de MT5
MAGIC_NUMBER: int = 300001


# ─── Futures → CFD Symbol Mapping ───────────────────────────────
# Futures symbols → Exness MT5 CFD names (same as MARK I)
SYMBOL_MAP: dict[str, dict] = {
    "ES": {
        "futures_name":   "E-mini S&P 500",
        "mt5_name":       "US500Cash",
        "display_name":   "S&P 500",
        "max_spread_pts": 1.0,
        "min_lot":        0.1,
        "max_lot":        0.3,
        "lot_step":       0.1,
    },
    "NQ": {
        "futures_name":   "E-mini Nasdaq 100",
        "mt5_name":       "US100Cash",
        "display_name":   "Nasdaq 100",
        "max_spread_pts": 3.5,
        "min_lot":        0.1,
        "max_lot":        0.3,
        "lot_step":       0.1,
    },
}

ACTIVE_SYMBOLS: list[str] = ["ES", "NQ"]


# ─── Strategy Parameters (v2 — Market Structure) ────────────────
# Two-timeframe model: M15 (trend/bias) + M5 (signal/entry)

# ATR for volatility filter and SL/TP calculation
ATR_PERIOD: int = 14
MIN_VOLATILITY_ATR_ES: float = float(os.getenv("MIN_ATR_ES", "1.0"))
MIN_VOLATILITY_ATR_NQ: float = float(os.getenv("MIN_ATR_NQ", "3.0"))

# Candle history to fetch from MT5
CANDLE_HISTORY_M5: int = int(os.getenv("CANDLE_HISTORY_M5", "50"))
CANDLE_HISTORY_M15: int = int(os.getenv("CANDLE_HISTORY_M15", "100"))

# Swing detection lookback (candles on each side for fractal detection)
SWING_LOOKBACK: int = int(os.getenv("SWING_LOOKBACK", "5"))


# ─── Scoring System (v2 — transparent, rule-based) ──────────────
# 6 factors × max 15-20 pts each = 100 pts max.
# Requires >= 3 independent factors to fire (no single-factor trades).
MIN_SCORE_TO_TRADE: int = int(os.getenv("MIN_SCORE", "50"))


# ─── Execution / Risk Management ────────────────────────────────
RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE", "1.0"))
ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULT", "1.5"))
ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULT", "2.5"))

MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "2"))
MAX_DAILY_DRAWDOWN_PCT: float = float(os.getenv("MAX_DAILY_DRAWDOWN", "3.0"))
MAX_LOSS_PER_TRADE_USD: float = float(os.getenv("MAX_LOSS_PER_TRADE_USD", "3.0"))
MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "10"))

# Caps de seguridad
MAX_LOT_ABSOLUTE: float = 0.3
MIN_LOT_ABSOLUTE: float = 0.1

# Cooldown between signals on the same symbol (seconds)
SIGNAL_COOLDOWN_SEC: int = int(os.getenv("SIGNAL_COOLDOWN", "300"))


# ─── Timing ─────────────────────────────────────────────────────
# How often the main loop checks for new signals (seconds)
SIGNAL_CHECK_INTERVAL_SEC: int = int(os.getenv("SIGNAL_INTERVAL", "15"))
HEARTBEAT_INTERVAL_SEC: int = 60
WARMUP_CANDLES: int = int(os.getenv("WARMUP_CANDLES", "10"))


# ─── Trade Health Monitor ───────────────────────────────────────
# Prevents stale trades from sitting forever (ported from MARK I)
TRADE_MAX_AGE_SEC: int = int(os.getenv("TRADE_MAX_AGE_SEC", "900"))       # 15 min
TRADE_GRACE_PERIOD_SEC: int = int(os.getenv("TRADE_GRACE_SEC", "120"))    # 2 min
HEALTH_CHECK_INTERVAL_SEC: int = int(os.getenv("HEALTH_CHECK_SEC", "30"))
HEALTH_DEGRADING_THRESHOLD: int = 60   # score < 60 → tighten SL
HEALTH_CRITICAL_THRESHOLD: int = 40    # score < 40 → exit trade


# ─── Regime-Aware Execution (v2.3) ──────────────────────────────
# When True, execution behavior adapts to detected market regime:
#   - Entry thresholds, SL/TP multipliers, trailing logic, and
#     health expectations all become regime-dependent.
# When False, all modules use NORMAL-profile defaults (exact v2.2).
# Toggle via .env for instant rollback without code changes.
REGIME_AWARE_EXECUTION: bool = os.getenv(
    "REGIME_AWARE_EXECUTION", "true"
).lower() == "true"


# ─── Daily Risk Manager ─────────────────────────────────────────
# Safety nets — generous limits, only activate in extreme scenarios
DAILY_PROFIT_TARGET_USD: float = float(os.getenv("DAILY_PROFIT_TARGET", "5.0"))
TRAILING_PROFIT_ACTIVATION_USD: float = float(os.getenv("TRAILING_ACTIVATION", "3.0"))
TRAILING_PROFIT_PULLBACK_USD: float = float(os.getenv("TRAILING_PULLBACK", "1.50"))


# ─── Market Hours (UTC) ─────────────────────────────────────────
# Trading sessions — system skips signal evaluation outside these
# hours. Format: list of (start_hour, end_hour) tuples in UTC.
# Default: CME E-mini futures hours (Sun 23:00 – Fri 22:00 UTC)
# with a daily maintenance break 22:00-23:00 UTC.
MARKET_HOURS_UTC: dict = {
    "ES": [(0, 22)],   # 00:00-22:00 UTC (covers US session + overnight)
    "NQ": [(0, 22)],   # Same as ES
}


# ─── Dashboard ──────────────────────────────────────────────────
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_ENABLED: bool = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"


# ─── Logging ────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


# ─── Telegram Notifier ──────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED: bool = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
TELEGRAM_TIMEOUT: float = float(os.getenv("TELEGRAM_TIMEOUT", "5.0"))
