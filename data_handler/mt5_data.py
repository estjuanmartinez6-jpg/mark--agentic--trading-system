"""
data_handler/mt5_data.py — Real MT5 Data Provider for MARK III v2

Fetches REAL OHLCV candle data from MetaTrader 5 across multiple
timeframes (M5, M15). Replaces the old MockOrderFlowProvider that
generated random data.

Data source: mt5.copy_rates_from_pos() — returns real market history.
Thread-safe: Uses the MT5Connector lock for all API calls.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import settings
from monitoring.logger import get_logger

logger = get_logger("MT5Data")

# ─── Timeframe mapping ──────────────────────────────────────────
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
}


class MT5DataProvider:
    """
    Fetches real OHLCV candle data from MetaTrader 5.

    Returns pandas DataFrames with columns:
        time, open, high, low, close, tick_volume, spread

    Plus pre-calculated:
        ema_21, ema_50, atr (14-period)
    """

    def __init__(self, connector) -> None:
        self._conn = connector
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._last_fetch: float = 0.0
        self._cache_ttl: float = 10.0  # Refresh every 10 seconds
        logger.info("MT5DataProvider initialized (real market data)")

    # ─── Main Entry Point ───────────────────────────────────────
    def get_all_symbols(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Returns market data for all active symbols across M5 + M15.

        Returns:
            {
                "ES": {"M5": DataFrame, "M15": DataFrame},
                "NQ": {"M5": DataFrame, "M15": DataFrame},
            }
        """
        now = time.time()
        if now - self._last_fetch < self._cache_ttl and self._cache:
            return self._cache

        result = {}
        for sym_key in settings.ACTIVE_SYMBOLS:
            sym_cfg = settings.SYMBOL_MAP.get(sym_key)
            if not sym_cfg:
                continue

            mt5_name = sym_cfg["mt5_name"]
            sym_data = {}

            for tf_name, tf_const in [("M5", TF_MAP["M5"]), ("M15", TF_MAP["M15"])]:
                count = settings.CANDLE_HISTORY_M5 if tf_name == "M5" else settings.CANDLE_HISTORY_M15
                df = self._fetch_candles(mt5_name, tf_const, count)
                if df is not None and len(df) >= 20:
                    df = self._add_indicators(df)
                    sym_data[tf_name] = df
                else:
                    logger.warning(
                        f"[{sym_key}] Insufficient {tf_name} data: "
                        f"got {len(df) if df is not None else 0} candles"
                    )

            if "M5" in sym_data and "M15" in sym_data:
                result[sym_key] = sym_data

        self._cache = result
        self._last_fetch = now
        return result

    # ─── Fetch Raw Candles ──────────────────────────────────────
    def _fetch_candles(
        self, mt5_symbol: str, timeframe: int, count: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV candles from MT5.
        Returns DataFrame or None on error.
        """
        try:
            with self._conn.lock:
                rates = mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, count)

            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

            # Keep only what we need
            cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
            available = [c for c in cols if c in df.columns]
            df = df[available].copy()

            # Ensure tick_volume exists
            if "tick_volume" not in df.columns:
                df["tick_volume"] = 0

            return df

        except Exception as exc:
            logger.error(f"Error fetching candles for {mt5_symbol}: {exc}")
            return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators:
          - EMA 21/50 (legacy trend filter)
          - ATR 14 (volatility/risk)
          - Ichimoku Cloud (trend + pullback levels)
          - RSI 14 (momentum)
          - VWAP (institutional reference)
          - Candle metrics (structure/pressure modules)
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ── EMA 21 (fast trend) ─────────────────────────────────
        df["ema_21"] = _ema(close, 21)

        # ── EMA 50 (slow trend) ─────────────────────────────────
        df["ema_50"] = _ema(close, 50)

        # ── ATR 14 ──────────────────────────────────────────────
        df["atr"] = _atr(high, low, close, 14)

        # ── Ichimoku Cloud ──────────────────────────────────────
        df["tenkan_sen"] = _donchian_mid(high, low, 9)       # Conversion line
        df["kijun_sen"] = _donchian_mid(high, low, 26)       # Base line
        tenkan = df["tenkan_sen"].values
        kijun = df["kijun_sen"].values
        # Senkou Span A = (Tenkan + Kijun) / 2, displaced 26 ahead
        senkou_a_raw = (tenkan + kijun) / 2.0
        df["senkou_a"] = np.concatenate([np.full(26, np.nan), senkou_a_raw[:-26]]) if len(senkou_a_raw) > 26 else senkou_a_raw
        # Senkou Span B = 52-period Donchian mid, displaced 26 ahead
        senkou_b_raw = _donchian_mid(high, low, 52)
        df["senkou_b"] = np.concatenate([np.full(26, np.nan), senkou_b_raw[:-26]]) if len(senkou_b_raw) > 26 else senkou_b_raw
        # Chikou Span = close displaced 26 periods back
        df["chikou_span"] = df["close"].shift(-26)

        # ── RSI 14 ──────────────────────────────────────────────
        df["rsi_14"] = _rsi(close, 14)

        # ── VWAP (Volume-Weighted Average Price) ────────────────
        df["vwap"] = _vwap(df)

        # ── Candle metrics (used by structure/pressure modules) ──
        df["body"] = df["close"] - df["open"]
        df["body_abs"] = df["body"].abs()
        df["range"] = df["high"] - df["low"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Body ratio: how much of the candle is body vs wick (0 to 1)
        df["body_ratio"] = np.where(
            df["range"] > 0,
            df["body_abs"] / df["range"],
            0.0,
        )

        # Close position within range (0=low, 1=high)
        df["close_position"] = np.where(
            df["range"] > 0,
            (df["close"] - df["low"]) / df["range"],
            0.5,
        )

        # Volume relative to rolling average
        vol = df["tick_volume"].values.astype(float)
        avg_vol = pd.Series(vol).rolling(20, min_periods=1).mean().values
        df["vol_ratio"] = np.where(avg_vol > 0, vol / avg_vol, 1.0)

        return df

    # ─── Cache Management ───────────────────────────────────────
    def invalidate_cache(self) -> None:
        """Force a fresh fetch on next call."""
        self._last_fetch = 0.0

    def get_cached(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Returns cached data without fetching."""
        return self._cache


# ═══════════════════════════════════════════════════════════════════
# Pure numpy helpers (no external dependencies)
# ═══════════════════════════════════════════════════════════════════

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average using numpy."""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)

    alpha = 2.0 / (period + 1)
    result = np.empty_like(data, dtype=float)
    result[:period] = np.nan
    result[period - 1] = np.mean(data[:period])

    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range using numpy."""
    if len(high) < 2:
        return np.zeros_like(high, dtype=float)

    tr = np.empty(len(high), dtype=float)
    tr[0] = high[0] - low[0]

    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Use EMA for ATR (Wilder's smoothing)
    atr_vals = np.empty_like(tr)
    atr_vals[:period] = np.nan
    atr_vals[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

    return atr_vals


def _donchian_mid(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Donchian Channel midline: (highest high + lowest low) / 2 over `period`."""
    n = len(high)
    result = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1: i + 1])
        ll = np.min(low[i - period + 1: i + 1])
        result[i] = (hh + ll) / 2.0
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder's smoothing."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=float)

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(n, np.nan, dtype=float)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def _vwap(df: pd.DataFrame) -> np.ndarray:
    """
    Volume-Weighted Average Price.
    Uses tick_volume as proxy. Resets daily if 'time' column exists.
    """
    typical_price = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    volume = df["tick_volume"].values.astype(float)

    # Simple cumulative VWAP (no daily reset in intraday context)
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)

    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_price)
    return vwap

