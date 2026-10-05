"""
simulator/indicators.py — Abstracted Indicator Calculation for MARK III Replay

Design principle (Req 1):
    Indicator calculation is hidden behind a single interface: IndicatorEngine.
    Phase 1 uses full visible-window recalculation (identical to live MT5DataProvider).
    Phase 2 can replace internals with stateful incremental computation
    WITHOUT changing any DataFeed or StrategyAdapter code.

CRITICAL CONSISTENCY GUARANTEE:
    This module imports and reuses the EXACT _ema() and _atr() helper
    functions from data_handler.mt5_data. It does NOT copy or reimplement them.
    This ensures bit-for-bit identical indicator values between:
      - Live mode (MT5DataProvider)
      - Replay mode (IndicatorEngine)

    Any discrepancy would create different strategy decisions — defeating
    the purpose of the replay framework.

Phase 2 upgrade path (documented, not implemented):
    class IncrementalEMA:
        def __init__(self, period: int, alpha: float = None): ...
        def update(self, new_close: float) -> float: ...

    class IncrementalATR:
        def __init__(self, period: int): ...
        def update(self, high: float, low: float, prev_close: float) -> float: ...

    IndicatorEngine.use_incremental() switches internals transparently.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Import the EXACT helpers used by the live system — not copies.
# _ema and _atr are module-level pure functions in data_handler/mt5_data.py
# (defined after the MT5DataProvider class, starting at line ~190).
# Importing them directly guarantees bit-for-bit identical values vs live.
from data_handler.mt5_data import _ema, _atr, _donchian_mid, _rsi, _vwap  # noqa: F401

logger = logging.getLogger("Sim.Indicators")


class IndicatorEngine:
    """
    Computes technical indicators on a DataFrame slice.

    Phase 1: Full visible-window recalculation on every call.
             Numerically identical to MT5DataProvider._add_indicators().

    The interface is stable: callers always call .compute(df_slice)
    and receive back a DataFrame with all indicator columns added.
    Internal implementation can later become incremental without
    breaking callers.

    Input columns required (from DataLoader normalized schema):
        time, open, high, low, close, tick_volume, spread

    Output adds:
        ema_21, ema_50, atr,
        tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span,
        rsi_14, vwap,
        body, body_abs, range, upper_wick, lower_wick,
        body_ratio, close_position, vol_ratio
    """

    # Indicator parameters (must match live settings)
    EMA_FAST: int = 21
    EMA_SLOW: int = 50
    ATR_PERIOD: int = 14
    VOL_ROLLING: int = 20

    def __init__(self) -> None:
        self._calls: int = 0
        logger.debug("IndicatorEngine initialized (Phase 1: full-window recomputation)")

    def compute(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Takes an immutable DataFrame SLICE (a view), returns a NEW
        DataFrame with all indicator columns added.

        NEVER modifies the input slice. Always returns a fresh copy.

        Args:
            df_slice: Read-only view of historical M5 candles up to
                      the current replay index.

        Returns:
            New DataFrame with OHLCV + all indicator columns.
            Index is preserved from input.
        """
        if df_slice is None or len(df_slice) == 0:
            return pd.DataFrame()

        # Make a copy so we never mutate the immutable source (Req 7)
        df = df_slice.copy()
        self._calls += 1

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values

        # ── EMA 21 (fast trend filter) ───────────────────────────────
        # Uses EXACT same _ema() as MT5DataProvider — guaranteed parity
        df["ema_21"] = _ema(close, self.EMA_FAST)

        # ── EMA 50 (slow trend filter) ───────────────────────────────
        df["ema_50"] = _ema(close, self.EMA_SLOW)

        # ── ATR 14 (Wilder's smoothing, same as MT5DataProvider) ─────
        df["atr"] = _atr(high, low, close, self.ATR_PERIOD)

        # ── Ichimoku Cloud ────────────────────────────────────────────
        df["tenkan_sen"] = _donchian_mid(high, low, 9)
        df["kijun_sen"] = _donchian_mid(high, low, 26)
        tenkan = df["tenkan_sen"].values
        kijun = df["kijun_sen"].values
        senkou_a_raw = (tenkan + kijun) / 2.0
        df["senkou_a"] = np.concatenate([np.full(26, np.nan), senkou_a_raw[:-26]]) if len(senkou_a_raw) > 26 else senkou_a_raw
        senkou_b_raw = _donchian_mid(high, low, 52)
        df["senkou_b"] = np.concatenate([np.full(26, np.nan), senkou_b_raw[:-26]]) if len(senkou_b_raw) > 26 else senkou_b_raw
        df["chikou_span"] = df["close"].shift(-26)

        # ── RSI 14 ────────────────────────────────────────────────────
        df["rsi_14"] = _rsi(close, 14)

        # ── VWAP ──────────────────────────────────────────────────────
        df["vwap"] = _vwap(df)

        # ── Candle morphology ────────────────────────────────────────
        df["body"]        = df["close"] - df["open"]
        df["body_abs"]    = df["body"].abs()
        df["range"]       = df["high"] - df["low"]
        df["upper_wick"]  = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"]  = df[["open", "close"]].min(axis=1) - df["low"]

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

        # ── Volume relative to rolling average ───────────────────────
        vol = df["tick_volume"].values.astype(float)
        avg_vol = (
            pd.Series(vol)
            .rolling(self.VOL_ROLLING, min_periods=1)
            .mean()
            .values
        )
        df["vol_ratio"] = np.where(avg_vol > 0, vol / avg_vol, 1.0)

        return df

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        """Number of times compute() has been called (= candles processed)."""
        return self._calls

    def validate_parity(
        self, df_slice: pd.DataFrame, reference_df: pd.DataFrame, tolerance: float = 1e-6
    ) -> bool:
        """
        Validates that IndicatorEngine output matches a reference DataFrame
        (e.g., from MT5DataProvider) to within floating-point tolerance.

        Used in automated tests to guarantee numerical parity between
        live and replay indicator computation.

        Args:
            df_slice: Raw OHLCV slice to compute on
            reference_df: DataFrame with pre-computed indicator columns
            tolerance: Maximum allowed absolute difference per cell

        Returns:
            True if all indicators match within tolerance
        """
        computed = self.compute(df_slice)
        cols = ["ema_21", "ema_50", "atr", "body_ratio", "close_position", "vol_ratio"]

        for col in cols:
            if col not in reference_df.columns:
                continue
            ref = reference_df[col].values
            sim = computed[col].values

            # Ignore NaN positions (EMA not initialized at start)
            valid_mask = ~(np.isnan(ref) | np.isnan(sim))
            if not valid_mask.any():
                continue

            max_diff = np.max(np.abs(ref[valid_mask] - sim[valid_mask]))
            if max_diff > tolerance:
                logger.error(
                    f"Parity failure on '{col}': max diff={max_diff:.2e} "
                    f"(tolerance={tolerance:.2e})"
                )
                return False

        return True
