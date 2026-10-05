"""
signal_engine/ichimoku_strategy.py — Ichimoku Cloud + VWAP + RSI Strategy (v4.0)

Replaces the 6-module SignalScorer with a clean, binary-logic strategy:
  - M15 Ichimoku Cloud for trend direction
  - M5 Kijun-sen for pullback entry level
  - VWAP for institutional price reference
  - RSI(14) for momentum confirmation

Entry requires ALL conditions to be true. No scoring, no factors.
Exit is handled by fixed SL/TP + breakeven lock.

At 1:2 R:R (SL=1.5×ATR, TP=3.0×ATR), only needs 34% WR to be profitable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from monitoring.logger import get_logger

logger = get_logger("Ichimoku")


# Reuse the Signal dataclass from scorer.py for compatibility with OrderManager
from signal_engine.scorer import Signal


# ── SL/TP Configuration ────────────────────────────────────────
SL_ATR_MULT = 1.5   # Stop loss distance
TP_ATR_MULT = 3.0   # Take profit distance → 1:2 R:R


class IchimokuStrategy:
    """
    Ichimoku Cloud + VWAP + RSI trend-following strategy.

    Entry Logic:
      BUY:  M15 price above cloud + TK bullish
            + M5 price near Kijun + price ≤ VWAP + RSI 30-55
      SELL: M15 price below cloud + TK bearish
            + M5 price near Kijun + price ≥ VWAP + RSI 45-70

    All conditions must be true simultaneously.
    """

    def __init__(self) -> None:
        self._last_signal_time: Dict[str, float] = {}
        logger.info(
            "IchimokuStrategy v4.0 initialized │ "
            f"SL={SL_ATR_MULT}×ATR │ TP={TP_ATR_MULT}×ATR │ "
            "Trend=Ichimoku Cloud │ Entry=Kijun pullback │ "
            "Confirm=VWAP+RSI"
        )

    def evaluate(
        self, market_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> List[Signal]:
        """
        Evaluates all symbols and returns actionable signals.
        Drop-in replacement for SignalScorer.evaluate().
        """
        if not market_data:
            return []

        signals = []

        for sym_key in ("ES", "NQ"):
            sym_data = market_data.get(sym_key)
            if not sym_data:
                continue

            signal = self._evaluate_symbol(sym_key, sym_data)

            if signal and signal.is_valid:
                if self._is_on_cooldown(sym_key):
                    continue
                self._mark_signal_fired(sym_key)
                signals.append(signal)

                logger.log(25,  # TRADE level
                    f"🎯 ICHIMOKU SIGNAL │ {sym_key}→{signal.mt5_symbol} │ "
                    f"{signal.action} │ Score={signal.score}/100 │ "
                    f"SL={signal.sl_distance:.2f} │ TP={signal.tp_distance:.2f} │ "
                    f"{signal.reason}"
                )

        return signals

    def _evaluate_symbol(
        self, sym_key: str, sym_data: Dict[str, pd.DataFrame]
    ) -> Optional[Signal]:
        """Evaluate a single symbol for Ichimoku entry conditions."""

        m5 = sym_data.get("M5")
        m15 = sym_data.get("M15")
        sym_cfg = settings.SYMBOL_MAP.get(sym_key)

        if m5 is None or m15 is None or sym_cfg is None:
            return None
        if len(m5) < 60 or len(m15) < 60:
            return None

        last_m5 = m5.iloc[-1]
        last_m15 = m15.iloc[-1]

        # ── Get indicator values ─────────────────────────────────
        atr = float(last_m5.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            return None

        # M15 Ichimoku
        m15_close = float(last_m15["close"])
        m15_tenkan = last_m15.get("tenkan_sen", np.nan)
        m15_kijun = last_m15.get("kijun_sen", np.nan)
        m15_senkou_a = last_m15.get("senkou_a", np.nan)
        m15_senkou_b = last_m15.get("senkou_b", np.nan)

        if any(pd.isna(v) for v in [m15_tenkan, m15_kijun, m15_senkou_a, m15_senkou_b]):
            logger.debug(f"[{sym_key}] Ichimoku not initialized on M15")
            return None

        m15_tenkan = float(m15_tenkan)
        m15_kijun = float(m15_kijun)
        m15_senkou_a = float(m15_senkou_a)
        m15_senkou_b = float(m15_senkou_b)

        # M5 indicators
        m5_close = float(last_m5["close"])
        m5_kijun = last_m5.get("kijun_sen", np.nan)
        m5_vwap = last_m5.get("vwap", np.nan)
        m5_rsi = last_m5.get("rsi_14", np.nan)

        if any(pd.isna(v) for v in [m5_kijun, m5_vwap, m5_rsi]):
            logger.debug(f"[{sym_key}] M5 indicators not ready")
            return None

        m5_kijun = float(m5_kijun)
        m5_vwap = float(m5_vwap)
        m5_rsi = float(m5_rsi)

        # ── Cloud boundaries ──────────────────────────────────────
        cloud_top = max(m15_senkou_a, m15_senkou_b)
        cloud_bottom = min(m15_senkou_a, m15_senkou_b)

        # ══════════════════════════════════════════════════════════
        # CHECK BUY CONDITIONS
        # ══════════════════════════════════════════════════════════
        buy_reasons = []
        buy_ok = True

        # 1. M15 price above cloud (bullish trend)
        if m15_close > cloud_top:
            buy_reasons.append(f"M15 above cloud ({m15_close:.0f}>{cloud_top:.0f})")
        else:
            buy_ok = False

        # 2. M15 Tenkan > Kijun (TK bullish cross)
        if buy_ok and m15_tenkan > m15_kijun:
            buy_reasons.append(f"TK bullish ({m15_tenkan:.1f}>{m15_kijun:.1f})")
        elif buy_ok:
            buy_ok = False

        # 3. M5 price near Kijun-sen (pullback to equilibrium)
        #    "Near" = within 1.0×ATR of Kijun
        if buy_ok:
            kijun_dist = abs(m5_close - m5_kijun) / atr
            if kijun_dist <= 1.0:
                buy_reasons.append(f"Near Kijun ({kijun_dist:.2f}×ATR)")
            else:
                buy_ok = False

        # 4. M5 price ≤ VWAP (buying at institutional discount)
        if buy_ok:
            if m5_close <= m5_vwap * 1.001:  # Small tolerance
                buy_reasons.append(f"Below VWAP ({m5_close:.1f}≤{m5_vwap:.1f})")
            else:
                buy_ok = False

        # 5. M5 RSI between 30-55 (oversold pullback, not exhausted)
        if buy_ok:
            if 30 <= m5_rsi <= 55:
                buy_reasons.append(f"RSI={m5_rsi:.0f}")
            else:
                buy_ok = False

        # ══════════════════════════════════════════════════════════
        # CHECK SELL CONDITIONS
        # ══════════════════════════════════════════════════════════
        sell_reasons = []
        sell_ok = True

        # 1. M15 price below cloud (bearish trend)
        if m15_close < cloud_bottom:
            sell_reasons.append(f"M15 below cloud ({m15_close:.0f}<{cloud_bottom:.0f})")
        else:
            sell_ok = False

        # 2. M15 Tenkan < Kijun (TK bearish cross)
        if sell_ok and m15_tenkan < m15_kijun:
            sell_reasons.append(f"TK bearish ({m15_tenkan:.1f}<{m15_kijun:.1f})")
        elif sell_ok:
            sell_ok = False

        # 3. M5 price near Kijun-sen (pullback to equilibrium)
        if sell_ok:
            kijun_dist = abs(m5_close - m5_kijun) / atr
            if kijun_dist <= 1.0:
                sell_reasons.append(f"Near Kijun ({kijun_dist:.2f}×ATR)")
            else:
                sell_ok = False

        # 4. M5 price ≥ VWAP (selling at institutional premium)
        if sell_ok:
            if m5_close >= m5_vwap * 0.999:  # Small tolerance
                sell_reasons.append(f"Above VWAP ({m5_close:.1f}≥{m5_vwap:.1f})")
            else:
                sell_ok = False

        # 5. M5 RSI between 45-70 (overbought pullback, not exhausted)
        if sell_ok:
            if 45 <= m5_rsi <= 70:
                sell_reasons.append(f"RSI={m5_rsi:.0f}")
            else:
                sell_ok = False

        # ══════════════════════════════════════════════════════════
        # BUILD SIGNAL
        # ══════════════════════════════════════════════════════════

        if buy_ok:
            action = "BUY"
            reason = " + ".join(buy_reasons)
            score = 80  # Fixed high score (all conditions met = high conviction)
        elif sell_ok:
            action = "SELL"
            reason = " + ".join(sell_reasons)
            score = 80
        else:
            # Log why no signal (for diagnostics, every ~60 cycles)
            cloud_pos = "ABOVE" if m15_close > cloud_top else ("BELOW" if m15_close < cloud_bottom else "INSIDE")
            logger.info(
                f"[{sym_key}] ICHIMOKU │ No signal │ "
                f"Cloud={cloud_pos} │ TK={'B' if m15_tenkan > m15_kijun else 'S'} │ "
                f"Kijun_dist={abs(m5_close - m5_kijun)/atr:.2f}×ATR │ "
                f"VWAP={'below' if m5_close < m5_vwap else 'above'} │ "
                f"RSI={m5_rsi:.0f}"
            )
            return None

        # Structural reference for health monitor
        struct_ref = float(m5.iloc[-1].get("kijun_sen", m5_close))

        return Signal(
            symbol=sym_key,
            mt5_symbol=sym_cfg["mt5_name"],
            action=action,
            score=score,
            atr=atr,
            sl_distance=atr * SL_ATR_MULT,
            tp_distance=atr * TP_ATR_MULT,
            reason=reason,
            timestamp=time.time(),
            trend_score=20,     # Cloud confirmed
            structure_score=0,
            rejection_score=0,
            volume_score=0,
            pressure_score=0,
            correlation_score=0,
            factors_active=5,   # All 5 conditions met
            regime="NORMAL",
            structural_ref=struct_ref,
            speed_score=50,
            profile=None,       # v4.0: no regime profiles needed
            setup_type="PULLBACK",
        )

    # ── Cooldown ────────────────────────────────────────────────
    def _is_on_cooldown(self, symbol: str) -> bool:
        last = self._last_signal_time.get(symbol, 0)
        elapsed = time.time() - last
        if elapsed < settings.SIGNAL_COOLDOWN_SEC:
            return True
        return False

    def _mark_signal_fired(self, symbol: str) -> None:
        self._last_signal_time[symbol] = time.time()
