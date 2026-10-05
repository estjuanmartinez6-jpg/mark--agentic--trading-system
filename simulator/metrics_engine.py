"""
simulator/metrics_engine.py — Post-Replay Performance Analytics for MARK III

Computes comprehensive statistics from closed trade records after a replay.
Designed to support multi-run comparison for parameter sensitivity analysis.

Metrics computed:
  Core:    Total trades, win rate, profit factor, expectancy
  Risk:    Max drawdown, Sharpe ratio, Calmar ratio, Ulcer index
  Streaks: Max consecutive wins/losses, recovery factor
  Session: PnL by hour, win rate by hour, trades by hour
  Setup:   Win rate by CONTINUATION vs REVERSAL, by regime
  Fill:    Spread and slippage impact analysis
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("Sim.Metrics")


class MetricsEngine:
    """
    Computes all performance statistics from a completed replay session's
    closed trade list.

    Usage:
        metrics = MetricsEngine(trades, initial_balance=30.0)
        report = metrics.generate()
        metrics.print_report(report)
        metrics.save_json(report, "results/metrics.json")
    """

    def __init__(
        self,
        trades: List[dict],
        initial_balance: float = 30.0,
    ) -> None:
        self._trades = trades
        self._initial_balance = initial_balance
        self._df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Main interface ───────────────────────────────────────────────

    def generate(self) -> dict:
        """
        Compute all metrics and return a structured report dict.
        Returns an empty skeleton if no trades were taken.
        """
        if self._df.empty:
            logger.info("[Metrics] No trades to analyze")
            return self._empty_report()

        df = self._df.copy()
        df["pnl_usd"] = pd.to_numeric(df.get("pnl_usd", 0), errors="coerce").fillna(0.0)

        report = {
            "summary":         self._core_metrics(df),
            "risk":            self._risk_metrics(df),
            "streaks":         self._streak_metrics(df),
            "by_hour":         self._by_hour(df),
            "by_session":      self._by_session(df),
            "by_setup":        self._by_setup(df),
            "by_regime":       self._by_regime(df),
            "fill_impact":     self._fill_impact(df),
            "equity_curve":    self._equity_curve(df),
        }
        return report

    # ── Core metrics ────────────────────────────────────────────────

    def _core_metrics(self, df: pd.DataFrame) -> dict:
        pnl = df["pnl_usd"].values
        winners = pnl[pnl > 0]
        losers  = pnl[pnl < 0]

        total   = len(pnl)
        wins    = len(winners)
        losses  = len(losers)
        win_rate = wins / total if total > 0 else 0.0

        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss   = float(abs(losers.sum())) if len(losers) > 0 else 0.0
        net_pnl      = float(pnl.sum())

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        avg_win  = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss = float(abs(losers.mean())) if len(losers) > 0 else 0.0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        avg_duration = (
            df["duration_min"].mean() if "duration_min" in df.columns else 0.0
        )

        return {
            "total_trades":   total,
            "winners":        wins,
            "losers":         losses,
            "breakeven":      total - wins - losses,
            "win_rate_pct":   round(win_rate * 100, 2),
            "profit_factor":  round(profit_factor, 3) if profit_factor != float("inf") else "∞",
            "expectancy_usd": round(expectancy, 4),
            "net_pnl_usd":    round(net_pnl, 4),
            "gross_profit":   round(gross_profit, 4),
            "gross_loss":     round(gross_loss, 4),
            "avg_win_usd":    round(avg_win, 4),
            "avg_loss_usd":   round(avg_loss, 4),
            "largest_win":    round(float(winners.max()), 4) if len(winners) > 0 else 0.0,
            "largest_loss":   round(float(abs(losers.min())), 4) if len(losers) > 0 else 0.0,
            "avg_duration_min": round(float(avg_duration), 2),
            "final_balance":  round(self._initial_balance + net_pnl, 4),
            "return_pct":     round((net_pnl / self._initial_balance) * 100, 3),
        }

    # ── Risk metrics ────────────────────────────────────────────────

    def _risk_metrics(self, df: pd.DataFrame) -> dict:
        pnl = df["pnl_usd"].values
        equity_curve = self._initial_balance + pnl.cumsum()

        # Max drawdown
        peak = equity_curve[0]
        max_dd_usd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd_usd:
                max_dd_usd = dd
        max_dd_pct = (max_dd_usd / self._initial_balance) * 100 if self._initial_balance > 0 else 0

        # Sharpe ratio (annualized, risk-free rate = 0)
        import numpy as np
        std = float(np.std(pnl)) if len(pnl) > 1 else 0.0
        mean = float(np.mean(pnl))
        # Annualize assuming ~252 trading days × 6 trades/day (rough)
        trades_per_year = 252 * 6
        sharpe = (
            (mean / std) * math.sqrt(trades_per_year) if std > 0 else 0.0
        )

        # Calmar ratio
        net_return_pct = ((sum(pnl)) / self._initial_balance) * 100
        calmar = net_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

        # Ulcer index (measure of depth and duration of drawdowns)
        drawdowns_pct = []
        peak = equity_curve[0]
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak) * 100 if peak > 0 else 0.0
            drawdowns_pct.append(dd_pct)
        import numpy as np
        ulcer = math.sqrt(float(np.mean(np.array(drawdowns_pct) ** 2)))

        # Recovery factor
        recovery = (
            abs(sum(pnl)) / max_dd_usd if max_dd_usd > 0 else 0.0
        )

        return {
            "max_drawdown_usd":  round(max_dd_usd, 4),
            "max_drawdown_pct":  round(max_dd_pct, 3),
            "sharpe_ratio":      round(sharpe, 4),
            "calmar_ratio":      round(calmar, 4),
            "ulcer_index":       round(ulcer, 4),
            "recovery_factor":   round(recovery, 4),
        }

    # ── Streak metrics ───────────────────────────────────────────────

    def _streak_metrics(self, df: pd.DataFrame) -> dict:
        pnl = df["pnl_usd"].values
        max_wins = max_losses = 0
        cur_wins = cur_losses = 0

        for p in pnl:
            if p > 0:
                cur_wins  += 1
                cur_losses = 0
                max_wins   = max(max_wins, cur_wins)
            elif p < 0:
                cur_losses += 1
                cur_wins   = 0
                max_losses  = max(max_losses, cur_losses)
            else:
                cur_wins = cur_losses = 0

        return {
            "max_consecutive_wins":   max_wins,
            "max_consecutive_losses": max_losses,
        }

    # ── By-hour breakdown ────────────────────────────────────────────

    def _by_hour(self, df: pd.DataFrame) -> dict:
        if "entry_time" not in df.columns:
            return {}
        try:
            df = df.copy()
            df["entry_hour_ny"] = pd.to_datetime(
                df["entry_time"], utc=True
            ).dt.tz_convert("America/New_York").dt.hour

            result = {}
            for hour in range(8, 17):
                h = df[df["entry_hour_ny"] == hour]
                if h.empty:
                    continue
                pnl_h = h["pnl_usd"]
                wins  = (pnl_h > 0).sum()
                result[f"{hour:02d}:xx NY"] = {
                    "trades":      int(len(h)),
                    "wins":        int(wins),
                    "win_rate":    round(wins / len(h) * 100, 1),
                    "net_pnl":     round(float(pnl_h.sum()), 4),
                }
            return result
        except Exception as exc:
            logger.debug(f"[Metrics] by_hour error: {exc}")
            return {}

    # ── By session ───────────────────────────────────────────────────

    def _by_session(self, df: pd.DataFrame) -> dict:
        if "simulated_session" not in df.columns:
            return {}
        result = {}
        # Fixed explicit ordering
        for session in ["MORNING", "AFTERNOON", "NIGHT"]:
            g = df[df["simulated_session"] == session]
            if len(g) == 0:
                continue
            wins = int((g["pnl_usd"] > 0).sum())
            result[session] = {
                "trades":    int(len(g)),
                "win_rate":  float(round(wins / len(g) * 100, 1)),
                "net_pnl":   float(round(float(g["pnl_usd"].sum()), 4)),
            }
        return result

    # ── By setup type ────────────────────────────────────────────────

    def _by_setup(self, df: pd.DataFrame) -> dict:
        if "setup_type" not in df.columns:
            return {}
        result = {}
        for stype in df["setup_type"].unique():
            g = df[df["setup_type"] == stype]
            wins = (g["pnl_usd"] > 0).sum()
            result[stype] = {
                "trades":    int(len(g)),
                "win_rate":  round(wins / len(g) * 100, 1) if len(g) > 0 else 0,
                "net_pnl":   round(float(g["pnl_usd"].sum()), 4),
            }
        return result

    # ── By regime ───────────────────────────────────────────────────

    def _by_regime(self, df: pd.DataFrame) -> dict:
        col = "market_regime" if "market_regime" in df.columns else "regime"
        if col not in df.columns:
            return {}
        result = {}
        for regime in df[col].unique():
            g = df[df[col] == regime]
            wins = (g["pnl_usd"] > 0).sum()
            result[regime] = {
                "trades":    int(len(g)),
                "win_rate":  round(wins / len(g) * 100, 1) if len(g) > 0 else 0,
                "net_pnl":   round(float(g["pnl_usd"].sum()), 4),
            }
        return result

    # ── Fill impact analysis ─────────────────────────────────────────

    def _fill_impact(self, df: pd.DataFrame) -> dict:
        result = {}
        if "spread_at_entry" in df.columns:
            result["avg_spread_pts"] = round(float(df["spread_at_entry"].mean()), 4)
            result["total_spread_cost_usd"] = round(
                float((df["spread_at_entry"] * df.get("lot_size", 1)).sum()), 4
            )
        if "slippage_applied" in df.columns:
            result["avg_slippage_pts"] = round(float(df["slippage_applied"].mean()), 4)
            result["total_slippage_cost_usd"] = round(
                float((df["slippage_applied"] * df.get("lot_size", 1)).sum()), 4
            )
        if "commission" in df.columns:
            result["total_commission_usd"] = round(float(df["commission"].sum()), 4)
        return result

    # ── Equity curve ─────────────────────────────────────────────────

    def _equity_curve(self, df: pd.DataFrame) -> list:
        pnl = df["pnl_usd"].values
        curve = []
        running = self._initial_balance
        for i, p in enumerate(pnl):
            running += float(p)
            curve.append(round(running, 4))
        return curve

    # ── Reporting ────────────────────────────────────────────────────

    def print_report(self, report: dict) -> None:
        s = report.get("summary", {})
        r = report.get("risk", {})
        k = report.get("streaks", {})

        print("\n" + "═" * 60)
        print(" MARK III REPLAY — PERFORMANCE REPORT")
        print("═" * 60)
        print(f"  Total trades:    {s.get('total_trades', 0)}")
        print(f"  Win rate:        {s.get('win_rate_pct', 0):.1f}%  ({s.get('winners',0)}W / {s.get('losers',0)}L)")
        print(f"  Net P&L:         ${s.get('net_pnl_usd', 0):+.2f}")
        print(f"  Return:          {s.get('return_pct', 0):+.2f}%")
        print(f"  Profit Factor:   {s.get('profit_factor', 0)}")
        print(f"  Expectancy:      ${s.get('expectancy_usd', 0):+.4f}")
        print(f"  Avg Win:         ${s.get('avg_win_usd', 0):.2f}")
        print(f"  Avg Loss:        ${s.get('avg_loss_usd', 0):.2f}")
        print(f"  Largest Win:     ${s.get('largest_win', 0):.2f}")
        print(f"  Largest Loss:    ${s.get('largest_loss', 0):.2f}")
        print(f"  ─────────────────────────────────")
        print(f"  Max Drawdown:    ${r.get('max_drawdown_usd', 0):.2f} ({r.get('max_drawdown_pct', 0):.2f}%)")
        print(f"  Sharpe Ratio:    {r.get('sharpe_ratio', 0):.3f}")
        print(f"  Calmar Ratio:    {r.get('calmar_ratio', 0):.3f}")
        print(f"  Recovery Factor: {r.get('recovery_factor', 0):.3f}")
        print(f"  ─────────────────────────────────")
        print(f"  Max Win Streak:  {k.get('max_consecutive_wins', 0)}")
        print(f"  Max Loss Streak: {k.get('max_consecutive_losses', 0)}")
        print("═" * 60)

        # By session
        by_session = report.get("by_session", {})
        if by_session:
            print("\n  BY SESSION:")
            for sess, data in by_session.items():
                print(f"    {sess:20s} {data['trades']:3d} trades │ "
                      f"WR={data['win_rate']:.0f}% │ P&L=${data['net_pnl']:+.2f}")

        # By setup
        by_setup = report.get("by_setup", {})
        if by_setup:
            print("\n  BY SETUP TYPE:")
            for setup, data in by_setup.items():
                print(f"    {setup:20s} {data['trades']:3d} trades │ "
                      f"WR={data['win_rate']:.0f}% │ P&L=${data['net_pnl']:+.2f}")

        # By regime
        by_regime = report.get("by_regime", {})
        if by_regime:
            print("\n  BY REGIME:")
            for reg, data in by_regime.items():
                print(f"    {reg:20s} {data['trades']:3d} trades │ "
                      f"WR={data['win_rate']:.0f}% │ P&L=${data['net_pnl']:+.2f}")

        print()

    def save_json(self, report: dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[Metrics] Saved to {path}")

    @staticmethod
    def compare(results: List[dict], labels: List[str]) -> None:
        """Print side-by-side comparison of multiple replay results."""
        print("\n" + "═" * 80)
        print(" REPLAY COMPARISON")
        print("═" * 80)
        metrics_keys = [
            ("summary", "total_trades"),
            ("summary", "win_rate_pct"),
            ("summary", "net_pnl_usd"),
            ("summary", "profit_factor"),
            ("risk",    "max_drawdown_pct"),
            ("risk",    "sharpe_ratio"),
        ]
        header = f"{'Metric':35s}" + "".join(f"{lbl:>15s}" for lbl in labels)
        print(header)
        print("─" * 80)
        for section, key in metrics_keys:
            vals = [r.get(section, {}).get(key, "N/A") for r in results]
            row = f"{section}.{key:25s}" + "".join(f"{str(v):>15s}" for v in vals)
            print(row)
        print()

    def _empty_report(self) -> dict:
        return {
            "summary":      {"total_trades": 0, "net_pnl_usd": 0.0, "win_rate_pct": 0.0},
            "risk":         {"max_drawdown_usd": 0.0, "sharpe_ratio": 0.0},
            "streaks":      {},
            "by_hour":      {},
            "by_setup":     {},
            "by_regime":    {},
            "fill_impact":  {},
            "equity_curve": [],
        }
