"""
simulator/visualizer.py — Post-Replay Charts for MARK III Replay

Subscribes to EventBus events and generates visual performance reports.
All charts saved as PNG to the session output directory.

Charts generated:
  1. equity_curve.png  — Running account balance + drawdown shading
  2. candles.png       — M5 candlestick with EMA21/EMA50 + trade markers
  3. pnl_dist.png      — P&L distribution histogram
  4. by_hour.png       — Win rate + P&L by NY hour

Requires: matplotlib, mplfinance
Optional: gracefully degrades if not installed (skips charts, logs warning)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("Sim.Visualizer")

# Check availability of plotting libraries at module load
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend — safe for headless runs
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("[Visualizer] matplotlib not installed. Charts disabled.")

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False


class ReplayVisualizer:
    """
    Generates post-replay visualization charts.

    Subscribes to EventBus "replay_end" event so charts are generated
    automatically when the session finishes.

    All charts are saved as PNG to output_dir. No GUI windows shown.
    """

    def __init__(self, output_dir: str, session_config: dict = None) -> None:
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._session_config = session_config or {}

        # Data accumulated during replay
        self._trade_opens: List[dict]   = []
        self._trade_closes: List[dict]  = []
        self._equity_history: List[dict] = []
        self._candle_history: List[pd.Series] = []
        self._initial_balance: float = 30.0

    def subscribe(self, bus, initial_balance: float = 30.0) -> None:
        """Subscribe to EventBus events."""
        from simulator.event_bus import EventBus
        self._initial_balance = initial_balance
        bus.subscribe(EventBus.NEW_CANDLE,   self._on_new_candle)
        bus.subscribe(EventBus.TRADE_OPEN,   self._on_trade_open)
        bus.subscribe(EventBus.TRADE_CLOSE,  self._on_trade_close)
        bus.subscribe(EventBus.REPLAY_END,   self._on_replay_end)

    # ── Event handlers ───────────────────────────────────────────────

    def _on_new_candle(
        self, timestamp, raw_candle: pd.Series, **kwargs
    ) -> None:
        """Accumulate candles for the chart."""
        self._candle_history.append(raw_candle)

    def _on_trade_open(
        self, ticket, fill_price, timestamp, signal, **kwargs
    ) -> None:
        self._trade_opens.append({
            "ticket":     ticket,
            "price":      fill_price,
            "timestamp":  timestamp,
            "direction":  signal.action,
            "score":      signal.score,
        })

    def _on_trade_close(
        self, ticket, exit_price, pnl_usd, reason, **kwargs
    ) -> None:
        self._trade_closes.append({
            "ticket":    ticket,
            "price":     exit_price,
            "pnl_usd":   pnl_usd,
            "reason":    reason,
            "timestamp": kwargs.get("timestamp"),
        })

    def _on_replay_end(self, metrics_summary: dict = None, **kwargs) -> None:
        """Generate all charts when replay finishes."""
        self.generate_all(metrics_summary or {})

    # ── Chart generation ─────────────────────────────────────────────

    def generate_all(self, metrics: dict = None) -> None:
        """Generate and save all post-replay charts."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("[Visualizer] matplotlib not available — skipping charts")
            return

        logger.info(f"[Visualizer] Generating charts → {self._out_dir}")

        trades_df = self._build_trades_df()

        self._plot_equity_curve(trades_df)
        self._plot_pnl_distribution(trades_df)

        if MPLFINANCE_AVAILABLE and self._candle_history:
            self._plot_candlestick()

        if not trades_df.empty and "by_hour" in (metrics or {}):
            self._plot_by_hour(metrics["by_hour"])

        logger.info("[Visualizer] All charts saved.")

    def _plot_equity_curve(self, trades_df: pd.DataFrame) -> None:
        """Equity curve with drawdown shading."""
        if trades_df.empty:
            return

        pnl_values = trades_df["pnl_usd"].values
        equity = self._initial_balance + pnl_values.cumsum()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.patch.set_facecolor("#0d1117")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            ax.spines["bottom"].set_color("#30363d")
            ax.spines["top"].set_color("#30363d")
            ax.spines["left"].set_color("#30363d")
            ax.spines["right"].set_color("#30363d")

        # Equity line
        ax1.plot(equity, color="#58a6ff", linewidth=1.5, label="Equity")
        ax1.axhline(
            y=self._initial_balance, color="#6e7681",
            linestyle="--", linewidth=1, alpha=0.7, label="Initial Balance"
        )

        # Drawdown shading
        peak = equity[0]
        for i, eq in enumerate(equity):
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > 0:
                ax1.fill_between([i - 1, i], [peak, peak], [eq, eq],
                                 color="#f85149", alpha=0.15)

        # Trade markers on equity
        for i, row in trades_df.iterrows():
            color = "#3fb950" if row["pnl_usd"] > 0 else "#f85149"
            ax1.scatter(i, equity[i], color=color, s=30, zorder=5, alpha=0.8)

        ax1.set_title("Equity Curve — MARK III Replay",
                      color="#e6edf3", fontsize=14, pad=12)
        ax1.set_ylabel("Balance (USD)", color="#8b949e")
        ax1.legend(facecolor="#161b22", labelcolor="#e6edf3", framealpha=0.8)
        ax1.grid(True, color="#21262d", linewidth=0.5)

        # P&L bars
        colors = ["#3fb950" if p > 0 else "#f85149" for p in pnl_values]
        ax2.bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.8)
        ax2.axhline(y=0, color="#6e7681", linewidth=0.8)
        ax2.set_ylabel("Trade P&L", color="#8b949e")
        ax2.set_xlabel("Trade #", color="#8b949e")
        ax2.grid(True, color="#21262d", linewidth=0.5)

        plt.tight_layout()
        out = self._out_dir / "equity_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[Visualizer] Saved: {out.name}")

    def _plot_candlestick(self) -> None:
        """M5 candlestick chart with EMA21/EMA50 and trade markers."""
        if not MPLFINANCE_AVAILABLE or not self._candle_history:
            return

        try:
            df = pd.DataFrame(self._candle_history)
            df = df.set_index("time")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            df = df[["open", "high", "low", "close", "tick_volume"]].rename(
                columns={"tick_volume": "volume"}
            )

            # EMA overlays
            import numpy as np
            from data_handler.mt5_data import _ema
            close_arr = df["close"].values
            ema21 = _ema(close_arr, 21)
            ema50 = _ema(close_arr, 50)

            add_plots = [
                mpf.make_addplot(
                    ema21, color="#f0883e", width=1.2, label="EMA21"
                ),
                mpf.make_addplot(
                    ema50, color="#58a6ff", width=1.2, label="EMA50"
                ),
            ]

            # Trade markers
            buy_markers  = [float("nan")] * len(df)
            sell_markers = [float("nan")] * len(df)
            exit_markers = [float("nan")] * len(df)

            for trade in self._trade_opens:
                ts = pd.Timestamp(trade["timestamp"])
                if ts.tz is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                if ts in df.index:
                    idx = df.index.get_loc(ts)
                    if trade["direction"] == "BUY":
                        buy_markers[idx] = float(df["low"].iloc[idx]) - 2
                    else:
                        sell_markers[idx] = float(df["high"].iloc[idx]) + 2

            if any(not pd.isna(v) for v in buy_markers):
                add_plots.append(mpf.make_addplot(
                    buy_markers, type="scatter", marker="^",
                    markersize=80, color="#3fb950"
                ))
            if any(not pd.isna(v) for v in sell_markers):
                add_plots.append(mpf.make_addplot(
                    sell_markers, type="scatter", marker="v",
                    markersize=80, color="#f85149"
                ))

            dark_style = mpf.make_mpf_style(
                base_mpf_style="nightclouds",
                facecolor="#0d1117",
                edgecolor="#30363d",
                gridcolor="#21262d",
            )

            out = self._out_dir / "candles.png"
            mpf.plot(
                df,
                type="candle",
                style=dark_style,
                addplot=add_plots,
                volume=True,
                title="\nMARK III Replay — M5 Candles",
                ylabel="Price",
                figratio=(16, 9),
                figscale=1.2,
                savefig=dict(fname=str(out), dpi=150, bbox_inches="tight"),
            )
            logger.info(f"[Visualizer] Saved: {out.name}")
        except Exception as exc:
            logger.warning(f"[Visualizer] Candlestick chart failed: {exc}")

    def _plot_pnl_distribution(self, trades_df: pd.DataFrame) -> None:
        """P&L distribution histogram."""
        if trades_df.empty:
            return

        pnl = trades_df["pnl_usd"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        bins = min(30, max(10, len(pnl) // 2))
        ax.hist(pnl[pnl > 0], bins=bins, color="#3fb950", alpha=0.7, label="Winners")
        ax.hist(pnl[pnl < 0], bins=bins, color="#f85149", alpha=0.7, label="Losers")
        ax.axvline(x=0, color="#6e7681", linewidth=1.5, linestyle="--")
        ax.axvline(
            x=float(pnl.mean()), color="#f0883e",
            linewidth=1.5, linestyle=":", label=f"Mean: ${pnl.mean():+.3f}"
        )

        ax.set_title("P&L Distribution", color="#e6edf3", fontsize=13)
        ax.set_xlabel("P&L (USD)", color="#8b949e")
        ax.set_ylabel("Count", color="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.grid(True, color="#21262d", linewidth=0.5)

        plt.tight_layout()
        out = self._out_dir / "pnl_distribution.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[Visualizer] Saved: {out.name}")

    def _plot_by_hour(self, by_hour: dict) -> None:
        """Win rate and P&L by NY hour bar chart."""
        if not by_hour:
            return

        hours   = list(by_hour.keys())
        wr      = [by_hour[h].get("win_rate", 0) for h in hours]
        net_pnl = [by_hour[h].get("net_pnl", 0) for h in hours]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True
        )
        fig.patch.set_facecolor("#0d1117")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.grid(True, color="#21262d", linewidth=0.5)

        x = range(len(hours))
        ax1.bar(x, wr, color="#58a6ff", alpha=0.8)
        ax1.axhline(y=50, color="#f0883e", linestyle="--", linewidth=1)
        ax1.set_ylabel("Win Rate (%)", color="#8b949e")
        ax1.set_title("Performance by NY Hour", color="#e6edf3", fontsize=13)

        pnl_colors = ["#3fb950" if p > 0 else "#f85149" for p in net_pnl]
        ax2.bar(x, net_pnl, color=pnl_colors, alpha=0.8)
        ax2.axhline(y=0, color="#6e7681", linewidth=0.8)
        ax2.set_ylabel("Net P&L (USD)", color="#8b949e")
        ax2.set_xlabel("NY Hour", color="#8b949e")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(hours, rotation=45, ha="right")

        plt.tight_layout()
        out = self._out_dir / "by_hour.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[Visualizer] Saved: {out.name}")

    # ── Helper ───────────────────────────────────────────────────────

    def _build_trades_df(self) -> pd.DataFrame:
        """Merge open/close events into a trade DataFrame."""
        if not self._trade_closes:
            return pd.DataFrame()

        rows = []
        open_lookup = {t["ticket"]: t for t in self._trade_opens}

        for close in self._trade_closes:
            ticket = close["ticket"]
            open_data = open_lookup.get(ticket, {})
            rows.append({
                "ticket":      ticket,
                "direction":   open_data.get("direction", ""),
                "entry_price": open_data.get("price", 0),
                "exit_price":  close["price"],
                "pnl_usd":     close["pnl_usd"],
                "reason":      close["reason"],
                "open_time":   open_data.get("timestamp"),
                "close_time":  close.get("timestamp"),
                "score":       open_data.get("score", 0),
            })
        return pd.DataFrame(rows)

    @property
    def charts_dir(self) -> Path:
        return self._out_dir
