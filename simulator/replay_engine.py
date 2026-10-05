"""
simulator/replay_engine.py — Main Orchestrator for MARK III Historical Replay

Coordinates all simulator components and exposes the public API.
Event-driven architecture: DataFeed emits → subscribers react.

Usage:
    engine = ReplayEngine(date="2026-05-22", speed="instant", seed=42)
    result = engine.run()

Full parameter reference: see __init__ docstring.

Determinism contract (Req 15):
    Given identical (date, config, seed), every run produces identical
    trade outcomes, signals, and metrics. Verified via RNG state isolation,
    SimulationClock isolation, and immutable source DataFrames.
"""
from __future__ import annotations

import logging
import time as wall_time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pytz

from simulator.broker_sim import FillMode, LatencyModel, SimulatedExecutionBackend
from simulator.checkpoint import Checkpointer, ReplaySnapshot
from simulator.data_feed import DataFeed
from simulator.data_loader import CSVDataLoader, DataFrameLoader, MT5DataLoader, _to_utc_ts
from simulator.event_bus import EventBus
from simulator.indicators import IndicatorEngine
from simulator.m15_builder import M15Builder
from simulator.metrics_engine import MetricsEngine
from simulator.session_controller import SessionController
from simulator.simulation_clock import SimulationClock
from simulator.strategy_adapter import StrategyAdapter
from simulator.trade_logger import SimTradeLogger
from simulator.visualizer import ReplayVisualizer

logger = logging.getLogger("Sim.ReplayEngine")

NY_TZ = pytz.timezone("America/New_York")


class ReplayResult:
    """Encapsulates all outputs from a completed replay session."""

    def __init__(
        self,
        metrics: dict,
        trades: List[dict],
        output_dir: str,
        config: dict,
        candles_processed: int,
        signals_fired: int,
        elapsed_wall_sec: float,
    ) -> None:
        self.metrics         = metrics
        self.trades          = trades
        self.output_dir      = output_dir
        self.config          = config
        self.candles_processed = candles_processed
        self.signals_fired   = signals_fired
        self.elapsed_wall_sec = elapsed_wall_sec

    def print_summary(self) -> None:
        """Print a concise replay summary to console."""
        s = self.metrics.get("summary", {})
        print(f"\n{'═'*55}")
        print(f"  MARK III REPLAY COMPLETE")
        print(f"{'─'*55}")
        print(f"  Date/Event:        {self.config.get('date')} {self.config.get('event','')}")
        print(f"  Candles processed: {self.candles_processed}")
        print(f"  Signals fired:     {self.signals_fired}")
        print(f"  Total trades:      {s.get('total_trades', 0)}")
        print(f"  Win rate:          {s.get('win_rate_pct', 0):.1f}%")
        print(f"  Net P&L:           ${s.get('net_pnl_usd', 0):+.2f}")
        print(f"  Max drawdown:      ${self.metrics.get('risk',{}).get('max_drawdown_usd',0):.2f}")
        print(f"  Wall time:         {self.elapsed_wall_sec:.1f}s")
        print(f"  Results saved:     {self.output_dir}")
        print(f"{'═'*55}\n")


class ReplayEngine:
    """
    Main MARK III historical replay orchestrator.

    Wires all simulator components together and runs the event loop.

    Args:
        date:           Date to replay (YYYY-MM-DD)
        start_ny:       Session start in NY time (HH:MM). Default: "09:30"
        end_ny:         Session end in NY time (HH:MM).   Default: "16:00"
        event:          Predefined event window (CPI/NFP/FOMC/OPEN/CLOSE/etc.)
        data_source:    "mt5" | "csv" | "dataframe"
        csv_path:       Path to CSV file (if data_source="csv")
        dataframe:      Pre-built DataFrame (if data_source="dataframe")
        initial_balance: Starting account balance in USD. Default: 30.0
        spread_model:   "historical" | "dynamic" | "fixed"
        slippage_max:   Max slippage in points. Default: 0.5
        fill_mode:      "pessimistic" | "optimistic" | "nearest" | "random"
        commission:     USD per trade (round-turn). Default: 0.0
        order_latency_ms:     Order transmission latency (ms). Default: 0
        execution_latency_ms: Fill execution latency (ms).   Default: 0
        speed:          1|10|100|"instant"|"step". Default: "instant"
        seed:           Random seed for determinism (Req 3, 15). Default: 42
        output_dir:     Directory for results. Default: auto-generated
        checkpoint_dir: Directory for checkpoints. None = disabled
        checkpoint_every_n: Candles between checkpoints. Default: 100
        on_replay_start / on_new_candle / on_trade_open /
        on_trade_close / on_replay_end: Optional callback hooks (Req 5)
        verbose:        Enable debug logging. Default: False
    """

    def __init__(
        self,
        # Session
        date: str = None,
        start_ny: str = None,
        end_ny: str = None,
        start_colombia: str = None,
        end_colombia: str = None,
        event: str = None,
        # Data source
        data_source: str = "mt5",
        csv_path: str = None,
        dataframe: pd.DataFrame = None,
        mt5_format: bool = False,
        cache_dir: str = "simulator/data",
        # Execution
        initial_balance: float = 30.0,
        spread_model: str = "historical",
        slippage_max: float = 0.5,
        fill_mode: str = "pessimistic",
        commission: float = 0.0,
        # Latency
        order_latency_ms: float = 0.0,
        execution_latency_ms: float = 0.0,
        # Replay behavior
        speed: Union[float, str] = "instant",
        seed: int = 42,
        # Output
        output_dir: str = None,
        generate_charts: bool = True,
        # Checkpointing
        checkpoint_dir: str = None,
        checkpoint_every_n: int = 100,
        # Event hooks (Req 5)
        on_replay_start: Callable = None,
        on_new_candle: Callable   = None,
        on_trade_open: Callable   = None,
        on_trade_close: Callable  = None,
        on_replay_end: Callable   = None,
        # Debug
        verbose: bool = False,
    ) -> None:

        if date is None and dataframe is None and csv_path is None:
            raise ValueError(
                "Must specify one of: date, csv_path, or dataframe"
            )

        # ── Configuration ────────────────────────────────────────────
        self._config = {
            "date":              date,
            "start_ny":          start_ny,
            "end_ny":            end_ny,
            "start_colombia":    start_colombia,
            "end_colombia":      end_colombia,
            "event":             event,
            "data_source":       data_source,
            "mt5_format":        mt5_format,
            "cache_dir":         cache_dir,
            "initial_balance":   initial_balance,
            "spread_model":      spread_model,
            "slippage_max":      slippage_max,
            "fill_mode":         fill_mode,
            "commission":        commission,
            "order_latency_ms":  order_latency_ms,
            "execution_latency_ms": execution_latency_ms,
            "speed":             speed,
            "seed":              seed,
        }

        self._date             = date
        self._start_ny         = start_ny
        self._end_ny           = end_ny
        self._start_colombia   = start_colombia
        self._end_colombia     = end_colombia
        self._event            = event
        self._data_source      = data_source
        self._csv_path         = csv_path
        self._dataframe        = dataframe
        self._mt5_format       = mt5_format
        self._cache_dir        = cache_dir
        self._initial_balance  = initial_balance
        self._spread_model     = spread_model
        self._slippage_max     = slippage_max
        self._fill_mode        = fill_mode
        self._commission       = commission
        self._speed            = speed
        self._seed             = seed
        self._checkpoint_every = checkpoint_every_n
        self._generate_charts  = generate_charts

        # ── Output directory ─────────────────────────────────────────
        event_label = f"_{event}" if event else ""
        if output_dir:
            self._out_dir = Path(output_dir)
        else:
            ts_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._out_dir = (
                Path("simulator/results") /
                f"replay_{date or 'custom'}{event_label}_seed{seed}_{ts_label}"
            )
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # ── Logging ──────────────────────────────────────────────────
        if verbose:
            logging.getLogger("Sim").setLevel(logging.DEBUG)

        # ── Control flags ────────────────────────────────────────────
        self._paused     = False
        self._stop_flag  = False
        self._started    = False

        # ── Latency model ────────────────────────────────────────────
        self._latency = LatencyModel(
            order_transmission_ms=order_latency_ms,
            execution_ms=execution_latency_ms,
        )

        # ── Checkpointing ────────────────────────────────────────────
        self._checkpointer = (
            Checkpointer(checkpoint_dir or str(self._out_dir / "checkpoints"))
            if checkpoint_dir is not None or checkpoint_every_n > 0
            else None
        )

        # ── User hooks (Req 5) ───────────────────────────────────────
        self._user_hooks = {
            EventBus.REPLAY_START: on_replay_start,
            EventBus.NEW_CANDLE:   on_new_candle,
            EventBus.TRADE_OPEN:   on_trade_open,
            EventBus.TRADE_CLOSE:  on_trade_close,
            EventBus.REPLAY_END:   on_replay_end,
        }

        # Components initialized lazily in run()
        self._bus:     Optional[EventBus]           = None
        self._broker:  Optional[SimulatedExecutionBackend] = None
        self._adapter: Optional[StrategyAdapter]    = None
        self._feed:    Optional[DataFeed]           = None
        self._tlogger: Optional[SimTradeLogger]     = None
        self._metrics: Optional[MetricsEngine]      = None
        self._viz:     Optional[ReplayVisualizer]   = None

    # ── Main interface ───────────────────────────────────────────────

    def run(self) -> ReplayResult:
        """
        Execute the full replay session.

        Returns:
            ReplayResult with metrics, trades, and output paths.
        """
        wall_start = wall_time.time()
        logger.info(
            f"\n{'═'*60}\n"
            f"  MARK III REPLAY ENGINE\n"
            f"  Date={self._date} │ Event={self._event} │ "
            f"Speed={self._speed} │ Seed={self._seed}\n"
            f"{'═'*60}"
        )

        # ── 1. Build session window ───────────────────────────────────
        session_ctrl = SessionController(
            date=self._date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            start_ny=self._start_ny,
            end_ny=self._end_ny,
            start_colombia=self._start_colombia,
            end_colombia=self._end_colombia,
            event=self._event,
        )
        warmup_start, session_end = session_ctrl.get_data_window()

        # ── 2. Load historical data ──────────────────────────────────
        source_df = self._load_data(warmup_start, session_end)
        if source_df.empty:
            raise ValueError(
                f"No historical data available for {self._date}. "
                "Check your data source or date range."
            )

        # Filter to session window only (warmup period included)
        warmup_start_ts = _to_utc_ts(warmup_start)
        session_end_ts  = _to_utc_ts(session_end)
        source_df = source_df[
            (source_df["time"] >= warmup_start_ts) &
            (source_df["time"] <= session_end_ts)
        ].reset_index(drop=True)

        warmup_bars = session_ctrl.get_warmup_bar_count(source_df)
        if warmup_bars < 1:
            raise ValueError(
                "No warmup candles found before session start. "
                "Extend data_source or session start time."
            )

        logger.info(
            f"Source data: {len(source_df)} M5 bars │ "
            f"Warmup: {warmup_bars} │ "
            f"Replay: {len(source_df) - warmup_bars}"
        )

        # ── 3. Initialize event bus ──────────────────────────────────
        self._bus = EventBus()

        # Register user hooks
        for event_name, hook in self._user_hooks.items():
            if hook is not None:
                self._bus.subscribe(event_name, hook)

        # ── 4. Initialize core components ────────────────────────────
        self._broker = SimulatedExecutionBackend(
            initial_balance=self._initial_balance,
            spread_model=self._spread_model,
            slippage_max_pts=self._slippage_max,
            fill_mode=self._fill_mode,
            commission_per_trade=self._commission,
            latency=self._latency,
            seed=self._seed,
        )

        clock     = SimulationClock()
        ind_eng   = IndicatorEngine()
        m15_build = M15Builder()

        self._feed = DataFeed(
            source_m5=source_df,
            indicator_engine=ind_eng,
            m15_builder=m15_build,
            sim_clock=clock,
            event_bus=self._bus,
            warmup_bars=warmup_bars,
        )

        self._adapter = StrategyAdapter(
            execution_backend=self._broker,
            sim_clock=clock,
            event_bus=self._bus,
        )

        # ── 5. Initialize logging + visualization ────────────────────
        self._tlogger = SimTradeLogger(
            output_dir=str(self._out_dir),
            fill_mode=self._fill_mode,
            commission=self._commission,
            session_config=self._config,
        )
        self._tlogger.subscribe(self._bus)

        if self._generate_charts:
            self._viz = ReplayVisualizer(
                output_dir=str(self._out_dir),
                session_config=self._config,
            )
            self._viz.subscribe(self._bus, initial_balance=self._initial_balance)

        # ── 6. Emit replay_start ─────────────────────────────────────
        self._bus.emit(
            EventBus.REPLAY_START,
            date=self._date,
            session_start_ny=self._start_ny or "04:00 COT",
            session_end_ny=self._end_ny or "17:00 COT",
            macro_event=self._event,
            config=self._config,
        )

        # ── 7. Main replay loop (event-driven) ───────────────────────
        # DataFeed.advance() emits "new_candle" → all subscribers fire
        candle_num = 0
        self._started = True

        while not self._stop_flag:
            # Handle pause/step mode
            if self._paused:
                if str(self._speed) == "step":
                    inp = input(
                        f"  [STEP {candle_num}] Press ENTER for next candle, "
                        f"'q' to finish: "
                    )
                    if inp.strip().lower() == "q":
                        break
                else:
                    wall_time.sleep(0.1)
                    continue

            if not self._feed.advance():
                break  # Exhausted

            candle_num += 1

            # Optional checkpoint
            if (
                self._checkpointer is not None
                and self._checkpoint_every > 0
                and candle_num % self._checkpoint_every == 0
            ):
                self._save_checkpoint(clock, candle_num)

            # Speed control
            self._apply_speed_delay()

        # ── 8. Close any open positions at session end ───────────────
        open_positions = self._broker.get_open_positions()
        if open_positions:
            logger.info(
                f"Closing {len(open_positions)} positions at session end"
            )
            for pos in open_positions:
                if self._broker.close_position(pos.ticket, "Session_End"):
                    closed = self._broker.closed_trades[-1]
                    try:
                        self._adapter._health_mon.clear_trade(pos.ticket)
                    except Exception:
                        pass

                    close_time = closed.get("close_time")
                    open_time = closed.get("open_time")
                    duration_sec = (
                        float(close_time) - float(open_time)
                        if close_time is not None and open_time is not None
                        else None
                    )
                    timestamp = (
                        pd.Timestamp(float(close_time), unit="s", tz="UTC")
                        if close_time is not None
                        else None
                    )
                    self._bus.emit(
                        EventBus.TRADE_CLOSE,
                        ticket=closed["ticket"],
                        exit_price=closed["exit_price"],
                        pnl_usd=closed["pnl_usd"],
                        reason=closed["close_reason"],
                        duration_sec=duration_sec,
                        timestamp=timestamp,
                    )

        # ── 9. Post-session analytics ────────────────────────────────
        self._tlogger.finalize()
        closed_trades = self._tlogger.closed_trades

        me = MetricsEngine(closed_trades, initial_balance=self._initial_balance)
        report = me.generate()
        me.print_report(report)
        me.save_json(report, str(self._out_dir / "metrics.json"))

        # ── 10. Emit replay_end (triggers charts) ─────────────────────
        self._bus.emit(
            EventBus.REPLAY_END,
            total_trades=len(closed_trades),
            final_equity=self._broker.equity,
            metrics_summary=report,
        )

        elapsed = wall_time.time() - wall_start

        result = ReplayResult(
            metrics=report,
            trades=closed_trades,
            output_dir=str(self._out_dir),
            config=self._config,
            candles_processed=candle_num,
            signals_fired=self._adapter.signals_fired,
            elapsed_wall_sec=elapsed,
        )
        result.print_summary()
        return result

    # ── Control methods ──────────────────────────────────────────────

    def pause(self) -> None:
        """Pause replay after current candle completes."""
        self._paused = True
        logger.info("[ReplayEngine] Paused")

    def resume(self) -> None:
        """Resume a paused replay."""
        self._paused = False
        logger.info("[ReplayEngine] Resumed")

    def stop(self) -> None:
        """Stop replay and trigger post-session analytics."""
        self._stop_flag = True
        logger.info("[ReplayEngine] Stop requested")

    # ── Private helpers ──────────────────────────────────────────────

    def _load_data(
        self, warmup_start: datetime, session_end: datetime
    ) -> pd.DataFrame:
        """Load historical M5 data from the appropriate source."""
        if self._data_source == "csv" and self._csv_path:
            loader = CSVDataLoader(self._csv_path, mt5_format=self._mt5_format)
            return loader.load("ES", warmup_start, session_end, "M5")

        elif self._data_source == "dataframe" and self._dataframe is not None:
            loader = DataFrameLoader(self._dataframe)
            return loader.load("ES", warmup_start, session_end, "M5")

        else:  # "mt5" (default)
            loader = MT5DataLoader(
                cache_dir=self._cache_dir
            )
            return loader.load("ES", warmup_start, session_end, "M5")

    def _apply_speed_delay(self) -> None:
        """
        Apply a wall-clock delay between candles based on speed setting.
        Speed = 1    → 300s sleep (real time M5)
        Speed = N    → 300/N seconds
        Speed = "instant" → no sleep
        Speed = "step"    → handled in main loop (user input)
        """
        if self._speed == "instant" or self._speed == "step":
            return
        try:
            speed_factor = float(self._speed)
            if speed_factor > 0:
                sleep_sec = 300.0 / speed_factor  # 300s = 1 M5 candle
                wall_time.sleep(sleep_sec)
        except (ValueError, TypeError):
            pass

    def _save_checkpoint(self, clock: SimulationClock, candle_num: int) -> None:
        """Save a replay checkpoint."""
        if not self._checkpointer:
            return
        try:
            from simulator.checkpoint import Checkpointer
            snap = ReplaySnapshot(
                replay_index=self._feed.candle_index,
                sim_timestamp=clock.simulated_timestamp,
                sim_timestamp_str=clock.now_str(),
                balance=self._broker.balance,
                equity=self._broker.equity,
                initial_balance=self._initial_balance,
                open_positions=[
                    {
                        "ticket":    p.ticket,
                        "symbol":    p.symbol,
                        "direction": p.direction,
                        "lot_size":  p.lot_size,
                        "entry_price": p.entry_price,
                        "sl_price":  p.sl_price,
                        "tp_price":  p.tp_price,
                        "open_time": p.open_time,
                    }
                    for p in self._broker.get_open_positions()
                ],
                closed_trades=self._broker.closed_trades,
                rng_state_b64=Checkpointer.encode_rng_state(self._broker._rng),
                config=self._config,
                candle_count=candle_num,
            )
            self._checkpointer.save(snap, label=f"candle_{candle_num:06d}")
        except Exception as exc:
            logger.warning(f"[ReplayEngine] Checkpoint failed: {exc}")
