"""
simulator/replay_cli.py — Command-Line Interface for MARK III Replay

Usage examples:

  # Replay a full day from MT5 (uses cached data if available)
  python -m simulator.replay_cli --date 2026-05-22 --speed instant --seed 42

  # Replay the CPI window
  python -m simulator.replay_cli --date 2026-05-22 --event CPI

  # Replay from a CSV file
  python -m simulator.replay_cli --csv path/to/us500_m5.csv --date 2026-05-22

  # Custom time window
  python -m simulator.replay_cli --date 2026-05-22 --start 09:30 --end 14:00

  # Step-through mode (pause after each candle)
  python -m simulator.replay_cli --date 2026-05-22 --speed step

  # Pessimistic fills, dynamic spread, latency
  python -m simulator.replay_cli --date 2026-05-22 \\
      --fill-mode pessimistic --spread-model dynamic \\
      --slippage 1.0 --order-latency 50

  # Determinism test: run twice with same seed, compare JSON output
  python -m simulator.replay_cli --date 2026-05-22 --seed 42 --output run_a
  python -m simulator.replay_cli --date 2026-05-22 --seed 42 --output run_b
  # run_a/trades.json and run_b/trades.json will be identical

  # Resume from checkpoint
  python -m simulator.replay_cli --checkpoint simulator/results/.../checkpoint_000100.json

  # Verbose mode (full debug logging)
  python -m simulator.replay_cli --date 2026-05-22 --verbose
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simulator.replay_cli",
        description="MARK III Historical Replay Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Session ───────────────────────────────────────────────────────
    session = p.add_argument_group("Session")
    session.add_argument(
        "--date", "-d",
        metavar="YYYY-MM-DD",
        help="Date to replay (required unless --csv with --date or --checkpoint)",
    )
    session.add_argument(
        "--event", "-e",
        metavar="EVENT",
        choices=["CPI", "NFP", "FOMC", "FOMC_MINS", "OPEN", "CLOSE",
                 "FULL_DAY", "FULL", "PPI", "RETAIL", "ISM", "JOBS", "PCE", "GDP"],
        help="Predefined macro-event window (overrides --start/--end)",
    )
    session.add_argument(
        "--start", "-S",
        metavar="HH:MM",
        default=None,
        help="Session start time (NY). Default: 09:30 unless Colombia local time is used.",
    )
    session.add_argument(
        "--end", "-E",
        metavar="HH:MM",
        default=None,
        help="Session end time (NY). Default: 16:00 unless Colombia local time is used.",
    )
    session.add_argument(
        "--start-colombia",
        metavar="HH:MM",
        default=None,
        help="Session start time in Colombia local time. Default: 09:00 if no NY start is set.",
    )
    session.add_argument(
        "--end-colombia",
        metavar="HH:MM",
        default=None,
        help="Session end time in Colombia local time. Default: 16:00 if no NY end is set.",
    )

    # ── Data source ───────────────────────────────────────────────────
    data = p.add_argument_group("Data Source")
    data.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to M5 CSV file (alternative to MT5)",
    )
    data.add_argument(
        "--mt5-format",
        action="store_true",
        help="Use MT5-exported tab-separated CSV format",
    )
    data.add_argument(
        "--cache-dir",
        metavar="DIR",
        default="simulator/data",
        help="MT5 data cache directory (default: simulator/data)",
    )

    # ── Execution ─────────────────────────────────────────────────────
    execution = p.add_argument_group("Execution")
    execution.add_argument(
        "--balance", "-b",
        type=float,
        default=30.0,
        metavar="USD",
        help="Initial account balance (default: 30.0)",
    )
    execution.add_argument(
        "--spread-model",
        choices=["historical", "dynamic", "fixed"],
        default="historical",
        help="Spread model (default: historical)",
    )
    execution.add_argument(
        "--slippage",
        type=float,
        default=0.5,
        metavar="POINTS",
        help="Max slippage in points (default: 0.5)",
    )
    execution.add_argument(
        "--fill-mode",
        choices=["pessimistic", "optimistic", "nearest", "random"],
        default="pessimistic",
        help="OHLC ambiguity fill mode (default: pessimistic)",
    )
    execution.add_argument(
        "--commission",
        type=float,
        default=0.0,
        metavar="USD",
        help="Commission per trade in USD (default: 0.0)",
    )
    execution.add_argument(
        "--order-latency",
        type=float,
        default=0.0,
        metavar="MS",
        help="Order transmission latency in ms (default: 0)",
    )
    execution.add_argument(
        "--exec-latency",
        type=float,
        default=0.0,
        metavar="MS",
        help="Execution fill latency in ms (default: 0)",
    )

    # ── Replay behavior ──────────────────────────────────────────────
    behavior = p.add_argument_group("Replay Behavior")
    behavior.add_argument(
        "--speed", "-s",
        default="instant",
        metavar="SPEED",
        help=(
            "Replay speed: 1=realtime, 10=10x, 100=100x, "
            "'instant'=max speed, 'step'=manual. (default: instant)"
        ),
    )
    behavior.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # ── Checkpointing ─────────────────────────────────────────────────
    cp = p.add_argument_group("Checkpointing")
    cp.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Resume from a checkpoint JSON file",
    )
    cp.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help="Save checkpoint every N candles (0=disabled, default: 0)",
    )
    cp.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        default=None,
        help="Directory for checkpoint files",
    )

    # ── Output ────────────────────────────────────────────────────────
    output = p.add_argument_group("Output")
    output.add_argument(
        "--output", "-o",
        metavar="DIR",
        help="Output directory for results (default: auto-named)",
    )
    output.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return p


def main(argv=None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Logging setup ─────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate args ─────────────────────────────────────────────────
    if args.date is None and args.checkpoint is None and args.csv is None:
        parser.error("Must specify --date, --csv, or --checkpoint")

    # Handle speed argument
    speed: object = args.speed
    if speed not in ("instant", "step"):
        try:
            speed = float(speed)
        except ValueError:
            parser.error(f"Invalid --speed value: {args.speed}")

    # ── Build ReplayEngine ────────────────────────────────────────────
    from simulator.replay_engine import ReplayEngine

    engine = ReplayEngine(
        # Session
        date=args.date,
        start_ny=args.start,
        end_ny=args.end,
        start_colombia=args.start_colombia,
        end_colombia=args.end_colombia,
        event=args.event,
        # Data source
        data_source="csv" if args.csv else "mt5",
        csv_path=args.csv,
        mt5_format=args.mt5_format,
        cache_dir=args.cache_dir,
        # Execution
        initial_balance=args.balance,
        spread_model=args.spread_model,
        slippage_max=args.slippage,
        fill_mode=args.fill_mode,
        commission=args.commission,
        order_latency_ms=args.order_latency,
        execution_latency_ms=args.exec_latency,
        # Behavior
        speed=speed,
        seed=args.seed,
        # Output
        output_dir=args.output,
        generate_charts=not args.no_charts,
        # Checkpointing
        checkpoint_every_n=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        # Debug
        verbose=args.verbose,
    )

    try:
        result = engine.run()
        return 0
    except KeyboardInterrupt:
        print("\n[!] Replay interrupted by user")
        return 1
    except Exception as exc:
        logging.error(f"Replay failed: {exc}", exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    sys.exit(main())
