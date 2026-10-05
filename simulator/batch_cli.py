"""
Batch replay runner for MARK III.

Runs the replay engine across many dates and writes aggregate metrics.
This is intended for simulation validation, not live trading.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def _date_range(start: str, end: str, include_weekends: bool) -> list[str]:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt < start_dt:
        raise argparse.ArgumentTypeError("--to-date must be >= --from-date")

    dates: list[str] = []
    current = start_dt
    while current <= end_dt:
        if include_weekends or current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulator.batch_cli",
        description="Run MARK III replay across multiple dates.",
    )

    dates = parser.add_argument_group("Dates")
    dates.add_argument("--dates", nargs="*", help="Explicit dates: YYYY-MM-DD ...")
    dates.add_argument("--from-date", dest="from_date", help="First date in range")
    dates.add_argument("--to-date", dest="to_date", help="Last date in range")
    dates.add_argument(
        "--include-weekends",
        action="store_true",
        help="Include Saturday/Sunday when using --from-date/--to-date",
    )

    session = parser.add_argument_group("Session")
    session.add_argument("--event", choices=[
        "CPI", "NFP", "FOMC", "FOMC_MINS", "OPEN", "CLOSE",
        "FULL_DAY", "FULL", "PPI", "RETAIL", "ISM", "JOBS", "PCE", "GDP",
    ])
    session.add_argument("--start", dest="start_ny", help="Start time in NY, HH:MM")
    session.add_argument("--end", dest="end_ny", help="End time in NY, HH:MM")
    session.add_argument("--start-colombia", help="Start time in Colombia, HH:MM")
    session.add_argument("--end-colombia", help="End time in Colombia, HH:MM")

    data = parser.add_argument_group("Data")
    data.add_argument("--cache-dir", default="simulator/data")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--balance", type=float, default=30.0)
    execution.add_argument(
        "--spread-model",
        choices=["historical", "dynamic", "fixed"],
        default="historical",
    )
    execution.add_argument("--slippage", type=float, default=0.5)
    execution.add_argument(
        "--fill-mode",
        choices=["pessimistic", "optimistic", "nearest", "random"],
        default="pessimistic",
    )
    execution.add_argument("--commission", type=float, default=0.0)
    execution.add_argument("--order-latency", type=float, default=0.0)
    execution.add_argument("--exec-latency", type=float, default=0.0)
    execution.add_argument("--seed", type=int, default=42)

    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output",
        default=None,
        help="Batch output directory. Default: simulator/results/batch_<utc_ts>",
    )
    output.add_argument(
        "--charts",
        action="store_true",
        help="Generate charts for each day. Default is metrics only.",
    )
    output.add_argument("--verbose", "-v", action="store_true")

    return parser


def _collect_dates(args: argparse.Namespace) -> list[str]:
    dates = list(args.dates or [])
    if args.from_date or args.to_date:
        if not args.from_date or not args.to_date:
            raise ValueError("Use --from-date and --to-date together.")
        dates.extend(_date_range(args.from_date, args.to_date, args.include_weekends))

    unique_dates = sorted(set(dates))
    if not unique_dates:
        raise ValueError("Specify --dates or --from-date/--to-date.")

    for date in unique_dates:
        _parse_date(date)
    return unique_dates


def _empty_day_row(date: str, status: str, error: str = "") -> dict[str, Any]:
    return {
        "date": date,
        "status": status,
        "error": error,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "net_pnl_usd": 0.0,
        "profit_factor": 0.0,
        "expectancy_usd": 0.0,
        "max_drawdown_usd": 0.0,
        "return_pct": 0.0,
        "output_dir": "",
    }


def _day_row(date: str, result) -> dict[str, Any]:
    summary = result.metrics.get("summary", {})
    risk = result.metrics.get("risk", {})
    return {
        "date": date,
        "status": "ok",
        "error": "",
        "trades": int(summary.get("total_trades", 0) or 0),
        "wins": int(summary.get("winners", 0) or 0),
        "losses": int(summary.get("losers", 0) or 0),
        "win_rate_pct": float(summary.get("win_rate_pct", 0.0) or 0.0),
        "net_pnl_usd": float(summary.get("net_pnl_usd", 0.0) or 0.0),
        "profit_factor": summary.get("profit_factor", 0.0),
        "expectancy_usd": float(summary.get("expectancy_usd", 0.0) or 0.0),
        "max_drawdown_usd": float(risk.get("max_drawdown_usd", 0.0) or 0.0),
        "return_pct": float(summary.get("return_pct", 0.0) or 0.0),
        "output_dir": result.output_dir,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    trade_rows = [row for row in ok_rows if row["trades"] > 0]
    total_trades = sum(int(row["trades"]) for row in ok_rows)
    total_wins = sum(int(row["wins"]) for row in ok_rows)
    total_losses = sum(int(row["losses"]) for row in ok_rows)
    total_pnl = sum(float(row["net_pnl_usd"]) for row in ok_rows)

    best = max(ok_rows, key=lambda row: row["net_pnl_usd"], default=None)
    worst = min(ok_rows, key=lambda row: row["net_pnl_usd"], default=None)

    return {
        "dates_requested": len(rows),
        "dates_completed": len(ok_rows),
        "dates_failed": len(rows) - len(ok_rows),
        "trade_days": len(trade_rows),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate_pct": round((total_wins / total_trades) * 100, 2)
        if total_trades else 0.0,
        "net_pnl_usd": round(total_pnl, 4),
        "expectancy_usd": round(total_pnl / total_trades, 4)
        if total_trades else 0.0,
        "avg_daily_pnl_usd": round(total_pnl / len(ok_rows), 4)
        if ok_rows else 0.0,
        "best_day": best["date"] if best else None,
        "best_day_pnl_usd": round(best["net_pnl_usd"], 4) if best else 0.0,
        "worst_day": worst["date"] if worst else None,
        "worst_day_pnl_usd": round(worst["net_pnl_usd"], 4) if worst else 0.0,
        "max_daily_drawdown_usd": round(
            max((row["max_drawdown_usd"] for row in ok_rows), default=0.0),
            4,
        ),
    }


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "summary": summary,
                "days": rows,
            },
            f,
            indent=2,
            default=str,
        )

    if rows:
        with open(out_dir / "batch_days.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\nBATCH REPLAY COMPLETE")
    print("-" * 48)
    print(f"Dates completed: {summary['dates_completed']}/{summary['dates_requested']}")
    print(f"Trade days:      {summary['trade_days']}")
    print(f"Total trades:    {summary['total_trades']}")
    print(f"Win rate:        {summary['win_rate_pct']:.2f}%")
    print(f"Net P&L:         ${summary['net_pnl_usd']:+.4f}")
    print(f"Expectancy:      ${summary['expectancy_usd']:+.4f}")
    print(f"Best day:        {summary['best_day']} (${summary['best_day_pnl_usd']:+.4f})")
    print(f"Worst day:       {summary['worst_day']} (${summary['worst_day_pnl_usd']:+.4f})")
    print(f"Saved:           {out_dir}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        dates = _collect_dates(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.output:
        out_dir = Path(args.output)
    else:
        label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("simulator/results") / f"batch_{label}"

    from simulator.replay_engine import ReplayEngine

    rows: list[dict[str, Any]] = []
    for date in dates:
        print(f"\n[{date}] Running replay...")
        day_out = out_dir / date
        try:
            engine = ReplayEngine(
                date=date,
                start_ny=args.start_ny,
                end_ny=args.end_ny,
                start_colombia=args.start_colombia,
                end_colombia=args.end_colombia,
                event=args.event,
                data_source="mt5",
                cache_dir=args.cache_dir,
                initial_balance=args.balance,
                spread_model=args.spread_model,
                slippage_max=args.slippage,
                fill_mode=args.fill_mode,
                commission=args.commission,
                order_latency_ms=args.order_latency,
                execution_latency_ms=args.exec_latency,
                speed="instant",
                seed=args.seed,
                output_dir=str(day_out),
                generate_charts=args.charts,
                checkpoint_every_n=0,
                verbose=args.verbose,
            )
            rows.append(_day_row(date, engine.run()))
        except Exception as exc:
            logging.error("[%s] Replay failed: %s", date, exc)
            rows.append(_empty_day_row(date, "failed", str(exc)))

    summary = _aggregate(rows)
    config = vars(args).copy()
    config["dates"] = dates
    _write_outputs(out_dir, rows, summary, config)
    _print_summary(summary, out_dir)

    return 0 if summary["dates_completed"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
