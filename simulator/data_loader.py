"""
simulator/data_loader.py — Historical Data Loaders for MARK III Replay

Provides multiple data sources through a unified abstract interface.
All loaders produce an immutable DataFrame with the normalized schema
used by MT5DataProvider, enabling the StrategyAdapter to pass data
directly to the unmodified SignalScorer.

Normalized output schema:
    time         datetime64[ns, UTC]  — candle open time (UTC)
    open         float64
    high         float64
    low          float64
    close        float64
    tick_volume  int64
    spread       float64              — points (0 if not available)

The IndicatorEngine adds computed columns (ema_21, ema_50, atr, etc.)
on top of this schema — loaders do NOT pre-compute indicators.

Immutability guarantee (Req 7):
    All loaders return DataFrames with a reset integer index.
    DataFeed accesses via iloc slicing — no in-place mutation.
    The returned DataFrame should be treated as read-only by callers.

Memory safety (Req 8):
    DataFrames are read once at load time and held in memory.
    For large datasets (months of M1), a chunked/lazy loader can be
    added as a future subclass without changing the interface.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("Sim.DataLoader")

# Standard timezone abbreviation for display
UTC = timezone.utc


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    """
    Safely convert a datetime (tz-aware or naive) to a UTC pandas Timestamp.
    Avoids ValueError when dt already has tzinfo set.
    """
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ═══════════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════

class DataLoaderBase(ABC):
    """
    Abstract interface for all historical data sources.

    Subclasses implement load() and return a normalized OHLCV DataFrame.
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close"}

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str = "M5",
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol within [start_utc, end_utc].

        Args:
            symbol:    Symbol key (e.g., "ES", "US500Cash")
            start_utc: Inclusive start time (UTC-aware datetime)
            end_utc:   Inclusive end time   (UTC-aware datetime)
            timeframe: "M5" or "M15" (default: M5)

        Returns:
            Normalized DataFrame (see module docstring for schema).
            Index is 0-based integer. Sorted ascending by time.
        """

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and normalize a loaded DataFrame to the required schema.
        Called by all concrete loaders before returning.
        """
        if df is None or df.empty:
            logger.warning(f"[DataLoader] Empty DataFrame for {symbol}")
            return pd.DataFrame()

        # Validate required columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame for {symbol} missing columns: {missing}")

        # Ensure 'time' column is a UTC-aware datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        else:
            raise ValueError("DataFrame must have a 'time' column")

        # Ensure numeric OHLCV columns
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Optional columns with defaults
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0
        df["tick_volume"] = df["tick_volume"].fillna(0).astype(int)

        if "spread" not in df.columns:
            df["spread"] = 0.0
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)

        # Sort ascending by time, reset integer index
        df = df.sort_values("time").reset_index(drop=True)

        # Drop any rows with NaN OHLC (corrupt data)
        n_before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(
                f"[DataLoader] Dropped {n_dropped} rows with NaN OHLC for {symbol}"
            )

        logger.info(
            f"[DataLoader] {symbol}: {len(df)} candles loaded "
            f"| {df['time'].iloc[0]} → {df['time'].iloc[-1]}"
        )
        return df


# ═══════════════════════════════════════════════════════════════════════
# CSV LOADER
# ═══════════════════════════════════════════════════════════════════════

class CSVDataLoader(DataLoaderBase):
    """
    Loads historical OHLCV data from a CSV file.

    Expected columns (flexible — extra columns are kept):
        time, open, high, low, close, tick_volume (optional), spread (optional)

    Time column formats accepted:
        - ISO 8601: "2026-05-22T08:00:00+00:00"
        - Unix timestamp (int or float seconds)
        - Human-readable: "2026-05-22 08:00:00"
        - pandas will infer format automatically

    For MT5-exported CSVs:
        MT5 exports use '<DATE>\t<TIME>' or a unified datetime column.
        Pass mt5_format=True to handle the tab-separated date+time.
    """

    def __init__(self, csv_path: str, mt5_format: bool = False) -> None:
        """
        Args:
            csv_path:   Path to the CSV file
            mt5_format: If True, handle MT5's date+time column format
        """
        self._path = Path(csv_path)
        self._mt5_format = mt5_format

        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

    def load(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str = "M5",
    ) -> pd.DataFrame:
        logger.info(f"[CSVLoader] Loading {symbol} from {self._path}")

        if self._mt5_format:
            df = self._load_mt5_csv()
        else:
            df = pd.read_csv(self._path)
            # Detect the time column
            time_col = next(
                (c for c in df.columns if c.lower() in ("time", "datetime", "date")),
                None,
            )
            if time_col and time_col != "time":
                df = df.rename(columns={time_col: "time"})

        df = self._normalize(df, symbol)

        # Filter to requested window
        df = df[
            (df["time"] >= _to_utc_ts(start_utc))
            & (df["time"] <= _to_utc_ts(end_utc))
        ].reset_index(drop=True)

        return df

    def _load_mt5_csv(self) -> pd.DataFrame:
        """Handles MT5's tab-separated '<DATE>\t<TIME>\t<OPEN>...' format."""
        df = pd.read_csv(self._path, sep="\t")
        # MT5 header: <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <SPREAD>
        col_map = {
            "<DATE>":    "date_str",
            "<TIME>":    "time_str",
            "<OPEN>":    "open",
            "<HIGH>":    "high",
            "<LOW>":     "low",
            "<CLOSE>":   "close",
            "<TICKVOL>": "tick_volume",
            "<SPREAD>":  "spread",
        }
        # Strip whitespace from headers
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={k.strip(): v for k, v in col_map.items() if k.strip() in df.columns})

        if "date_str" in df.columns and "time_str" in df.columns:
            df["time"] = pd.to_datetime(
                df["date_str"].astype(str) + " " + df["time_str"].astype(str),
                utc=True,
            )
            df = df.drop(columns=["date_str", "time_str"], errors="ignore")

        return df


# ═══════════════════════════════════════════════════════════════════════
# MT5 LOADER
# ═══════════════════════════════════════════════════════════════════════

class MT5DataLoader(DataLoaderBase):
    """
    Downloads historical data directly from MetaTrader 5 via Python API.

    Requires MT5 to be running and connected.
    Uses mt5.copy_rates_range() for precise date-range queries.

    Data is cached to a local CSV after first download to avoid
    repeated API calls for the same date range.
    """

    TF_MAP = {
        "M1":  1,
        "M5":  5,
        "M15": 15,
        "H1":  60,
    }

    def __init__(self, cache_dir: str = "simulator/data") -> None:
        """
        Args:
            cache_dir: Directory to cache downloaded data as CSV.
                       None = no caching (re-download every time).
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str = "M5",
    ) -> pd.DataFrame:
        # Check cache first
        cache_path = self._cache_path(symbol, start_utc, end_utc, timeframe)
        if cache_path and cache_path.exists():
            logger.info(f"[MT5Loader] Cache hit: {cache_path}")
            loader = CSVDataLoader(str(cache_path))
            return loader.load(symbol, start_utc, end_utc, timeframe)

        # Download from MT5
        df = self._download(symbol, start_utc, end_utc, timeframe)
        df = self._normalize(df, symbol)

        # Save to cache
        if cache_path and not df.empty:
            df.to_csv(cache_path, index=False)
            logger.info(f"[MT5Loader] Cached {len(df)} candles → {cache_path}")

        return df

    def _download(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise RuntimeError(
                "MetaTrader5 Python package not installed. "
                "Install it or use CSVDataLoader instead."
            )

        # Resolve MT5 symbol name
        from config import settings
        sym_cfg = settings.SYMBOL_MAP.get(symbol, {})
        mt5_name = sym_cfg.get("mt5_name", symbol)

        tf_minutes = self.TF_MAP.get(timeframe, 5)
        # Map to MT5 timeframe constant
        tf_const_map = {
            1:  getattr(mt5, "TIMEFRAME_M1",  1),
            5:  getattr(mt5, "TIMEFRAME_M5",  5),
            15: getattr(mt5, "TIMEFRAME_M15", 15),
            60: getattr(mt5, "TIMEFRAME_H1",  60),
        }
        tf_const = tf_const_map.get(tf_minutes, mt5.TIMEFRAME_M5)

        logger.info(
            f"[MT5Loader] Downloading {mt5_name} {timeframe} "
            f"{start_utc.date()} → {end_utc.date()}"
        )

        if not mt5.initialize():
            logger.error("[MT5Loader] Failed to initialize MetaTrader 5")
            return pd.DataFrame()

        rates = mt5.copy_rates_range(mt5_name, tf_const, start_utc, end_utc)

        if rates is None or len(rates) == 0:
            logger.warning(
                f"[MT5Loader] No data returned for {mt5_name}. "
                f"Check MT5 terminal connection and symbol availability."
            )
            val = mt5.last_error()
            logger.debug(f"[MT5Loader] MT5 error code: {val}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        # MT5 broker server runs on UTC+3 (EET/EEST). The 'time' field
        # from copy_rates_range() is a Unix timestamp in BROKER local
        # time, NOT real UTC. We must subtract the broker offset (3h)
        # to get true UTC timestamps.
        BROKER_OFFSET_SEC = 3 * 3600  # UTC+3
        df["time"] = pd.to_datetime(df["time"] - BROKER_OFFSET_SEC, unit="s", utc=True)
        return df

    def _cache_path(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str,
    ) -> Optional[Path]:
        if not self._cache_dir:
            return None
        fname = (
            f"{symbol}_{timeframe}_"
            f"{start_utc.strftime('%Y%m%d')}_"
            f"{end_utc.strftime('%Y%m%d')}.csv"
        )
        return self._cache_dir / fname


# ═══════════════════════════════════════════════════════════════════════
# DATAFRAME LOADER
# ═══════════════════════════════════════════════════════════════════════

class DataFrameLoader(DataLoaderBase):
    """
    Wraps a pre-built pandas DataFrame.

    Useful for:
    - Unit tests with synthetic data
    - Loading data from external Python pipelines
    - Passing data already in memory without file I/O
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        self._df = dataframe.copy()  # Take ownership; immutable after this

    def load(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        timeframe: str = "M5",
    ) -> pd.DataFrame:
        df = self._normalize(self._df.copy(), symbol)

        df = df[
            (df["time"] >= _to_utc_ts(start_utc))
            & (df["time"] <= _to_utc_ts(end_utc))
        ].reset_index(drop=True)

        return df
