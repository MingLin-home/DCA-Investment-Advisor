'''
Usage:
python impute_data.py \
    --config ./config.yaml

"--config" is optional. By default, it loads ./config.yaml

It reads all stock symbols in "stock_symbols" (with optional legacy tickers in
"etf_symbols", when present) from ./config.yaml.

For each symbol:
- it reads "avg_price", "date" from <output_dir>/raw_data/<symbol>_price.csv
- if EPS data exists, it reads "date" , "EPS" from <output_dir>/raw_data/<symbol>_eps.csv
- it then combines "date" , "avg_price", "EPS" into a new dataframe and writes to <output_dir>/data/<symbol>.csv
- it adds a column "timestamp" which is the unix timestamp in seconds of the "date"

The "date" should be sorted and continuous. That is, the first row of date must be "stock_start_date", the last row of date must be stock_end_date. For any two row i and (i+1), their timestamp must increase by exactly 24 hours * 3600 seconds / hour.
Insert a new row if some date is missing.

For "avg_price" and "EPS" column, if value is missing, use the nearest non-missing value to fill the missing value.
'''
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml

from config_dates import parse_config_date


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Impute stock price and EPS data")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational log messages",
    )
    return parser.parse_args()


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return cfg


def ensure_required_keys(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Config file is missing required keys: {', '.join(missing)}")


def parse_date(value: Any, field: str) -> datetime:
    return parse_config_date(value, field=field)


def normalize_symbols(raw_symbols: Any, field_name: str, required: bool = True) -> list[str]:
    if raw_symbols is None:
        if not required:
            return []
        raise ValueError(f"'{field_name}' must provide at least one ticker")

    if not isinstance(raw_symbols, Iterable) or isinstance(raw_symbols, (str, bytes)):
        raise TypeError(f"'{field_name}' must be a list of tickers")

    symbols: list[str] = []
    for symbol in raw_symbols:
        normalized = str(symbol).strip().upper()
        if not normalized or normalized in symbols:
            continue
        symbols.append(normalized)

    if required and not symbols:
        raise ValueError(f"'{field_name}' must contain at least one valid ticker")
    return symbols


def read_value_series(csv_path: Path, value_column: str) -> pd.Series:
    if not csv_path.exists():
        LOGGER.warning("Missing CSV file: %s", csv_path)
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(f"Failed to read CSV {csv_path}: {exc}") from exc

    if df.empty or "date" not in df.columns or value_column not in df.columns:
        LOGGER.warning("CSV %s missing required data columns", csv_path)
        return pd.Series(dtype=float)

    df = df[["date", value_column]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["date", value_column])
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")
    series = df.set_index("date")[value_column]
    series.index = pd.DatetimeIndex(series.index)  # ensure consistent dtype
    return series.astype(float)


def reindex_with_nearest(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if series.empty:
        return pd.Series([float("nan")] * len(index), index=index, dtype=float)

    cleaned = series.dropna()
    if cleaned.empty:
        return pd.Series([float("nan")] * len(index), index=index, dtype=float)

    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    try:
        reindexed = cleaned.reindex(index, method="nearest")
    except ValueError:
        # Fallback: ensure monotonic index before reindexing
        cleaned = cleaned.sort_index()
        reindexed = cleaned.reindex(index, method="nearest")
    return reindexed.astype(float)


def build_output_dataframe(
    dates: pd.DatetimeIndex,
    price_series: pd.Series,
    eps_series: pd.Series,
) -> pd.DataFrame:
    price_full = reindex_with_nearest(price_series, dates)
    eps_full = reindex_with_nearest(eps_series, dates)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "avg_price": price_full.to_numpy(dtype=float),
            "EPS": eps_full.to_numpy(dtype=float),
        }
    )

    timestamps = (pd.Series(dates).astype("int64") // 10**9).to_numpy(dtype="int64")
    df["timestamp"] = timestamps
    return df


def process_symbol(
    symbol: str,
    cfg: Dict[str, Any],
    date_index: pd.DatetimeIndex,
    raw_dir: Path,
    data_dir: Path,
    has_eps: bool = True,
) -> bool:
    price_path = raw_dir / f"{symbol}_price.csv"
    eps_path = raw_dir / f"{symbol}_eps.csv"

    price_series = read_value_series(price_path, "avg_price")
    if has_eps:
        eps_series = read_value_series(eps_path, "EPS")
    else:
        eps_series = pd.Series(dtype=float)

    if price_series.empty and eps_series.empty:
        LOGGER.warning("No data available for %s; skipping", symbol)
        return False

    df = build_output_dataframe(date_index, price_series, eps_series)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{symbol}.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s with %d rows", output_path, len(df))
    return True


def main() -> None:
    args = parse_args()
    configure_logging(args.quiet)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    ensure_required_keys(cfg, ["stock_symbols", "stock_start_date", "stock_end_date", "output_dir"])

    stock_symbols = normalize_symbols(cfg["stock_symbols"], "stock_symbols")
    etf_symbols = normalize_symbols(cfg.get("etf_symbols"), "etf_symbols", required=False)
    symbol_has_eps: dict[str, bool] = {}
    for symbol in stock_symbols:
        symbol_has_eps.setdefault(symbol, True)
    for symbol in etf_symbols:
        symbol_has_eps.setdefault(symbol, False)  # keep stock preference if symbol overlaps
    start_date = parse_date(cfg["stock_start_date"], "stock_start_date")
    end_date = parse_date(cfg["stock_end_date"], "stock_end_date")
    if end_date < start_date:
        raise ValueError("'stock_end_date' must not be earlier than 'stock_start_date'")

    date_index = pd.date_range(start=start_date, end=end_date, freq="D", tz=None, name="date")

    output_root = Path(cfg["output_dir"]).expanduser()
    raw_dir = output_root / "raw_data"
    data_dir = output_root / "data"

    any_success = False
    for symbol, has_eps in symbol_has_eps.items():
        try:
            if process_symbol(symbol, cfg, date_index, raw_dir, data_dir, has_eps=has_eps):
                any_success = True
        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", symbol, exc)

    if not any_success:
        raise SystemExit("No output files were created; check input data")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surfacing to CLI
        LOGGER.error("%s", exc)
        sys.exit(1)
