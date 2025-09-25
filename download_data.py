"""Fetch historical stock prices using yfinance and store daily averages."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download stock price data using yfinance")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return cfg


def ensure_required_keys(cfg: Dict, keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Config file is missing required keys: {', '.join(missing)}")


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Date '{date_str}' must use YYYY-MM-DD format") from exc


def fetch_stock_prices(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    end_inclusive = end + timedelta(days=1)
    data = yf.download(
        tickers=symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end_inclusive.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="column",
    )
    if data.empty:
        return pd.DataFrame(columns=["stock_symbol", "avg_price", "date"])

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)

    prices = data[["High", "Low", "Close"]].dropna()
    if prices.empty:
        return pd.DataFrame(columns=["stock_symbol", "avg_price", "date"])

    avg_price = (prices["High"] + prices["Low"] + prices["Close"]) / 3.0
    index = pd.to_datetime(prices.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)

    result = pd.DataFrame(
        {
            "stock_symbol": symbol,
            "avg_price": avg_price.astype(float),
            "date": index.strftime("%Y-%m-%d"),
        }
    )
    return result.sort_values("date").reset_index(drop=True)


def save_prices(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    required_keys = ["stock_symbols", "stock_start_date", "stock_end_date", "output_dir"]
    ensure_required_keys(cfg, required_keys)

    stock_symbols = cfg["stock_symbols"]
    if not isinstance(stock_symbols, (list, tuple)) or not stock_symbols:
        raise ValueError("'stock_symbols' must be a non-empty list")

    etf_symbols = cfg.get("etf_symbols", [])
    if etf_symbols is None:
        etf_symbols = []
    if not isinstance(etf_symbols, (list, tuple)):
        raise ValueError("'etf_symbols' must be a list when provided")

    all_symbols: List[str] = list(dict.fromkeys(list(stock_symbols) + list(etf_symbols)))

    start_date = parse_date(str(cfg["stock_start_date"]))
    end_date = parse_date(str(cfg["stock_end_date"]))
    if end_date < start_date:
        raise ValueError("'stock_end_date' must be on or after 'stock_start_date'")

    output_dir = Path(cfg["output_dir"]).expanduser()
    raw_data_dir = output_dir / "raw_data"

    for symbol in all_symbols:
        symbol_str = str(symbol).strip().upper()
        if not symbol_str:
            continue
        print(f"Fetching {symbol_str} from {start_date.date()} to {end_date.date()}")
        df = fetch_stock_prices(symbol_str, start_date, end_date)
        if df.empty:
            print(f"No data returned for {symbol_str}; skipping")
            continue
        output_path = raw_data_dir / f"{symbol_str}_price.csv"
        save_prices(df, output_path)
        print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
